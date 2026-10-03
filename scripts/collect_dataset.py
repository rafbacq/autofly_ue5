"""M3's pilot (Plan 4 B7): the expert collects a dataset, the validator checks it, it is exported as RLDS, and the run
is recorded as M3's gate.

    env -u PYTHONPATH .venv/bin/python scripts/collect_dataset.py --scene s01 \\
        --model runs/expert/s01_r2/best/best_model.zip --scene-config scene_autofly_s01_fast.jsonc \\
        --name s01_pilot --episodes 100 --instance 1 [--check-python <python with tensorflow-datasets>]

- Writes data/<name>/ (the raw store, spec §10) and data/rejects/<name>/; refuses a name that already holds a dataset.
- `--scene-config` is required: collect on the clock the expert was trained and gated on (s01_r2: the 1 ms config).
- Seeds come from `COLLECTION_SEED_BASE` (400e6), disjoint from training, evaluation, the gate and the probes.
- Stochastic by default (spec §9 step 5; docs/decisions/2026-10-02-m2-closeout.md); `--deterministic` for comparison.
- The gate record (default docs/gates/m3_gate.json, never written over) passes when the requested episodes were kept,
  the validator passes, and the engine-fault audit is clean. The TFDS read-back is recorded next to it.

Simulators: one, on `--instance`, stopped through its own pid record at the end. Ends with `os._exit` like every
entry point that talks to projectairsim (its client leaves a non-daemon thread behind).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable  # noqa: E402

from autofly_ue5 import evidence  # noqa: E402
from autofly_ue5.collect.collector import collect  # noqa: E402
from autofly_ue5.dataset.raw import RawDatasetWriter  # noqa: E402
from autofly_ue5.dataset.rlds import VERSION, export_rlds  # noqa: E402
from autofly_ue5.expert.reward import REWARD_VERSION  # noqa: E402
from autofly_ue5.expert.seeds import COLLECTION_SEED_BASE  # noqa: E402
from autofly_ue5.expert.train import sha256_of  # noqa: E402
from autofly_ue5.expert.vec import teardown  # noqa: E402
from autofly_ue5.paths import PACKAGED_BINARY, ROOT, UE_PROJECT_DIR  # noqa: E402
from autofly_ue5.scenes.resolve import resolve_scene  # noqa: E402
from autofly_ue5.sim.airsim_backend import scene_config_factory, scene_config_record  # noqa: E402
from autofly_ue5.sim.process import SIM_RUN_DIR, instance_dir, slot_busy, stop_instances, sweep_orphaned_instances  # noqa: E402
from autofly_ue5.validate.dataset import validate_dataset  # noqa: E402
from autofly_ue5.validate.engine_check import audit_engine_faults, boot_id, xid_count  # noqa: E402
from scripts.export_rlds import check_with_tfds  # noqa: E402
from scripts.m2_gate import build_eval_env, checkpoint_obs_config, default_sac_loader  # noqa: E402

DEFAULT_TARGET_NAME = "orange cylinder"  # U3: s01's target until M4's pool exists
TARGET_NAME_STATUS = "placeholder_until_M4"  # U3 (docs/decisions/2026-10-03-m3-a0-and-collection.md)
PAK = UE_PROJECT_DIR / "Packaged" / "Development" / "Linux" / "Blocks" / "Content" / "Paks" / "Blocks-Linux.pak"


def _git(*args: str, cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def platform_provenance() -> dict:
    """What produced the frames (spec §10.2): the client library, the platform tree, the engine and the package."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        client = version("projectairsim")
    except PackageNotFoundError:
        client = None
    build = ROOT / "engine" / "Engine" / "Build" / "Build.version"
    return {
        "projectairsim_client": client,
        "platform_git_head": _git("rev-parse", "HEAD", cwd=ROOT / "platform") if (ROOT / "platform").is_dir() else None,
        "engine_build_version": json.loads(build.read_text()) if build.is_file() else None,
        "package_binary_sha256": sha256_of(PACKAGED_BINARY) if PACKAGED_BINARY.is_file() else None,
        "package_pak_sha256": sha256_of(PAK) if PAK.is_file() else None,
    }


def _write_exclusive(destination: Path, record: dict) -> Path:
    """Write the record without ever replacing a file: main() refuses an existing record at startup, but another pilot
    could write one while this one runs. A clash goes to a sibling `.conflict-<time>` file instead."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for candidate in (destination, destination.with_name(f"{destination.stem}.conflict-{stamp}{destination.suffix}"),
                      destination.with_name(f"{destination.stem}.conflict-{stamp}-{os.getpid()}{destination.suffix}")):
        record["written_to"] = str(candidate)
        try:
            with open(candidate, "x") as handle:
                handle.write(json.dumps(record, indent=2) + "\n")
        except FileExistsError:
            print(f"WARNING: {candidate} appeared while this pilot ran; not replacing it", file=sys.stderr)
            continue
        return candidate
    raise FileExistsError(f"could not write the record next to {destination}")


def run(*, scene: str, model_path: Path, scene_config: str, name: str, n_episodes: int, instance: int, out_path: Path,
        data_root: Path, deterministic: bool = False, max_attempts: int | None = None,
        target_name: str = DEFAULT_TARGET_NAME, seed_base: int = COLLECTION_SEED_BASE, check_python: Path | None = None,
        device: str = "auto", sim_factory: Callable[[], Any] | None = None, load_model: Callable[[Path], Any] | None = None,
        obs_config_reader: Callable[[Path], Any] | None = None, sim_root: Path = SIM_RUN_DIR,
        platform: dict | None = None) -> dict[str, Any]:
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    run_start_epoch = time.time()
    xid_before = xid_count(run_started)
    boot_before = boot_id()
    resolved = resolve_scene(scene)
    sim_factory = sim_factory or scene_config_factory(scene_config, resolved.movable_objects, run_root=sim_root)
    load_model = load_model or (lambda p: default_sac_loader(p, device=device))
    obs_config = (obs_config_reader or checkpoint_obs_config)(model_path)
    record: dict[str, Any] = {
        "milestone": "M3",
        "description": "M3 pilot: the SAC expert collects AutoFly-format episodes; the dataset validator (spec §11) "
                       "must pass. Pass = the requested episodes kept, validator pass, engine-fault audit clean.",
        "scene": scene, "scene_config": scene_config_record(scene_config), "dataset": {"name": name,
                                                                                        "path": str(Path(data_root) / name)},
        "expert": {"path": str(model_path), "sha256": sha256_of(model_path), "deterministic": deterministic,
                   "obs_config": obs_config.to_json()},
        "episodes_requested": n_episodes, "seed_base": seed_base, "instance": instance, "target_name": target_name,
        "a0": "sector8", "reward_version": REWARD_VERSION, "collector_git_head": _git("rev-parse", "HEAD"),
        "collector_tree_dirty": bool(_git("status", "--porcelain")), "run_started": run_started,
        "status": "failed", "pass": False, "error": None, "collection": None, "validation": None, "rlds": None,
    }
    provenance = {"expert": record["expert"], "platform": platform if platform is not None else platform_provenance(),
                  "scene_config": record["scene_config"], "collector_git_head": record["collector_git_head"],
                  "collector_tree_dirty": record["collector_tree_dirty"], "reward_version": REWARD_VERSION,
                  "dataset": name}
    record["platform"] = provenance["platform"]
    swept = sweep_orphaned_instances(sim_root)  # a crashed earlier run's simulators; never a live run's
    if swept:
        print(f"swept orphaned instances before starting: {swept}", file=sys.stderr)
    env = writer = None
    progress: dict[str, Any] = {}  # collect()'s live counts: they survive an exception partway (the M3 review)
    try:
        writer = RawDatasetWriter(data_root, name, scene=resolved.scene, provenance=provenance,
                                  card={"target_names": {target_name: TARGET_NAME_STATUS}})
        model = load_model(model_path)
        env = build_eval_env(scene, instance=instance, sim_factory=sim_factory, seed_base=seed_base, sim_root=sim_root,
                             obs_config=obs_config)
        t0 = time.monotonic()
        before = env.get_fault_summary()
        try:
            collect(model, env, scene=resolved.scene, writer=writer, seed_base=seed_base, n_keep=n_episodes,
                    target_name=target_name, deterministic=deterministic, max_attempts=max_attempts,
                    layout_sha256=resolved.layout_sha256, progress=progress, target_name_status=TARGET_NAME_STATUS)
        finally:
            progress["wall_s"] = round(time.monotonic() - t0, 1)
            try:
                progress["backend_faults"] = {"before": before, "after": env.get_fault_summary()}
            except Exception as err:
                progress["backend_faults"] = {"error": f"{type(err).__name__}: {err}"}
        record["status"] = progress["status"]
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"
        print(f"collect_dataset failed: {record['error']}", file=sys.stderr)
        traceback.print_exc()
    finally:
        try:
            if env is not None:
                teardown(env, [instance], sim_root)  # bounded close + stop this run's own slot, nothing else
            else:
                stop_instances([instance], sim_root)
        except Exception as err:
            print(f"WARNING: teardown raised {type(err).__name__}: {err}", file=sys.stderr)
    record["collection"] = progress or None
    kept, attempted = progress.get("kept", 0), progress.get("attempted", 0)
    root = Path(data_root) / name
    if writer is not None and writer.claimed:  # only a store this run created: never report on someone else's dataset
        try:
            record["validation"] = validate_dataset(root)
        except Exception as err:
            record["validation"] = {"pass": False, "failures": [f"the validator raised {type(err).__name__}: {err}"]}
        if writer.manifest["counts"]["episodes"]:  # whatever was kept is exported, even from a failed run
            try:
                dataset = f"autofly_ue5_{name}"
                record["rlds"] = export_rlds(root, root / "rlds", dataset)
                if check_python is not None:
                    record["rlds"]["tfds_check"] = check_with_tfds(check_python, root / "rlds" / dataset / VERSION, root)
            except Exception as err:
                record["rlds"] = {"error": f"{type(err).__name__}: {err}"}
    try:
        engine_faults = audit_engine_faults(since=run_started, since_epoch=run_start_epoch, xid_before=xid_before,
                                            boot_before=boot_before, log_dirs=[instance_dir(instance, sim_root)])
        record["faults_ok"] = engine_faults.pop("ok")
    except Exception as err:
        engine_faults = {"error": f"the audit raised {type(err).__name__}: {err}"}
        record["faults_ok"] = False
    record["engine_faults"] = engine_faults
    rlds = record["rlds"] or {}
    record["pass"] = bool(record["status"] == "ok" and kept >= n_episodes and (record["validation"] or {}).get("pass")
                          and record["faults_ok"] and rlds and "error" not in rlds
                          and (check_python is None or (rlds.get("tfds_check") or {}).get("pass") is True))
    record["run_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # Evidence once any episode was flown, kept or not: an all-rejected pilot is a failed gate, not one never started.
    destination = evidence.record_destination(out_path, did_work=attempted > 0)
    written = _write_exclusive(destination, record)
    print(f"record written to {written}", file=sys.stderr)
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="s01")
    p.add_argument("--model", type=Path, required=True, help="the expert checkpoint (M2 closeout: s01_r2 best_model)")
    p.add_argument("--scene-config", required=True, help="the Project AirSim config the expert was trained and gated on")
    p.add_argument("--name", required=True, help="the dataset's directory under --data-root; never written over")
    p.add_argument("--episodes", type=int, default=100, help="successful episodes to keep")
    p.add_argument("--max-attempts", type=int, default=None, help="seeds to fly at most (default 10 x --episodes)")
    p.add_argument("--instance", type=int, default=0,
                   help="the simulator slot; refused while another live run holds it (an --instances N training run "
                        "holds 0..N)")
    p.add_argument("--sim-root", type=Path, default=SIM_RUN_DIR, help=argparse.SUPPRESS)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=None, help="the gate record (default docs/gates/m3_gate.json)")
    p.add_argument("--target-name", default=DEFAULT_TARGET_NAME)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--check-python", type=Path, default=None, help="a python with tensorflow-datasets, to read it back")
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_path = args.out if args.out is not None else evidence.GATES_DIR / "m3_gate.json"
    try:
        evidence.refuse_existing_evidence(out_path)
        busy = slot_busy(args.instance, args.sim_root)
        if busy:
            raise ValueError(f"{busy}; pass a free --instance")
        resolve_scene(args.scene)
        scene_config_record(args.scene_config)
        if not args.model.is_file():
            raise FileNotFoundError(f"no checkpoint at {args.model}")
        if (args.data_root / args.name / "manifest.json").exists():
            raise FileExistsError(f"{args.data_root / args.name} already holds a dataset; pick a new --name")
        checkpoint_obs_config(args.model)  # offline: an unreadable checkpoint is refused before anything launches
    except (FileNotFoundError, FileExistsError, ValueError) as err:
        print(f"refusing to start: {err}", file=sys.stderr)
        return 2
    record = run(scene=args.scene, model_path=args.model, scene_config=args.scene_config, name=args.name,
                 n_episodes=args.episodes, instance=args.instance, out_path=out_path, data_root=args.data_root,
                 deterministic=args.deterministic, max_attempts=args.max_attempts, target_name=args.target_name,
                 check_python=args.check_python, device=args.device, sim_root=args.sim_root)
    print(json.dumps({"status": record["status"], "pass": record["pass"], "error": record["error"],
                      "kept": (record["collection"] or {}).get("kept"),
                      "validator_pass": (record["validation"] or {}).get("pass")}, indent=2))
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
