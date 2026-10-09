"""M3's pilot (Plan 4 B7): the expert collects a dataset, the validator checks it, it is exported as RLDS, and the run
is recorded as M3's gate.

    env -u PYTHONPATH .venv/bin/python scripts/collect_dataset.py --scene s01 \\
        --model runs/expert/s01_r2/best/best_model.zip --scene-config scene_autofly_s01_fast.jsonc \\
        --name s01_pilot --episodes 100 --instance 1 [--check-python <python with tensorflow-datasets>]

- Writes data/<name>/ (the raw store, spec §10) and data/rejects/<name>/; refuses a name that already holds a dataset.
- `--scene-config` is required: collect on the clock the expert was trained and gated on (s01_r2: the 1 ms config).
- Seeds come from `--seed-slice` k of the collection range (400e6 + k * 1e6), disjoint from training, evaluation, the
  gate and the probes. One slice per (scene, collection run); 99 is for smokes and diagnostics.
- Stochastic by default (spec §9 step 5; docs/decisions/2026-10-02-m2-closeout.md); `--deterministic` for comparison.
- A mesh scene's footprint rule (expert/footprint_contact.py) is the shape the expert's run trained under, read from its
  sessions.json (expert/run_shape.py), as the gate reads it. `--footprint-shape` names it only for a model outside any
  run, and is refused where it contradicts the run's unless `--against-trained-shape` says that is the point. Recorded
  as `contact_rules` in the record and every episode's provenance (a pillar scene has none).
- `--targets assets/targets.json` (plan 5 C5): each episode's target and distractors are drawn from the scene split's
  pool, less what the scene's entry of the pool's exclusion table names (assets/target_exclusions.json, plan 6 step
  2c), and the instruction names the target; the card lists the scene's names with status "pool". Each episode's
  provenance records its target's and distractors' entries, their `stand_in` mark included (plan 6 step 2b), and
  `prewarm_steps`: the hover steps its reset held before the first record (every pool episode: under load a spawned
  object is drawn only frames after its spawn), with `first_use` saying whether it spawned an object new to the
  simulator session (plan 6 step 2a), and `first_frame_visibility`: whether that first recorded frame shows the target
  (visible, missing, occluded, occluded_unmodeled, out_of_view or undetermined; validate/visibility.py), counted in the
  record's collection summary, every `missing` said on stderr. Without it, M3's way: the orange cylinder under `--target-name` (default "orange
  cylinder", a placeholder in the card). Not both.
- The gate record (default docs/gates/m3_gate.json, never written over) passes when the requested episodes were kept,
  the validator passes, and the engine-fault audit is clean. The TFDS read-back is recorded next to it.
- `--no-export` skips the store's own RLDS export (and so any `--check-python` read-back, refused beside it): plan 6
  revision 1 exports one merged dataset from every store, and a copy per store would double the disk and the time.
  The record's `rlds` then says so, and the validator alone decides the store.

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
from autofly_ue5.collect.collector import POOL_NAME_STATUS, collect  # noqa: E402
from autofly_ue5.dataset.raw import RawDatasetWriter  # noqa: E402
from autofly_ue5.dataset.rlds import VERSION, export_rlds  # noqa: E402
from autofly_ue5.dataset.targets import TargetPool  # noqa: E402
from autofly_ue5.expert.footprint_contact import SHAPES, task_contact_rules  # noqa: E402
from autofly_ue5.expert.reward import REWARD_VERSION  # noqa: E402
from autofly_ue5.expert.run_shape import scored_shape  # noqa: E402
from autofly_ue5.expert.seeds import COLLECTION_SEED_BASE, collection_seed_base  # noqa: E402
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
EXPORT_SKIPPED = {"skipped": "merged export (plan 6 revision 1)"}  # the record's rlds under --no-export
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
        target_name: str | None = None, targets: Path | None = None, seed_base: int = COLLECTION_SEED_BASE,
        check_python: Path | None = None, device: str = "auto", sim_factory: Callable[[], Any] | None = None,
        load_model: Callable[[Path], Any] | None = None, obs_config_reader: Callable[[Path], Any] | None = None,
        sim_root: Path = SIM_RUN_DIR, platform: dict | None = None, footprint_shape: str | None = None,
        against_trained_shape: bool = False, export: bool = True) -> dict[str, Any]:
    if not export and check_python is not None:
        raise ValueError("--check-python reads the store's own RLDS export back, and --no-export skips that export")
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    run_start_epoch = time.time()
    xid_before = xid_count(run_started)
    boot_before = boot_id()
    resolved = resolve_scene(scene)
    # A mesh scene's footprint rule: the shape the expert's run trained under, `footprint_shape` only where the run
    # names none or `against_trained_shape` says so (expert/run_shape.py), and the outline's fields; read before
    # anything is claimed or launched. None on a pillar scene, whose record and provenance stay M3's, key for key.
    footprint_shape, trained_shape = scored_shape(resolved.layout, {"the expert": Path(model_path)}, footprint_shape,
                                                  against_trained=against_trained_shape)
    contact_rules = task_contact_rules(resolved.layout, shape=footprint_shape)
    if contact_rules is not None and trained_shape not in (None, footprint_shape):
        contact_rules["footprint"]["trained_shape"] = trained_shape
    # The target pool (plan 5 C5), read and checked against the scene's split before anything is claimed or launched.
    pool = TargetPool.load(targets) if targets is not None else None
    if pool is not None and target_name is not None:
        raise ValueError("--target-name is the pool-free fallback; with --targets each episode's entry names its target")
    if pool is None and target_name is None:
        target_name = DEFAULT_TARGET_NAME
    # As this scene draws from it: after its own exclusions (the pool's target_exclusions.json, plan 6 step 2c).
    pool_record = pool.to_json(resolved.scene.split, scene_id=resolved.scene.id) if pool is not None else None
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
        "status": "failed", "pass": False, "error": None, "collection": None, "validation": None,
        "rlds": None if export else dict(EXPORT_SKIPPED),
    }
    provenance = {"expert": record["expert"], "platform": platform if platform is not None else platform_provenance(),
                  "scene_config": record["scene_config"], "collector_git_head": record["collector_git_head"],
                  "collector_tree_dirty": record["collector_tree_dirty"], "reward_version": REWARD_VERSION,
                  "dataset": name}
    record["platform"] = provenance["platform"]
    if contact_rules is not None:  # what ended (or would have ended) each episode at a tree, and measured how
        record["contact_rules"] = provenance["contact_rules"] = contact_rules
    if pool_record is not None:  # only with a pool: a pool-free pilot's record and provenance are M3's, key for key
        record["target_pool"] = provenance["target_pool"] = pool_record
        card = {"target_names": {pool_name: POOL_NAME_STATUS for pool_name in pool_record["names"]}}
    else:
        card = {"target_names": {target_name: TARGET_NAME_STATUS}}
    swept = sweep_orphaned_instances(sim_root)  # a crashed earlier run's simulators; never a live run's
    if swept:
        print(f"swept orphaned instances before starting: {swept}", file=sys.stderr)
    env = writer = None
    progress: dict[str, Any] = {}  # collect()'s live counts: they survive an exception partway (the M3 review)
    try:
        writer = RawDatasetWriter(data_root, name, scene=resolved.scene, provenance=provenance, card=card)
        model = load_model(model_path)
        env = build_eval_env(scene, instance=instance, sim_factory=sim_factory, seed_base=seed_base, sim_root=sim_root,
                             obs_config=obs_config, target_pool=pool, footprint_shape=footprint_shape)
        t0 = time.monotonic()
        before = env.get_fault_summary()
        try:
            collect(model, env, scene=resolved.scene, writer=writer, seed_base=seed_base, n_keep=n_episodes,
                    target_name=target_name, deterministic=deterministic, max_attempts=max_attempts,
                    layout_sha256=resolved.layout_sha256, progress=progress,
                    target_name_status=TARGET_NAME_STATUS if pool is None else None)
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
        if export and writer.manifest["counts"]["episodes"]:  # whatever was kept is exported, even from a failed run
            try:
                record["rlds"] = export_rlds(root, Path(data_root), name)  # spec §10.1: data/<dataset_name>/1.0.0/
                if check_python is not None:
                    record["rlds"]["tfds_check"] = check_with_tfds(check_python, root / VERSION, root)
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
    # Under --no-export rlds is EXPORT_SKIPPED, which has no error: the skipped export is no failure, the validator decides
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
    p.add_argument("--seed-slice", type=int, default=0,
                   help="this collection run's slice of the collection seeds (expert/seeds.py): one per (scene, run); "
                        "0 = s01's pilot, 99 = smokes and diagnostics")
    p.add_argument("--max-attempts", type=int, default=None, help="seeds to fly at most (default 10 x --episodes)")
    p.add_argument("--instance", type=int, default=0,
                   help="the simulator slot; refused while another live run holds it (an --instances N training run "
                        "holds 0..N)")
    p.add_argument("--sim-root", type=Path, default=SIM_RUN_DIR, help=argparse.SUPPRESS)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=None, help="the gate record (default docs/gates/m3_gate.json)")
    p.add_argument("--target-name", default=None,
                   help=f"every episode's target name without --targets (default {DEFAULT_TARGET_NAME!r}, M3's placeholder)")
    p.add_argument("--targets", type=Path, default=None,
                   help="the target pool (assets/targets.json, plan 5 C5): each episode draws its target and distractors "
                        "from the scene split's usable entries, and the instruction names the target")
    p.add_argument("--footprint-shape", choices=SHAPES, default=None,
                   help="on a scene with mesh obstacles, what the footprint rule measures to: the model's outline or its "
                        "footprint circle. Default: the shape the expert's run trained under (its sessions.json, "
                        "expert/run_shape.py); needed only for a model outside any run, refused when it contradicts the "
                        "run's. Recorded in the record and each episode's provenance")
    p.add_argument("--against-trained-shape", action="store_true",
                   help="collect under --footprint-shape although the expert's run trained under the other shape; "
                        "contact_rules then names the shape it trained under as trained_shape")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--check-python", type=Path, default=None, help="a python with tensorflow-datasets, to read it back")
    p.add_argument("--no-export", action="store_true",
                   help="skip this store's own RLDS export (plan 6 revision 1 exports one merged dataset from every "
                        "store); the record's rlds says so and the validator still runs")
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_path = args.out if args.out is not None else evidence.GATES_DIR / "m3_gate.json"
    try:
        evidence.refuse_existing_evidence(out_path)
        if args.no_export and args.check_python is not None:
            raise ValueError("--check-python reads the store's own RLDS export back, and --no-export skips that export")
        seed_base = collection_seed_base(args.seed_slice)
        busy = slot_busy(args.instance, args.sim_root)
        if busy:
            raise ValueError(f"{busy}; pass a free --instance")
        if not args.model.is_file():
            raise FileNotFoundError(f"no checkpoint at {args.model}")
        # The expert's footprint shape (its run's, the flag's where it names none, or against it on purpose), and a mesh
        # asset without an outline field under the outline: offline, before anything launches.
        layout = resolve_scene(args.scene).layout
        footprint_shape, _trained = scored_shape(layout, {"the expert": args.model}, args.footprint_shape,
                                                 against_trained=args.against_trained_shape)
        task_contact_rules(layout, shape=footprint_shape)
        scene_config_record(args.scene_config)
        if (args.data_root / args.name / "manifest.json").exists():
            raise FileExistsError(f"{args.data_root / args.name} already holds a dataset; pick a new --name")
        checkpoint_obs_config(args.model)  # offline: an unreadable checkpoint is refused before anything launches
        if args.targets is not None:  # offline too: a pool the scene cannot draw from, after its exclusions
            if args.target_name is not None:
                raise ValueError("--target-name is the pool-free fallback; with --targets each episode's entry names its "
                                 "target")
            scene_file = resolve_scene(args.scene).scene
            TargetPool.load(args.targets).for_split(scene_file.split, scene_id=scene_file.id)
    except (FileNotFoundError, FileExistsError, ValueError) as err:
        print(f"refusing to start: {err}", file=sys.stderr)
        return 2
    record = run(scene=args.scene, model_path=args.model, scene_config=args.scene_config, name=args.name,
                 n_episodes=args.episodes, instance=args.instance, out_path=out_path, data_root=args.data_root,
                 deterministic=args.deterministic, max_attempts=args.max_attempts, target_name=args.target_name,
                 targets=args.targets, check_python=args.check_python, device=args.device, sim_root=args.sim_root,
                 seed_base=seed_base, footprint_shape=footprint_shape, against_trained_shape=args.against_trained_shape,
                 export=not args.no_export)
    print(json.dumps({"status": record["status"], "pass": record["pass"], "error": record["error"],
                      "kept": (record["collection"] or {}).get("kept"),
                      "validator_pass": (record["validation"] or {}).get("pass")}, indent=2))
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
