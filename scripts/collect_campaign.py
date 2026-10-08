"""A collection campaign (plan 6 step 2e): one scripts/collect_dataset.py run per (scene, batch), each into its own
store, on the simulator slots the campaign names, one job per slot at a time.

    PY="env -u PYTHONPATH .venv/bin/python"
    $PY scripts/collect_campaign.py plan --campaign runs/collect/campaign_thin.json --episodes 50 --batches 1 \\
        --slots 6 7 8 9 --pilot s01=runs/expert/s01_r2/best/best_model.zip \\
        --pilot s02=runs/expert/s02_r1/averaged/avg_350k_end.zip ...
    $PY scripts/collect_campaign.py launch --campaign runs/collect/campaign_thin.json [--dry-run] [--retry]
    $PY scripts/collect_campaign.py status --campaign runs/collect/campaign_thin.json

plan (offline: it writes the campaign file and nothing else, never over an existing one). One job per (scene, batch):
the scenes in expert/seeds.py's CAMPAIGN_SCENE_ORDER, batches --first-batch to --first-batch + --batches - 1, the
--slots handed out round robin over the jobs in that order. Each job is one collect_dataset.py command:
  --scene; --model (the scene's --pilot); --scene-config scene_autofly_<base>_fast.jsonc (the 1 ms clock the experts
  were trained and gated on, for the level the scene flies: s01d flies s01's, scenes/resolve.py); --name
  <scene>_c<batch>; --episodes N; --max-attempts (default 4 x N, against the collector's own 10 x N); --seed-slice
  campaign_seed_slice(scene, batch) (expert/seeds.py: 1 + 10 x the scene's place in the order + batch); --instance
  <slot>; --targets; --out <records dir>/<name>.json; --no-export (plan 6 revision 1 exports one merged dataset, not a
  copy per store); and on a scene with mesh obstacles --footprint-shape outline, the task's rule, with
  --against-trained-shape where the pilot's run trained under the circle (expert/run_shape.py: s02_r1). A pilot whose
  run had no footprint rule (the s01 and s01d experts') or no run record needs only the shape.
  Refused before anything is written: a scene outside the order or named twice, a pilot that is not a file, a scene
  config or an outline field that does not exist, a pool the scene cannot draw from after its exclusions, a store,
  rejects or a record already under a job's name, a slice outside 1-98 or one that a store under the data root or a
  record in the records dir already flew, a batch past the scene's ten, a slot that is negative or named twice.
launch: starts the jobs it may, in plan order, with `scripts/run_job.sh start collect_<name> -- <command>` from the
  repository root, --interval-s (30 s) apart. Not a job whose store exists, whose job runs, or whose job ended (with
  --retry, one that ended without a store, refused at startup say, starts again). Never on a slot whose
  runs/sim/inst<k>/pid.json names a live simulator or a live run (a training run's evaluation slot is down between
  evaluations while its trainer lives) or cannot be read (look at it first), nor on a slot a job of this campaign
  holds: a slot runs one job at a time, so run launch again as batches end. --dry-run says what it would start.
status: per job, its state (pending, running, finished and its exit code, stopped, died) and its store's counts from
  its manifest and its rejects: kept of those asked, flown, rejected by reason; then per scene and in all. Then the
  first-frame checks (validate/visibility.py) of every episode, kept or rejected, from its provenance: counts of each
  status per store and per target, and every `missing` episode by name.

The campaign file holds every job's exact command, so what runs is on disk before anything runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from typing import Any  # noqa: E402

from autofly_ue5.dataset.raw import NAME_RE  # noqa: E402
from autofly_ue5.dataset.targets import TargetPool  # noqa: E402
from autofly_ue5.durable import write_text_durably  # noqa: E402
from autofly_ue5.expert.footprint_contact import has_footprint_obstacles, task_contact_rules  # noqa: E402
from autofly_ue5.expert.run_shape import scored_shape, trained_shape  # noqa: E402
from autofly_ue5.expert.seeds import (  # noqa: E402
    CAMPAIGN_SCENE_ORDER,
    COLLECTION_SEED_BASE,
    COLLECTION_SLICE_STRIDE,
    DIAGNOSTIC_COLLECTION_SLICE,
    MAX_COLLECTION_SLICES,
    campaign_seed_slice,
    collection_seed_base,
)
from autofly_ue5.paths import CONFIGS_DIR, JOBS_DIR, ROOT, RUNS_DIR  # noqa: E402
from autofly_ue5.scenes.resolve import resolve_scene  # noqa: E402
from autofly_ue5.sim.process import (  # noqa: E402
    SIM_RUN_DIR,
    UNREADABLE_RECORD_ERRORS,
    RunOwner,
    instance_dir,
    is_alive,
    read_pid_file,
)
from autofly_ue5.validate.visibility import STATUSES as FIRST_FRAME_STATUSES  # noqa: E402

VERSION = 1
DATA_ROOT = ROOT / "data"  # collect_dataset.py's default --data-root
RECORDS_DIR = RUNS_DIR / "collect"
TARGETS = ROOT / "assets" / "targets.json"
MAX_ATTEMPTS_FACTOR = 4  # seeds flown at most per kept episode asked: a kept-rate of 0.25, plan 6 revision 1's floor
INTERVAL_S = 30.0  # between two launches: two simulators' first resets at once race the GPU guard (expert/vec.py)
COMMAND_HEAD = ("env", "-u", "PYTHONPATH", ".venv/bin/python", "scripts/collect_dataset.py")
FOOTPRINT_SHAPE = "outline"  # the task's rule on a mesh scene (expert/footprint_contact.py)
FIRST_SLICE, LAST_SLICE = 1, DIAGNOSTIC_COLLECTION_SLICE - 1  # slice 0 is the M3 pilot's, 99 the smokes'


class CampaignError(ValueError):
    """A campaign that cannot be planned or read as it stands."""


def _rel(path: Path) -> str:
    """`path` relative to the repository root when it lies inside it (the jobs run from there), else absolute."""
    path = Path(path).absolute()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _slice_of(seed: Any) -> int | None:
    if isinstance(seed, int) and COLLECTION_SEED_BASE <= seed < collection_seed_base(MAX_COLLECTION_SLICES - 1) \
            + COLLECTION_SLICE_STRIDE:
        return (seed - COLLECTION_SEED_BASE) // COLLECTION_SLICE_STRIDE
    return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def used_slices(data_root: Path, records_dir: Path) -> dict[int, str]:
    """{collection slice: what flew it} over every store under `data_root` (a store's seeds all lie in one slice: its
    first episode's, or its first reject's) and every collection record in `records_dir` (its seed_base)."""
    used: dict[int, str] = {}
    root = Path(data_root)
    for manifest in sorted(root.glob("*/manifest.json")):
        episodes = (_read_json(manifest) or {}).get("episodes") or []
        if episodes and _slice_of(episodes[0].get("seed")) is not None:
            used.setdefault(_slice_of(episodes[0]["seed"]), f"store {manifest.parent}")
    for rejects in sorted(p for p in (root / "rejects").glob("*") if p.is_dir()):
        for reject in sorted(rejects.glob("*.json")):
            seed = ((_read_json(reject) or {}).get("provenance") or {}).get("seed")
            if _slice_of(seed) is not None:
                used.setdefault(_slice_of(seed), f"the rejects in {rejects}")
                break
    for record in sorted(Path(records_dir).glob("*.json")):
        data = _read_json(record)
        if isinstance(data, dict) and _slice_of(data.get("seed_base")) is not None:
            used.setdefault(_slice_of(data["seed_base"]), f"the collection record {record}")
    return used


def _footprint(resolved, model: Path) -> dict | None:
    """The footprint flags of a scene's jobs: none on a scene without mesh obstacles; else the task's outline, against
    the pilot's run where it trained under the circle. Checked as collect_dataset.py will check it, and the outline's
    fields read, before the campaign is written."""
    if not has_footprint_obstacles(resolved.layout):
        return None
    trained, _sessions = trained_shape(model)
    against = trained not in (None, FOOTPRINT_SHAPE)
    scored_shape(resolved.layout, {"the pilot": model}, FOOTPRINT_SHAPE, against_trained=against)
    task_contact_rules(resolved.layout, shape=FOOTPRINT_SHAPE)
    return {"shape": FOOTPRINT_SHAPE, "trained_shape": trained, "against_trained_shape": against}


def plan_campaign(pilots: dict[str, Path], *, episodes: int, batches: int, first_batch: int, slots: list[int],
                  max_attempts: int | None = None, targets: Path = TARGETS, data_root: Path = DATA_ROOT,
                  records_dir: Path = RECORDS_DIR) -> dict:
    """The campaign file's content (see the module's doc), every refusal raised as a CampaignError (or the error of the
    check that refused) before anything is written."""
    unknown = [scene for scene in pilots if scene not in CAMPAIGN_SCENE_ORDER]
    if unknown:
        raise CampaignError(f"{unknown}: not campaign scenes; a campaign collects {CAMPAIGN_SCENE_ORDER} "
                            f"(expert/seeds.py, which gives each its seed slices)")
    if not slots:
        raise CampaignError("name at least one slot")
    if len(set(slots)) != len(slots) or any(slot < 0 for slot in slots):
        raise CampaignError(f"slots {slots}: each slot once, none negative")
    if episodes < 1 or batches < 1 or first_batch < 0:
        raise CampaignError(f"--episodes {episodes}, --batches {batches}, --first-batch {first_batch}: at least 1, "
                            f"1 and 0")
    max_attempts = MAX_ATTEMPTS_FACTOR * episodes if max_attempts is None else max_attempts
    if max_attempts < episodes:
        raise CampaignError(f"--max-attempts {max_attempts} is below --episodes {episodes}: no job could keep them all")
    data_root, records_dir = Path(data_root).absolute(), Path(records_dir).absolute()
    used = used_slices(data_root, records_dir)
    pool = TargetPool.load(targets)
    scenes: dict[str, dict] = {}
    jobs: list[dict] = []
    for scene_id in (s for s in CAMPAIGN_SCENE_ORDER if s in pilots):
        model = Path(pilots[scene_id])
        if not model.is_file():
            raise CampaignError(f"{scene_id}: no pilot at {model}")
        resolved = resolve_scene(scene_id)
        config = f"scene_autofly_{resolved.base_id}_fast.jsonc"
        if not (CONFIGS_DIR / config).is_file():
            raise CampaignError(f"{scene_id}: no scene config {CONFIGS_DIR / config} (the level it flies is "
                                f"{resolved.base_id}'s)")
        pool.for_split(resolved.scene.split, scene_id=scene_id)  # what the scene draws, after its exclusions
        footprint = _footprint(resolved, model)
        scene_slots = []
        for batch in range(first_batch, first_batch + batches):
            seed_slice = campaign_seed_slice(scene_id, batch)
            if not FIRST_SLICE <= seed_slice <= LAST_SLICE:
                raise CampaignError(f"{scene_id} batch {batch}: slice {seed_slice} is outside "
                                    f"{FIRST_SLICE}-{LAST_SLICE} (0 is the M3 pilot's, 99 the smokes')")
            if seed_slice in used:
                raise CampaignError(f"{scene_id} batch {batch}: slice {seed_slice} was already flown by "
                                    f"{used[seed_slice]}; plan from a later --first-batch")
            name = f"{scene_id}_c{batch}"
            if not NAME_RE.match(name):
                raise CampaignError(f"{name!r} is not a dataset name collect_dataset.py accepts")
            store, rejects, record = data_root / name, data_root / "rejects" / name, records_dir / f"{name}.json"
            if store.exists():
                raise CampaignError(f"{store} exists: {name} was collected (or begun) already")
            if rejects.is_dir() and any(rejects.iterdir()):
                raise CampaignError(f"{rejects} holds rejects: {name} was begun already")
            if record.exists():
                raise CampaignError(f"{record} exists: {name} was run already")
            slot = slots[len(jobs) % len(slots)]
            scene_slots.append(slot)
            argv = [*COMMAND_HEAD, "--scene", scene_id, "--model", _rel(model), "--scene-config", config,
                    "--name", name, "--episodes", str(episodes), "--max-attempts", str(max_attempts),
                    "--seed-slice", str(seed_slice), "--instance", str(slot), "--targets", _rel(targets),
                    "--out", _rel(record), "--no-export"]
            if data_root != DATA_ROOT.absolute():
                argv += ["--data-root", _rel(data_root)]
            if footprint is not None:
                argv += ["--footprint-shape", footprint["shape"]]
                if footprint["against_trained_shape"]:
                    argv.append("--against-trained-shape")
            jobs.append({"scene": scene_id, "batch": batch, "name": name, "job": f"collect_{name}", "slot": slot,
                         "seed_slice": seed_slice, "seed_base": collection_seed_base(seed_slice), "episodes": episodes,
                         "max_attempts": max_attempts, "store": _rel(store), "record": _rel(record), "argv": argv,
                         "command": shlex.join(argv)})
        scenes[scene_id] = {"pilot": _rel(model), "pilot_sha256": _sha256(model), "scene_config": config,
                            "split": resolved.scene.split, "episodes_per_batch": episodes, "batches": batches,
                            "first_batch": first_batch, "slots": scene_slots, "footprint": footprint}
    return {"version": VERSION, "kind": "autofly_ue5 collection campaign (scripts/collect_campaign.py)",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "targets": _rel(targets), "targets_sha256": pool.sha256,
            "data_root": _rel(data_root), "records_dir": _rel(records_dir), "episodes_per_batch": episodes,
            "max_attempts": max_attempts, "slots": list(slots), "scenes": scenes, "jobs": jobs}


def load_campaign(path: Path) -> dict:
    campaign = _read_json(path)
    if not isinstance(campaign, dict) or campaign.get("version") != VERSION or not isinstance(campaign.get("jobs"), list):
        raise CampaignError(f"{path}: not a version-{VERSION} campaign file (scripts/collect_campaign.py plan writes one)")
    return campaign


# ----------------------------------------------------------------------------------------------------------------------
# What runs where: scripts/run_job.sh's job records, and each slot's simulator record.
# ----------------------------------------------------------------------------------------------------------------------
def jobs_dir_default() -> Path:
    """Where scripts/run_job.sh keeps its records (its AUTOFLY_JOBS_DIR, default runs/jobs)."""
    return Path(os.environ.get("AUTOFLY_JOBS_DIR", JOBS_DIR))


def job_state(job: str, jobs_dir: Path) -> dict:
    """A run_job.sh job's state from its records: pending (never started), running (its recorded pid, the job's bash
    wrapper, is alive), finished (its exit code), stopped (run_job.sh stop), died (gone without an exit code: a host
    crash), unreadable."""
    jobs_dir = Path(jobs_dir)
    exit_file, pid_file = jobs_dir / f"{job}.exit", jobs_dir / f"{job}.pid.json"
    if exit_file.is_file():
        text = exit_file.read_text().strip()
        return {"state": "finished", "exit": int(text) if text.lstrip("-").isdigit() else text}
    if not pid_file.is_file():
        return {"state": "pending"}
    if (jobs_dir / f"{job}.stopped").is_file():
        return {"state": "stopped"}
    try:
        pid = int(json.loads(pid_file.read_text())["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return {"state": "unreadable"}
    return {"state": "running", "pid": pid} if is_alive(pid) else {"state": "died", "pid": pid}


def slot_taken(instance: int, sim_root: Path = SIM_RUN_DIR) -> str | None:
    """Why no collector may start on slot `instance` now, or None. Its runs/sim/inst<k>/pid.json names a live simulator,
    or a live run whose simulator is down between uses (an --instances N training run's evaluation slot N between
    evaluations), or cannot be read (a host freeze left one at 0 bytes, MEMORY.md: look at it before using the slot)."""
    path = instance_dir(instance, sim_root) / "pid.json"
    try:
        record = read_pid_file(path)
    except FileNotFoundError:
        return None
    except UNREADABLE_RECORD_ERRORS as err:
        return f"slot {instance}'s record {path} is unreadable ({type(err).__name__}): look at it before using the slot"
    if is_alive(record.pid):
        return f"slot {instance} holds a live simulator (pid {record.pid})"
    if record.owner_pid is not None and RunOwner(record.owner_pid, record.owner_start_ticks).is_alive():
        return f"slot {instance} belongs to the live run pid {record.owner_pid} (its simulator is down between uses)"
    return None


def start_job(job: str, argv: list[str]) -> tuple[bool, str]:
    """`scripts/run_job.sh start <job> -- <argv>` from the repository root (the jobs' paths are relative to it)."""
    result = subprocess.run(["bash", str(ROOT / "scripts" / "run_job.sh"), "start", job, "--", *argv], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def launch(campaign: dict, *, interval_s: float = INTERVAL_S, dry_run: bool = False, retry: bool = False,
           sim_root: Path = SIM_RUN_DIR, jobs_dir: Path | None = None) -> list[dict]:
    """Start what may start now (see the module's doc); returns one row per job: its name, slot and what was done."""
    jobs_dir = Path(jobs_dir) if jobs_dir is not None else jobs_dir_default()
    data_root = _abs(campaign["data_root"])
    states = {job["job"]: job_state(job["job"], jobs_dir) for job in campaign["jobs"]}
    held = {job["slot"]: job["job"] for job in campaign["jobs"] if states[job["job"]]["state"] in ("running", "unreadable")}
    rows, started = [], 0
    for job in campaign["jobs"]:
        state, slot = states[job["job"]], job["slot"]
        has_store = (data_root / job["name"] / "manifest.json").is_file()
        if state["state"] in ("running", "unreadable"):
            verdict = f"{state['state']}" + (f" (pid {state['pid']})" if "pid" in state else "")
        elif has_store:
            verdict = f"collected: {data_root / job['name']} exists"
        elif state["state"] != "pending" and not retry:
            ended = f"finished with exit {state['exit']}" if state["state"] == "finished" else state["state"]
            verdict = f"{ended} without a store; see {jobs_dir / (job['job'] + '.log')}, --retry starts it again"
        elif slot in held:
            verdict = f"waiting: slot {slot} is held by {held[slot]}"
        elif (reason := slot_taken(slot, sim_root)) is not None:
            verdict = f"waiting: {reason}"
        elif dry_run:
            verdict = f"would start {job['job']}: {job['command']}"
            held[slot] = job["job"]
        else:
            if started:
                time.sleep(interval_s)
            ok, output = start_job(job["job"], job["argv"])
            started += 1
            if ok:
                held[slot] = job["job"]
                verdict = f"started {job['job']} ({output})"
            else:
                verdict = f"FAILED to start {job['job']}: {output}"
        rows.append({"name": job["name"], "job": job["job"], "slot": slot, "verdict": verdict})
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# status
# ----------------------------------------------------------------------------------------------------------------------
def store_counts(data_root: Path, name: str) -> dict:
    """A store's counts from its manifest (kept) and its rejects (one file each, with its reason)."""
    manifest = _read_json(Path(data_root) / name / "manifest.json") or {}
    kept = int((manifest.get("counts") or {}).get("episodes", 0))
    reasons = Counter()
    for reject in sorted((Path(data_root) / "rejects" / name).glob("*.json")):
        reasons[str((_read_json(reject) or {}).get("reason"))] += 1
    rejected = sum(reasons.values())
    return {"store": bool(manifest), "kept": kept, "rejected": rejected, "flown": kept + rejected,
            "rejected_by_reason": dict(sorted(reasons.items()))}


def store_first_frames(data_root: Path, name: str) -> tuple[Counter, dict[str, Counter], list[str]]:
    """A store's first-frame checks (validate/visibility.py) from its episodes' provenance, kept and rejected: (counts
    by status, counts by status per target name, the ids of the episodes whose first frame missed its target)."""
    counts, by_target, missing = Counter(), {}, []
    kept = sorted((Path(data_root) / name / "provenance").glob("*.json"))
    rejected = sorted((Path(data_root) / "rejects" / name).glob("*.json"))
    for path, is_reject in [*((p, False) for p in kept), *((p, True) for p in rejected)]:
        data = _read_json(path) or {}
        provenance = (data.get("provenance") or {}) if is_reject else data
        record = provenance.get("first_frame_visibility")
        if not isinstance(record, dict):
            continue  # a pool-free episode records none
        status = str(record.get("status"))
        target = str((provenance.get("target") or {}).get("name"))
        counts[status] += 1
        by_target.setdefault(target, Counter())[status] += 1
        if status == "missing":
            missing.append(f"{path.stem} ({target})")
    return counts, by_target, missing


def _ordered(counts: Counter) -> dict[str, int]:
    """Counts by first-frame status, in validate/visibility.py's order, an unknown status after them."""
    known = [s for s in FIRST_FRAME_STATUSES if counts.get(s)]
    return {s: counts[s] for s in [*known, *sorted(s for s in counts if s not in FIRST_FRAME_STATUSES)]}


def _totals(rows: list[dict]) -> dict:
    reasons = Counter()
    for row in rows:
        reasons.update(row["rejected_by_reason"])
    return {"kept": sum(r["kept"] for r in rows), "episodes": sum(r["episodes"] for r in rows),
            "flown": sum(r["flown"] for r in rows), "rejected": sum(r["rejected"] for r in rows),
            "rejected_by_reason": dict(sorted(reasons.items()))}


def campaign_status(campaign: dict, *, jobs_dir: Path | None = None) -> dict:
    jobs_dir = Path(jobs_dir) if jobs_dir is not None else jobs_dir_default()
    data_root, records_dir = _abs(campaign["data_root"]), _abs(campaign.get("records_dir", RECORDS_DIR))
    rows, targets = [], {}
    for job in campaign["jobs"]:
        record = _read_json(records_dir / f"{job['name']}.json")
        first_frame, by_target, missing = store_first_frames(data_root, job["name"])
        for target, counts in by_target.items():
            targets.setdefault(target, Counter()).update(counts)
        rows.append({"name": job["name"], "scene": job["scene"], "batch": job["batch"], "slot": job["slot"],
                     "seed_slice": job.get("seed_slice"), "episodes": job["episodes"],
                     **job_state(job["job"], jobs_dir), **store_counts(data_root, job["name"]),
                     "first_frame": _ordered(first_frame), "first_frame_missing": missing,
                     "record": ({k: record.get(k) for k in ("status", "pass", "error")}
                                if isinstance(record, dict) else None)})
    scenes = {scene: _totals([r for r in rows if r["scene"] == scene]) for scene in dict.fromkeys(r["scene"] for r in rows)}
    return {"jobs": rows, "scenes": scenes, "all": _totals(rows),
            "targets": {name: _ordered(counts) for name, counts in sorted(targets.items())}}


def _reasons(counts: dict) -> str:
    return f" ({', '.join(f'{reason} {n}' for reason, n in counts.items())})" if counts else ""


def print_status(summary: dict) -> None:
    print("store      slot slice state                 kept      flown  rejected")
    for row in summary["jobs"]:
        state = row["state"] + (f" {row['exit']}" if row["state"] == "finished" else "")
        record = row["record"]
        verdict = f"  record: {record['status']}, pass {record['pass']}" if record else ""
        print(f"{row['name']:10s} {row['slot']:<4d} {row['seed_slice'] if row['seed_slice'] is not None else '-':<5} "
              f"{state:21s} kept {row['kept']}/{row['episodes']}  flown {row['flown']}  rejected {row['rejected']}"
              f"{_reasons(row['rejected_by_reason'])}{verdict}")
    for scene, totals in summary["scenes"].items():
        print(f"{scene}: kept {totals['kept']}/{totals['episodes']}, flown {totals['flown']}, rejected "
              f"{totals['rejected']}{_reasons(totals['rejected_by_reason'])}")
    totals = summary["all"]
    print(f"all: kept {totals['kept']}/{totals['episodes']}, flown {totals['flown']}, rejected {totals['rejected']}"
          f"{_reasons(totals['rejected_by_reason'])}")
    if not summary["targets"]:
        return
    print("first frames (validate/visibility.py), kept and rejected, per store:")
    for row in summary["jobs"]:
        if row["first_frame"]:
            print(f"{row['name']} first frame: {', '.join(f'{s} {n}' for s, n in row['first_frame'].items())}")
    print("per target:")
    width = max(len(name) for name in summary["targets"]) + 2
    for name, counts in summary["targets"].items():
        print(f"  {name:{width}s}  {', '.join(f'{s} {n}' for s, n in counts.items())}")
    for row in summary["jobs"]:
        for episode in row["first_frame_missing"]:  # every miss by name: the failure the first-use hold guards against
            print(f"MISSING {row['name']} {episode}")


# ----------------------------------------------------------------------------------------------------------------------
def _pilot_arg(text: str) -> tuple[str, Path]:
    scene, sep, path = text.partition("=")
    if not sep or not scene or not path:
        raise argparse.ArgumentTypeError(f"{text!r}: give SCENE=CHECKPOINT")
    return scene, Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="write the campaign file: one collect_dataset.py command per (scene, batch)")
    plan.add_argument("--campaign", type=Path, required=True, help="the campaign file to write (never over one)")
    plan.add_argument("--pilot", type=_pilot_arg, action="append", required=True, metavar="SCENE=CHECKPOINT",
                      help="a scene and the checkpoint that collects it; once per scene")
    plan.add_argument("--episodes", type=int, required=True, help="kept episodes per batch")
    plan.add_argument("--batches", type=int, default=1, help="batches per scene (each its own store and slice)")
    plan.add_argument("--first-batch", type=int, default=0, help="the first batch's index (a later campaign tops up)")
    plan.add_argument("--slots", type=int, nargs="+", required=True, help="the simulator slots the jobs take in turn")
    plan.add_argument("--max-attempts", type=int, default=None,
                      help=f"seeds each job flies at most (default {MAX_ATTEMPTS_FACTOR} x --episodes)")
    plan.add_argument("--targets", type=Path, default=TARGETS)
    plan.add_argument("--data-root", type=Path, default=DATA_ROOT)
    plan.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    run = sub.add_parser("launch", help="start the jobs that may start now, one per free slot")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--interval-s", type=float, default=INTERVAL_S)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--retry", action="store_true", help="start again a job that ended without leaving a store")
    run.add_argument("--sim-root", type=Path, default=SIM_RUN_DIR, help=argparse.SUPPRESS)
    run.add_argument("--jobs-dir", type=Path, default=None, help=argparse.SUPPRESS)
    status = sub.add_parser("status", help="each store's kept / flown / rejected by reason")
    status.add_argument("--campaign", type=Path, required=True)
    status.add_argument("--jobs-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "plan":
        try:
            pilots: dict[str, Path] = {}
            for scene, path in args.pilot:
                if scene in pilots:
                    raise CampaignError(f"--pilot {scene} is named twice")
                pilots[scene] = path
            if args.campaign.exists():
                raise CampaignError(f"{args.campaign} exists; a campaign file is never written over")
            campaign = plan_campaign(pilots, episodes=args.episodes, batches=args.batches, first_batch=args.first_batch,
                                     slots=args.slots, max_attempts=args.max_attempts, targets=args.targets,
                                     data_root=args.data_root, records_dir=args.records_dir)
        except (ValueError, OSError) as err:
            print(f"refusing to plan: {err}", file=sys.stderr)
            return 2
        args.campaign.parent.mkdir(parents=True, exist_ok=True)
        write_text_durably(args.campaign, json.dumps(campaign, indent=2) + "\n")
        for job in campaign["jobs"]:
            print(f"{job['job']}: {job['command']}")
        print(f"{len(campaign['jobs'])} jobs planned in {args.campaign}")
        return 0
    try:
        campaign = load_campaign(args.campaign)
    except CampaignError as err:
        print(f"refusing: {err}", file=sys.stderr)
        return 2
    if args.command == "launch":
        rows = launch(campaign, interval_s=args.interval_s, dry_run=args.dry_run, retry=args.retry,
                      sim_root=args.sim_root, jobs_dir=args.jobs_dir)
        for row in rows:
            print(f"{row['name']:10s} slot {row['slot']:<3d} {row['verdict']}")
        return 1 if any(row["verdict"].startswith("FAILED") for row in rows) else 0
    print_status(campaign_status(campaign, jobs_dir=args.jobs_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
