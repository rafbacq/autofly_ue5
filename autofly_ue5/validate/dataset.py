"""The dataset validator (spec §11; Plan 4 task B6). M3's gate: the 100-episode pilot must pass it.

Failures, any one of which fails the dataset:
- the manifest and the files disagree (counts, steps, a missing frame, a missing provenance file);
- a record is not AutoFly's (state not float32[9], action not float32[3], an image that is not 256 x 256 RGB);
- NaN or inf anywhere; an action outside the action space;
- time order broken, or not exactly one 0.2 s simulator step per record;
- a state that contradicts itself (state[2] is altitude - 1.47 m; the first record is at the start) or leaves the
  altitude band;
- a kept episode that did not end at its target, or that is longer than the step limit;
- an instruction that is not one of the release's templates filled with the episode's own target name and its scene's
  obstacle phrase (Plan 2's placeholder "the target" fails);
- a provenance record missing a field spec §10.2 needs, or whose seed or simulator times differ from the record's.

A malformed episode (wrong shapes) is reported and its remaining checks skipped, never a crash of the validator.

The speed, altitude, action and length distributions are reported next to the real released episodes' (spec §3.2),
and differences are warnings: they describe the data rather than break it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from autofly_ue5.dataset.raw import FORMAT
from autofly_ue5.dataset.state import STATE2_OFFSET_M
from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES

STEP_NS = 200_000_000
STEP_LIMIT = 300
ALTITUDE_BAND_M = (1.0, 3.0)
# Live poses are float32: the first record sat 1 um from the start; one step later the drone was 5-11 mm away (the
# 2026-10-03 smoke). 1 mm passes the first and fails a recording that starts a step late.
START_TOLERANCE_M = 1e-3
LAST_RECORD_MAX_DISTANCE_M = 5.0 + 0.5  # the success radius, plus at most one 0.2 s step at 2 m/s before success
ACTION_LOW = np.array([0.0, -1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([2.0, 1.0, 1.0], dtype=np.float32)
PROVENANCE_KEYS = ("scene", "seed", "start_pose", "target", "distractors", "sim_time_ns", "termination", "expert", "a0",
                   "platform")
# The two real released episodes (spec §3.2), for comparison only.
RELEASE = {"speed_m_s_median": [1.87, 1.96], "altitude_m_range": [1.17, 2.57], "episode_steps": [96, 78],
           "action_forward_m_s_typical": [1.97, 1.99]}


def _stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    return {"median": float(np.median(a)), "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}


def validate_dataset(root: Path, *, decode_every_frame: bool = True) -> dict:
    root = Path(root)
    failures: list[str] = []
    warnings: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"pass": False, "failures": [f"{manifest_path} is missing"], "warnings": [], "counts": {}, "distributions": {}}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        failures.append(f"manifest format {manifest.get('format')!r}, expected {FORMAT!r}")
    speeds, altitudes, forwards, lengths = [], [], [], []
    records = 0
    by_scene: dict[str, int] = {}
    by_target: dict[str, int] = {}
    for entry in manifest.get("episodes", []):
        eid = entry["id"]
        ep = root / entry["path"]
        try:
            steps = np.load(ep / "steps.npz")
            state, action, times = steps["state"], steps["action"], steps["sim_time_ns"]
        except Exception as err:
            failures.append(f"{eid}: steps.npz unreadable ({type(err).__name__}: {err})")
            continue
        n = len(state)
        records += n
        lengths.append(n)
        by_scene[entry["scene"]] = by_scene.get(entry["scene"], 0) + 1
        by_target[entry["target_name"]] = by_target.get(entry["target_name"], 0) + 1
        if n != entry["steps"]:
            failures.append(f"{eid}: {n} records, the manifest says {entry['steps']}")
        if state.shape != (n, 9) or state.dtype != np.float32:
            failures.append(f"{eid}: state is {state.dtype}{state.shape}, not float32 (T, 9)")
        if action.shape != (n, 3) or action.dtype != np.float32:
            failures.append(f"{eid}: action is {action.dtype}{action.shape}, not float32 (T, 3)")
        if times.shape != (n,):
            failures.append(f"{eid}: sim_time_ns has shape {times.shape}")
        if state.shape != (n, 9) or action.shape != (n, 3) or times.shape != (n,):
            continue  # every check below indexes these shapes; the failure above already fails the dataset
        if not (np.all(np.isfinite(state)) and np.all(np.isfinite(action))):
            failures.append(f"{eid}: NaN or inf in state or action")  # and keep checking: report everything wrong
        if np.any(action < ACTION_LOW - 1e-6) or np.any(action > ACTION_HIGH + 1e-6):
            failures.append(f"{eid}: an action outside [0, 2] x [-1, 1] x [-1, 1]")
        if n > 1 and not np.all(np.diff(times) == STEP_NS):
            failures.append(f"{eid}: records are not exactly one 0.2 s step apart (or not in time order)")
        if not (1 <= n <= STEP_LIMIT):
            failures.append(f"{eid}: {n} records, outside 1-{STEP_LIMIT}")
        if np.any(np.abs(state[:, 2] - (state[:, 8] - STATE2_OFFSET_M)) > 1e-4):
            failures.append(f"{eid}: state[2] is not altitude - {STATE2_OFFSET_M} m")
        if np.any(state[:, 8] < ALTITUDE_BAND_M[0] - 1e-6) or np.any(state[:, 8] > ALTITUDE_BAND_M[1] + 1e-6):
            failures.append(f"{eid}: an altitude outside the {ALTITUDE_BAND_M} m band")
        if np.any(np.abs(state[0, 6:8]) > START_TOLERANCE_M):
            failures.append(f"{eid}: the first record is not at the start (state[6:8] = {state[0, 6:8].tolist()})")
        if state[-1, 0] > LAST_RECORD_MAX_DISTANCE_M:
            failures.append(f"{eid}: the last record is {state[-1, 0]:.2f} m from the target; a kept episode ends there")
        obstacle = manifest.get("scenes", {}).get(entry["scene"], {}).get("instruction_obstacle")
        if obstacle is None:
            failures.append(f"{eid}: the manifest gives no instruction_obstacle for scene {entry['scene']}")
        elif entry.get("instruction") not in {t.format(target=entry["target_name"], obstacle=obstacle)
                                              for t in INSTRUCTION_TEMPLATES}:
            failures.append(f"{eid}: instruction {entry.get('instruction')!r} is not a release template naming "
                            f"{entry['target_name']!r} and {obstacle!r}")
        frames = sorted((ep / "frames").glob("*.png"))
        if len(frames) != n:
            failures.append(f"{eid}: {len(frames)} frames for {n} records")
        for path in (frames if decode_every_frame else frames[:1] + frames[-1:]):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None or img.shape != (256, 256, 3) or img.dtype != np.uint8:
                failures.append(f"{eid}: frame {path.name} does not decode to 256x256 RGB")
                break
        prov_path = root / "provenance" / f"{eid}.json"
        if not prov_path.is_file():
            failures.append(f"{eid}: provenance missing")
        else:
            prov = json.loads(prov_path.read_text())
            missing = [k for k in PROVENANCE_KEYS if k not in prov]
            if missing:
                failures.append(f"{eid}: provenance lacks {missing}")
            elif prov["termination"] != "success":
                failures.append(f"{eid}: provenance says {prov['termination']}")
            else:
                if prov["seed"] != entry["seed"]:
                    failures.append(f"{eid}: provenance seed {prov['seed']} != the manifest's {entry['seed']}")
                if [int(t) for t in prov["sim_time_ns"]] != times.tolist():
                    failures.append(f"{eid}: provenance simulator times differ from steps.npz")
        speeds.extend(state[:, 3].tolist())
        altitudes.extend(state[:, 8].tolist())
        forwards.extend(action[:, 0].tolist())
    counts = manifest.get("counts", {})
    if counts.get("episodes") != len(manifest.get("episodes", [])) or counts.get("records") != records:
        failures.append(f"manifest counts {counts.get('episodes')} episodes / {counts.get('records')} records, files hold "
                        f"{len(manifest.get('episodes', []))} / {records}")
    distributions = {"speed_m_s": _stats(speeds), "altitude_m": _stats(altitudes), "action_forward_m_s": _stats(forwards),
                     "episode_steps": _stats(lengths)}
    if distributions["speed_m_s"] and not (1.0 <= distributions["speed_m_s"]["median"] <= 2.0):
        warnings.append(f"median speed {distributions['speed_m_s']['median']:.2f} m/s; the release's is 1.87-1.96")
    return {"pass": not failures, "failures": failures, "warnings": warnings,
            "counts": {"episodes": len(manifest.get("episodes", [])), "records": records, "rejects": counts.get("rejects"),
                       "by_scene": by_scene, "by_target": by_target},
            "distributions": distributions, "release_for_comparison": RELEASE}
