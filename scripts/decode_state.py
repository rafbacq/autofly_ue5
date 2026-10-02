"""Decode the released AutoFly state[9] from the two real episodes on this host (spec §10.1, Plan 4 task B1).

Spec §10.1 knows state[0] (horizontal distance to target), state[3] (speed) and state[6..8] (position, z up). It
leaves state[1], [2], [4] and [5] to "hypotheses tested against the two real episodes", each adopted only at
|correlation| >= 0.95 on both. This script makes that test reproducible, and reports what it needs:

- **Time order.** The exported rows are shuffled. Sorting by state[0] (distance, decreasing) recovers time order: the
  median step falls from 13-18 m to 0.35-0.41 m. Both episodes list their time steps in one shared order: with one
  step of the longer episode set aside, 70 of 78 file rows land on the same time step (`order.shared_permutation`).
- **Hypotheses** computed from the time-ordered trajectory:
  - heading of motion, yaw rate and vertical speed;
  - forward and lateral acceleration, stand-ins for pitch and roll;
  - the target's position (trilaterated from state[0] by linear least squares) and the bearing to it, world and
    relative, with sin/cos;
  - altitude, height above the start;
  - the row's own action.
- **a0** (spec §9 step 4): the time-ordered first actions, and where the turn-in-place actions really are.

Read-only. Writes one JSON (default runs/m3/state_decoding.json):

    env -u PYTHONPATH .venv/bin/python scripts/decode_state.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from autofly_ue5.paths import RUNS_DIR  # noqa: E402

REAL_EPISODES = Path("/home/nvidiasims/research_uav/results/qwen_autofly/data_smoke_v1")
ADOPT_ABS_R = 0.95  # spec §10.1
UNKNOWN_FIELDS = (1, 2, 4, 5)
DT_S = 0.2
TURN_IN_PLACE = {"max_forward_m_s": 0.5, "min_abs_yaw_rate": 0.5}


def load_episode(path: Path) -> dict:
    d = json.loads(Path(path).read_text().splitlines()[0])
    return {"states": np.asarray(d["states"], dtype=float), "actions": np.asarray(d["actions"], dtype=float),
            "instruction": d["instructions"][0], "source": d.get("source_id"), "split": d.get("split")}


def time_order(states: np.ndarray) -> np.ndarray:
    """File rows in time order: distance to the target only falls along an episode that ends at the target."""
    return np.argsort(-states[:, 0], kind="stable")


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def trilaterate(x, y, d) -> tuple[float, float, float]:
    """(tx, ty, rms residual): |p - t| = d is linear in (tx, ty, |t|^2) after squaring."""
    m = np.stack([-2 * x, -2 * y, np.ones_like(x)], 1)
    sol, *_ = np.linalg.lstsq(m, d ** 2 - x ** 2 - y ** 2, rcond=None)
    tx, ty = float(sol[0]), float(sol[1])
    return tx, ty, float(np.sqrt(np.mean((np.hypot(x - tx, y - ty) - d) ** 2)))


def hypotheses(states: np.ndarray, actions: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Candidate quantities per time-ordered step, and what was fitted to build them."""
    x, y, z, d = states[:, 6], states[:, 7], states[:, 8], states[:, 0]
    vx, vy, vz = np.gradient(x, DT_S), np.gradient(y, DT_S), np.gradient(z, DT_S)
    heading = np.unwrap(np.arctan2(vy, vx))
    yaw_rate = np.gradient(heading, DT_S)
    speed = np.hypot(vx, vy)
    tx, ty, resid = trilaterate(x, y, d)
    bearing_world = np.arctan2(ty - y, tx - x)
    relative = _wrap(bearing_world - heading)
    return {
        "heading": _wrap(heading), "sin_heading": np.sin(heading), "cos_heading": np.cos(heading),
        "yaw_rate": yaw_rate, "vertical_speed": vz, "forward_accel": np.gradient(speed, DT_S),
        "lateral_accel": speed * yaw_rate,
        "bearing_world": bearing_world, "bearing_relative": relative, "sin_bearing_relative": np.sin(relative),
        "cos_bearing_relative": np.cos(relative), "altitude": z, "height_above_start": z - z[0],
        "action_forward": actions[:, 0], "action_yaw_rate": actions[:, 1], "action_vertical": actions[:, 2],
    }, {"target_xy": [tx, ty], "trilateration_rms_m": resid}


def _r(a, b) -> float | None:
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def shared_order(long: np.ndarray, short: np.ndarray) -> dict:
    """Do two episodes' file orders list their time steps in the same order? The longer episode has steps the shorter
    lacks, so drop the one step of the longer whose removal lines the two up best, renumber, and count the file rows
    that land on the same time step."""
    long, short = [int(t) for t in long], [int(t) for t in short]
    best = (-1, None)
    for drop in [None, *range(len(long))]:
        mapped = long if drop is None else [t - (t > drop) for t in long if t != drop]
        mapped = [t for t in mapped if t < len(short)]
        exact = sum(a == b for a, b in zip(mapped, short))
        if exact > best[0]:
            best = (exact, drop)
    return {"dropped_step_of_longer": best[1], "file_rows_on_the_same_time_step": best[0], "of": len(short)}


def decode(episodes: dict[str, dict]) -> dict:
    per_episode, correlations = {}, {k: {} for k in UNKNOWN_FIELDS}
    ranks = {}
    for name, ep in episodes.items():
        order = time_order(ep["states"])
        s, a = ep["states"][order], ep["actions"][order]
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(len(order))
        ranks[name] = rank
        steps = np.hypot(np.diff(s[:, 6]), np.diff(s[:, 7]))
        file_steps = np.hypot(np.diff(ep["states"][:, 6]), np.diff(ep["states"][:, 7]))
        cands, fit = hypotheses(s, a)
        for k in UNKNOWN_FIELDS:
            correlations[k][name] = {h: _r(s[:, k], v) for h, v in cands.items()}
        slope, intercept = np.polyfit(s[:, 8], s[:, 2], 1)
        turns = [int(i) for i in range(len(a)) if a[i, 0] <= TURN_IN_PLACE["max_forward_m_s"]
                 and abs(a[i, 1]) >= TURN_IN_PLACE["min_abs_yaw_rate"]]
        per_episode[name] = {
            "source": ep["source"], "instruction": ep["instruction"], "steps": len(order),
            "median_step_m": {"file_order": float(np.median(file_steps)), "time_order": float(np.median(steps))},
            "jumps_over_1_2_m_in_time_order": int(np.sum(steps > 1.2)),
            "start_state": s[0].tolist(), "end_state": s[-1].tolist(),
            "first_actions_in_time_order": a[:5].tolist(),
            "file_row_0": {"time_step": int(rank[0]), "action": ep["actions"][0].tolist(), "state": ep["states"][0].tolist()},
            "turn_in_place_time_steps": turns,
            "state2_vs_altitude_fit": {"slope": float(slope), "intercept": float(intercept),
                                        "residual_rms": float(np.std(s[:, 2] - (slope * s[:, 8] + intercept)))},
            **fit,
        }
    names = list(episodes)
    adopted, best = {}, {}
    for k in UNKNOWN_FIELDS:
        both = [h for h in correlations[k][names[0]]
                if all((correlations[k][n].get(h) is not None and abs(correlations[k][n][h]) >= ADOPT_ABS_R) for n in names)]
        adopted[f"state[{k}]"] = both
        ranked = sorted(correlations[k][names[0]], key=lambda h: -min(abs(correlations[k][n][h] or 0.0) for n in names))
        best[f"state[{k}]"] = [{"hypothesis": h, **{n: correlations[k][n][h] for n in names}} for h in ranked[:4]]
    shared = shared_order(ranks[names[0]], ranks[names[1]]) if len(names) == 2 else None
    return {
        "description": __doc__.split("\n\n")[0],
        "adopt_rule": f"|r| >= {ADOPT_ABS_R} on both episodes (spec §10.1)",
        "adopted": adopted,
        "best_hypotheses": best,
        "order": {"method": "sort by state[0], decreasing", "shared_permutation": shared},
        "episodes": per_episode,
        "correlations": {f"state[{k}]": v for k, v in correlations.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REAL_EPISODES)
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "m3" / "state_decoding.json")
    args = parser.parse_args(argv)
    episodes = {split: load_episode(args.data / f"{split}.jsonl") for split in ("train", "val")}
    report = decode(episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"adopted": report["adopted"], "best_hypotheses": report["best_hypotheses"],
                      "shared_permutation": report["order"]["shared_permutation"],
                      "a0": {n: {"file_row_0": e["file_row_0"]["time_step"], "first_action": e["first_actions_in_time_order"][0],
                                 "turn_in_place_steps": e["turn_in_place_time_steps"]}
                             for n, e in report["episodes"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
