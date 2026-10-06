"""Path efficiency of a raw dataset store's episodes (the paper's PER, Eq. 4, spec §8's L_opt), from provenance.

    env -u PYTHONPATH .venv/bin/python scripts/dataset_stats.py --raw data/<name> [--out runs/<...>.json]

For every kept episode: L, the length of the flown trajectory (`poses_per_record` plus `final_pose`, 3-D), and L_opt,
the shortest path on the scene's occupancy grid from the start to the 5 m success disc (`scenes/paths.py`), at two
inflations: the rotor half-span (0.48 m: the shortest physically collision-free path) and the sampler's clearance
(1.4 m: a path with the margin the expert was trained to keep). PER_i = L_opt / max(L, L_opt); the report's PER is the
mean over the episodes, as the paper's Eq. 4 averages over successful trials. The paper reports PER 73-78 % for its
models and does not define L_opt; both conventions are recorded with the numbers.

The grid path overestimates the true optimum (8-connected moves: up to about 8 %; the inflation: whatever the real
drone could have cut closer), so `per_physical` and `per_clearance` are upper bounds on the paper's PER. The straight
line to the success disc is a lower bound on L_opt, so `per_straight` = straight / max(L, straight) is a lower bound
on PER. The true value lies between them; on the pilot the two bounds are 0.96 and 0.99 (2026-10-06).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

from autofly_ue5.expert.reward import RewardConfig  # noqa: E402
from autofly_ue5.scenes.model import DRONE_HALF_SPAN_M, Layout  # noqa: E402
from autofly_ue5.scenes.paths import optimal_path_m  # noqa: E402
from autofly_ue5.scenes.resolve import resolve_scene  # noqa: E402

INFLATIONS_M = {"physical": DRONE_HALF_SPAN_M, "clearance": 1.4}
RESOLUTION_M = 0.5


def trajectory_length_m(poses_per_record: list, final_pose: list) -> float:
    """3-D length of the flown path: the recorded poses (one per record, before each action) and where it ended."""
    pts = np.asarray([p[:3] for p in list(poses_per_record) + [final_pose]], dtype=float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def episode_efficiency(layout: Layout, provenance: dict, *, success_radius_m: float = RewardConfig().success_radius_m,
                       resolution_m: float = RESOLUTION_M) -> dict:
    start = provenance["start_pose"]
    target = provenance["target"]["xyz"]
    flown = trajectory_length_m(provenance["poses_per_record"], provenance["final_pose"])
    out = {"records": len(provenance["poses_per_record"]), "flown_m": round(flown, 3),
           "straight_m": round(max(math.hypot(target[0] - start[0], target[1] - start[1]) - success_radius_m, 0.0), 3)}
    out["per_straight"] = round(out["straight_m"] / max(flown, out["straight_m"]), 4) if out["straight_m"] > 0 else None
    for name, inflate in INFLATIONS_M.items():
        l_opt = optimal_path_m(layout, (start[0], start[1]), (target[0], target[1]), goal_radius_m=success_radius_m,
                               inflate_m=inflate, resolution_m=resolution_m)
        out[f"l_opt_{name}_m"] = None if l_opt is None else round(l_opt, 3)
        out[f"per_{name}"] = None if l_opt is None else round(l_opt / max(flown, l_opt), 4)
    return out


def dataset_stats(raw: Path, *, layout: Layout | None = None) -> dict:
    raw = Path(raw)
    manifest = json.loads((raw / "manifest.json").read_text())
    scenes = set(e["scene"] for e in manifest["episodes"])
    if layout is None:
        if len(scenes) != 1:
            raise ValueError(f"the store holds scenes {sorted(scenes)}; pass one layout per scene (not supported yet)")
        layout = resolve_scene(next(iter(scenes))).layout
    episodes = []
    for entry in manifest["episodes"]:
        provenance = json.loads((raw / "provenance" / f"{entry['id']}.json").read_text())
        episodes.append({"id": entry["id"], **episode_efficiency(layout, provenance)})
    summary = {"dataset": manifest["name"], "episodes": len(episodes), "resolution_m": RESOLUTION_M,
               "inflations_m": INFLATIONS_M, "success_radius_m": RewardConfig().success_radius_m}
    for key in ("flown_m", "straight_m", "l_opt_physical_m", "l_opt_clearance_m", "per_straight", "per_physical", "per_clearance"):
        values = [e[key] for e in episodes if e.get(key) is not None]
        summary[key] = ({"mean": round(float(np.mean(values)), 4), "median": round(float(np.median(values)), 4),
                         "min": round(float(np.min(values)), 4), "max": round(float(np.max(values)), 4), "n": len(values)}
                        if values else None)
    return {"summary": summary, "episodes": episodes}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="write the full report here (refused if it exists)")
    args = p.parse_args(argv)
    report = dataset_stats(args.raw)
    print(json.dumps(report["summary"], indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "x") as handle:
            handle.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
