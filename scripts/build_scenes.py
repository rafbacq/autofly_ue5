"""Expand a scene file into a layout, check reachability, write <id>.layout.json and <id>.level.json.

A scene that reuses another's level (e.g. s01d, spec §6.1) gets no level of its own. For a dynamic scene this writes
<id>.dynamic_report.json instead: what its mover sampler does over training-range seeds (never the gate's), the numbers
docs/decisions/2026-10-02-dynamic-obstacles.md quotes.

env -u PYTHONPATH .venv/bin/python scripts/build_scenes.py scenes/s01_white_pillars.json
env -u PYTHONPATH .venv/bin/python scripts/build_scenes.py scenes/s01d_moving_pillars.json   # the dynamic report
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap, as in m2_gate.py: a direct `python scripts/build_scenes.py` puts scripts/ at sys.path[0].
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

from autofly_ue5.paths import RUNS_DIR  # noqa: E402
from autofly_ue5.scenes.build import build_scene  # noqa: E402
from autofly_ue5.scenes.model import load_scene_file  # noqa: E402

REPORT_SEEDS = 500


def dynamic_report(scene_id: str, n_seeds: int = REPORT_SEEDS) -> dict:
    """The dynamic scene's mover sampling over training worker 0's first `n_seeds` episodes."""
    from autofly_ue5.expert.episode import sample_setup
    from autofly_ue5.expert.seeds import worker_seed_base
    from autofly_ue5.scenes.resolve import resolve_scene

    resolved = resolve_scene(scene_id)
    seeds = range(worker_seed_base(0), worker_seed_base(0) + n_seeds)
    placed, ratios, times, kinds, guard, unplaced, near = [], [], [], Counter(), Counter(), Counter(), []
    for seed in seeds:
        t0 = time.perf_counter()
        setup = sample_setup(resolved.scene, resolved.layout, np.random.default_rng(seed))
        times.append(1000.0 * (time.perf_counter() - t0))
        stats = setup.mover_stats
        placed.append(len(setup.movers))
        near.append(sum(1 for r in setup.movers
                        if _near_line(r, (setup.start.x, setup.start.y), setup.target_xy_z[:2],
                                      resolved.scene.dynamic.path_corridor_m)))
        guard[stats["guard"]] += 1
        if stats["path_ratio"] is not None:
            ratios.append(stats["path_ratio"])
        for reasons in stats["unplaced"].values():
            unplaced[reasons[0].split(":")[0] if reasons else "unknown"] += 1
        kinds.update(r.kind for r in setup.movers)
    return {
        "description": "Mover sampling of a dynamic scene (spec §6.5) over training-range seeds (scripts/build_scenes.py).",
        "scene": resolved.scene.id,
        "scene_sha256": resolved.scene.sha256,
        "base_scene": resolved.base_id,
        "layout_sha256": resolved.layout_sha256,
        "n_seeds": n_seeds,
        "seeds": [seeds[0], seeds[-1]],
        "movers_placed": {"mean": statistics.fmean(placed), "min": min(placed), "max": max(placed),
                          "histogram": {str(k): v for k, v in sorted(Counter(placed).items())}},
        "movers_near_flight_line": {"mean": statistics.fmean(near), "episodes_with_none": sum(n == 0 for n in near)},
        "unplaced_movers_by_first_reason": dict(unplaced),
        "guard": dict(guard),
        "path_ratio": {"median": statistics.median(ratios), "p95": float(np.percentile(ratios, 95)), "max": max(ratios)},
        "route_kinds": dict(kinds),
        "reset_ms": {"median": statistics.median(times), "p95": float(np.percentile(times, 95)), "max": max(times)},
    }


def _near_line(route, a, b, corridor_m: float) -> bool:
    from autofly_ue5.scenes.motion import _segment_distance

    return _segment_distance(route.home_x, route.home_y, *a, *b) <= corridor_m


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR / "levels")
    parser.add_argument("--report-seeds", type=int, default=REPORT_SEEDS, help="dynamic scenes: seeds in the report")
    args = parser.parse_args(argv)
    scene = load_scene_file(args.scene)
    if scene.level is not None:
        if scene.dynamic is None:
            print(f"{scene.id} reuses {scene.level}'s level and has nothing of its own to build", file=sys.stderr)
            return 0
        report = dynamic_report(scene.id, n_seeds=args.report_seeds)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"{scene.id}.dynamic_report.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("scene", "movers_placed", "guard", "path_ratio", "reset_ms")}))
        return 0
    summary = build_scene(args.scene, args.out_dir)
    print(json.dumps(summary))
    return 0 if summary["reachability"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
