"""Closed-loop evaluation metrics (spec §3.1, §12 M6): the paper's success rate SR, collision rate CR and path efficiency
PER, with the out-of-bounds and timeout rates that complete the four outcomes.

Rates carry 95 % Wilson score intervals. At 20 episodes per scene a rate's interval is 0.3-0.4 wide, and a comparison
without it reads noise as a difference (MEMORY.md: never decide a run's fate on 20-episode scores). The project's earlier
intervals are Wilson's (370/400 -> 0.895-0.947, CLAUDE.md's M2d row), so these compare with them.

PER_i = L_opt / max(L_i, L_opt), averaged over successful episodes (the paper's Sec. 4.1). L is the flown path's
horizontal length: the success test and L_opt are horizontal too. The paper does not define L_opt, so it is reported as
an interval (MEMORY.md, "Scene learnings"). The straight line from the start to the 5 m success disc bounds it from
below: no obstacle makes a path shorter. The shortest 8-connected path on the scene's occupancy grid, inflated by the
sampler's 1.4 m clearance, bounds it from above: the grid overestimates a straight line by up to 8 %, and the inflation
keeps a path farther from obstacles than a drone must stay. PER from the first is a lower bound, from the second an upper
one, both averaged over the same successes (those with a grid optimum), as scripts/dataset_stats.py does.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from typing import Iterable, Sequence

from autofly_ue5.expert.episode import spawn_clearance_m
from autofly_ue5.scenes.model import Layout
from autofly_ue5.scenes.paths import optimal_path_m

WILSON_Z_95 = 1.959963984540054  # the standard normal's 97.5th percentile
RATE_OUTCOMES = ("success", "collision", "out_of_bounds", "timeout")  # reward.Outcome's endings, in the record's order
RATE_KEYS = {"success": "success_rate", "collision": "collision_rate", "out_of_bounds": "out_of_bounds_rate",
             "timeout": "timeout_rate"}
L_OPT_RESOLUTION_M = 0.5  # scripts/dataset_stats.py's grid, so the harness's PER compares with the datasets'
L_OPT_INFLATE_M = spawn_clearance_m()  # 1.4 m: the inflation every layout is accepted at, so a grid path exists


def wilson_interval(successes: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float] | None:
    """The Wilson score interval of `successes` out of `n` (95 % at the default z); None for n = 0."""
    if n < 0 or not 0 <= successes <= max(n, 0):
        raise ValueError(f"a proportion needs 0 <= successes <= n, got {successes} of {n}")
    if n == 0:
        return None
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def horizontal_path_length_m(points_xy: Sequence[tuple[float, float]]) -> float:
    """The length of the flown path through `points_xy` (the drone's x, y at the start and after every step)."""
    return float(sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(points_xy, points_xy[1:])))


def l_opt_bounds(layout: Layout, start_xy, target_xy, *, success_radius_m: float, inflate_m: float = L_OPT_INFLATE_M,
                 resolution_m: float = L_OPT_RESOLUTION_M, exclude_tags: Iterable[str] = ()) -> dict:
    """L_opt's two bounds for one episode: `straight_m`, the straight line to the success disc, and `grid_m`, the
    shortest grid path to it (scenes/paths.optimal_path_m), floored by the straight line, None when the disc cannot be
    reached on the grid. `exclude_tags`: obstacles left off the grid, a dynamic scene's movers, which leave their homes."""
    excluded = set(exclude_tags)
    kept = tuple(inst for inst in layout.instances if inst.tag not in excluded)
    grid_layout = dataclasses.replace(layout, instances=kept) if len(kept) != len(layout.instances) else layout
    straight = max(math.hypot(target_xy[0] - start_xy[0], target_xy[1] - start_xy[1]) - success_radius_m, 0.0)
    grid = optimal_path_m(grid_layout, (start_xy[0], start_xy[1]), (target_xy[0], target_xy[1]),
                          goal_radius_m=success_radius_m, inflate_m=inflate_m, resolution_m=resolution_m)
    return {"straight_m": straight, "grid_m": None if grid is None else max(grid, straight), "inflate_m": inflate_m,
            "resolution_m": resolution_m, "excluded_obstacles": len(layout.instances) - len(kept)}


def per_bounds(path_length_m: float, l_opt: dict) -> dict:
    """PER = L_opt / max(L, L_opt) at each end of L_opt's interval: `lower` from the straight line, `upper` from the grid
    path; None where that L_opt is missing or both lengths are zero."""

    def per(optimum: float | None) -> float | None:
        if optimum is None:
            return None
        longer = max(path_length_m, optimum)
        return None if longer <= 0 else optimum / longer

    return {"lower": per(l_opt["straight_m"]), "upper": per(l_opt.get("grid_m"))}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise(episodes: Sequence[dict]) -> dict:
    """The metrics of a set of episode records (a scene's, or every scene's): outcome counts, each rate with its 95 %
    Wilson interval over every episode, PER's interval over the successes with both bounds, what the collisions were
    (expert/env.py's collision_source), mean steps and path length, and the policy's time per action."""
    n = len(episodes)
    counts = Counter(e["outcome"] for e in episodes)
    outcomes = {o: counts.get(o, 0) for o in RATE_OUTCOMES}
    outcomes.update({o: c for o, c in sorted(counts.items()) if o not in RATE_OUTCOMES})  # e.g. an outer guard's ending
    summary: dict = {"episodes": n, "outcomes": outcomes}
    for outcome, key in RATE_KEYS.items():
        interval = wilson_interval(counts.get(outcome, 0), n)
        summary[key] = {"value": counts.get(outcome, 0) / n if n else None,
                        "wilson95": list(interval) if interval is not None else None}
    successes = [e for e in episodes if e["outcome"] == "success"]
    both = [e["per"] for e in successes
            if (e.get("per") or {}).get("lower") is not None and (e.get("per") or {}).get("upper") is not None]
    summary["per"] = {"successes": len(successes), "episodes": len(both), "without_grid_optimum": len(successes) - len(both),
                      "lower": _mean([p["lower"] for p in both]), "upper": _mean([p["upper"] for p in both])}
    summary["collision_sources"] = dict(Counter(e.get("collision_source") or "sim"
                                                for e in episodes if e["outcome"] == "collision"))
    summary["mean_steps"] = _mean([e["steps"] for e in episodes if e.get("steps") is not None])
    summary["mean_path_length_m"] = _mean([e["path_length_m"] for e in episodes if e.get("path_length_m") is not None])
    timed = [(e["policy_act_s"], e["steps"]) for e in episodes if e.get("policy_act_s") and e.get("steps")]
    total_steps = sum(steps for _act, steps in timed)
    summary["policy_act_s"] = {
        "mean": sum(act["mean"] * steps for act, steps in timed) / total_steps if total_steps else None,
        "max": max((act["max"] for act, _steps in timed), default=None)}
    return summary
