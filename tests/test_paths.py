"""autofly_ue5/scenes/paths.py: shortest paths on the §6.2 occupancy grid, for spec §13's detour metric (does a layout
let a crossing go through the field, or only around it?) and spec §8's L_opt (the paper's path efficiency)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autofly_ue5.scenes.model import Bounds, Instance, Layout

BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)


def _pillar(tag: str, x: float, y: float, radius_m: float = 0.5) -> Instance:
    return Instance(tag=tag, asset="cylinder", x=x, y=y, z_center=-5.0, yaw=0.0, scale=(1.0, 1.0, 10.0), material="white",
                    radius_m=radius_m, height_m=10.0)


def _layout(instances) -> Layout:
    return Layout(scene_id="t", seed=0, bounds=BOUNDS, instances=tuple(instances))


def _wall(x: float, gap_y: float | None = None, half: float = 2.0) -> list[Instance]:
    """Pillars every metre across the scene at `x`, with a gap of `2 * half` m around `gap_y` if given."""
    ys = [y for y in np.arange(-34.5, 35.0, 1.0) if gap_y is None or abs(y - gap_y) > half]
    return [_pillar(f"w{i:03d}", x, float(y)) for i, y in enumerate(ys)]


def test_dijkstra_matches_euclid_on_an_empty_grid_within_the_eight_connected_error():
    from autofly_ue5.scenes.paths import dijkstra_path_m, free_grid

    free, xs, ys = free_grid(_layout([]), resolution_m=0.5, inflate_m=1.4)
    sources = np.zeros_like(free)
    goals = np.zeros_like(free)
    sources[np.argmin(np.abs(xs + 30.0)), np.argmin(np.abs(ys + 20.0))] = True
    goals[np.argmin(np.abs(xs - 30.0)), np.argmin(np.abs(ys - 10.0))] = True
    length = dijkstra_path_m(free, sources, goals, resolution_m=0.5)
    straight = math.hypot(60.0, 30.0)
    assert straight <= length <= straight * 1.09, "8-connected paths are at most ~8 % longer than straight lines"
    assert dijkstra_path_m(free, sources, np.zeros_like(free), resolution_m=0.5) is None


def test_optimal_path_goes_through_a_gap_and_is_none_when_sealed():
    from autofly_ue5.scenes.paths import optimal_path_m

    start, target = (-30.0, 0.0), (30.0, 0.0)
    open_field = optimal_path_m(_layout([]), start, target, goal_radius_m=5.0, inflate_m=0.5, resolution_m=0.5)
    assert open_field == pytest.approx(55.0, abs=1.0), "straight to the 5 m success radius"
    gap = optimal_path_m(_layout(_wall(0.0, gap_y=10.0, half=3.0)), start, target, goal_radius_m=5.0, inflate_m=0.5, resolution_m=0.5)
    direct_via_gap = math.hypot(30.0, 10.0) + math.hypot(30.0, 10.0) - 5.0
    assert direct_via_gap <= gap <= direct_via_gap * 1.09
    assert optimal_path_m(_layout(_wall(0.0)), start, target, goal_radius_m=5.0, inflate_m=0.5, resolution_m=0.5) is None, \
        "a wall edge to edge: nothing gets through"


def test_optimal_path_tolerates_a_start_on_the_inflation_boundary():
    from autofly_ue5.scenes.paths import optimal_path_m

    # episode.py draws starts exactly spawn_clearance_m() from a pillar's surface, which the inflated grid may mark occupied
    layout = _layout([_pillar("p", -28.0, 0.0)])
    start = (-28.0 + 0.5 + 1.4, 0.0)
    assert optimal_path_m(layout, start, (30.0, 0.0), goal_radius_m=5.0, inflate_m=1.4, resolution_m=0.25) is not None


def test_crossing_detours_tell_a_permeable_field_from_a_sealed_one():
    from autofly_ue5.scenes.paths import crossing_detours

    permeable = _wall(0.0, gap_y=0.0, half=3.0)  # one 6 m gap in the middle
    report = crossing_detours(_layout(permeable), start_band=(2.0, 6.0), target_band=(0.0, 3.0), inflate_m=1.4,
                              resolution_m=0.5, rng=np.random.default_rng(0))
    assert set(report["edges"]) == {"x_min", "x_max", "y_min", "y_max"}
    assert report["edges"]["y_min"]["median"] == pytest.approx(1.0, abs=0.09), "along the wall: straight"
    assert 1.0 <= report["edges"]["x_min"]["median"] <= 1.6, "across the wall: through the one gap, from wherever the start is"
    assert report["max"] == max(e["max"] for e in report["edges"].values())

    sealed = _wall(0.0)  # no gap: a crossing can only go around, outside the field, which the §6.2 rule still accepts
    report = crossing_detours(_layout(sealed), start_band=(2.0, 6.0), target_band=(0.0, 3.0), inflate_m=1.4,
                              resolution_m=0.5, rng=np.random.default_rng(0))
    assert report["edges"]["x_min"]["unreachable"] == report["samples"], "the wall reaches the bounds: nothing crosses"

    almost = _wall(0.0)[2:-2]  # the wall stops 2.5 m short of each side: the only way across is around its ends
    report = crossing_detours(_layout(almost), start_band=(2.0, 6.0), target_band=(0.0, 3.0), inflate_m=0.5,
                              resolution_m=0.5, rng=np.random.default_rng(0), samples_per_edge=12)
    assert report["edges"]["x_min"]["max"] > 1.3, "a start near the middle has a long way round (about 1.46 from y = 0)"
    assert report["edges"]["y_min"]["median"] == pytest.approx(1.0, abs=0.09), "along the wall nothing changed"


def test_s01_crosses_with_a_small_detour():
    from autofly_ue5.scenes.paths import crossing_detours
    from tests.test_expert_episode import scene_and_layout

    scene, layout = scene_and_layout()
    report = crossing_detours(layout, start_band=scene.start_band, target_band=scene.target_band, inflate_m=1.4,
                              resolution_m=0.5, rng=np.random.default_rng(0))
    assert report["max"] < 1.3, report


def test_dataset_stats_report_path_efficiency_per_kept_episode(tmp_path):
    from scripts.dataset_stats import dataset_stats, trajectory_length_m
    from tests.test_collect import _collect
    from tests.test_expert_episode import scene_and_layout

    assert trajectory_length_m([[0, 0, -2, 0], [3, 4, -2, 0]], [3, 4, -4, 0]) == pytest.approx(5.0 + 2.0)
    _summary, root = _collect(tmp_path, n_keep=2)
    _scene, layout = scene_and_layout()
    report = dataset_stats(root, layout=layout)
    assert report["summary"]["episodes"] == 2 and len(report["episodes"]) == 2
    for ep in report["episodes"]:
        assert ep["flown_m"] >= ep["straight_m"] > 0
        assert ep["l_opt_physical_m"] is not None and ep["l_opt_physical_m"] >= ep["straight_m"] - 0.6
        assert ep["l_opt_clearance_m"] >= ep["l_opt_physical_m"] - 0.6, "the wider inflation never finds a shorter path"
        assert 0.0 < ep["per_physical"] <= 1.0 and 0.0 < ep["per_clearance"] <= 1.0
        assert 0.0 < ep["per_straight"] <= ep["per_physical"], "the straight line bounds PER from below, the grid from above"
    assert report["summary"]["per_physical"]["n"] == 2 and report["summary"]["per_straight"]["n"] == 2
