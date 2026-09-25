import json

import numpy as np
import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import Bounds, Instance, Layout, load_registry, load_scene_file
from autofly_ue5.scenes.reachability import (
    EDGES,
    band_mask,
    cell_centers,
    check_reachability,
    occupancy,
    reachable_from,
)

BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)


def _post(tag: str, x: float, y: float, radius: float = 0.2) -> Instance:
    return Instance(tag=tag, asset="cylinder", x=x, y=y, z_center=-5.0, yaw=0.0, scale=(2 * radius, 2 * radius, 10.0),
                    material="white", radius_m=radius, height_m=10.0)


def test_cell_centers_and_band_mask():
    xs, ys = cell_centers(BOUNDS, 0.5)
    assert len(xs) == 140 and xs[0] == -34.75 and ys[-1] == 34.75
    band = band_mask(BOUNDS, 0.5, 2.0, 6.0)
    ix = int(np.argmin(np.abs(xs - (-31.0))))
    iy = int(np.argmin(np.abs(ys - 0.0)))
    assert band[ix, iy]  # about 4 m inside the south (x_min) edge
    assert not band[int(np.argmin(np.abs(xs - 0.0))), iy]  # centre of the scene


def test_occupancy_inflates_by_radius():
    layout = Layout("t", 0, BOUNDS, (_post("obs_0000", 0.0, 0.0, radius=0.5),))
    occ = occupancy(layout, 0.25, inflate_m=0.0)
    xs, ys = cell_centers(BOUNDS, 0.25)
    assert occ[int(np.argmin(np.abs(xs - 0.125))), int(np.argmin(np.abs(ys - 0.125)))]
    assert not occ[int(np.argmin(np.abs(xs - 1.125))), int(np.argmin(np.abs(ys - 0.125)))]
    assert occupancy(layout, 0.25, inflate_m=1.4).sum() > occ.sum()


def test_reachable_from_grows_through_free_cells_only():
    free = np.ones((5, 5), dtype=bool)
    free[2, :] = False
    seeds = np.zeros((5, 5), dtype=bool)
    seeds[0, 0] = True
    reach = reachable_from(free, seeds)
    assert reach[1, 4] and not reach[3, 0]


def test_empty_layout_is_reachable():
    result = check_reachability(Layout("t", 0, BOUNDS, ()), (2.0, 6.0), (0.0, 3.0))
    assert result.ok and result.unreachable_start_cells == 0 and result.free_start_cells > 0
    # No obstacles => p_extent is 0 on both axes (R13), but the corridor half-width is still 2*inflate_m (R14),
    # a plain strip through open space with nothing to block it -- genuinely 0, not vacuous.
    assert result.p_extent_m == {"x": 0.0, "y": 0.0}
    assert result.corridor_half_width_m == {"x": 2 * result.inflate_m, "y": 2 * result.inflate_m}
    assert result.crossing_unreachable_cells == {edge: 0 for edge in EDGES}


def test_full_width_wall_passes_the_spec_rule_but_fails_the_crossing_rule():
    wall = tuple(_post(f"w{k}", 0.0, -35.0 + 0.25 * k) for k in range(281))  # x = 0, from y = -35 to y = 35
    result = check_reachability(Layout("t", 0, BOUNDS, wall), (2.0, 6.0), (0.0, 3.0))
    assert result.unreachable_start_cells == 0  # every start cell still reaches its own edge's target ring
    assert not result.ok and "cross" in result.reason
    # The wall reaches the full y-extent of the scene (p_extent_y = 35.2 m), so the x-crossing corridor
    # (half-width min(35.2 + 2*1.4, 35) = 35 m, clipped to the scene bounds) covers essentially the whole
    # scene -- and the wall still splits it into two components end to end.
    assert result.crossing_unreachable_cells["x_min"] > 0 and result.crossing_unreachable_cells["x_max"] > 0
    # The wall sits at x = 0 (p_extent_x = 0.2 m), so the y-crossing corridor is only half-width
    # 0.2 + 2*1.4 = 3.0 m wide -- room to round the wall's flat face (radius 0.2 m + 1.4 m inflation = 1.6 m)
    # on each side, leaving a free lane. The wall spans the full y-range, so that lane runs start to target
    # uninterrupted -- 0 is a genuine pass here, not a vacuous one.
    assert result.crossing_unreachable_cells["y_min"] == 0 and result.crossing_unreachable_cells["y_max"] == 0


def test_enclosed_start_pocket_is_rejected():
    posts = []
    for k in range(0, 41):  # vertical walls at x=-33 and x=-25 from y=-5 to y=5
        y = -5.0 + 0.25 * k
        posts += [_post(f"w{k}a", -33.0, y), _post(f"w{k}b", -25.0, y)]
    for k in range(0, 33):  # horizontal walls at y=-5 and y=5 from x=-33 to x=-25
        x = -33.0 + 0.25 * k
        posts += [_post(f"h{k}a", x, -5.0), _post(f"h{k}b", x, 5.0)]
    result = check_reachability(Layout("t", 0, BOUNDS, tuple(posts)), (2.0, 6.0), (0.0, 3.0))
    assert not result.ok and result.unreachable_start_cells > 0
    assert "cannot reach" in result.reason


def test_s01_layout_is_reachable():
    scene = load_scene_file(SCENES_DIR / "s01_white_pillars.json")
    layout = generate_layout(scene, load_registry())
    result = check_reachability(layout, scene.start_band, scene.target_band)
    assert result.ok, result.reason
    assert result.inflate_m == 1.4
    assert result.crossing_unreachable_cells == {edge: 0 for edge in EDGES}


def test_single_isolated_obstacle_does_not_sever_its_own_corridor(tmp_path):
    # R13 regression (Critical, caught in review): _obstacle_extent sized the corridor from the bare radius_m,
    # while occupancy blocks radius_m + inflate_m. A single obstacle alone in the scene is then wide enough to
    # occupy the ENTIRE corridor R13 derived from its own (uninflated) extent, severing its own corridor and
    # rejecting a layout the spec rule already knows is flyable. count = 1 is a plausible easy/curriculum scene.
    data = json.loads((SCENES_DIR / "s01_white_pillars.json").read_text())
    data["obstacle_groups"][0]["count"] = 1
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(data))
    scene = load_scene_file(path)
    layout = generate_layout(scene, load_registry())
    assert len(layout.instances) == 1
    result = check_reachability(layout, scene.start_band, scene.target_band)
    assert result.ok, result.reason
    assert result.crossing_unreachable_cells == {edge: 0 for edge in EDGES}


def test_wall_confined_to_the_middle_is_accepted_under_r14():
    # A full-width wall at x = 0, confined to |y| <= 10. Under R13 alone (corridor half-width == p_extent,
    # about 10.2 m) this wall filled its own corridor edge to edge and was rejected. R14 widens the corridor
    # by 2*inflate_m -- one inflated obstacle's width of room on each side -- which is exactly enough to round
    # the two ends of this short wall: a route can go from the x_min band, out to |y| just past the wall's
    # inflated end, across x = 0, and back to the x_max band, all inside the (now wider) corridor. That route
    # rounds the wall -- here the whole obstacle field -- within inflate_m of it, rather than through open space
    # far away as the OLD start_band-derived corridor (fixed at |y| < 29) allowed. It does not pass THROUGH the
    # field: R14's corridor always leaves such a lane beside it (see the solid-block test below), so the rule
    # proves solvability only.
    wall = tuple(_post(f"w{k}", 0.0, -10.0 + 0.25 * k) for k in range(81))  # x = 0, from y = -10 to y = 10
    layout = Layout("t", 0, BOUNDS, wall)
    result = check_reachability(layout, (2.0, 6.0), (0.0, 3.0))
    assert result.ok, result.reason
    assert result.p_extent_m == {"x": pytest.approx(0.2), "y": pytest.approx(10.2)}
    assert result.corridor_half_width_m == {"x": pytest.approx(3.0), "y": pytest.approx(13.0)}
    assert result.crossing_unreachable_cells == {edge: 0 for edge in EDGES}


def test_corridor_half_width_is_p_extent_plus_twice_inflate_clipped_to_bounds():
    layout = Layout("t", 0, BOUNDS, (
        _post("a", 5.0, 3.0, radius=0.5),   # p_extent_x = 5.5 (unclipped), p_extent_y = 3.5 (unclipped)
        _post("b", 1.0, 33.0, radius=0.5),  # pushes p_extent_y to 33.5, so half-width 33.5 + 2.8 = 36.3 clips to 35
    ))
    result = check_reachability(layout, (2.0, 6.0), (0.0, 3.0))
    inflate = result.inflate_m
    assert result.p_extent_m == {"x": pytest.approx(5.5), "y": pytest.approx(33.5)}
    assert result.corridor_half_width_m["x"] == pytest.approx(5.5 + 2 * inflate)  # well inside the 35 m bound
    assert result.corridor_half_width_m["y"] == pytest.approx(35.0)  # 33.5 + 2*1.4 = 36.3 clipped to the bound


def test_a_solid_obstacle_block_passes_the_crossing_rule_which_proves_solvability_only():
    # C5 (2026-09-24 review), a documented limitation, pinned so it cannot silently change: the crossing corridor is
    # p_extent + 2*inflate wide, and every obstacle's inflated footprint ends by p_extent + inflate, so a free lane
    # `inflate` wide always runs beside the field from edge to edge. The rule therefore proves the crossing is
    # solvable; it does not force a route THROUGH the field -- even a solid 54 x 54 m block is accepted.
    block = tuple(_post(f"b{i}_{j}", float(i), float(j), radius=1.0) for i in range(-26, 27) for j in range(-26, 27))
    result = check_reachability(Layout("t", 0, BOUNDS, block), (2.0, 6.0), (0.0, 3.0))
    assert result.ok, result.reason
    assert result.corridor_half_width_m["y"] - (result.p_extent_m["y"] + result.inflate_m) == pytest.approx(result.inflate_m)
