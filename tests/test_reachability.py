import numpy as np

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import Bounds, Instance, Layout, load_registry, load_scene_file
from autofly_ue5.scenes.reachability import (
    EDGES,
    band_mask,
    cell_centers,
    check_reachability,
    edge_distances,
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
    # No obstacles => p_extent is 0 for both crossing directions (R13) => every corridor is empty => vacuously 0.
    assert result.crossing_unreachable_cells == {edge: 0 for edge in EDGES}


def test_full_width_wall_passes_the_spec_rule_but_fails_the_crossing_rule():
    wall = tuple(_post(f"w{k}", 0.0, -35.0 + 0.25 * k) for k in range(281))  # x = 0, from y = -35 to y = 35
    result = check_reachability(Layout("t", 0, BOUNDS, wall), (2.0, 6.0), (0.0, 3.0))
    assert result.unreachable_start_cells == 0  # every start cell still reaches its own edge's target ring
    assert not result.ok and "cross" in result.reason
    # The wall reaches the full y-extent of the scene (p_extent_y ~= 35.2 m), so under R13 the x-crossing
    # corridor covers essentially the whole scene -- and the wall still splits it into two components.
    assert result.crossing_unreachable_cells["x_min"] > 0 and result.crossing_unreachable_cells["x_max"] > 0
    # The wall sits at x = 0 (p_extent_x = 0.2 m), so the y-crossing corridor (|x| <= 0.2 m) is entirely
    # inside the wall's own inflated footprint and contains no free cells at all -- vacuously 0, not "passes".
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


def _old_style_crossing_unreachable(layout, start_band, target_band, resolution_m=0.25, inflate_m=1.4):
    """Re-creates the brief's original (pre-R13) crossing rule: the corridor is fixed by start_band[1] alone,
    not by the layout's own obstacle extent. Used only to prove the R13 regression this task fixes -- that a
    fixed corridor can be evaded by a layout whose obstacles do not reach it."""
    perpendicular = {"x_min": ("y_min", "y_max"), "x_max": ("y_min", "y_max"),
                      "y_min": ("x_min", "x_max"), "y_max": ("x_min", "x_max")}
    opposite = {"x_min": "x_max", "x_max": "x_min", "y_min": "y_max", "y_max": "y_min"}
    free = ~occupancy(layout, resolution_m, inflate_m)
    d = edge_distances(layout.bounds, resolution_m)
    counts = {}
    for edge in EDGES:
        side_a, side_b = perpendicular[edge]
        corridor = free & (d[side_a] > start_band[1]) & (d[side_b] > start_band[1])
        start = corridor & (d[edge] >= start_band[0]) & (d[edge] <= start_band[1])
        far = opposite[edge]
        target = corridor & (d[far] >= target_band[0]) & (d[far] <= target_band[1])
        counts[edge] = int((start & ~reachable_from(corridor, target)).sum())
    return counts


def test_obstacles_confined_to_the_middle_evade_the_old_fixed_corridor_but_not_r13():
    # A full-width wall at x = 0, but confined to |y| <= 10 (well inside the old fixed corridor boundary of
    # |y| < 29, which was derived from start_band[1] = 6). This is exactly the s01-style geometry the Task 12
    # review found: a route can go around the ends of the wall, through open space the OLD corridor still
    # counted as "inside the obstacle field", and cross the scene without ever weaving between obstacles.
    wall = tuple(_post(f"w{k}", 0.0, -10.0 + 0.25 * k) for k in range(81))  # x = 0, from y = -10 to y = 10
    layout = Layout("t", 0, BOUNDS, wall)
    start_band, target_band = (2.0, 6.0), (0.0, 3.0)

    old = _old_style_crossing_unreachable(layout, start_band, target_band)
    assert old["x_min"] == 0 and old["x_max"] == 0  # OLD rule: a detour through |y| in (10, 29) counts as crossing

    result = check_reachability(layout, start_band, target_band)
    assert result.unreachable_start_cells == 0  # the spec rule alone still can't tell (whole scene is one blob)
    assert not result.ok and "cross" in result.reason
    # R13: p_extent_y ~= 10.2 m, so the x-crossing corridor is |y| <= 10.2 m -- exactly where the wall sits --
    # and the wall fills that corridor edge to edge, so there is no detour left inside it.
    assert result.crossing_unreachable_cells["x_min"] > 0 and result.crossing_unreachable_cells["x_max"] > 0
