"""autofly_ue5/scenes/motion.py: routes, mover sampling with its constraints and path-length guard, yielding and
contact (spec §6.5). Statistics come from training-range seeds only, never the gate's EVAL_SEED_BASE episodes."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from autofly_ue5.scenes.model import Bounds, DynamicSpec, Instance
from autofly_ue5.scenes.motion import (
    CircleFootprint,
    MoverRoute,
    first_contact,
    in_view,
    sample_movers,
    surface_gaps,
    sweep_gap,
    yield_clocks,
)
from tests.test_expert_episode import scene_and_layout

SPEC = DynamicSpec(count=(8, 12), path_movers=(2, 4), path_corridor_m=4.0, route_kinds=("pingpong", "orbit"),
                   speed_m_s=(0.4, 1.2), pingpong_half_length_m=(1.0, 2.5), orbit_radius_m=(1.0, 2.0), min_gap_m=1.0,
                   contact_m=1.0, yield_margin_m=1.5, start_keepout_m=6.0, target_keepout_m=4.0, max_path_ratio=1.2)
TRAINING_SEEDS = range(1_000_000, 1_000_150)  # worker 0's range (autofly_ue5.expert.seeds)


def _pingpong(**kw) -> MoverRoute:
    args = dict(tag="obs_0000", home_x=0.0, home_y=0.0, z=-5.0, footprint=CircleFootprint(0.5), kind="pingpong",
                speed_m_s=1.0, phase=0.0, heading_rad=0.0, half_length_m=2.0)
    args.update(kw)
    return MoverRoute(**args)


def _orbit(**kw) -> MoverRoute:
    args = dict(tag="obs_0001", home_x=0.0, home_y=0.0, z=-5.0, footprint=CircleFootprint(0.5), kind="orbit",
                speed_m_s=1.0, phase=0.0, orbit_radius_m=1.5, direction=1)
    args.update(kw)
    return MoverRoute(**args)


# --------------------------------------------------------------------------------------------------------
# Routes.
# --------------------------------------------------------------------------------------------------------
def test_a_pingpong_goes_out_back_through_home_and_out_the_other_way_at_constant_speed():
    r = _pingpong(heading_rad=math.pi / 2)
    assert r.period_s == pytest.approx(8.0)
    for tau, expected in ((0.0, (0.0, 0.0)), (2.0, (0.0, 2.0)), (4.0, (0.0, 0.0)), (6.0, (0.0, -2.0)), (8.0, (0.0, 0.0))):
        assert r.position(tau) == pytest.approx(expected, abs=1e-9)
    for tau in np.linspace(0.05, 7.9, 60):
        if min(abs(tau - t) for t in (2.0, 6.0)) < 0.02:
            continue  # the turnarounds
        (x0, y0), (x1, y1) = r.position(tau), r.position(tau + 0.01)
        assert math.hypot(x1 - x0, y1 - y0) / 0.01 == pytest.approx(1.0, rel=1e-6)
    assert r.position(1.0) == _pingpong(heading_rad=math.pi / 2, phase=0.125).position(0.0)


def test_an_orbit_circles_home_at_its_speed_and_direction():
    r = _orbit(speed_m_s=0.75, phase=0.25)
    assert r.position(0.0) == pytest.approx((0.0, 1.5))
    for tau in np.linspace(0.0, 20.0, 50):
        x, y = r.position(tau)
        assert math.hypot(x, y) == pytest.approx(1.5)
        (x1, y1) = r.position(tau + 0.01)
        assert math.hypot(x1 - x, y1 - y) / 0.01 == pytest.approx(0.75, rel=1e-4)
    ccw, cw = r.position(0.1), _orbit(speed_m_s=0.75, phase=0.25, direction=-1).position(0.1)
    assert ccw[0] < 0.0 < cw[0], "+1 turns counter-clockwise in x-y"


def test_path_distances_are_exact():
    p = _pingpong()  # from (-2, 0) to (2, 0)
    assert p.path_distance(0.0, 3.0) == pytest.approx(3.0)
    assert p.path_distance(5.0, 4.0) == pytest.approx(5.0)
    assert p.surface_distance(0.0, 3.0) == pytest.approx(2.5)
    o = _orbit()
    assert o.path_distance(0.0, 0.0) == pytest.approx(1.5) and o.path_distance(4.0, 0.0) == pytest.approx(2.5)
    xs, ys = np.array([0.0, 5.0, 1.0]), np.array([3.0, 4.0, 0.0])
    assert p.path_distances(xs, ys) == pytest.approx([p.path_distance(x, y) for x, y in zip(xs, ys)])
    assert o.path_distances(xs, ys) == pytest.approx([o.path_distance(x, y) for x, y in zip(xs, ys)])


def test_the_sweep_gap_never_overstates_the_true_gap():
    a, b = _pingpong(), _pingpong(tag="obs_0002", home_y=3.0)
    assert 2.0 - 0.05 <= sweep_gap(a, b) <= 2.0
    c = _orbit(home_x=5.0)  # circle around (5, 0) of radius 1.5: nearest path point (3.5, 0); a ends at (2, 0)
    assert 1.5 - 1.0 - 0.05 <= sweep_gap(a, c) <= 0.5
    assert sweep_gap(a, _orbit(home_x=60.0)) == math.inf


# --------------------------------------------------------------------------------------------------------
# Sampling.
# --------------------------------------------------------------------------------------------------------
def _episode(seed: int):
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()  # static s01: its draws come first, exactly as in s01d
    rng = np.random.default_rng(seed)
    setup = sample_setup(scene, layout, rng)
    sample = sample_movers(SPEC, layout, start_xy=(setup.start.x, setup.start.y), target_xy=setup.target_xy_z[:2],
                           distractors_xy=[d[:2] for d in setup.distractors], rng=rng, inflate_m=1.4,
                           success_radius_m=5.0)
    return layout, setup, sample


def _dense(route: MoverRoute, spacing: float = 0.02) -> np.ndarray:
    return route.path_points(spacing)


def _independent_violations(layout, setup, routes) -> list[str]:
    """Every constraint re-checked by brute force on densely sampled positions, independently of motion.py's own."""
    out = []
    moving = {r.tag for r in routes}
    b = layout.bounds
    for r in routes:
        pts = _dense(r)
        rad = r.footprint.radius_m
        if (pts[:, 0] - rad < b.x_min - 1e-9).any() or (pts[:, 0] + rad > b.x_max + 1e-9).any() or \
                (pts[:, 1] - rad < b.y_min - 1e-9).any() or (pts[:, 1] + rad > b.y_max + 1e-9).any():
            out.append(f"{r.tag}: leaves the bounds")
        start = np.hypot(pts[:, 0] - setup.start.x, pts[:, 1] - setup.start.y).min() - rad
        if start < SPEC.start_keepout_m - 1e-9:
            out.append(f"{r.tag}: {start:.2f} m from the start")
        for x, y, _z in (setup.target_xy_z, *setup.distractors):
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y).min() - rad
            if d < SPEC.target_keepout_m - 1e-9:
                out.append(f"{r.tag}: {d:.2f} m from the target or a distractor")
        for inst in layout.instances:
            if inst.tag == r.tag or inst.tag in moving:
                continue
            d = np.hypot(pts[:, 0] - inst.x, pts[:, 1] - inst.y).min() - rad - inst.radius_m
            if d < SPEC.min_gap_m - 1e-9:
                out.append(f"{r.tag}: {d:.2f} m from pillar {inst.tag}")
        for other in routes:
            if other.tag <= r.tag:
                continue
            if math.hypot(r.home_x - other.home_x, r.home_y - other.home_y) > r.reach_m() + other.reach_m() + SPEC.min_gap_m:
                continue  # too far apart for any two points of their sweeps to come within min_gap
            q = _dense(other)
            d = np.sqrt(((pts[:, None, :] - q[None, :, :]) ** 2).sum(-1)).min() - rad - other.footprint.radius_m
            if d < SPEC.min_gap_m - 0.02:
                out.append(f"{r.tag}/{other.tag}: sweeps {d:.2f} m apart")
    return out


def test_every_constraint_and_the_path_guard_hold_on_training_seeds():
    placed, near, durations, repaired = [], 0, [], 0
    for seed in TRAINING_SEEDS:
        t0 = time.perf_counter()
        layout, setup, sample = _episode(seed)
        durations.append(time.perf_counter() - t0)
        routes, stats = sample.routes, sample.stats
        assert _independent_violations(layout, setup, routes) == [], f"seed {seed}"
        assert len(routes) <= stats["k"] and SPEC.count[0] <= stats["k"] <= SPEC.count[1]
        assert len({r.tag for r in routes}) == len(routes)
        if stats["path_ratio"] is not None:
            assert stats["path_ratio"] <= SPEC.max_path_ratio + 1e-9, f"seed {seed}: {stats}"
        assert stats["guard"] in ("ok", "repaired")
        repaired += stats["guard"] == "repaired"
        placed.append(len(routes))
        segment = ((setup.start.x, setup.start.y), setup.target_xy_z[:2])
        near += any(_near_line(r, *segment) for r in routes)
    assert np.mean(placed) >= 8.0 and min(placed) >= 2, (np.mean(placed), min(placed))
    assert near / len(placed) >= 0.9, "nearly every episode must put a mover by the flight line"
    assert repaired <= 0.2 * len(placed)
    assert np.median(durations) < 0.5, f"median {np.median(durations):.3f} s per reset"


def _near_line(route: MoverRoute, a, b) -> bool:
    from autofly_ue5.scenes.motion import _segment_distance

    return _segment_distance(route.home_x, route.home_y, *a, *b) <= SPEC.path_corridor_m


def test_the_same_seed_draws_the_same_movers_and_routes_cover_both_kinds():
    _l, _s, first = _episode(1_000_007)
    _l, _s, again = _episode(1_000_007)
    assert first.routes == again.routes and first.stats == again.stats
    kinds = {r.kind for seed in range(2_000_000, 2_000_010) for r in _episode(seed)[2].routes}
    assert kinds == {"pingpong", "orbit"}
    for r in first.routes:
        assert SPEC.speed_m_s[0] <= r.speed_m_s <= SPEC.speed_m_s[1] and 0.0 <= r.phase < 1.0
        if r.kind == "pingpong":
            assert SPEC.pingpong_half_length_m[0] <= r.half_length_m <= SPEC.pingpong_half_length_m[1]
        else:
            assert SPEC.orbit_radius_m[0] <= r.orbit_radius_m <= SPEC.orbit_radius_m[1] and r.direction in (-1, 1)


def test_the_guard_drops_a_mover_that_seals_the_only_gap_and_never_raises():
    # A wall of pillars along x = 0 (their inflated disks overlap) with one gap around y = 0, and one pillar in the
    # middle of that gap. Statically the gap stays open on both sides of that pillar; orbiting at 2 m it sweeps the
    # whole gap shut, forcing a detour round the end of the wall, so the guard must drop it -- and not raise.
    from autofly_ue5.scenes.model import Layout

    ys = [0.0] + [sign * y for y in range(6, 31, 3) for sign in (1, -1)]
    pillars = tuple(Instance(tag=f"obs_{i:04d}", asset="cylinder", x=0.0, y=float(y), z_center=-5.0, yaw=0.0,
                             scale=(1.0, 1.0, 10.0), material="white", radius_m=0.5, height_m=10.0)
                    for i, y in enumerate(ys))
    layout = Layout(scene_id="s99", seed=0, bounds=Bounds(-35.0, 35.0, -35.0, 35.0), instances=pillars)
    spec = DynamicSpec(**{**SPEC.__dict__, "count": (1, 1), "path_movers": (1, 1), "path_corridor_m": 1.0,
                          "min_gap_m": 0.0, "start_keepout_m": 0.0, "target_keepout_m": 0.0, "max_path_ratio": 1.2,
                          "orbit_radius_m": (2.0, 2.0), "route_kinds": ("orbit",)})
    for seed in range(5):
        result = sample_movers(spec, layout, start_xy=(-20.0, 0.0), target_xy=(20.0, 0.0), distractors_xy=[],
                               rng=np.random.default_rng(seed), inflate_m=1.4, success_radius_m=5.0)
        assert result.stats["chosen"] == ["obs_0000"]
        assert result.stats["guard"] == "repaired" and result.stats["dropped_by_guard"] == ["obs_0000"]
        assert result.routes == () and result.stats["path_ratio"] == pytest.approx(1.0)
    # The same mover swinging along the wall (a pingpong on the y axis) leaves a gap: kept.
    spec_kept = DynamicSpec(**{**spec.__dict__, "route_kinds": ("pingpong",), "pingpong_half_length_m": (0.5, 0.5)})
    kept = sample_movers(spec_kept, layout, start_xy=(-20.0, 0.0), target_xy=(20.0, 0.0), distractors_xy=[],
                         rng=np.random.default_rng(0), inflate_m=1.4, success_radius_m=5.0)
    assert kept.stats["guard"] == "ok" and len(kept.routes) == 1 and kept.stats["path_ratio"] <= 1.2


# --------------------------------------------------------------------------------------------------------
# Per-step rules.
# --------------------------------------------------------------------------------------------------------
def test_a_mover_never_closes_on_a_hovering_drone_but_may_leave():
    route = _pingpong(half_length_m=2.5, speed_m_s=1.2, heading_rad=0.0)  # sweeps x in [-2.5, 2.5] along y = 0
    yield_m = 2.5
    for drone in ((5.6, 0.0), (0.0, 2.0), (-4.0, 0.5)):
        taus = [0.0]
        start_gap = surface_gaps((route,), [route.position(0.0)], drone)[0]
        floor = min(start_gap, yield_m)
        for _ in range(400):
            taus = yield_clocks((route,), taus, drone, 0.2, yield_distance_m=yield_m)
            gap = surface_gaps((route,), [route.position(taus[0])], drone)[0]
            assert gap >= floor - 1e-9, f"drone at {drone}: the mover closed to {gap:.3f} m"
    # Far away nothing waits.
    assert yield_clocks((route,), [1.0], (30.0, 30.0), 0.2, yield_distance_m=yield_m) == [1.2]


def test_a_waiting_mover_resumes_once_the_drone_has_passed():
    route = _pingpong(half_length_m=2.5, speed_m_s=1.0)
    taus = [0.0]
    for _ in range(3):
        taus = yield_clocks((route,), taus, (2.9, 0.0), 0.2, yield_distance_m=2.5)  # 2.4 m off, on the outward leg
    assert taus == [0.0]
    assert yield_clocks((route,), taus, (2.9, 20.0), 0.2, yield_distance_m=2.5) == [0.2]


def test_contact_uses_the_swept_step_not_just_its_end():
    route = _pingpong()
    positions = [(0.0, 0.0)]
    # The drone crosses right over the mover between two observations: both ends are far, the segment is not.
    assert first_contact((route,), positions, (-3.0, 0.3), (3.0, 0.3), 1.0) == (0, pytest.approx(-0.2))
    assert first_contact((route,), positions, (-3.0, 2.0), (3.0, 2.0), 1.0) is None
    assert first_contact((), [], (0.0, 0.0), (1.0, 0.0), 1.0) is None


def test_in_view_is_the_cameras_horizontal_field_of_view():
    assert in_view((0.0, 0.0, 0.0), (5.0, 4.9))
    assert not in_view((0.0, 0.0, 0.0), (5.0, 5.2))
    assert not in_view((0.0, 0.0, 0.0), (-5.0, 0.0))
    assert in_view((0.0, 0.0, math.pi), (-5.0, 0.0))
