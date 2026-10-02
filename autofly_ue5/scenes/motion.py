"""Moving obstacles (spec §6.5): routes, per-episode mover sampling, and the per-step rules. Pure: no simulator.

A mover is one of the layout's obstacles that leaves home for an episode and follows a route whose position is an
analytic function of its own clock tau. Everything here works in the horizontal plane (NED x, y): s01's pillars are
8-12 m tall and the drone flies at 1-3 m, so every pillar spans the whole flight band.

- `sample_movers` picks the movers and their routes for one episode, enforces the sweep constraints and the
  path-length guard, and never raises: an explicit-seed evaluation replays a seed whose reset faulted, so a raise
  would loop.
- `yield_clocks` advances each mover's clock unless that would bring it toward the drone inside the yield distance:
  movers never ram the drone, so every contact is the policy's own doing.
- `first_contact` scores the paper's d_col against movers: the drone's swept step passes within `contact_m` of a
  mover's surface.

The rules and their evidence: docs/decisions/2026-10-02-dynamic-obstacles.md.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from autofly_ue5.frames import wrap_pi
from autofly_ue5.scenes.model import Bounds, DynamicSpec, Instance, Layout
from autofly_ue5.scenes.reachability import cell_centers, occupancy

MAX_ROUTE_TRIES = 20         # route draws per chosen pillar before it stays home
SWEEP_SAMPLE_M = 0.05        # spacing of the points a route is sampled at when comparing two sweeps
GUARD_RESOLUTION_M = 0.25    # the reachability grid's resolution (spec §6.2)
START_SEED_RADII_M = (0.5, 1.0)
VIEW_HALF_ANGLE_RAD = math.radians(45.0)  # the camera's 90 deg HFOV
VIEW_HISTORY_FRAMES = 3


@dataclass(frozen=True)
class CircleFootprint:
    """A mover's footprint in the flight band. Only circles today (s01's pillars); M4 adds others behind the same
    `radius_m`-style queries (canopy at flight altitude for trees, oriented boxes for vehicles)."""
    radius_m: float


@dataclass(frozen=True)
class MoverRoute:
    """One mover's route. `phase` is the fraction of the period at which tau = 0 starts. Pingpong: back and forth
    through home along `heading_rad`, `half_length_m` each way, starting from home outward at phase 0. Orbit: a circle
    of `orbit_radius_m` around home, at angle 2 pi phase when tau = 0, turning `direction` (+1 counter-clockwise in
    x-y, -1 clockwise)."""
    tag: str
    home_x: float
    home_y: float
    z: float
    footprint: CircleFootprint
    kind: str
    speed_m_s: float
    phase: float
    heading_rad: float = 0.0
    half_length_m: float = 0.0
    orbit_radius_m: float = 0.0
    direction: int = 1
    yaw: float = 0.0  # the pillar's own orientation, kept while it moves (the layout instance's yaw)

    @property
    def period_s(self) -> float:
        if self.kind == "pingpong":
            return 4.0 * self.half_length_m / self.speed_m_s
        return 2.0 * math.pi * self.orbit_radius_m / self.speed_m_s

    def position(self, tau: float) -> tuple[float, float]:
        if self.kind == "pingpong":
            u = (self.phase + tau / self.period_s) % 1.0
            if u < 0.25:
                s = 4.0 * u
            elif u < 0.75:
                s = 2.0 - 4.0 * u
            else:
                s = 4.0 * u - 4.0
            s *= self.half_length_m
            return self.home_x + s * math.cos(self.heading_rad), self.home_y + s * math.sin(self.heading_rad)
        angle = 2.0 * math.pi * self.phase + self.direction * self.speed_m_s * tau / self.orbit_radius_m
        return self.home_x + self.orbit_radius_m * math.cos(angle), self.home_y + self.orbit_radius_m * math.sin(angle)

    def path_distance(self, x: float, y: float) -> float:
        """Exact distance from (x, y) to the path the mover's centre sweeps."""
        if self.kind == "pingpong":
            ux, uy = math.cos(self.heading_rad), math.sin(self.heading_rad)
            t = max(-self.half_length_m, min(self.half_length_m, (x - self.home_x) * ux + (y - self.home_y) * uy))
            return math.hypot(x - (self.home_x + t * ux), y - (self.home_y + t * uy))
        return abs(math.hypot(x - self.home_x, y - self.home_y) - self.orbit_radius_m)

    def surface_distance(self, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest point the mover's footprint can ever occupy (negative inside)."""
        return self.path_distance(x, y) - self.footprint.radius_m

    def path_points(self, spacing_m: float = SWEEP_SAMPLE_M) -> np.ndarray:
        """(n, 2) points along the centre path, at most `spacing_m` apart."""
        if self.kind == "pingpong":
            n = max(2, math.ceil(2.0 * self.half_length_m / spacing_m) + 1)
            s = np.linspace(-self.half_length_m, self.half_length_m, n)
            return np.stack([self.home_x + s * math.cos(self.heading_rad), self.home_y + s * math.sin(self.heading_rad)], 1)
        n = max(8, math.ceil(2.0 * math.pi * self.orbit_radius_m / spacing_m))
        a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        return np.stack([self.home_x + self.orbit_radius_m * np.cos(a), self.home_y + self.orbit_radius_m * np.sin(a)], 1)

    def path_distances(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """`path_distance` over arrays (broadcast)."""
        if self.kind == "pingpong":
            ux, uy = math.cos(self.heading_rad), math.sin(self.heading_rad)
            t = np.clip((xs - self.home_x) * ux + (ys - self.home_y) * uy, -self.half_length_m, self.half_length_m)
            return np.hypot(xs - (self.home_x + t * ux), ys - (self.home_y + t * uy))
        return np.abs(np.hypot(xs - self.home_x, ys - self.home_y) - self.orbit_radius_m)

    def reach_m(self) -> float:
        """How far from home the footprint's edge can get."""
        extent = self.half_length_m if self.kind == "pingpong" else self.orbit_radius_m
        return extent + self.footprint.radius_m

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def sweep_gap(a: MoverRoute, b: MoverRoute) -> float:
    """A lower bound on the surface gap between two sweeps: a's path sampled every SWEEP_SAMPLE_M against b's exact
    path distance, less half the spacing (the most the sampling can overstate it)."""
    if math.hypot(a.home_x - b.home_x, a.home_y - b.home_y) > a.reach_m() + b.reach_m() + 10.0:
        return math.inf
    pts = a.path_points()
    centre_gap = float(b.path_distances(pts[:, 0], pts[:, 1]).min()) - SWEEP_SAMPLE_M / 2.0
    return centre_gap - a.footprint.radius_m - b.footprint.radius_m


def _segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    t = 0.0 if length2 == 0.0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


# --------------------------------------------------------------------------------------------------------
# Sampling.
# --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MoverSample:
    routes: tuple[MoverRoute, ...]
    stats: dict


def _draw_route(inst: Instance, spec: DynamicSpec, rng: np.random.Generator) -> MoverRoute:
    # Every parameter is drawn on every try, whatever the kind, so a try always consumes the same draws.
    kind = spec.route_kinds[int(rng.integers(len(spec.route_kinds)))]
    speed = float(rng.uniform(*spec.speed_m_s))
    phase = float(rng.uniform(0.0, 1.0))
    heading = float(rng.uniform(0.0, math.pi))
    half_length = float(rng.uniform(*spec.pingpong_half_length_m))
    orbit_radius = float(rng.uniform(*spec.orbit_radius_m))
    direction = 1 if rng.uniform() < 0.5 else -1
    common = dict(tag=inst.tag, home_x=inst.x, home_y=inst.y, z=inst.z_center, footprint=CircleFootprint(inst.radius_m),
                  kind=kind, speed_m_s=speed, phase=phase, yaw=inst.yaw)
    if kind == "pingpong":
        return MoverRoute(**common, heading_rad=heading, half_length_m=half_length)
    return MoverRoute(**common, orbit_radius_m=orbit_radius, direction=direction)


def _inside(route: MoverRoute, b: Bounds) -> bool:
    """The whole sweep, footprint included, lies inside the bounds: a pingpong's two ends, an orbit's extreme
    points (home +- radius on each axis)."""
    r = route.footprint.radius_m
    if route.kind == "pingpong":
        pts = route.path_points(2.0 * route.half_length_m)  # exactly the two ends
        xs, ys = pts[:, 0], pts[:, 1]
        return bool(xs.min() - r >= b.x_min and xs.max() + r <= b.x_max and ys.min() - r >= b.y_min and ys.max() + r <= b.y_max)
    reach = route.orbit_radius_m + r
    return (route.home_x - reach >= b.x_min and route.home_x + reach <= b.x_max
            and route.home_y - reach >= b.y_min and route.home_y + reach <= b.y_max)


def route_violations(route: MoverRoute, *, fixed: list[Instance], accepted: list[MoverRoute], bounds: Bounds,
                     start_xy: tuple[float, float], keep_clear_xy: list[tuple[float, float]], spec: DynamicSpec) -> list[str]:
    """Every §6.5 constraint the route breaks (empty: none). `fixed`: pillars that may stand at home this episode;
    `keep_clear_xy`: the target and distractors."""
    out = []
    if not _inside(route, bounds):
        out.append("bounds")
    if route.surface_distance(*start_xy) < spec.start_keepout_m:
        out.append("start")
    if any(route.surface_distance(x, y) < spec.target_keepout_m for x, y in keep_clear_xy):
        out.append("target")
    reach = route.reach_m() + spec.min_gap_m
    for inst in fixed:
        if math.hypot(inst.x - route.home_x, inst.y - route.home_y) > reach + inst.radius_m + 1.0:
            continue
        if route.surface_distance(inst.x, inst.y) - inst.radius_m < spec.min_gap_m:
            out.append(f"pillar:{inst.tag}")
            break
    for other in accepted:
        if sweep_gap(route, other) < spec.min_gap_m:
            out.append(f"sweep:{other.tag}")
            break
    return out


def revalidate_after_drops(routes: list[MoverRoute], instances: list[Instance], spec: DynamicSpec,
                           ) -> tuple[list[MoverRoute], list[str]]:
    """Routes that still keep min_gap_m from every pillar now standing at home, and the tags dropped for it. A mover the
    guard dropped goes home, but movers accepted after it were checked against its sweep only: for an orbit, home is
    the circle's centre, so a later sweep inside the ring can be too close to it. Dropping one can expose another, so
    this repeats until nothing changes. (Impossible on s01, whose centres are >= 4.42 m apart and whose sweeps reach at
    most 2.5 m; kept for the layouts and parameters to come.)"""
    kept, dropped = list(routes), []
    while True:
        moving = {r.tag for r in kept}
        homes = [i for i in instances if i.tag not in moving]
        bad = next((r for r in kept if any(
            i.tag != r.tag and r.surface_distance(i.x, i.y) - i.radius_m < spec.min_gap_m for i in homes)), None)
        if bad is None:
            return kept, dropped
        kept.remove(bad)
        dropped.append(bad.tag)


def _sweep_mask(route: MoverRoute, xs: np.ndarray, ys: np.ndarray, b: Bounds, resolution_m: float, grow_m: float) -> np.ndarray:
    mask = np.zeros((len(xs), len(ys)), dtype=bool)
    r = route.reach_m() + grow_m
    ix0 = max(0, int(math.floor((route.home_x - r - b.x_min) / resolution_m)))
    ix1 = min(len(xs), int(math.ceil((route.home_x + r - b.x_min) / resolution_m)) + 1)
    iy0 = max(0, int(math.floor((route.home_y - r - b.y_min) / resolution_m)))
    iy1 = min(len(ys), int(math.ceil((route.home_y + r - b.y_min) / resolution_m)) + 1)
    if ix0 >= ix1 or iy0 >= iy1:
        return mask
    gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], indexing="ij")
    mask[ix0:ix1, iy0:iy1] = route.path_distances(gx, gy) < route.footprint.radius_m + grow_m
    return mask


def _disk(xs: np.ndarray, ys: np.ndarray, x: float, y: float, radius: float) -> np.ndarray:
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return (gx - x) ** 2 + (gy - y) ** 2 <= radius * radius


def bfs_path_m(free: np.ndarray, sources: np.ndarray, goals: np.ndarray, resolution_m: float) -> float | None:
    """4-connected shortest path, in metres, from any free source cell to any free goal cell; None if unreachable."""
    reach = sources & free
    goals = goals & free
    if not reach.any() or not goals.any():
        return None
    frontier, steps = reach.copy(), 0
    while True:
        if (frontier & goals).any():
            return steps * resolution_m
        grown = np.zeros_like(frontier)
        grown[1:, :] |= frontier[:-1, :]
        grown[:-1, :] |= frontier[1:, :]
        grown[:, 1:] |= frontier[:, :-1]
        grown[:, :-1] |= frontier[:, 1:]
        grown &= free & ~reach
        if not grown.any():
            return None
        reach |= grown
        frontier = grown
        steps += 1


@dataclass(frozen=True)
class _Guard:
    layout: Layout
    xs: np.ndarray
    ys: np.ndarray
    sources: np.ndarray  # the target's success disk
    starts: np.ndarray
    inflate_m: float
    resolution_m: float

    def length(self, homes_kept: tuple[Instance, ...], routes: list[MoverRoute]) -> float | None:
        occ = occupancy(dataclasses.replace(self.layout, instances=homes_kept), self.resolution_m, self.inflate_m)
        for route in routes:
            occ |= _sweep_mask(route, self.xs, self.ys, self.layout.bounds, self.resolution_m, self.inflate_m)
        return bfs_path_m(~occ, self.sources, self.starts, self.resolution_m)


def _guard(layout: Layout, start_xy, target_xy, *, inflate_m: float, success_radius_m: float, resolution_m: float) -> _Guard:
    xs, ys = cell_centers(layout.bounds, resolution_m)
    free_static = ~occupancy(layout, resolution_m, inflate_m)
    starts = np.zeros_like(free_static)
    for radius in START_SEED_RADII_M:  # the start sits on the inflation boundary by construction (episode.py)
        starts = _disk(xs, ys, *start_xy, radius) & free_static
        if starts.any():
            break
    return _Guard(layout, xs, ys, _disk(xs, ys, *target_xy, success_radius_m), starts, inflate_m, resolution_m)


def sample_movers(
    spec: DynamicSpec, layout: Layout, *, start_xy: tuple[float, float], target_xy: tuple[float, float],
    distractors_xy: list[tuple[float, float]], rng: np.random.Generator, inflate_m: float, success_radius_m: float,
    resolution_m: float = GUARD_RESOLUTION_M,
) -> MoverSample:
    """The movers of one episode (spec §6.5). Fixed draw order: mover count, path-mover count, path movers, other
    movers, then each mover's route tries in turn. Never raises."""
    k = int(rng.integers(spec.count[0], spec.count[1] + 1))
    n_path = int(rng.integers(spec.path_movers[0], spec.path_movers[1] + 1))
    instances = list(layout.instances)
    near = [i for i in instances if _segment_distance(i.x, i.y, *start_xy, *target_xy) <= spec.path_corridor_m]
    chosen = [near[j] for j in rng.permutation(len(near))[:min(n_path, len(near), k)]]
    chosen_tags = {i.tag for i in chosen}
    rest = [i for i in instances if i.tag not in chosen_tags]
    chosen += [rest[j] for j in rng.permutation(len(rest))[:max(0, min(k, len(instances)) - len(chosen))]]
    n_near_chosen = sum(1 for i in chosen if i in near)

    accepted: list[MoverRoute] = []
    accepted_tags: set[str] = set()
    unplaced: dict[str, list[str]] = {}
    keep_clear = [tuple(target_xy)] + [tuple(d) for d in distractors_xy]
    for inst in chosen:
        fixed = [i for i in instances if i.tag != inst.tag and i.tag not in accepted_tags]
        reasons: list[str] = []
        for _ in range(MAX_ROUTE_TRIES):
            route = _draw_route(inst, spec, rng)
            reasons = route_violations(route, fixed=fixed, accepted=accepted, bounds=layout.bounds, start_xy=start_xy,
                                       keep_clear_xy=keep_clear, spec=spec)
            if not reasons:
                accepted.append(route)
                accepted_tags.add(inst.tag)
                break
        else:
            unplaced[inst.tag] = reasons

    guard = _guard(layout, start_xy, target_xy, inflate_m=inflate_m, success_radius_m=success_radius_m,
                   resolution_m=resolution_m)
    static_m = guard.length(layout.instances, [])

    def homes_without(routes: list[MoverRoute]) -> tuple[Instance, ...]:
        moving = {r.tag for r in routes}
        return tuple(i for i in layout.instances if i.tag not in moving)

    dropped_by_guard: list[str] = []
    if static_m is None:
        # Never seen on s01: the static scene has no path the guard can measure, so nothing moves this episode.
        dropped_by_guard = [r.tag for r in accepted]
        accepted, dynamic_m, outcome = [], None, "static_unreachable"
    else:
        dynamic_m = guard.length(homes_without(accepted), accepted)
        outcome = "ok"
        limit = spec.max_path_ratio * static_m
        while accepted and (dynamic_m is None or dynamic_m > limit):
            outcome = "repaired"
            best_i, best_m = 0, None
            for i in range(len(accepted)):
                trial = accepted[:i] + accepted[i + 1:]
                m = guard.length(homes_without(trial), trial)
                if m is not None and (best_m is None or m < best_m):
                    best_i, best_m = i, m
            dropped_by_guard.append(accepted[best_i].tag)
            accepted = accepted[:best_i] + accepted[best_i + 1:]
            dynamic_m = best_m if accepted else static_m
    dropped_after_guard: list[str] = []
    if dropped_by_guard:
        accepted, dropped_after_guard = revalidate_after_drops(accepted, list(layout.instances), spec)
        if dropped_after_guard:  # fewer sweeps can only shorten the path; measure what is flown
            dynamic_m = guard.length(homes_without(accepted), accepted) if accepted else static_m
    stats = {
        "k": k, "n_path_requested": n_path, "n_near_line": len(near), "n_near_chosen": n_near_chosen,
        "chosen": [i.tag for i in chosen], "placed": len(accepted), "unplaced": unplaced,
        "dropped_by_guard": dropped_by_guard, "dropped_after_guard": dropped_after_guard, "guard": outcome,
        "path_static_m": static_m, "path_dynamic_m": dynamic_m,
        "path_ratio": (dynamic_m / static_m) if (static_m and dynamic_m is not None) else None,
    }
    return MoverSample(routes=tuple(accepted), stats=stats)


# --------------------------------------------------------------------------------------------------------
# Per-step rules.
# --------------------------------------------------------------------------------------------------------
def yield_clocks(routes: tuple[MoverRoute, ...], taus: list[float], drone_xy: tuple[float, float], dt: float,
                 *, yield_distance_m: float) -> list[float]:
    """Each mover's next clock: tau + dt, unless that pose brings its surface toward the drone while inside
    `yield_distance_m` of it, in which case it waits (tau unchanged)."""
    out = []
    for route, tau in zip(routes, taus):
        cx, cy = route.position(tau)
        nx, ny = route.position(tau + dt)
        r = route.footprint.radius_m
        d_now = math.hypot(cx - drone_xy[0], cy - drone_xy[1]) - r
        d_next = math.hypot(nx - drone_xy[0], ny - drone_xy[1]) - r
        out.append(tau + dt if d_next >= yield_distance_m or d_next >= d_now else tau)
    return out


def surface_gaps(routes: tuple[MoverRoute, ...], positions: list[tuple[float, float]], a_xy: tuple[float, float],
                 b_xy: tuple[float, float] | None = None) -> list[float]:
    """Each mover's surface distance from the drone's point a_xy, or from its swept segment a_xy -> b_xy."""
    b_xy = a_xy if b_xy is None else b_xy
    return [_segment_distance(px, py, *a_xy, *b_xy) - route.footprint.radius_m
            for route, (px, py) in zip(routes, positions)]


def first_contact(routes, positions, a_xy, b_xy, contact_m: float) -> tuple[int, float] | None:
    """(index, surface gap) of the mover the drone's swept step a_xy -> b_xy came closest to, if within contact_m."""
    gaps = surface_gaps(routes, positions, a_xy, b_xy)
    if not gaps:
        return None
    i = int(np.argmin(gaps))
    return (i, gaps[i]) if gaps[i] <= contact_m else None


def in_view(drone_pose: tuple[float, float, float], point_xy: tuple[float, float],
            half_angle_rad: float = VIEW_HALF_ANGLE_RAD) -> bool:
    """Whether point_xy lies within the camera's horizontal field of view from drone_pose (x, y, yaw)."""
    x, y, yaw = drone_pose
    return abs(wrap_pi(math.atan2(point_xy[1] - y, point_xy[0] - x) - yaw)) <= half_angle_rad
