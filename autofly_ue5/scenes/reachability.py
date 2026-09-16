"""2D occupancy reachability (spec §6.2) plus a crossing rule (controller ruling R13).

Spec rule: every free start-band cell must reach a free target-band cell, where the band is measured as
distance to the NEAREST of the four edges (spec §6.2). This alone accepts any layout whose outer ring is
free, including a full-width wall down the centre of the scene, because a route can always detour around the
obstacle field near the perimeter instead of crossing it.

Crossing rule (R13): a route must additionally be able to cross the scene through the obstacle field itself.
For each pair of opposite edges, the corridor is derived from the layout's OWN obstacle extent along the
perpendicular axis -- p_extent is the farthest any obstacle's surface reaches from the centreline -- rather
than from start_band. Tying the corridor to the obstacles' own footprint means a route cannot evade the field
by detouring through open space that the obstacles never reached, regardless of the scene's margin, jitter or
object sizes. When a layout has no obstacles, p_extent is 0, every corridor is empty, and the crossing rule is
vacuously satisfied -- the spec rule alone still governs such a layout.

The check only guarantees solvable layouts; it never produces actions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from autofly_ue5.scenes.model import Bounds, Instance, Layout

EDGES = ("x_min", "x_max", "y_min", "y_max")
OPPOSITE = {"x_min": "x_max", "x_max": "x_min", "y_min": "y_max", "y_max": "y_min"}
# Which coordinate axis bounds each edge's crossing corridor: an x_min<->x_max crossing is confined by how far
# obstacles reach in y, and a y_min<->y_max crossing by how far they reach in x.
CORRIDOR_AXIS = {"x_min": "y", "x_max": "y", "y_min": "x", "y_max": "x"}


@dataclass(frozen=True)
class ReachabilityResult:
    ok: bool
    resolution_m: float
    inflate_m: float
    free_start_cells: int
    free_target_cells: int
    unreachable_start_cells: int
    crossing_unreachable_cells: dict[str, int]
    reason: str

    def to_json(self) -> dict:
        return asdict(self)


def cell_centers(bounds: Bounds, resolution_m: float) -> tuple[np.ndarray, np.ndarray]:
    nx = int(round(bounds.width / resolution_m))
    ny = int(round(bounds.height / resolution_m))
    xs = bounds.x_min + (np.arange(nx) + 0.5) * resolution_m
    ys = bounds.y_min + (np.arange(ny) + 0.5) * resolution_m
    return xs, ys


def occupancy(layout: Layout, resolution_m: float, inflate_m: float) -> np.ndarray:
    b = layout.bounds
    xs, ys = cell_centers(b, resolution_m)
    occ = np.zeros((len(xs), len(ys)), dtype=bool)
    for inst in layout.instances:
        r = inst.radius_m + inflate_m
        ix0 = max(0, int(math.floor((inst.x - r - b.x_min) / resolution_m)))
        ix1 = min(len(xs), int(math.ceil((inst.x + r - b.x_min) / resolution_m)) + 1)
        iy0 = max(0, int(math.floor((inst.y - r - b.y_min) / resolution_m)))
        iy1 = min(len(ys), int(math.ceil((inst.y + r - b.y_min) / resolution_m)) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], indexing="ij")
        occ[ix0:ix1, iy0:iy1] |= (gx - inst.x) ** 2 + (gy - inst.y) ** 2 < r * r
    return occ


def edge_distances(bounds: Bounds, resolution_m: float) -> dict[str, np.ndarray]:
    xs, ys = cell_centers(bounds, resolution_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return {"x_min": gx - bounds.x_min, "x_max": bounds.x_max - gx, "y_min": gy - bounds.y_min, "y_max": bounds.y_max - gy}


def band_mask(bounds: Bounds, resolution_m: float, d_min: float, d_max: float) -> np.ndarray:
    d = np.minimum.reduce(list(edge_distances(bounds, resolution_m).values()))
    return (d >= d_min) & (d <= d_max)


def reachable_from(free: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    reach = seeds & free
    while True:
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= free
        if np.array_equal(grown, reach):
            return reach
        reach = grown


def _obstacle_extent(instances: tuple[Instance, ...], axis: str) -> float:
    """Farthest any obstacle's surface reaches from the centreline along `axis` ('x' or 'y'); 0 with none."""
    reach = [abs(getattr(inst, axis)) + inst.radius_m for inst in instances]
    return max(reach) if reach else 0.0


def crossing_unreachable(
    free: np.ndarray, layout: Layout, resolution_m: float, start_band: tuple[float, float], target_band: tuple[float, float]
) -> dict[str, int]:
    xs, ys = cell_centers(layout.bounds, resolution_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid = {"x": gx, "y": gy}
    p_extent = {axis: _obstacle_extent(layout.instances, axis) for axis in ("x", "y")}
    d = edge_distances(layout.bounds, resolution_m)
    counts: dict[str, int] = {}
    for edge in EDGES:
        axis = CORRIDOR_AXIS[edge]
        corridor = free & (np.abs(grid[axis]) <= p_extent[axis])
        far = OPPOSITE[edge]
        start = corridor & (d[edge] >= start_band[0]) & (d[edge] <= start_band[1])
        target = corridor & (d[far] >= target_band[0]) & (d[far] <= target_band[1])
        counts[edge] = int((start & ~reachable_from(corridor, target)).sum())
    return counts


def check_reachability(
    layout: Layout,
    start_band: tuple[float, float],
    target_band: tuple[float, float],
    clearance_m: float = 1.0,
    drone_radius_m: float = 0.4,
    resolution_m: float = 0.25,
) -> ReachabilityResult:
    inflate = drone_radius_m + clearance_m
    free = ~occupancy(layout, resolution_m, inflate)
    start = band_mask(layout.bounds, resolution_m, *start_band) & free
    target = band_mask(layout.bounds, resolution_m, *target_band) & free
    n_start, n_target = int(start.sum()), int(target.sum())
    if n_start == 0 or n_target == 0:
        return ReachabilityResult(False, resolution_m, inflate, n_start, n_target, n_start, {},
                                  "no free start cell" if n_start == 0 else "no free target cell")
    unreachable = int((start & ~reachable_from(free, target)).sum())
    crossing = crossing_unreachable(free, layout, resolution_m, start_band, target_band)
    if unreachable:
        reason = f"{unreachable} free start cells cannot reach any target cell"
    elif sum(crossing.values()):
        reason = f"free start cells cannot cross the obstacle field to the opposite edge: {crossing}"
    else:
        reason = "ok"
    ok = unreachable == 0 and sum(crossing.values()) == 0
    return ReachabilityResult(ok, resolution_m, inflate, n_start, n_target, unreachable, crossing, reason)
