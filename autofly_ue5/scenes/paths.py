"""Shortest paths on the §6.2 occupancy grid, for two measurements the project needs and the paper leaves to us.

Spec §13 (M4): the reachability rule proves a crossing is *solvable*, not that it passes through the obstacle field; a
sealed field passes when the free lane beside it is open (`reachability.py`). `crossing_detours` flies evenly spaced
start-band cells on each edge straight across to the opposite target band and reports two things: how much longer the
shortest crossing is than the straight line (a permeable field costs a few percent, a sealed one the way round), and
whether a crossing exists *inside the field's own lateral extent*, where going round is not allowed. The second test is
what catches a compact sealed block, which costs little to round (a 20 x 20 m block of pillars reads 1.14) but which no
episode would ever fly through.

Spec §8 (M6): the paper's path efficiency L_opt / max(L, L_opt) needs an optimal path length. `optimal_path_m` is the
shortest grid path from the start to the success disc around the target. Two conventions are ours, recorded with
every result: the grid is inflated by the caller's `inflate_m` (the drone's rotor half-span 0.48 m gives the
shortest physically collision-free path; 1.4 m, the sampler's clearance, a path the expert was trained to keep), and
8-connected moves with sqrt(2) diagonals and no corner cutting overestimate a straight line by at most about 8 %.

`motion.py` keeps its 4-connected BFS: the mover guard compares two path lengths on the same metric, where the
Manhattan bias cancels; here lengths are compared with straight lines and with flown trajectories, so they must be
near-Euclidean.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from autofly_ue5.scenes.model import Layout
from autofly_ue5.scenes.reachability import (
    CORRIDOR_AXIS,
    EDGES,
    OPPOSITE,
    _obstacle_extent,
    cell_centers,
    edge_distances,
    occupancy,
)

START_SEED_RADII_M = (0.0, 0.5, 1.0, 1.5)  # a start may sit on the inflation boundary (episode.py draws it exactly there)
_MOVES = tuple((dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0))


def free_grid(layout: Layout, *, resolution_m: float, inflate_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(free cells, cell-centre xs, cell-centre ys) of the layout's occupancy grid inflated by `inflate_m`."""
    xs, ys = cell_centers(layout.bounds, resolution_m)
    return ~occupancy(layout, resolution_m, inflate_m), xs, ys


def dijkstra_path_m(free: np.ndarray, sources: np.ndarray, goals: np.ndarray, resolution_m: float, *,
                    source_costs_m: np.ndarray | None = None) -> float | None:
    """Length in metres of the shortest 8-connected path from any free source cell to any free goal cell, diagonals
    costing sqrt(2) and never cutting the corner of an occupied cell; None if no goal is reachable. `source_costs_m`
    gives each source cell a starting length (its distance from the real start), 0 without."""
    free = np.asarray(free, dtype=bool)
    starts = np.argwhere(sources & free)
    goal = goals & free
    if len(starts) == 0 or not goal.any():
        return None
    nx, ny = free.shape
    dist = np.full(free.shape, np.inf)
    heap: list[tuple[float, int, int]] = []
    for ix, iy in starts:
        cost = 0.0 if source_costs_m is None else float(source_costs_m[ix, iy]) / resolution_m
        dist[ix, iy] = cost
        heap.append((cost, int(ix), int(iy)))
    heapq.heapify(heap)
    while heap:
        d, ix, iy = heapq.heappop(heap)
        if d > dist[ix, iy]:
            continue
        if goal[ix, iy]:
            return d * resolution_m
        for dx, dy, cost in _MOVES:
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < nx and 0 <= jy < ny) or not free[jx, jy]:
                continue
            if dx and dy and not (free[ix + dx, iy] and free[ix, iy + dy]):
                continue  # no squeezing diagonally between two occupied cells
            nd = d + cost
            if nd < dist[jx, jy]:
                dist[jx, jy] = nd
                heapq.heappush(heap, (nd, jx, jy))
    return None


def _disk(xs: np.ndarray, ys: np.ndarray, x: float, y: float, radius_m: float) -> np.ndarray:
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return (gx - x) ** 2 + (gy - y) ** 2 <= radius_m * radius_m


def optimal_path_m(layout: Layout, start_xy, goal_xy, *, goal_radius_m: float, inflate_m: float, resolution_m: float) -> float | None:
    """L_opt: the shortest path from the start to within `goal_radius_m` of the goal on the inflated grid, in metres (to
    about one cell); None when the start has no free cell within START_SEED_RADII_M or the goal disc is unreachable.
    Each candidate source cell starts at its own distance from the real start, so a start on the inflation boundary
    (episode.py draws them there) costs the way to the nearest free cell instead of being granted it."""
    free, xs, ys = free_grid(layout, resolution_m=resolution_m, inflate_m=inflate_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    from_start = np.hypot(gx - start_xy[0], gy - start_xy[1])
    sources = np.zeros_like(free)
    for radius in START_SEED_RADII_M:
        sources = (from_start <= max(radius, resolution_m * 0.75)) & free
        if sources.any():
            break
    else:
        return None
    return dijkstra_path_m(free, sources, _disk(xs, ys, goal_xy[0], goal_xy[1], goal_radius_m), resolution_m,
                           source_costs_m=from_start)


def _spaced(values: np.ndarray, n: int) -> np.ndarray:
    """Indices of `n` entries spread evenly over the sorted order of `values` (all of them if there are fewer)."""
    order = np.argsort(values, kind="stable")
    if len(order) <= n:
        return order
    return order[np.round(np.linspace(0, len(order) - 1, n)).astype(int)]


def crossing_detours(layout: Layout, *, start_band: tuple[float, float], target_band: tuple[float, float], inflate_m: float,
                     resolution_m: float, samples_per_edge: int = 12, goal_half_width_m: float = 3.0) -> dict:
    """For each edge, `samples_per_edge` free start-band cells spread evenly along it, each flown to the opposite target
    band *straight across* (the band's cells within `goal_half_width_m` of the start's lateral position, widened until
    the window holds a free cell). Per sample: the shortest path over the straight line to the nearest goal cell, and
    whether a path exists inside the field's own lateral extent (reachability's p_extent, the farthest any obstacle's
    surface reaches from the centreline), where rounding the field is not allowed. Aiming straight across is what tells
    "through the field" from "around it": with the whole far band as the goal, a start near the field's end would round
    it almost for free.

    Returns per edge the median and max detour (1.0 is a straight flight), the samples that could not cross at all
    (`unreachable`), those inside the field's extent that could not cross through it (`through_unreachable`), those
    beside the field (`beside_field`) and how often the goal window had to widen; and overall `max` (inf when any sample
    could not cross), `unreachable` and `through_unreachable`. A layout for M4 wants max under its bound and both
    totals 0.
    """
    free, xs, ys = free_grid(layout, resolution_m=resolution_m, inflate_m=inflate_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid = {"x": gx, "y": gy}
    d = edge_distances(layout.bounds, resolution_m)
    extent = {axis: _obstacle_extent(layout.instances, axis) for axis in ("x", "y")}
    band_span = max(layout.bounds.width, layout.bounds.height)
    edges: dict[str, dict] = {}
    for edge in EDGES:
        far = OPPOSITE[edge]
        axis = CORRIDOR_AXIS[edge]
        lateral = grid[axis]  # the coordinate along the edge: y for an x_min/x_max crossing, x otherwise
        field = np.abs(lateral) <= extent[axis]  # the field's own lateral extent: through it, not round it
        starts = free & (d[edge] >= start_band[0]) & (d[edge] <= start_band[1])
        band = free & (d[far] >= target_band[0]) & (d[far] <= target_band[1])
        candidates = np.argwhere(starts)
        counts = {"unreachable": 0, "through_unreachable": 0, "beside_field": 0, "goal_window_widened": 0}
        ratios: list[float] = []
        if len(candidates) and band.any():
            picks = candidates[_spaced(lateral[starts], samples_per_edge)]
            for ix, iy in picks:
                half = goal_half_width_m
                goals = band & (np.abs(lateral - lateral[ix, iy]) <= half)
                while not goals.any() and half < band_span:
                    half *= 2.0
                    goals = band & (np.abs(lateral - lateral[ix, iy]) <= half)
                    counts["goal_window_widened"] += 1 if goals.any() else 0
                if not goals.any():
                    counts["unreachable"] += 1
                    continue
                source = np.zeros_like(free)
                source[ix, iy] = True
                length = dijkstra_path_m(free, source, goals, resolution_m)
                if length is None:
                    counts["unreachable"] += 1
                    continue
                straight = float(np.hypot(gx[goals] - gx[ix, iy], gy[goals] - gy[ix, iy]).min())
                ratios.append(length / straight if straight > 0 else 1.0)
                if field[ix, iy]:
                    if dijkstra_path_m(free & field, source, goals & field, resolution_m) is None:
                        counts["through_unreachable"] += 1
                else:
                    counts["beside_field"] += 1
            n_samples = len(picks)
        else:
            n_samples = samples_per_edge
            counts["unreachable"] = n_samples
        edges[edge] = {"samples": n_samples, "median": float(np.median(ratios)) if ratios else None,
                       "max": (math.inf if counts["unreachable"] else max(ratios)) if (ratios or counts["unreachable"]) else None,
                       **counts, "ratios": [round(r, 4) for r in ratios]}
    maxima = [e["max"] for e in edges.values() if e["max"] is not None]
    return {"samples": samples_per_edge, "inflate_m": inflate_m, "resolution_m": resolution_m,
            "goal_half_width_m": goal_half_width_m, "field_extent_m": extent, "edges": edges,
            "max": max(maxima) if maxima else None,
            "unreachable": sum(e["unreachable"] for e in edges.values()),
            "through_unreachable": sum(e["through_unreachable"] for e in edges.values())}
