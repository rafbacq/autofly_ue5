"""Seeded expansion of a scene file into concrete obstacle instances (spec §6.2).

Placements (spec §6.1): `jittered_grid` (pillars, poles: s01's, whose draw order is pinned by tests/test_generate.py's
S01_LAYOUT_SHA256 and must never change), `poisson` (rocks, stones, ruins: dart-throwing with a minimum centre
distance), `clusters` (tree clusters: cluster centres, then members inside each cluster's disc) and `stacks` (boxes:
stack positions, then boxes on top of each other). Footprints: `circle` (radius from the x scale) and `box` (the
circumscribed circle of the scaled base, conservative for the occupancy grid and every clearance check, which are
circle-only until spec §13's M4 item replaces them).

Several groups per scene (s04's rocks and stones, s10's walls and containers): a later group's instances are placed
clear of every earlier instance (surface gap >= 0), by rejection for the samplers and by refusal for a grid, which
cannot move its points. A group never spends more than MAX_PLACEMENT_TRIES candidates: a scene that asks for more
obstacles than its area holds fails here, loudly, not in the editor.
"""

from __future__ import annotations

import math
import random

from autofly_ue5.scenes.model import AssetEntry, AssetRegistry, Bounds, Instance, Layout, ObstacleGroup, SceneFile

MAX_PLACEMENT_TRIES = 20_000
SUPPORTED_FOOTPRINTS = ("circle", "box")

Keepout = list[tuple[float, float, float]]  # (x, y, footprint radius) of instances already placed


class UnsupportedPlacementError(ValueError):
    """The asset's footprint or pivot is one the generator cannot place (spec §6.1 lists what it can)."""


def _inset(bounds: Bounds, margin_m: float) -> tuple[float, float, float, float]:
    x0, x1 = bounds.x_min + margin_m, bounds.x_max - margin_m
    y0, y1 = bounds.y_min + margin_m, bounds.y_max - margin_m
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"margin {margin_m} m leaves no placement area")
    return x0, x1, y0, y1


def _clear(x: float, y: float, radius_m: float, keepout: Keepout) -> bool:
    """No overlap with any earlier instance: centre distance at least the two footprint radii."""
    return all(math.hypot(x - kx, y - ky) >= radius_m + kr for kx, ky, kr in keepout)


def jittered_grid(rng: random.Random, bounds: Bounds, count: int, margin_m: float, jitter_m: float) -> list[tuple[float, float]]:
    x0, x1, y0, y1 = _inset(bounds, margin_m)
    width, height = x1 - x0, y1 - y0
    cols = max(1, math.ceil(math.sqrt(count * width / height)))
    rows = math.ceil(count / cols)
    cell_w, cell_h = width / cols, height / rows
    if jitter_m > min(cell_w, cell_h) / 2:
        raise ValueError(f"jitter {jitter_m} m exceeds half a grid cell ({min(cell_w, cell_h) / 2:.2f} m)")
    cells = sorted(rng.sample([(r, c) for r in range(rows) for c in range(cols)], count))
    return [
        (x0 + (c + 0.5) * cell_w + rng.uniform(-jitter_m, jitter_m), y0 + (r + 0.5) * cell_h + rng.uniform(-jitter_m, jitter_m))
        for r, c in cells
    ]


def poisson(rng: random.Random, bounds: Bounds, count: int, margin_m: float, min_distance_m: float, *,
            keepout: Keepout = (), radius_m: float = 0.0) -> list[tuple[float, float]]:
    """`count` points inside the margin, every pair at least `min_distance_m` apart (and at least `2 * radius_m`, so
    instances of that footprint never touch), each clear of `keepout` by `radius_m` plus the other's radius."""
    x0, x1, y0, y1 = _inset(bounds, margin_m)
    spacing = max(min_distance_m, 2.0 * radius_m)
    points: list[tuple[float, float]] = []
    for _ in range(MAX_PLACEMENT_TRIES):
        if len(points) == count:
            return points
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if all(math.hypot(x - px, y - py) >= spacing for px, py in points) and _clear(x, y, radius_m, keepout):
            points.append((x, y))
    if len(points) == count:
        return points
    raise ValueError(f"poisson placement put {len(points)} of {count} points at >= {spacing:.2f} m spacing inside the "
                     f"{margin_m} m margin in {MAX_PLACEMENT_TRIES} tries: fewer obstacles, a smaller distance or a smaller margin")


def clusters(rng: random.Random, bounds: Bounds, count: int, margin_m: float, cluster_count: int, per_cluster: tuple[int, int],
             radius_m: float, *, keepout: Keepout = (), member_radius_m: float = 0.0
             ) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """`cluster_count` cluster centres whose discs of `radius_m` do not overlap, stay inside the margin and are clear of
    `keepout`, and `count` members split over them with `per_cluster[0] <= size <= per_cluster[1]`, each member uniform
    in its disc, every member at least `2 * member_radius_m` from every other (across clusters too) and clear of
    `keepout`."""
    sizes = _partition(rng, count, cluster_count, per_cluster, what="cluster")
    try:
        centres = poisson(rng, bounds, cluster_count, margin_m + radius_m, 2.0 * radius_m, keepout=keepout, radius_m=radius_m)
    except ValueError as err:
        raise ValueError(f"{cluster_count} clusters of radius {radius_m} m do not fit inside the {margin_m} m margin"
                         f"{' beside the earlier groups' if keepout else ''}: {err}") from err
    members: list[list[tuple[float, float]]] = []
    placed: list[tuple[float, float]] = []
    for (cx, cy), size in zip(centres, sizes):
        group: list[tuple[float, float]] = []
        for _ in range(MAX_PLACEMENT_TRIES):
            if len(group) == size:
                break
            r, theta = radius_m * math.sqrt(rng.random()), rng.uniform(-math.pi, math.pi)
            x, y = cx + r * math.cos(theta), cy + r * math.sin(theta)
            if (all(math.hypot(x - px, y - py) >= 2.0 * member_radius_m for px, py in placed)
                    and _clear(x, y, member_radius_m, keepout)):
                group.append((x, y))
                placed.append((x, y))
        if len(group) != size:
            raise ValueError(f"cluster at ({cx:.1f}, {cy:.1f}) holds {len(group)} of {size} members of radius "
                             f"{member_radius_m} m inside {radius_m} m in {MAX_PLACEMENT_TRIES} tries"
                             f"{' (earlier groups and neighbouring clusters take room too)' if keepout or placed else ''}")
        members.append(group)
    return centres, members


def _partition(rng: random.Random, total: int, parts: int, size_range: tuple[int, int], *, what: str) -> list[int]:
    """`total` split into `parts` sizes, each within `size_range`, the surplus over the minimum spread at random."""
    lo, hi = size_range
    if not parts * lo <= total <= parts * hi:
        raise ValueError(f"count {total} cannot be split into {parts} {what}s of {lo}-{hi} each "
                         f"(needs {parts * lo} to {parts * hi})")
    sizes = [lo] * parts
    for _ in range(total - parts * lo):
        sizes[rng.choice([i for i, s in enumerate(sizes) if s < hi])] += 1
    return sizes


def _footprint_radius_m(asset: AssetEntry, s_xy: float) -> float:
    if asset.footprint == "circle":
        return round(asset.base_size_m[0] * s_xy / 2.0, 4)
    return round(math.hypot(asset.base_size_m[0] * s_xy, asset.base_size_m[1] * s_xy) / 2.0, 4)  # box: circumscribed


def _check_group(group: ObstacleGroup, registry: AssetRegistry) -> AssetEntry:
    if group.asset not in registry.assets:
        raise ValueError(f"asset {group.asset!r} is not in the registry")
    asset = registry.assets[group.asset]
    if asset.footprint not in SUPPORTED_FOOTPRINTS or asset.pivot != "center":
        raise UnsupportedPlacementError(f"asset {group.asset!r} footprint={asset.footprint} pivot={asset.pivot} is not "
                                        f"supported (footprints {SUPPORTED_FOOTPRINTS}, pivot 'center')")
    for material in group.palette:
        if material not in registry.materials:
            raise ValueError(f"material {material!r} is not in the registry")
    return asset


def _instance(tag: str, group: ObstacleGroup, asset: AssetEntry, x: float, y: float, *, s_xy: float, s_z: float,
              material: str, yaw: float = 0.0, z_top_m: float = 0.0) -> Instance:
    """One instance standing on `z_top_m` metres of whatever is below it (0: the ground)."""
    height = round(asset.base_size_m[2] * s_z, 4)
    return Instance(
        tag=tag, asset=group.asset, x=round(x, 4), y=round(y, 4), z_center=round(-(z_top_m + height / 2.0), 4),
        yaw=yaw, scale=(s_xy, s_xy, s_z), material=material, radius_m=_footprint_radius_m(asset, s_xy), height_m=height,
    )


def generate_layout(scene: SceneFile, registry: AssetRegistry, seed: int | None = None) -> Layout:
    used_seed = scene.seed if seed is None else seed
    rng = random.Random(used_seed)
    instances: list[Instance] = []
    for group in scene.obstacle_groups:
        asset = _check_group(group, registry)
        placement = group.placement["type"]
        keepout: Keepout = [(i.x, i.y, i.radius_m) for i in instances]
        radius_max = _footprint_radius_m(asset, group.scale_xy[1])  # spacing uses the largest footprint the group can draw
        tag = lambda: f"obs_{len(instances):04d}"  # noqa: E731

        def draw_standing(x: float, y: float, *, random_yaw: bool) -> Instance:
            # The pinned draw order of a grid instance: position first, then x-y scale, z scale, material. Scattered and
            # clustered obstacles (rocks, trees, containers) also face a random way; a grid's pillars keep yaw 0, as s01's.
            s_xy = round(rng.uniform(*group.scale_xy), 4)
            s_z = round(rng.uniform(*group.scale_z), 4)
            material = rng.choice(group.palette)
            yaw = round(rng.uniform(-math.pi, math.pi), 4) if random_yaw else 0.0
            return _instance(tag(), group, asset, x, y, s_xy=s_xy, s_z=s_z, material=material, yaw=yaw)

        if placement == "jittered_grid":
            points = jittered_grid(rng, scene.bounds, group.count, group.placement["margin_m"], group.placement["jitter_m"])
            for x, y in points:
                if not _clear(x, y, radius_max, keepout):
                    raise ValueError(f"group {group.asset!r}: a jittered_grid point at ({x:.1f}, {y:.1f}) would overlap an "
                                     f"earlier group's obstacle; a grid cannot move its points, so place it first or widen "
                                     f"its margin")
                instances.append(draw_standing(x, y, random_yaw=False))
        elif placement == "poisson":
            points = poisson(rng, scene.bounds, group.count, group.placement["margin_m"], group.placement["min_distance_m"],
                             keepout=keepout, radius_m=radius_max)
            for x, y in points:
                instances.append(draw_standing(x, y, random_yaw=True))
        elif placement == "clusters":
            p = group.placement
            _centres, members = clusters(rng, scene.bounds, group.count, p["margin_m"], p["cluster_count"],
                                         (p["per_cluster"][0], p["per_cluster"][1]), p["radius_m"],
                                         keepout=keepout, member_radius_m=radius_max)
            for x, y in (point for group_points in members for point in group_points):
                instances.append(draw_standing(x, y, random_yaw=True))
        elif placement == "stacks":
            p = group.placement
            heights = _partition(rng, group.count, p["stack_count"], (p["height_range"][0], p["height_range"][1]), what="stack")
            points = poisson(rng, scene.bounds, p["stack_count"], p["margin_m"], 2.0 * radius_max, keepout=keepout, radius_m=radius_max)
            for (x, y), n_boxes in zip(points, heights):
                s_xy = round(rng.uniform(*group.scale_xy), 4)  # one footprint and one heading per stack: boxes align
                yaw = round(rng.uniform(-math.pi, math.pi), 4)
                z_top = 0.0
                for _ in range(n_boxes):
                    s_z = round(rng.uniform(*group.scale_z), 4)
                    box = _instance(tag(), group, asset, x, y, s_xy=s_xy, s_z=s_z, material=rng.choice(group.palette), yaw=yaw,
                                    z_top_m=z_top)
                    instances.append(box)
                    z_top += box.height_m
        else:  # the schema admits no other type; a new one must be placed here before a scene may use it
            raise UnsupportedPlacementError(f"placement '{placement}' is not implemented")
    return Layout(scene_id=scene.id, seed=used_seed, bounds=scene.bounds, instances=tuple(instances))
