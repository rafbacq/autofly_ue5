"""Seeded expansion of a scene file into concrete obstacle instances (spec §6.2)."""

from __future__ import annotations

import math
import random

from autofly_ue5.scenes.model import AssetRegistry, Bounds, Instance, Layout, SceneFile


class UnsupportedPlacementError(ValueError):
    """The placement type or asset footprint is not implemented in Plan 1 (only jittered_grid circles)."""


def jittered_grid(rng: random.Random, bounds: Bounds, count: int, margin_m: float, jitter_m: float) -> list[tuple[float, float]]:
    x0, x1 = bounds.x_min + margin_m, bounds.x_max - margin_m
    y0, y1 = bounds.y_min + margin_m, bounds.y_max - margin_m
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError(f"margin {margin_m} m leaves no placement area")
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


def generate_layout(scene: SceneFile, registry: AssetRegistry, seed: int | None = None) -> Layout:
    used_seed = scene.seed if seed is None else seed
    rng = random.Random(used_seed)
    instances: list[Instance] = []
    for group in scene.obstacle_groups:
        if group.asset not in registry.assets:
            raise ValueError(f"asset {group.asset!r} is not in the registry")
        asset = registry.assets[group.asset]
        if asset.footprint != "circle" or asset.pivot != "center":
            raise UnsupportedPlacementError(f"asset {group.asset!r} footprint={asset.footprint} pivot={asset.pivot} is not supported in Plan 1")
        for material in group.palette:
            if material not in registry.materials:
                raise ValueError(f"material {material!r} is not in the registry")
        placement = group.placement["type"]
        if placement != "jittered_grid":
            raise UnsupportedPlacementError(f"placement '{placement}' is not implemented in Plan 1 (planned for M4)")
        points = jittered_grid(rng, scene.bounds, group.count, group.placement["margin_m"], group.placement["jitter_m"])
        for x, y in points:
            s_xy = round(rng.uniform(*group.scale_xy), 4)
            s_z = round(rng.uniform(*group.scale_z), 4)
            material = rng.choice(group.palette)
            height = round(asset.base_size_m[2] * s_z, 4)
            instances.append(Instance(
                tag=f"obs_{len(instances):04d}", asset=group.asset, x=round(x, 4), y=round(y, 4),
                z_center=round(-height / 2.0, 4), yaw=0.0, scale=(s_xy, s_xy, s_z), material=material,
                radius_m=round(asset.base_size_m[0] * s_xy / 2.0, 4), height_m=height,
            ))
    return Layout(scene_id=scene.id, seed=used_seed, bounds=scene.bounds, instances=tuple(instances))
