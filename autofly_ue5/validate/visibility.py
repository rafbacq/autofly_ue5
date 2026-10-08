"""First-frame visibility (plan 6 revision 1, the "Targets" done check): does an episode's first recorded frame show its
target where the target stands?

The env's first-use hold (expert/env.py PREWARM_HOLD_STEPS) exists because a pool object spawned for the first time in a
simulator session could be missing from the first frames after it spawned (runs/m4/targets_settle_diag/settle.json).
This check proves, episode by episode under true collection conditions, that the hold suffices. It reads the frame the
collector records first (the reset's observation, a0 applied) with the live checks' camera model and geometry
(validate/obstacles.py: the level 90 deg pinhole 0.40 m ahead of the body, planar depth; cylinders, yawed boxes and
glTF meshes, ray-cast), and gives one of five statuses:

  out_of_view   the centre of the target's box (its standing position, half its registry height up; the box
                `spawned_box` gives) projects outside the image, or behind the camera;
  occluded      the ray from the camera to that centre meets layout geometry (or a spawned distractor's box) before the
                target's box; the occluder is recorded, and for a mesh whether it was its solid part or its leaf cards;
  visible       a pixel of the 5 x 5 window about the projected centre reads the target: its depth lies within the
                target box's extent along that pixel's ray, give or take max(1.0 m, 5 % of the centre's depth);
  missing       pixels of the window could tell the target from what lies behind it, and none reads it: the frame does
                not show a target that is in view and unobstructed. The failure this record exists for: never dropped;
  undetermined  no pixel of the window can tell: none of their rays meets the target's box in front of everything else
                (a target smaller than a pixel at its range), or what lies behind reads within the same tolerance (a
                target's base against the ground). Said, rather than counted as either.

A window pixel can tell when its ray meets the target's box before everything else the geometry predicts there (the
ground plane, the layout's obstacles near the line of sight, the distractors' boxes) and that background lies beyond the
target's far face by more than the tolerance. Without that test, a far target would read "visible" off the ground just
behind its base: the first frame's target stands on the far edge, 64-78 m away in the collector's open-field tests of
s01's ground.

What the env exposes, and so what is used: the depth is the simulator's own frame at full resolution
(`Observation.depth`, 256 x 256 planar metres, no hit = inf: sim/decode.py), not the policy's encoded input; the camera's
pose is the body's at that frame plus the 0.40 m mount, level, because the observation carries the yaw only (the drone
hovers level at that frame). A spawned object's box is its registry bounds box, axis-aligned as spawned at yaw 0.
"""

from __future__ import annotations

import math

import numpy as np

from autofly_ue5.expert.episode import EpisodeSetup, standing_z_ned
from autofly_ue5.scenes.model import AssetEntry, AssetRegistry, Layout, load_registry
from autofly_ue5.sim.types import Observation
from autofly_ue5.validate.obstacles import (
    Box,
    MeshLibrary,
    camera_origin,
    focal_px,
    mesh_keys,
    obstacles_from_layout_json,
    pixel_rays,
    project_point,
)

STATUSES = ("visible", "missing", "occluded", "out_of_view", "undetermined")
WINDOW_HALF_PX = 2  # a 5 x 5 window about the projected centre, as scripts/probe_targets.py reads a target's centre
MIN_TOLERANCE_M = 1.0
RANGE_TOLERANCE = 0.05
SIGHT_MARGIN_M = 1.0  # obstacles whose footprint comes this near the line of sight (plus the window's spread) are cast


def spawned_box(tag: str, asset: AssetEntry, scale, x: float, y: float) -> Box:
    """The bounds box of an object spawned at (x, y) as expert/episode.py's apply_setup spawns it (standing on the
    ground at yaw 0): the registry's base size at `scale`, about the standing position plus the scaled bounds origin,
    as scripts/probe_targets.py predicts the box the simulator reports."""
    scale = tuple(float(s) for s in scale)
    origin = asset.bounds_origin_cm or (0.0, 0.0, 0.0)
    half = tuple(asset.base_size_m[i] * scale[i] / 2 for i in range(3))
    return Box(tag=tag, x=x + origin[0] * scale[0] / 100.0, y=y + origin[1] * scale[1] / 100.0,
               radius_m=math.hypot(half[0], half[1]), z_center=standing_z_ned(asset, scale) - origin[2] * scale[2] / 100.0,
               yaw=0.0, half=half)


def _segment_point_distance(a, b, p) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    length2 = vx * vx + vy * vy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / length2))
    return math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy))


class FirstFrameVisibility:
    """Built once per collection from the scene's layout (a mesh scene's glTFs are loaded here, once), then applied to
    each episode's first recorded observation."""

    def __init__(self, layout: Layout, registry: AssetRegistry | None = None, library: MeshLibrary | None = None):
        self._registry = registry if registry is not None else load_registry()
        data = {"layout": layout.to_json()}
        keys = mesh_keys(data, self._registry)
        if keys and library is None:
            library = MeshLibrary.from_files(keys)
        self._obstacles = obstacles_from_layout_json(data, self._registry, library)

    def _boxes(self, setup: EpisodeSetup) -> tuple[Box, list[Box]]:
        if setup.target is not None:
            target = (setup.target.asset, setup.target.scale)
            distractors = [(e.asset, e.scale) for e in setup.distractor_entries]
        else:  # the pool-free orange cylinder and its unpainted copies (M3)
            target = ("cylinder", setup.target_scale)
            distractors = [("cylinder", setup.target_scale)] * len(setup.distractors)
        tx, ty, _tz = setup.target_xy_z
        box = spawned_box("target", self._registry.assets[target[0]], target[1], tx, ty)
        others = [spawned_box(f"distractor_{i}", self._registry.assets[asset], scale, dx, dy)
                  for i, ((asset, scale), (dx, dy, _dz)) in enumerate(zip(distractors, setup.distractors, strict=True))]
        return box, others

    def __call__(self, obs: Observation, setup: EpisodeSetup, *, prewarm_steps: int | None) -> dict:
        depth = np.asarray(obs.depth, dtype=np.float64)
        if depth.ndim != 2 or depth.shape[0] != depth.shape[1]:
            raise ValueError(f"the first-frame check reads a square depth frame, got shape {depth.shape}")
        n = depth.shape[0]
        pose = obs.pose
        target, distractors = self._boxes(setup)
        centre = (target.x, target.y, target.z_center)
        origin = camera_origin(pose)
        row, col, planar = project_point(pose, centre, image_size=n)
        tolerance = max(MIN_TOLERANCE_M, RANGE_TOLERANCE * planar)
        record = {"status": None, "range_m": round(math.dist(origin, centre), 3), "planar_depth_m": round(planar, 3),
                  "pixel": None, "depth_window_min_m": None, "prewarm_steps": prewarm_steps, "depth_frame": [n, n],
                  "tolerance_m": round(tolerance, 3), "expected_m": None, "occluder": None, "informative_px": 0,
                  "matching_px": 0}
        if planar <= 0 or not (0 <= row < n and 0 <= col < n):
            record["status"] = "out_of_view"
            return record
        r, c = int(row), int(col)
        record["pixel"] = [r, c]
        ray = tuple((centre[i] - origin[i]) / planar for i in range(3))  # its parameter is the planar depth
        near, far = target.surface_interval(origin, ray)
        record["expected_m"] = [round(near, 3), round(far, 3)]

        # What could stand in front of the target or right behind it: the layout's obstacles and the distractors near
        # the line of sight up to the target's far face plus the tolerance, widened by the window's spread there.
        reach = far + tolerance
        end = tuple(origin[i] + ray[i] * reach for i in range(3))
        margin = SIGHT_MARGIN_M + (WINDOW_HALF_PX + 1) / focal_px(n) * reach
        near_sight = [o for o in self._obstacles if _segment_point_distance(origin, end, (o.x, o.y)) <= o.radius_m + margin]
        boxes_near = [b for b in distractors if _segment_point_distance(origin, end, (b.x, b.y)) <= b.radius_m + margin]

        occluder = self._occluder(origin, ray, near, near_sight, boxes_near)
        if occluder is not None:
            record.update(status="occluded", occluder=occluder)
            return record

        rows = slice(max(r - WINDOW_HALF_PX, 0), min(r + WINDOW_HALF_PX + 1, n))
        cols = slice(max(c - WINDOW_HALF_PX, 0), min(c + WINDOW_HALF_PX + 1, n))
        rays = np.asarray(pixel_rays(pose, rows, cols, image_size=n), dtype=np.float64)
        seen = depth[rows, cols].reshape(-1)
        finite = seen[np.isfinite(seen)]
        record["depth_window_min_m"] = round(float(finite.min()), 3) if finite.size else None
        with np.errstate(divide="ignore", invalid="ignore"):
            behind = np.where(rays[:, 2] > 0, -origin[2] / rays[:, 2], np.inf)  # the ground plane, z = 0
        for obstacle in [*near_sight, *boxes_near]:
            behind = np.minimum(behind, obstacle.surface_distances(origin, rays))
        informative = matching = 0
        for direction, value, background in zip(rays, seen, behind):
            interval = target.surface_interval(origin, tuple(direction))
            if interval is None or interval[1] <= 0:
                continue  # this pixel's ray passes the target by
            lo, hi = interval
            if background < lo or background <= hi + tolerance:
                continue  # something stands in front at this pixel, or what lies behind would read as the target
            informative += 1
            if lo - tolerance <= value <= hi + tolerance:
                matching += 1
        record.update(informative_px=informative, matching_px=matching,
                      status="visible" if matching else "missing" if informative else "undetermined")
        return record

    @staticmethod
    def _occluder(origin, ray, near: float, obstacles, boxes) -> dict | None:
        """The first layout obstacle or distractor box the ray from the camera to the target's centre meets before the
        target's box: {"by": its tag, "part": "solid", "cards" (a mesh's leaf cards only) or "box" (a distractor's
        bounds box)}; None when nothing does."""
        rays = np.asarray([ray], dtype=np.float64)
        best: tuple[float, dict] | None = None
        for obstacle in obstacles:
            t = float(obstacle.surface_distances(origin, rays)[0])
            if not t < near:
                continue
            part = "solid"
            if obstacle.kind == "mesh" and not float(obstacle.surface_distances(origin, rays, opaque_only=True)[0]) < near:
                part = "cards"
            if best is None or t < best[0]:
                best = (t, {"by": obstacle.tag, "part": part})
        for box in boxes:
            t = box.surface_distance(origin, ray)
            if t is not None and t < near and (best is None or t < best[0]):
                best = (t, {"by": box.tag, "part": "box"})
        return best[1] if best is not None else None
