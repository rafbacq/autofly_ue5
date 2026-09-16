"""Geometry and metric helpers for the M1 live checks (NED metres, yaw radians)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autofly_ue5.frames import wrap_pi
from autofly_ue5.scenes.model import Bounds
from autofly_ue5.sim.types import Pose

CAMERA_OFFSET_M = 0.40


@dataclass(frozen=True)
class Pillar:
    tag: str
    x: float
    y: float
    radius: float
    height: float


@dataclass(frozen=True)
class DepthProbe:
    tag: str
    distance_m: float
    pose: Pose
    pillar: Pillar
    expected_depth_m: float


def pillars_from_layout_json(data: dict) -> list[Pillar]:
    return [Pillar(i["tag"], float(i["x"]), float(i["y"]), float(i["radius_m"]), float(i["height_m"]))
            for i in data["layout"]["instances"]]


def ray_circle_distance(ox: float, oy: float, yaw: float, cx: float, cy: float, r: float) -> float | None:
    dx, dy = math.cos(yaw), math.sin(yaw)
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    for t in (-b - root, -b + root):
        if t > 0:
            return t
    return None


def expected_center_depth(pose: Pose, pillar: Pillar, camera_offset_m: float = CAMERA_OFFSET_M) -> float | None:
    cam_x = pose.x + camera_offset_m * math.cos(pose.yaw)
    cam_y = pose.y + camera_offset_m * math.sin(pose.yaw)
    return ray_circle_distance(cam_x, cam_y, pose.yaw, pillar.x, pillar.y, pillar.radius)


def _segment_point_distance(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def choose_depth_probes(
    pillars: list[Pillar],
    bounds: Bounds,
    distances: tuple[float, ...] = (3.0, 6.0, 12.0),
    altitude_m: float = 2.0,
    camera_offset_m: float = CAMERA_OFFSET_M,
    body_clearance_m: float = 1.5,
    ray_clearance_m: float = 0.3,
    edge_margin_m: float = 2.0,
) -> list[DepthProbe]:
    probes = []
    for distance in distances:
        found = None
        for pillar in sorted(pillars, key=lambda p: p.tag):
            others = [q for q in pillars if q.tag != pillar.tag]
            for k in range(8):
                yaw = wrap_pi(k * math.pi / 4)
                ux, uy = math.cos(yaw), math.sin(yaw)
                standoff = pillar.radius + distance + camera_offset_m
                x, y = pillar.x - standoff * ux, pillar.y - standoff * uy
                if not (bounds.x_min + edge_margin_m <= x <= bounds.x_max - edge_margin_m
                        and bounds.y_min + edge_margin_m <= y <= bounds.y_max - edge_margin_m):
                    continue
                if any(math.hypot(x - q.x, y - q.y) < q.radius + body_clearance_m for q in others):
                    continue
                cam = (x + camera_offset_m * ux, y + camera_offset_m * uy)
                surface = (pillar.x - pillar.radius * ux, pillar.y - pillar.radius * uy)
                if any(_segment_point_distance(*cam, *surface, q.x, q.y) < q.radius + ray_clearance_m for q in others):
                    continue
                pose = Pose(x, y, -altitude_m, yaw)
                found = DepthProbe(pillar.tag, distance, pose, pillar, expected_center_depth(pose, pillar, camera_offset_m))
                break
            if found is not None:
                break
        if found is None:
            raise ValueError(f"no unobstructed depth probe at {distance} m")
        probes.append(found)
    return probes


def depth_agreement(a: np.ndarray, b: np.ndarray, far_m: float = 30.0) -> dict:
    """Compare two planar-depth images of the same pose: which pixels are nearer than far_m, and by how much they differ."""
    near_a = np.isfinite(a) & (a < far_m)
    near_b = np.isfinite(b) & (b < far_m)
    both = near_a & near_b
    relative = np.abs(a[both] - b[both]) / np.maximum(a[both], 1e-3)
    return {"mask_agreement": float(np.mean(near_a == near_b)), "pixels_compared": int(both.sum()),
            "median_relative_diff": float(np.median(relative)) if both.any() else None}


def tracking_error(commanded: list[float], measured: list[float], settle: int) -> dict:
    if len(commanded) != len(measured) or settle >= len(commanded):
        raise ValueError("commanded and measured must have equal length greater than settle")
    c, m = commanded[settle:], measured[settle:]
    mae = sum(abs(a - b) for a, b in zip(m, c)) / len(c)
    scale = sum(abs(v) for v in c) / len(c)
    return {"mean_abs_error": mae, "mean_commanded": sum(c) / len(c), "mean_measured": sum(m) / len(m),
            "relative_error": mae / scale if scale > 0 else None, "samples": len(c)}


def yaw_rates_from_yaws(yaws: list[float], dt: float) -> list[float]:
    return [wrap_pi(b - a) / dt for a, b in zip(yaws, yaws[1:])]


def speed_along_heading(velocity_ned: tuple[float, float, float], yaw: float) -> float:
    return velocity_ned[0] * math.cos(yaw) + velocity_ned[1] * math.sin(yaw)


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def pillar_width_px(axis_distance_m: float, radius_m: float, image_width: int = 256, hfov_deg: float = 90.0) -> float:
    """Pinhole image width of a vertical cylinder whose axis lies on the optical axis at axis_distance_m.

    The silhouette's tangent rays make asin(r / D) with the axis, i.e. an image half-width of f * r / sqrt(D^2 - r^2),
    with focal length f = (image_width / 2) / tan(hfov / 2)."""
    return image_width * radius_m / (math.sqrt(axis_distance_m**2 - radius_m**2) * math.tan(math.radians(hfov_deg) / 2.0))


def contiguous_width(mask_row: np.ndarray, center: int = 128) -> int:
    """Length of the run of True values in mask_row that contains index center (0 when center is False)."""
    if not mask_row[center]:
        return 0
    left = center
    while left > 0 and mask_row[left - 1]:
        left -= 1
    right = center
    while right < len(mask_row) - 1 and mask_row[right + 1]:
        right += 1
    return right - left + 1
