"""Whether a spawned target is drawn: a clear view on the scene's open ground, the target stood straight ahead of it, and
the depth check against the frame before it spawned. One implementation for the live probe of every pool target
(scripts/probe_targets.py) and the env's session warm-up (expert/env.py), so both judge a target the same way. Moved here
from scripts/probe_targets.py (2026-10-08) unchanged, with `placement` its distance and position arithmetic.

From the view (2 m up, heading along a world axis, a REACH_M line of view keeping VIEW_CLEARANCE_M from every obstacle),
a target stands 1.5 x its largest side + 3 m ahead of the camera, clamped (the probe: 4-10 m; the warm-up: 6-10 m), on
the ground as expert/episode.py's apply_setup stands it. It shows when, inside its projected box, the depth changed from
the frame before the spawn to within its near and far faces on at least 3 % of the pixels (4 at least), and nothing
there is nearer than its near face less 0.3 m.
"""

from __future__ import annotations

import math

import numpy as np

from autofly_ue5.expert.episode import standing_z_ned
from autofly_ue5.scenes.model import AssetEntry, Layout
from autofly_ue5.sim.types import CONTROL_DT_S, Observation, Pose
from autofly_ue5.validate.geometry import CAMERA_OFFSET_M
from autofly_ue5.validate.visibility import spawned_box

ALTITUDE_M = 2.0
REACH_M = 13.0  # the farthest target's far face from the body: 0.4 m mount + 10 m + half a 4 m car
VIEW_CLEARANCE_M = 3.0  # every pillar's surface this far from the line of view (a 4 m car's half-diagonal is 2.2 m)
MIN_DISTANCE_M, MAX_DISTANCE_M = 4.0, 10.0  # camera to the target's centre: 1.5 x its largest side + 3 m, clamped
CHANGE_M = 0.05  # a depth pixel changed by the spawn (depth arrives as float16: 4 mm at 4 m, 8 mm at 10 m)
DEPTH_TOLERANCE_M = 0.3  # around the box's near and far faces (pixel footprints at grazing rows, float16)
MIN_HIT_SHARE = 0.03  # of the projected box: a ladder or a chair fills little of its box
MIN_TARGET_PX = 4


def choose_view(layout: Layout, *, reach_m: float = REACH_M, clearance_m: float = VIEW_CLEARANCE_M) -> Pose:
    """The pose, at ALTITUDE_M and heading along a world axis, whose line of view (reach_m ahead of the body) keeps the
    most room to every obstacle (at least clearance_m to its surface), both ends 2 m inside the layout's bounds."""
    b = layout.bounds
    best, best_gap = None, -math.inf
    for x in np.linspace(b.x_min + 2.0, b.x_max - 2.0, 29):
        for y in np.linspace(b.y_min + 2.0, b.y_max - 2.0, 29):
            for yaw in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0):
                ux, uy = round(math.cos(yaw)), round(math.sin(yaw))
                ex, ey = x + reach_m * ux, y + reach_m * uy
                if not (b.x_min + 2.0 <= ex <= b.x_max - 2.0 and b.y_min + 2.0 <= ey <= b.y_max - 2.0):
                    continue
                gap = math.inf
                for inst in layout.instances:
                    t = max(0.0, min(reach_m, (inst.x - x) * ux + (inst.y - y) * uy))
                    gap = min(gap, math.hypot(inst.x - (x + t * ux), inst.y - (y + t * uy)) - inst.radius_m)
                if gap >= clearance_m and gap > best_gap:
                    best, best_gap = Pose(float(x), float(y), -ALTITUDE_M, yaw), gap
    if best is None:
        raise ValueError(f"no {reach_m} m line of view with {clearance_m} m clearance in {layout.scene_id}")
    return best


def camera(view: Pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwd = np.array([math.cos(view.yaw), math.sin(view.yaw), 0.0])
    right = np.array([-math.sin(view.yaw), math.cos(view.yaw), 0.0])
    return np.array([view.x, view.y, view.z]) + CAMERA_OFFSET_M * fwd, fwd, right


def project(points: np.ndarray, view: Pose, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates (u right, v down; pixel k spans [k, k+1)) and planar depth of NED points, for the 90 deg pinhole
    camera CAMERA_OFFSET_M ahead of the body (sim/fake.py renders the same camera)."""
    cam, fwd, right = camera(view)
    rel = points - cam
    f = rel @ fwd
    return n / 2.0 + (rel @ right) / f * n / 2.0, n / 2.0 + rel[:, 2] / f * n / 2.0, f


def expected_box(asset, scale, position) -> tuple[list[float], list[float]]:
    """(centre, size) of the box a spawned entry stands in at `position` (its standing height): one definition with the
    collector's first-frame check (validate/visibility.py spawned_box)."""
    box = spawned_box("target", asset, scale, position[0], position[1])
    return [box.x, box.y, box.z_center], [2 * h for h in box.half]


def placement(asset: AssetEntry, scale, view: Pose, *, min_distance_m: float = MIN_DISTANCE_M,
              max_distance_m: float = MAX_DISTANCE_M) -> tuple[list[float], float]:
    """(the spawn position straight ahead of the view, standing on the ground; its distance from the camera): 1.5 x the
    target's largest side + 3 m, clamped to [min_distance_m, max_distance_m]."""
    size = [b * s for b, s in zip(asset.base_size_m, scale)]
    distance = float(np.clip(1.5 * max(size) + 3.0, min_distance_m, max_distance_m))
    cam, fwd, _right = camera(view)
    return [float(cam[0] + distance * fwd[0]), float(cam[1] + distance * fwd[1]), standing_z_ned(asset, tuple(scale))], distance


def region(centre, size, view: Pose, n: int) -> tuple[np.ndarray, dict]:
    corners = np.array([[centre[0] + sx * size[0] / 2, centre[1] + sy * size[1] / 2, centre[2] + sz * size[2] / 2]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    u, v, f = project(corners, view, n)
    c0, c1 = int(np.clip(math.floor(u.min()), 0, n - 1)), int(np.clip(math.ceil(u.max()) - 1, 0, n - 1))
    r0, r1 = int(np.clip(math.floor(v.min()), 0, n - 1)), int(np.clip(math.ceil(v.max()) - 1, 0, n - 1))
    mask = np.zeros((n, n), bool)
    mask[r0:r1 + 1, c0:c1 + 1] = True
    cu, cv, _ = project(np.array([centre]), view, n)
    return mask, {"rows": [r0, r1], "cols": [c0, c1], "near_m": float(f.min()), "far_m": float(f.max()),
                  "centre_px": [float(cv[0]), float(cu[0])]}


def depth_check(before: np.ndarray, after: np.ndarray, region: np.ndarray, box: dict) -> tuple[dict, np.ndarray]:
    fin_b, fin_a = np.isfinite(before), np.isfinite(after)
    with np.errstate(invalid="ignore"):
        changed = (fin_a != fin_b) | (fin_a & fin_b & (np.abs(after - before) > CHANGE_M))
        in_range = fin_a & (after >= box["near_m"] - DEPTH_TOLERANCE_M) & (after <= box["far_m"] + DEPTH_TOLERANCE_M)
    target = region & changed & in_range
    n_region = int(region.sum())
    seen = after[region & fin_a]
    r, c = (int(round(x - 0.5)) for x in box["centre_px"])
    window = after[max(r - 2, 0): r + 3, max(c - 2, 0): c + 3]
    window = window[np.isfinite(window)]
    region_min = float(seen.min()) if seen.size else math.inf
    hit_share = target.sum() / max(n_region, 1)
    ok = bool(target.sum() >= MIN_TARGET_PX and hit_share >= MIN_HIT_SHARE and region_min >= box["near_m"] - DEPTH_TOLERANCE_M)
    return {"expected_near_m": round(box["near_m"], 3), "expected_far_m": round(box["far_m"], 3),
            "centre_depth_m": round(float(np.median(window)), 3) if window.size else None,
            "region_min_m": round(region_min, 3) if math.isfinite(region_min) else None,
            "region_px": n_region, "target_px": int(target.sum()), "hit_share": round(float(hit_share), 4),
            "changed_share": round(float((region & changed).sum() / max(n_region, 1)), 4), "ok": ok}, target


def hold(sim) -> Observation:
    """One hover step, and the observation after it."""
    sim.command_velocity(0.0, 0.0, 0.0)
    sim.step(CONTROL_DT_S)
    return sim.observe()
