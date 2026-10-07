"""Look at a built scene through the drone's own cameras: RGB and depth frames at each edge's start and at obstacle
close-ups, with the numbers a build must be judged by before anything flies in it (plan 5, tasks C7-C8).

    bash scripts/run_job.sh start snap_s09 -- env -u PYTHONPATH .venv/bin/python scripts/scene_snapshots.py --scene s09 \\
        --instance 5 --out runs/snapshots/s09

Writes pose_<k>_rgb.png, pose_<k>_depth.png, sheet.png (RGB over depth for every pose, labelled) and summary.json with,
per pose, the mean RGB brightness (the exposure calibration input: s01 was tuned to 109 of 255 on its grid floor), the
depth's minimum, median and no-hit share, and for a close-up the depth expected at the targeted obstacle's surface
against the depth measured in the frame's central columns. Reset-only (no steps), on one slot the caller chooses; it
refuses nothing a live run owns because the simulator's own launch guard does that.

Ends with os._exit, like scripts/m2_gate.py: the projectairsim client leaves a non-daemon thread behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autofly_ue5.expert.episode import _nearest_obstacle_gap  # noqa: E402
from autofly_ue5.scenes.resolve import resolve_scene  # noqa: E402
from autofly_ue5.sim.types import Pose  # noqa: E402
from autofly_ue5.validate.geometry import CAMERA_OFFSET_M  # noqa: E402

ALTITUDE_M = 2.0
EDGE_INSET_M = 4.0  # inside the start band's 2-6 m
CLOSEUP_GAP_M = 6.0  # camera to the obstacle's surface
CLEARANCE_M = 1.5  # the camera pose must be this clear of every other obstacle
DEPTH_VIEW_MAX_M = 60.0
CENTRE_COLUMNS = 8  # the depth's central window, where a targeted obstacle sits


def edge_poses(bounds) -> list[tuple[str, Pose]]:
    cx, cy = (bounds.x_min + bounds.x_max) / 2.0, (bounds.y_min + bounds.y_max) / 2.0
    return [("edge_x_min", Pose(bounds.x_min + EDGE_INSET_M, cy, -ALTITUDE_M, 0.0)),
            ("edge_x_max", Pose(bounds.x_max - EDGE_INSET_M, cy, -ALTITUDE_M, math.pi)),
            ("edge_y_min", Pose(cx, bounds.y_min + EDGE_INSET_M, -ALTITUDE_M, math.pi / 2.0)),
            ("edge_y_max", Pose(cx, bounds.y_max - EDGE_INSET_M, -ALTITUDE_M, -math.pi / 2.0))]


def closeup_poses(layout, picks: int = 3) -> list[tuple[str, Pose, float, str]]:
    """A camera CLOSEUP_GAP_M from the surface of a few obstacles, facing them, from whichever side is clear."""
    instances = layout.instances
    if not instances:
        return []
    indices = sorted({0, len(instances) // 2, len(instances) - 1})[:picks]
    poses = []
    for i in indices:
        inst = instances[i]
        others = tuple(o for o in instances if (o.x, o.y) != (inst.x, inst.y))
        for bearing in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0):
            d = inst.radius_m + CLOSEUP_GAP_M
            x, y = inst.x - d * math.cos(bearing), inst.y - d * math.sin(bearing)
            b = layout.bounds
            if not (b.x_min + 1.0 <= x <= b.x_max - 1.0 and b.y_min + 1.0 <= y <= b.y_max - 1.0):
                continue
            if _nearest_obstacle_gap(x, y, others) < CLEARANCE_M:
                continue
            poses.append((f"closeup_{inst.tag}", Pose(x, y, -ALTITUDE_M, bearing), CLOSEUP_GAP_M, inst.tag))
            break
    return poses


def depth_image(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[finite] = np.clip(255.0 * np.log1p(depth[finite]) / math.log1p(DEPTH_VIEW_MAX_M), 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
    img[~finite] = (20, 20, 20)  # no hit: dark
    return img


def frame_stats(obs, expected_gap_m: float | None) -> dict:
    rgb = obs.rgb
    depth = obs.depth
    finite = np.isfinite(depth)
    h, w = depth.shape
    # the window around the image centre, where a targeted obstacle sits at the camera's own altitude (whole central
    # columns would reach the ground 2 m below the camera and report that instead)
    centre = depth[h // 2 - CENTRE_COLUMNS // 2: h // 2 + CENTRE_COLUMNS // 2, w // 2 - CENTRE_COLUMNS // 2: w // 2 + CENTRE_COLUMNS // 2]
    centre_finite = centre[np.isfinite(centre)]
    stats = {"rgb_mean": round(float(rgb.mean()), 2), "rgb_mean_rgb": [round(float(c), 2) for c in rgb.reshape(-1, 3).mean(axis=0)],
             "rgb_std": round(float(rgb.std()), 2), "saturated_share": round(float((rgb >= 250).mean()), 4),
             "depth_min_m": round(float(depth[finite].min()), 3) if finite.any() else None,
             "depth_median_m": round(float(np.median(depth[finite])), 3) if finite.any() else None,
             "no_hit_share": round(float((~finite).mean()), 4),
             "centre_depth_min_m": round(float(centre_finite.min()), 3) if centre_finite.size else None}
    if expected_gap_m is not None and centre_finite.size:
        # the camera sits CAMERA_OFFSET_M ahead of the body origin the pose places; a footprint circle is conservative, so
        # a positive error (the surface farther than the circle) is expected for anything but a cylinder seen head-on
        stats["expected_surface_m"] = round(expected_gap_m - CAMERA_OFFSET_M, 3)
        stats["surface_error_m"] = round(float(centre_finite.min() - (expected_gap_m - CAMERA_OFFSET_M)), 3)
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--instance", type=int, required=True, help="a slot no live run owns")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scene-config", default=None, help="default: scene_autofly_<base>_fast.jsonc")
    args = p.parse_args(argv)
    from autofly_ue5.sim.airsim_backend import scene_config_factory

    resolved = resolve_scene(args.scene)
    config = args.scene_config or resolved.default_scene_config.replace(".jsonc", "_fast.jsonc")
    poses = [(name, pose, None, None) for name, pose in edge_poses(resolved.layout.bounds)] + closeup_poses(resolved.layout)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"scene": args.scene, "map_path": resolved.map_path, "scene_config": config, "layout_sha256": resolved.layout_sha256,
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "poses": []}
    sim = scene_config_factory(config, resolved.movable_objects)()
    tiles = []
    try:
        sim.launch(resolved.map_path, args.instance)
        for k, (name, pose, gap, tag) in enumerate(poses):
            obs = sim.reset(pose)
            cv2.imwrite(str(args.out / f"pose_{k}_rgb.png"), np.ascontiguousarray(obs.rgb[:, :, ::-1]))
            cv2.imwrite(str(args.out / f"pose_{k}_depth.png"), depth_image(obs.depth))
            stats = frame_stats(obs, gap)
            entry = {"index": k, "name": name, "target_tag": tag, "pose": [pose.x, pose.y, pose.z, pose.yaw], **stats}
            summary["poses"].append(entry)
            print(json.dumps(entry), flush=True)
            tile = np.concatenate([np.ascontiguousarray(obs.rgb[:, :, ::-1]), depth_image(obs.depth)], axis=0)
            label = f"{name}  rgb {stats['rgb_mean']:.0f}  d_min {stats['depth_min_m']}"
            cv2.putText(tile, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
    finally:
        try:
            sim.close()
        finally:
            if tiles:
                cv2.imwrite(str(args.out / "sheet.png"), np.concatenate(tiles, axis=1))
            summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{len(summary['poses'])} poses in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
