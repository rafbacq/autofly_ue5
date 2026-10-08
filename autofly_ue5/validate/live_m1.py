"""Live checks of spec section 11 on a packaged scene, through the Simulator interface: M1's gate for s01, and plan 5's
C8 gate for every scene after it.

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.live_m1 --out runs/m1/m1_gate.json
env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.live_m1 --scene s09 --instance 5   # -> docs/gates/m4_s09_gate.json

The checks were written for s01's centred cylinders; since 2026-10-07 they take any obstacle kind through
validate/obstacles.py (yawed boxes, imported meshes ray-cast against their glTF triangles), derive their poses from the
scene's bounds (the x_min start band, as s01's constants did), check every actor's world box inside the same session
(`map_matches_layout`, formerly the separate sim/check_map.py step) and save each depth probe's frames for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from autofly_ue5.evidence import m4_evidence_path, record_destination, refuse_existing_evidence
from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import PACKAGED_BINARY, RUNS_DIR
from autofly_ue5.scenes.model import Bounds, load_registry
from autofly_ue5.scenes.resolve import resolve_scene
from autofly_ue5.sim.airsim_backend import ProjectAirSimSimulator, scene_config_record
from autofly_ue5.sim.process import instance_dir, route_client_log
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.types import Pose
from autofly_ue5.validate.engine_check import (
    boot_id,
    count_device_lost,
    instance_logs_since,
    kernel_journal_readable,
    xid_count,
)
from autofly_ue5.validate.geometry import (
    CAMERA_OFFSET_M,
    contiguous_width,
    depth_agreement,
    mean_abs_diff,
    speed_along_heading,
    tracking_error,
    yaw_rates_from_yaws,
)
from autofly_ue5.validate.obstacles import (
    MeshLibrary,
    Obstacle,
    Probe,
    bbox_error,
    choose_probes,
    mesh_keys,
    obstacles_from_layout_json,
    row_silhouette,
)

DT = 0.2
# Opaque base UMaterial cooked with /Game/Geometry; M0 (Task 7 Step 11) showed it renders orange at runtime.
ORANGE_MATERIAL = "/Game/Geometry/Materials/M_Orange"
ALTITUDE_M = 2.0
BAND_INSET_M = 4.0  # inside the 2-6 m start band every scene keeps free (spec section 5)
GROUND_TAG = "AF_Ground"
GROUND_TOP_TOL_M = 0.02
# the near-surface band of a probe's silhouette: pixels within this much depth of the window's surface, in the frame and
# in the prediction alike, so the widths compared are of the same thing for a round pillar, a box face or a trunk
SILHOUETTE_BAND_M = 0.3
# the share of a probe window's on-obstacle pixels that must measure within their own tolerance of the ray cast
MIN_INLIER_SHARE = 0.8
CRASH_STEPS = 40  # 16 m at 2 m/s: the 6 m probe's solid surface is up to 9 m off for a tree whose footprint is its canopy


@dataclass(frozen=True)
class ScenePoses:
    """Where the checks stand, in the x_min start band (s01's M1 constants, derived from any scene's bounds)."""

    open: Pose        # facing +y along the band: tracking and one-step fly here
    field_view: Pose  # the band's middle, facing +x into the obstacle field
    empty_view: Pose  # the same spot facing -x, out of the scene over bare ground
    spawn: Pose       # the runtime cube, in the free band
    spawn_view: Pose  # 3 m from the cube, facing it


def scene_poses(bounds: Bounds, altitude_m: float = ALTITUDE_M) -> ScenePoses:
    x = bounds.x_min + BAND_INSET_M
    cy = (bounds.y_min + bounds.y_max) / 2.0
    return ScenePoses(open=Pose(x, bounds.y_min + 10.0, -altitude_m, math.pi / 2),
                      field_view=Pose(x, cy, -altitude_m, 0.0),
                      empty_view=Pose(x, cy, -altitude_m, math.pi),
                      spawn=Pose(x, bounds.y_min + 25.0, -altitude_m, 0.0),
                      spawn_view=Pose(x, bounds.y_min + 22.0, -altitude_m, math.pi / 2))


S01_POSES = scene_poses(Bounds(-35.0, 35.0, -35.0, 35.0))  # M1's OPEN_POSE (-31, -25), PILLAR_VIEW (-31, 0), SPAWN_POSE (-31, -10)


def _finite(v: float | None) -> float | None:
    return v if v is not None and math.isfinite(v) else None


def check_rgb_changes(sim: Simulator, poses: ScenePoses) -> dict:
    a = sim.reset(poses.field_view).rgb
    b = sim.reset(poses.empty_view).rgb
    a2 = sim.reset(poses.field_view).rgb
    out = {"diff_pillar_vs_empty": mean_abs_diff(a, b), "diff_pillar_repeat": mean_abs_diff(a, a2),
           "brightness_pillar_view": float(a.mean()), "brightness_empty_view": float(b.mean())}
    out["pass"] = out["diff_pillar_vs_empty"] > 10.0 and 20.0 < out["brightness_pillar_view"] < 235.0
    return out


def _save_frames(frames_dir: Path | None, stem: str, obs) -> None:
    if frames_dir is None:
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frames_dir / f"{stem}_rgb.png"), np.ascontiguousarray(obs.rgb[:, :, ::-1]))
    depth_mm = np.where(np.isfinite(obs.depth), np.clip(obs.depth * 1000.0, 0, 65535), 65535).astype(np.uint16)
    cv2.imwrite(str(frames_dir / f"{stem}_depth_mm.png"), depth_mm)  # 16-bit millimetres, 65535 = no hit


def check_depth(sim: Simulator, probes: list[Probe], frames_dir: Path | None = None) -> dict:
    """At each probe, the depth the camera reports against the depth ray-cast from the layout: the window medians
    agree within max(0.15 m, 2 %), at least `MIN_INLIER_SHARE` of the pixels predicted to be on the obstacle measure
    within max(0.1 m, 2 %) of their own prediction, and for a solid obstacle the row silhouette has the pinhole width."""
    rows = []
    for k, probe in enumerate(probes):
        obs = sim.reset(probe.pose)
        _save_frames(frames_dir, f"probe_{k}_{probe.tag}_{probe.distance_m:g}m", obs)
        window = obs.depth[probe.window].reshape(-1)
        predicted, opaque = probe.predicted_m, probe.predicted_opaque_m
        # three kinds of pixel (validate/obstacles.Probe): solid geometry in the clear, a leaf card first, nothing;
        # the per-pixel comparison is made on the interior of the solid patch, where half a pixel changes nothing
        is_solid = probe.solid_pixels
        interior = probe.interior_pixels
        is_card = np.isfinite(predicted) & ~is_solid
        is_empty = ~np.isfinite(predicted)
        measured_solid = window[interior]
        measured = float(np.median(measured_solid))
        expected = probe.expected_depth_m
        tolerance = max(0.15, 0.02 * probe.distance_m)
        error = abs(measured - expected) if math.isfinite(measured) else None
        pixel_tolerance = np.maximum(0.1, 0.02 * opaque[interior])
        inliers = np.isfinite(measured_solid) & (np.abs(measured_solid - opaque[interior]) <= pixel_tolerance)
        inlier_share = float(inliers.mean())
        # a card pixel measures the card (an opaque texel) or something behind it (a transparent one): never nearer
        card_ok = window[is_card] >= predicted[is_card] - np.maximum(0.1, 0.02 * predicted[is_card])
        card_share = float(card_ok.mean()) if is_card.any() else None
        # pixels predicted to pass the obstacle should see something farther (or nothing): the obstacle is not wider
        # than the geometry says either
        off = window[is_empty]
        beyond_share = float((~np.isfinite(off) | (off > expected + SILHOUETTE_BAND_M)).mean()) if is_empty.any() else None
        row = {"tag": probe.tag, "kind": probe.obstacle.kind, "solid": probe.solid, "distance_m": probe.distance_m,
               "altitude_m": -probe.pose.z, "window": [[probe.window[0].start, probe.window[0].stop], [probe.window[1].start, probe.window[1].stop]],
               "pixels": {"solid": int(is_solid.sum()), "interior": int(interior.sum()), "card": int(is_card.sum()), "empty": int(is_empty.sum())},
               "measured_m": _finite(measured), "expected_m": expected, "error_m": error, "tolerance_m": tolerance,
               "inlier_share": inlier_share, "min_inlier_share": MIN_INLIER_SHARE, "card_no_nearer_share": card_share,
               "beyond_share_off_target": beyond_share, "window_spread_predicted_m": probe.window_spread_m,
               "window_spread_measured_m": _finite(float(measured_solid.max() - measured_solid.min())) if np.isfinite(measured_solid).all() else None,
               "pose": [obs.pose.x, obs.pose.y, obs.pose.z, obs.pose.yaw]}
        hfov_consistent = None
        if probe.solid:
            # Field of view (spec section 4.1): the obstacle's near-surface silhouette in row 128 (camera height)
            # against the same silhouette ray-cast through the 90 deg pinhole model. For a pillar on the optical axis
            # this is M1's pinhole-width check; for a box face it is the same comparison of the same band of pixels.
            near_m = expected + SILHOUETTE_BAND_M
            predicted_mask = row_silhouette(probe.obstacle, probe.pose, near_m=near_m)
            depth_row = obs.depth[128]
            measured_mask = np.isfinite(depth_row) & (depth_row < near_m)
            width, expected_width = contiguous_width(measured_mask), contiguous_width(predicted_mask)
            width_tolerance = max(2.0, 0.1 * expected_width)
            hfov_consistent = expected_width > 0 and abs(width - expected_width) <= width_tolerance
            row.update({"width_px": width, "expected_width_px": expected_width, "width_tolerance_px": width_tolerance})
        row["hfov_consistent"] = hfov_consistent
        row["pass"] = (error is not None and error <= tolerance and inlier_share >= MIN_INLIER_SHARE
                       and (card_share is None or card_share >= MIN_INLIER_SHARE)
                       and (hfov_consistent is True or not probe.solid))
        rows.append(row)
    return {"probes": rows, "pass": len(rows) == len(probes) and all(r["pass"] for r in rows)}


def check_crash(sim: Simulator, probe: Probe, poses: ScenePoses) -> dict:
    """Fly at the probe's obstacle from its pose until the simulator reports a collision: the impact must lie within
    0.5 m of the obstacle's footprint circle and about where the depth probe put the solid surface (the camera's planar
    depth plus its 0.4 m offset from the body, within 1 m: a box corner or a branch may meet the body before the ray's
    surface does), and a reset afterwards must restore the view. A tree's trunk from the 6 m probe is 9 m off: the
    flight allows CRASH_STEPS steps, 2 m/s from a hover."""
    reference = sim.reset(poses.field_view)
    sim.reset(probe.pose)
    ux, uy = math.cos(probe.pose.yaw), math.sin(probe.pose.yaw)
    # the collision mesh is the solid geometry (a tree's leaf cards are their own actor without collision): the body
    # meets the nearest of it across its width, which the 16 x 16 window spans at the 6 m probe; a solid obstacle's
    # window is its surface
    contact_m = float(np.min(probe.predicted_opaque_m))
    out: dict = {"pass": False, "reason": f"no collision within {CRASH_STEPS} steps", "expected_contact_m": contact_m}
    for step in range(CRASH_STEPS):
        sim.command_velocity(2.0, 0.0, 0.0)
        sim.step(DT)
        obs = sim.observe()
        if obs.step_collisions:
            event = obs.step_collisions[0]
            o = probe.obstacle
            impact_from_footprint = math.hypot(event.impact_point[0] - o.x, event.impact_point[1] - o.y) - o.radius_m
            travel = (event.impact_point[0] - probe.pose.x) * ux + (event.impact_point[1] - probe.pose.y) * uy
            expected_travel = contact_m + CAMERA_OFFSET_M
            out = {"steps_to_collision": step + 1, "object_name": event.object_name, "target": o.tag, "kind": o.kind,
                   "impact_point": list(event.impact_point), "normal": list(event.normal),
                   "impact_from_footprint_m": impact_from_footprint, "footprint_radius_m": o.radius_m,
                   "travel_to_impact_m": travel, "expected_travel_m": expected_travel, "collided_flag": obs.collided,
                   "camera_pose_error_at_collision_m": obs.camera_pose_error_m,
                   "pass": obs.collided and impact_from_footprint <= 0.5 and abs(travel - expected_travel) <= 1.0}
            break
    # A reset after a crash must bring the Unreal actor and its camera back (research gotcha 6); the backend raises
    # CameraPoseError when it cannot. For a mesh that is recorded, not required: a body that reaches wood can wedge
    # the swept actor beyond any teleport (s02, 2026-10-07: 2 of 4 crashes), so tree scenes end an episode at the
    # footprint circle before contact and this crash is the gate's last act.
    try:
        after = sim.reset(poses.field_view)
    except Exception as err:
        out["reset_after_crash"] = {"error": f"{type(err).__name__}: {err}", "pass": False, "required": probe.solid}
        out["pass"] = bool(out["pass"]) and not probe.solid
        return out
    depth = depth_agreement(reference.depth, after.depth)
    reset_ok = (
        after.camera_pose_error_m < 0.1 and after.collided is False and depth["mask_agreement"] >= 0.98
        and (depth["median_relative_diff"] is None or depth["median_relative_diff"] <= 0.05)
    )
    out["reset_after_crash"] = {"camera_pose_error_m": after.camera_pose_error_m, "depth": depth,
                                "rgb_mean_abs_diff": mean_abs_diff(reference.rgb, after.rgb), "pass": reset_ok, "required": probe.solid,
                                # the escape the backend's reset climbed from (sim/airsim_backend._escape_to)
                                "escape": list(getattr(sim, "last_reset_escape", ("climb", 0.0))),
                                "attempts": getattr(sim, "last_reset_attempts", 0)}
    out["pass"] = bool(out["pass"]) and (reset_ok or not probe.solid)
    return out


def check_tracking(sim: Simulator, poses: ScenePoses) -> dict:
    sim.reset(poses.open)
    forward = []
    for _ in range(15):
        sim.command_velocity(2.0, 0.0, 0.0)
        sim.step(DT)
        obs = sim.observe()
        forward.append(speed_along_heading(obs.velocity_ned, obs.pose.yaw))
    sim.reset(poses.open)
    up, down = [], []
    for v_z, sink in [(0.5, up)] * 10 + [(-0.5, down)] * 10:
        sim.command_velocity(0.0, 0.0, v_z)
        sim.step(DT)
        sink.append(-sim.observe().velocity_ned[2])
    obs = sim.reset(poses.open)
    yaws, body_rates = [obs.pose.yaw], []
    for _ in range(15):
        sim.command_velocity(0.0, 0.5, 0.0)
        sim.step(DT)
        obs = sim.observe()
        yaws.append(obs.pose.yaw)
        body_rates.append(obs.yaw_rate)
    axes = {
        "forward": tracking_error([2.0] * 15, forward, settle=5),
        "vertical_up": tracking_error([0.5] * 10, up, settle=5),
        "vertical_down": tracking_error([-0.5] * 10, down, settle=5),
        "yaw_rate": tracking_error([0.5] * 15, yaw_rates_from_yaws(yaws, DT), settle=5),
        "yaw_rate_body_angular_z": tracking_error([0.5] * 15, body_rates, settle=5),
    }
    gated = ("forward", "vertical_up", "vertical_down", "yaw_rate")
    return {"axes": axes, "threshold_relative_error": 0.2,
            "pass": all(axes[k]["relative_error"] is not None and axes[k]["relative_error"] <= 0.2 for k in gated)}


NOTE_ONE_STEP = (
    "This check cannot actually fail: frame_times_match is guaranteed by FrameCollector.wait() "
    "(FrameTimestampError otherwise, sync.py), dt_ns==200_000_000/steps==1 by Observation.sim_time_ns "
    "and StepTimingError (airsim_backend.py), and kinematics agreement by StaleStateError. An "
    "Observation cannot exist unless all three already held, so a pass here is not independent "
    "evidence of them -- it reduces to 'the backend ran 20 steps without raising'. Kept as a gate "
    "check anyway because that is still worth recording: it demonstrates the invariants held in a "
    "live run, on the packaged binary, not just in the unit tests that pin the raises."
)


def check_one_step(sim: Simulator, poses: ScenePoses) -> dict:
    """Each record: one clock step of 200 ms, and the RGB, depth and kinematics that built it all carry that step's
    time. This criterion cannot fail on its own: it passes iff the backend's own invariants held (see NOTE_ONE_STEP)
    and is not independent evidence of them -- only evidence that 20 steps ran without one of those raising."""
    sim.reset(poses.open)
    rows = []
    start = time.monotonic()
    for _ in range(20):
        steps_before, time_before = sim.steps_taken, sim.observe().sim_time_ns
        sim.command_velocity(1.0, 0.0, 0.0)
        t = sim.step(DT)
        obs = sim.observe()
        rows.append({"steps": sim.steps_taken - steps_before, "dt_ns": t - time_before,
                     "times_ns": [obs.rgb_time_ns, obs.depth_time_ns, obs.kinematics_time_ns],
                     "frame_times_match": obs.rgb_time_ns == obs.depth_time_ns == obs.kinematics_time_ns == t})
    wall = time.monotonic() - start
    ok = all(r["steps"] == 1 and r["dt_ns"] == 200_000_000 and r["frame_times_match"] for r in rows)
    return {"records": len(rows), "records_per_s": len(rows) / wall, "rows": rows, "pass": ok, "note": NOTE_ONE_STEP}


def check_spawn_destroy(sim: Simulator, poses: ScenePoses) -> dict:
    """Runtime spawn of a cooked Blocks mesh with a runtime material in the packaged build (spec §4.1), seen in depth and
    colour, then destroyed. The backend raises if set_object_material returns False."""
    name = sim.spawn("AF_M1_Cube", "1M_Cube", poses.spawn, (1.0, 1.0, 1.0), ORANGE_MATERIAL)
    with_obs = sim.reset(poses.spawn_view)
    with_cube = float(np.median(with_obs.depth[126:130, 126:130]))
    red, green, blue = (float(v) for v in with_obs.rgb[120:136, 120:136].reshape(-1, 3).mean(axis=0))
    orange_dominant = red > green > blue and red - blue > 30.0
    sim.destroy(name)
    without_cube = float(np.median(sim.reset(poses.spawn_view).depth[126:130, 126:130]))
    expected = 3.0 - 0.5 - CAMERA_OFFSET_M
    gone = math.isinf(without_cube) or without_cube > expected + 1.0
    return {"spawned_name": name, "material": ORANGE_MATERIAL, "center_depth_with_cube_m": _finite(with_cube),
            "expected_m": expected, "center_rgb_with_cube": [red, green, blue], "orange_dominant": orange_dominant,
            "center_depth_after_destroy_m": _finite(without_cube),
            "pass": name.startswith("AF_M1_Cube") and abs(with_cube - expected) <= 0.15 and orange_dominant and gone}


def check_map(sim: Simulator, obstacles: list[Obstacle], *, tolerance_m: float = 0.05) -> dict:
    """Every layout actor's world-axis box as the simulator reports it against the layout's prediction (centre and
    size within tolerance_m), and the ground's top at z = 0: the packaged binary carries this layout, placed as built.
    A tree's cards actor must be found; it has no box to report (no collision), so its placement rests on the level
    builder's check and on the depth probes' card pixels."""
    rows = []
    for o in obstacles:
        row = bbox_error(o, sim.object_bounds(o.tag))
        row["tag"], row["kind"] = o.tag, o.kind
        rows.append(row)
        cards_tag = getattr(o, "cards_tag", None)
        if cards_tag:  # a tree's leaf cards: their own actor, without collision, at the same transform
            row = bbox_error(o, sim.object_bounds(cards_tag), part="cards")
            row["tag"], row["kind"] = cards_tag, "cards"
            rows.append(row)
    ground = sim.object_bounds(GROUND_TAG)
    ground_top = ground[0][2] - ground[1][2] / 2.0 if ground else None
    found = [r for r in rows if r["found"]]
    boxed = [r for r in found if not r.get("box_unavailable")]  # a cards actor is found but has no box (no collision)
    return {"obstacles": rows, "checked": len(rows), "found": len(found), "boxed": len(boxed),
            "max_center_error_m": max((r["center_error_m"] for r in boxed), default=None),
            "max_size_error_m": max((r["size_error_m"] for r in boxed), default=None),
            "ground_top_z_ned": ground_top, "tolerance_m": tolerance_m,
            "pass": (len(found) == len(rows) > 0 and len(boxed) > 0
                     and all(r["center_error_m"] < tolerance_m and r["size_error_m"] < tolerance_m for r in boxed)
                     and ground_top is not None and abs(ground_top) < GROUND_TOP_TOL_M)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="s01")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--layout", type=Path, default=None, help="default: the scene's stored layout (scenes/resolve.py)")
    parser.add_argument("--check-map", type=Path, default=None,
                        help="a sim/check_map.py record to embed as well (M1's separate step); the live check runs regardless")
    parser.add_argument("--package-manifest", type=Path, default=RUNS_DIR / "package" / "package_manifest.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: runs/m1/m1_gate.json for s01, docs/gates/m4_<scene>_gate.json otherwise (never overwritten)")
    parser.add_argument("--frames-dir", type=Path, default=None, help="default: runs/m1/frames or runs/m4/<scene>/frames")
    parser.add_argument("--scene-config", default=None,
                        help="default: the scene's own; e.g. scene_autofly_s01_fast.jsonc: M1 must pass on a clock before M2 trains on it")
    args = parser.parse_args(argv)
    resolved = resolve_scene(args.scene)
    is_s01 = args.scene == "s01"
    out = args.out or (RUNS_DIR / "m1" / "m1_gate.json" if is_s01 else m4_evidence_path(args.scene, "gate"))
    refuse_existing_evidence(out)
    frames_dir = args.frames_dir or (RUNS_DIR / "m1" / "frames" if is_s01 else RUNS_DIR / "m4" / args.scene / "frames")
    scene_config = args.scene_config or resolved.default_scene_config
    layout_path = args.layout or resolved.layout_path
    data = json.loads(layout_path.read_text())
    b = data["layout"]["bounds"]
    bounds = Bounds(b["x_min"], b["x_max"], b["y_min"], b["y_max"])
    registry = load_registry()
    keys = mesh_keys(data, registry)
    library = MeshLibrary.from_files(keys) if keys else None
    obstacles = obstacles_from_layout_json(data, registry, library)
    probes = choose_probes(obstacles, bounds, altitude_m=ALTITUDE_M)
    poses = scene_poses(bounds, ALTITUDE_M)
    kinds = {}
    for o in obstacles:
        kinds[o.kind] = kinds.get(o.kind, 0) + 1
    report: dict = {"scene": args.scene, "map": resolved.map_path, "layout": str(layout_path), "scene_sha256": data["scene_sha256"],
                    "layout_sha256": resolved.layout_sha256, "obstacle_kinds": kinds, "checks": {},
                    "probes": [{"tag": p.tag, "kind": p.obstacle.kind, "distance_m": p.distance_m, "expected_depth_m": p.expected_depth_m,
                                "window_spread_m": p.window_spread_m, "pose": [p.pose.x, p.pose.y, p.pose.z, p.pose.yaw]} for p in probes],
                    "poses": {k: [v.x, v.y, v.z, v.yaw] for k, v in vars(poses).items()}, "frames_dir": str(frames_dir),
                    "scene_config": scene_config_record(scene_config),
                    "check_map": json.loads(args.check_map.read_text()) if args.check_map and args.check_map.exists() else {},
                    "package_manifest": json.loads(args.package_manifest.read_text()) if args.package_manifest.exists() else {}}
    binary_digest = hashlib.sha256()
    with open(PACKAGED_BINARY, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            binary_digest.update(chunk)
    report["package_matches_manifest"] = (
        report["package_manifest"].get("binary", {}).get("sha256") == binary_digest.hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    route_client_log((RUNS_DIR / "m1" / "m1_gate.client.log") if is_s01 else (RUNS_DIR / "m4" / args.scene / "gate.client.log"))
    sim = ProjectAirSimSimulator(scene_config=scene_config)
    checks = [
        ("map_matches_layout", lambda: check_map(sim, obstacles)),
        ("rgb_changes_with_pose", lambda: check_rgb_changes(sim, poses)),
        ("depth_matches_geometry", lambda: check_depth(sim, probes, frames_dir)),
        ("velocity_tracking", lambda: check_tracking(sim, poses)),
        ("one_step_per_record", lambda: check_one_step(sim, poses)),
        ("spawn_destroy_packaged", lambda: check_spawn_destroy(sim, poses)),
        # last: a crash into a tree can leave the actor wedged (check_crash), and nothing may follow it
        ("crash_raises_collision", lambda: check_crash(sim, probes[1], poses)),
    ]
    # Xid lines are counted from this run's start without journalctl's -k (which implies -b, the current boot only).
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    run_start_epoch = time.time()
    xid_before = xid_count(run_started)
    launched = False
    try:
        start = time.monotonic()
        sim.launch(resolved.map_path, args.instance)
        launched = True
        report["launch_s"] = time.monotonic() - start
        for name, run in checks:
            report["checks"][name] = run()
            out.write_text(json.dumps(report, indent=2))
        report["vram_used_mib"] = gpu_memory_mib()[0]
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
    finally:
        sim.close()
    # This run's logs only (every sim*.log ever written would include other runs'), and a readable journal (C7).
    logs = instance_logs_since(instance_dir(args.instance), run_start_epoch)
    readable = kernel_journal_readable()
    report["faults"] = {"xid_since": run_started, "xid_before": xid_before, "xid_after": xid_count(run_started),
                        "boot_id": boot_id(), "device_lost": count_device_lost(logs), "kernel_journal_readable": readable}
    report["faults_ok"] = (readable and report["faults"]["xid_after"] == xid_before and len(logs) > 0
                           and sum(report["faults"]["device_lost"].values()) == 0)
    report["pass"] = ("error" not in report and len(report["checks"]) == len(checks)
                      and all(c["pass"] for c in report["checks"].values())
                      and report["package_matches_manifest"] and report["faults_ok"])
    destination = record_destination(out, did_work=launched)
    if destination != out and out.exists():
        out.unlink()  # the partial record written during the checks, before the run turned out never to have started
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": report["pass"], "error": report.get("error"), "record": str(destination),
                      "package_matches_manifest": report["package_matches_manifest"], "faults_ok": report["faults_ok"],
                      "checks": {k: v["pass"] for k, v in report["checks"].items()}}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    _code = main()
    # Not sys.exit(): an abandoned close() can leave projectairsim's non-daemon thread blocking interpreter
    # shutdown forever (see expert/train.py). Everything durable is already written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
