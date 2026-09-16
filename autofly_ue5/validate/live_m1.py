"""M1 gate: live checks of spec section 11 on the packaged s01 simulator, through the Simulator interface.

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.live_m1 --out runs/m1/m1_gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import PACKAGED_BINARY, RUNS_DIR
from autofly_ue5.scenes.model import Bounds
from autofly_ue5.sim.airsim_backend import ProjectAirSimSimulator
from autofly_ue5.sim.process import instance_dir, route_client_log
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.types import Pose
from autofly_ue5.validate.engine_check import boot_id, count_device_lost, xid_count
from autofly_ue5.validate.geometry import (
    CAMERA_OFFSET_M,
    DepthProbe,
    choose_depth_probes,
    contiguous_width,
    depth_agreement,
    expected_center_depth,
    mean_abs_diff,
    pillar_width_px,
    pillars_from_layout_json,
    speed_along_heading,
    tracking_error,
    yaw_rates_from_yaws,
)

DT = 0.2
# Opaque base UMaterial cooked with /Game/Geometry; M0 (Task 7 Step 11) showed it renders orange at runtime.
ORANGE_MATERIAL = "/Game/Geometry/Materials/M_Orange"
OPEN_POSE = Pose(-31.0, -25.0, -2.0, math.pi / 2)  # south start band (x_min edge), facing east along the band
PILLAR_VIEW = Pose(-31.0, 0.0, -2.0, 0.0)          # facing north into the pillar field
EMPTY_VIEW = Pose(-31.0, 0.0, -2.0, math.pi)        # facing south (-x), out of the scene over bare ground
SPAWN_POSE = Pose(-31.0, -10.0, -2.0, 0.0)          # runtime cube in the free start band (pillars start at x = -28)
SPAWN_VIEW = Pose(-31.0, -13.0, -2.0, math.pi / 2)  # 3 m west of the cube, facing east at it


def _finite(v: float | None) -> float | None:
    return v if v is not None and math.isfinite(v) else None


def check_rgb_changes(sim: Simulator) -> dict:
    a = sim.reset(PILLAR_VIEW).rgb
    b = sim.reset(EMPTY_VIEW).rgb
    a2 = sim.reset(PILLAR_VIEW).rgb
    out = {"diff_pillar_vs_empty": mean_abs_diff(a, b), "diff_pillar_repeat": mean_abs_diff(a, a2),
           "brightness_pillar_view": float(a.mean()), "brightness_empty_view": float(b.mean())}
    out["pass"] = out["diff_pillar_vs_empty"] > 10.0 and 20.0 < out["brightness_pillar_view"] < 235.0
    return out


def check_depth(sim: Simulator, probes: list[DepthProbe]) -> dict:
    rows = []
    for probe in probes:
        obs = sim.reset(probe.pose)
        measured = float(np.median(obs.depth[126:130, 126:130]))
        expected = expected_center_depth(obs.pose, probe.pillar)
        tolerance = max(0.15, 0.02 * probe.distance_m)
        error = abs(measured - expected) if expected is not None and math.isfinite(measured) else None
        # Field of view (spec §4.1): the pillar silhouette in row 128 (camera height) against the 90° pinhole prediction.
        # Silhouette pixels have planar depth below the axis distance; background pixels are farther or +inf.
        cam_x = obs.pose.x + CAMERA_OFFSET_M * math.cos(obs.pose.yaw)
        cam_y = obs.pose.y + CAMERA_OFFSET_M * math.sin(obs.pose.yaw)
        axis_distance = math.hypot(probe.pillar.x - cam_x, probe.pillar.y - cam_y)
        expected_width = pillar_width_px(axis_distance, probe.pillar.radius)
        row = obs.depth[128]
        width = contiguous_width(np.isfinite(row) & (row < axis_distance + 0.3))
        width_tolerance = max(2.0, 0.1 * expected_width)
        hfov_consistent = abs(width - expected_width) <= width_tolerance
        rows.append({"tag": probe.tag, "distance_m": probe.distance_m, "measured_m": _finite(measured), "expected_m": expected,
                     "error_m": error, "tolerance_m": tolerance, "width_px": width, "expected_width_px": expected_width,
                     "width_tolerance_px": width_tolerance, "hfov_consistent": hfov_consistent,
                     "pose": [obs.pose.x, obs.pose.y, obs.pose.z, obs.pose.yaw],
                     "pass": error is not None and error <= tolerance and hfov_consistent})
    return {"probes": rows, "pass": len(rows) == len(probes) and all(r["pass"] for r in rows)}


def check_crash(sim: Simulator, probe: DepthProbe) -> dict:
    reference = sim.reset(PILLAR_VIEW)
    sim.reset(probe.pose)
    out: dict = {"pass": False, "reason": "no collision within 25 steps"}
    for step in range(25):
        sim.command_velocity(2.0, 0.0, 0.0)
        sim.step(DT)
        obs = sim.observe()
        if obs.step_collisions:
            event = obs.step_collisions[0]
            axis_distance = math.hypot(event.impact_point[0] - probe.pillar.x, event.impact_point[1] - probe.pillar.y)
            out = {"steps_to_collision": step + 1, "object_name": event.object_name, "impact_to_pillar_axis_m": axis_distance,
                   "pillar_radius_m": probe.pillar.radius, "collided_flag": obs.collided,
                   "camera_pose_error_at_collision_m": obs.camera_pose_error_m,
                   "pass": obs.collided and axis_distance <= probe.pillar.radius + 0.5}
            break
    # A reset after a crash must bring the Unreal actor and its camera back (research gotcha 6); the backend raises
    # CameraPoseError when it cannot.
    try:
        after = sim.reset(PILLAR_VIEW)
    except Exception as err:
        out["reset_after_crash"] = {"error": f"{type(err).__name__}: {err}", "pass": False}
        out["pass"] = False
        return out
    depth = depth_agreement(reference.depth, after.depth)
    reset_ok = (
        after.camera_pose_error_m < 0.1 and after.collided is False and depth["mask_agreement"] >= 0.98
        and (depth["median_relative_diff"] is None or depth["median_relative_diff"] <= 0.05)
    )
    out["reset_after_crash"] = {"camera_pose_error_m": after.camera_pose_error_m, "depth": depth,
                                "rgb_mean_abs_diff": mean_abs_diff(reference.rgb, after.rgb), "pass": reset_ok}
    out["pass"] = bool(out["pass"]) and reset_ok
    return out


def check_tracking(sim: Simulator) -> dict:
    sim.reset(OPEN_POSE)
    forward = []
    for _ in range(15):
        sim.command_velocity(2.0, 0.0, 0.0)
        sim.step(DT)
        obs = sim.observe()
        forward.append(speed_along_heading(obs.velocity_ned, obs.pose.yaw))
    sim.reset(OPEN_POSE)
    up, down = [], []
    for v_z, sink in [(0.5, up)] * 10 + [(-0.5, down)] * 10:
        sim.command_velocity(0.0, 0.0, v_z)
        sim.step(DT)
        sink.append(-sim.observe().velocity_ned[2])
    obs = sim.reset(OPEN_POSE)
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


def check_one_step(sim: Simulator) -> dict:
    """Each record: one clock step of 200 ms, and the RGB, depth and kinematics that built it all carry that step's time."""
    sim.reset(OPEN_POSE)
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
    return {"records": len(rows), "records_per_s": len(rows) / wall, "rows": rows, "pass": ok}


def check_spawn_destroy(sim: Simulator) -> dict:
    """Runtime spawn of a cooked Blocks mesh with a runtime material in the packaged build (spec §4.1), seen in depth and
    colour, then destroyed. The backend raises if set_object_material returns False."""
    name = sim.spawn("AF_M1_Cube", "1M_Cube", SPAWN_POSE, (1.0, 1.0, 1.0), ORANGE_MATERIAL)
    with_obs = sim.reset(SPAWN_VIEW)
    with_cube = float(np.median(with_obs.depth[126:130, 126:130]))
    red, green, blue = (float(v) for v in with_obs.rgb[120:136, 120:136].reshape(-1, 3).mean(axis=0))
    orange_dominant = red > green > blue and red - blue > 30.0
    sim.destroy(name)
    without_cube = float(np.median(sim.reset(SPAWN_VIEW).depth[126:130, 126:130]))
    expected = 3.0 - 0.5 - CAMERA_OFFSET_M
    gone = math.isinf(without_cube) or without_cube > expected + 1.0
    return {"spawned_name": name, "material": ORANGE_MATERIAL, "center_depth_with_cube_m": _finite(with_cube),
            "expected_m": expected, "center_rgb_with_cube": [red, green, blue], "orange_dominant": orange_dominant,
            "center_depth_after_destroy_m": _finite(without_cube),
            "pass": name.startswith("AF_M1_Cube") and abs(with_cube - expected) <= 0.15 and orange_dominant and gone}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="/Game/AutoFly/Maps/S01")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--layout", type=Path, default=RUNS_DIR / "levels" / "s01.layout.json")
    parser.add_argument("--check-map", type=Path, default=RUNS_DIR / "m1" / "check_map.json")
    parser.add_argument("--package-manifest", type=Path, default=RUNS_DIR / "package" / "package_manifest.json")
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "m1" / "m1_gate.json")
    args = parser.parse_args(argv)
    data = json.loads(args.layout.read_text())
    b = data["layout"]["bounds"]
    probes = choose_depth_probes(pillars_from_layout_json(data), Bounds(b["x_min"], b["x_max"], b["y_min"], b["y_max"]))
    report: dict = {"map": args.map, "layout": str(args.layout), "scene_sha256": data["scene_sha256"], "checks": {},
                    "check_map": json.loads(args.check_map.read_text()) if args.check_map.exists() else {},
                    "package_manifest": json.loads(args.package_manifest.read_text()) if args.package_manifest.exists() else {}}
    binary_digest = hashlib.sha256()
    with open(PACKAGED_BINARY, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            binary_digest.update(chunk)
    report["package_matches_manifest"] = (
        report["package_manifest"].get("binary", {}).get("sha256") == binary_digest.hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    route_client_log(args.out.with_suffix(".client.log"))
    sim = ProjectAirSimSimulator(scene_config="scene_autofly_s01.jsonc")
    checks = [
        ("rgb_changes_with_pose", lambda: check_rgb_changes(sim)),
        ("depth_matches_geometry", lambda: check_depth(sim, probes)),
        ("crash_raises_collision", lambda: check_crash(sim, probes[1])),
        ("velocity_tracking", lambda: check_tracking(sim)),
        ("one_step_per_record", lambda: check_one_step(sim)),
        ("spawn_destroy_packaged", lambda: check_spawn_destroy(sim)),
    ]
    # Xid lines are counted from this run's start without journalctl's -k (which implies -b, the current boot only).
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    xid_before = xid_count(run_started)
    try:
        start = time.monotonic()
        sim.launch(args.map, args.instance)
        report["launch_s"] = time.monotonic() - start
        for name, run in checks:
            report["checks"][name] = run()
            args.out.write_text(json.dumps(report, indent=2))
        report["vram_used_mib"] = gpu_memory_mib()[0]
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
    finally:
        sim.close()
    logs = sorted(instance_dir(args.instance).glob("sim*.log"))
    report["faults"] = {"xid_since": run_started, "xid_before": xid_before, "xid_after": xid_count(run_started),
                        "boot_id": boot_id(), "device_lost": count_device_lost(logs)}
    report["faults_ok"] = (report["faults"]["xid_after"] == xid_before and len(logs) > 0
                           and sum(report["faults"]["device_lost"].values()) == 0)
    report["pass"] = ("error" not in report and len(report["checks"]) == len(checks)
                      and all(c["pass"] for c in report["checks"].values())
                      and report["check_map"].get("pass") is True and report["package_matches_manifest"]
                      and report["faults_ok"])
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": report["pass"], "error": report.get("error"), "check_map": report["check_map"].get("pass"),
                      "package_matches_manifest": report["package_matches_manifest"], "faults_ok": report["faults_ok"],
                      "checks": {k: v["pass"] for k, v in report["checks"].items()}}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
