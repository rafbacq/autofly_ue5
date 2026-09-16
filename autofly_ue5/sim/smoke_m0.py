"""M0 gate smoke test against a running Project AirSim simulator (launched with scripts/launch_sim.py).

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.smoke_m0 --instance 0 --scene scene_autofly_m0.jsonc \
    --out runs/m0/smoke_inst0.json --fixtures-dir runs/m0/fixtures
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.drone import YawControlMode
from projectairsim.types import BoxAlignment, Pose

from autofly_ue5.frames import body_to_ned, heading_unit, quat_to_yaw, wrap_pi, yaw_to_quat
from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import CONFIGS_DIR
from autofly_ue5.sim.decode import decode_depth, decode_rgb
from autofly_ue5.sim.process import ports_for_instance, route_client_log

DT_NS = 200_000_000
DT_S = 0.2
STEP_NS = 5_000_000
# The server starts a command's duration when its request job runs, possibly one clock tick after Step started
# (service_manager.cpp:440); two ticks of margin keep the reply inside the step.
COMMAND_DURATION_S = (DT_NS - 2 * STEP_NS) / 1e9
ROBOT = "Drone1"
CAMERA = "FrontCamera"
CAMERA_X_OFFSET_M = 0.40
IMAGE_W = 256
HFOV_DEG = 90.0
CUBE_ASSET = "1M_Cube"
PILLAR_SCALE = [1.0, 3.0, 12.0]  # 1 m deep along the view, 3 m wide face, 12 m tall
# Opaque base UMaterial (BLEND_Opaque) cooked with /Game/Geometry; the server loads materials with a UMaterial class
# filter (WorldSimApi.cpp:1030-1032). M_Blue is a MaterialInstanceConstant: attempted once, predicted to fail.
BASE_MATERIAL = "/Game/Geometry/Materials/M_Orange"
INSTANCE_MATERIAL = "/Game/Geometry/Materials/M_Blue"
PILLAR_NAME = "AF_M0_Pillar"
TMP_NAME = "AF_M0_Tmp"
SAFE_Z = -40.0
PHASE_ORDER = ["lockstep", "velocity", "reset", "spawn", "collision", "teleport"]


class CommandTimeoutError(RuntimeError):
    """A velocity command's reply did not arrive after its step."""


def make_pose(x: float, y: float, z: float, yaw: float) -> Pose:
    w, qx, qy, qz = yaw_to_quat(yaw)
    return Pose({"frame_id": "DEFAULT_FRAME", "translation": {"x": x, "y": y, "z": z},
                 "rotation": {"w": w, "x": qx, "y": qy, "z": qz}})


def center_depth(depth: np.ndarray) -> float:
    return float(np.median(depth[124:132, 124:132]))


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


def flat_face_width_px(half_width_m: float, depth_m: float, image_width: int = IMAGE_W, hfov_deg: float = HFOV_DEG) -> float:
    """Pinhole image width of a flat face perpendicular to the optical axis and centred on it."""
    return image_width * half_width_m / (depth_m * math.tan(math.radians(hfov_deg) / 2.0))


def camera_pose_error(msg: dict | None, rec: dict) -> float:
    """Distance between the camera position stamped in an image message and kinematics + the rotated mount offset."""
    if msg is None:
        return math.inf
    off = body_to_ned(*rec["quat"], (CAMERA_X_OFFSET_M, 0.0, 0.0))
    expected = (rec["x"] + off[0], rec["y"] + off[1], rec["z"] + off[2])
    return math.dist((float(msg["pos_x"]), float(msg["pos_y"]), float(msg["pos_z"])), expected)


class FrameSlots:
    """Newest RGB and depth message per stream, filled from the client's receive thread."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._latest: dict[str, dict | None] = {"rgb": None, "depth": None}

    def callback(self, key: str):
        def _on_message(_topic, msg):
            with self._cond:
                self._latest[key] = msg
                self._cond.notify_all()
        return _on_message

    def wait(self, t_ns: int, timeout_s: float) -> dict[str, dict | None]:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                msgs = dict(self._latest)
                if all(m is not None and m["time_stamp"] >= t_ns for m in msgs.values()):
                    return msgs
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return msgs
                self._cond.wait(remaining)


class Smoke:
    def __init__(self, client: ProjectAirSimClient, world: World, drone: Drone, frame_timeout_s: float,
                 first_frame_timeout_s: float, command_timeout_s: float) -> None:
        self.client, self.world, self.drone = client, world, drone
        self.frame_timeout_s = frame_timeout_s
        self.first_frame_timeout_s = first_frame_timeout_s
        self.command_timeout_s = command_timeout_s
        self.frames = FrameSlots()
        self.collision_msgs: list[dict] = []
        self.loop = asyncio.new_event_loop()
        self.t_ns = int(world.get_sim_time())
        self.records = 0
        self.origin: dict | None = None
        self.pillar_name: str | None = None
        self.pillar_center: tuple[float, float, float] | None = None
        self.pillar_yaw = 0.0
        cams = drone.sensors[CAMERA]
        client.subscribe(cams["scene_camera"], self.frames.callback("rgb"))
        client.subscribe(cams["depth_planar_camera"], self.frames.callback("depth"))
        client.subscribe(drone.robot_info["collision_info"], lambda _t, m: self.collision_msgs.append(m))

    def record(self, v_forward: float = 0.0, yaw_rate: float = 0.0, v_up: float = 0.0) -> dict:
        """Advance one lock-step and return the resulting frame. R8 (controller ruling, M0 smoke test):
        the first record() of a session returns a frame captured before the depth buffer has stabilised
        (measured: frame 0 had no_hit_px=14, min_finite_m=0.0425, max_finite_m=0.998, vs. no_hit_px~24,690,
        min_finite_m~1.4 on every later frame) and must be discarded, never treated as real data."""
        target = self.t_ns + DT_NS
        wall_start = time.monotonic()
        task = self.loop.run_until_complete(self.drone.move_by_velocity_body_frame_async(
            v_forward, 0.0, -v_up, duration=COMMAND_DURATION_S,
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom, yaw_is_rate=True, yaw=yaw_rate))
        self.loop.run_until_complete(asyncio.sleep(0.002))  # let the server job read the command start time
        result = self.world.step(DT_NS)
        try:
            self.loop.run_until_complete(asyncio.wait_for(task, timeout=self.command_timeout_s))
        except asyncio.TimeoutError as err:
            raise CommandTimeoutError(
                f"move command reply missing {self.command_timeout_s} s after the step to {target} ns") from err
        timeout = self.first_frame_timeout_s if self.records == 0 else self.frame_timeout_s
        msgs = self.frames.wait(target, timeout)
        kin = self.drone.get_ground_truth_kinematics()
        for _ in range(4):
            if int(kin["time_stamp"]) == target:
                break
            time.sleep(0.01)
            kin = self.drone.get_ground_truth_kinematics()
        wall_s = time.monotonic() - wall_start
        self.t_ns = int(result["sim_time_ns"])
        self.records += 1
        pos, ori = kin["pose"]["position"], kin["pose"]["orientation"]
        rec = {
            "target_ns": target,
            "sim_time_ns": int(result["sim_time_ns"]),
            "rgb_ts": msgs["rgb"]["time_stamp"] if msgs["rgb"] else None,
            "depth_ts": msgs["depth"]["time_stamp"] if msgs["depth"] else None,
            "kin_ts": int(kin["time_stamp"]),
            "x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"]),
            "yaw": quat_to_yaw(ori["w"], ori["x"], ori["y"], ori["z"]),
            "quat": [float(ori[k]) for k in ("w", "x", "y", "z")],
            "vel": [float(kin["twist"]["linear"][k]) for k in ("x", "y", "z")],
            "events": [e for e in result["robots"][ROBOT]["events"] if e.get("type") == "collision"],
            "rgb_msg": msgs["rgb"],
            "depth_msg": msgs["depth"],
            "wall_s": wall_s,
        }
        rec["camera_pose_error_m"] = camera_pose_error(msgs["rgb"], rec)
        if self.origin is None:
            self.origin = {"x": rec["x"], "y": rec["y"], "z": rec["z"], "yaw": rec["yaw"]}
        return rec


def frames_at_target(rec: dict) -> bool:
    return rec["rgb_ts"] == rec["target_ns"] and rec["depth_ts"] == rec["target_ns"]


def probe_get_images(smoke: Smoke) -> dict:
    start = time.monotonic()
    try:
        images = smoke.drone.get_images(CAMERA, [0, 1])
    except RuntimeError as err:
        return {"ok": False, "error": str(err), "elapsed_s": round(time.monotonic() - start, 3)}
    stamps = {str(k): int(v["time_stamp"]) for k, v in images.items()}
    ok = set(images) == {0, 1} and all(s >= smoke.t_ns for s in stamps.values())
    return {"ok": ok, "time_stamps": stamps, "sim_time_ns": smoke.t_ns, "elapsed_s": round(time.monotonic() - start, 3)}


def phase_lockstep(smoke: Smoke, args: argparse.Namespace) -> dict:
    warm = [smoke.record() for _ in range(args.warmup_steps)]  # first-frame wait and shader warm-up stay untimed
    start_late_s = None
    if args.start_at is not None:  # lets two instances time overlapping windows
        start_late_s = time.time() - args.start_at
        if start_late_s < 0:
            time.sleep(-start_late_s)
    n = args.steps
    used = total = vram_sample_unix = None
    recs = []
    timed_start_unix = time.time()
    wall_start = time.monotonic()
    for i in range(n):
        recs.append(smoke.record())
        if i == n // 2:  # mid-window sample, while this camera (and any concurrent instance) is capturing
            used, total = gpu_memory_mib()
            vram_sample_unix = time.time()
    wall = time.monotonic() - wall_start
    timed_end_unix = time.time()
    checked = warm + recs
    m = len(checked)
    last = recs[-1]
    rgb = decode_rgb(last["rgb_msg"]) if last["rgb_msg"] else None
    depth = decode_depth(last["depth_msg"]) if last["depth_msg"] else None
    z0 = checked[0]["z"]
    # R6 (controller ruling): decode_depth() already canonicalises this build's 0.0 no-hit sentinel to
    # +inf (R5), so a finite-and-positive minimum now means "closest real hit", not "closest to sky".
    # Extremes are recorded per checked frame (decode is cheap for <= steps+warmup_steps 256x256 half-float
    # frames) so a future regression can be localised straight from the report, without a re-run.
    depth_extremes = []
    for i, r in enumerate(checked):
        d = decode_depth(r["depth_msg"]) if r["depth_msg"] else None
        if d is None:
            depth_extremes.append({"index": i, "no_hit_px": None, "min_finite_m": None, "max_finite_m": None})
            continue
        finite = d[np.isfinite(d) & (d > 0.0)]
        depth_extremes.append({
            "index": i,
            "no_hit_px": int(np.sum(~np.isfinite(d))),
            "min_finite_m": float(finite.min()) if finite.size else None,
            "max_finite_m": float(finite.max()) if finite.size else None,
        })
    last_extremes = depth_extremes[-1]
    out = {
        "steps": n,
        "warmup_steps": len(warm),
        "sim_time_exact": sum(r["sim_time_ns"] == r["target_ns"] for r in checked),
        "frames_eq_target": sum(frames_at_target(r) for r in checked),
        "kin_ts_eq_target": sum(r["kin_ts"] == r["target_ns"] for r in checked),
        "max_camera_pose_error_m": max(r["camera_pose_error_m"] for r in checked),
        "z_drift_m": max(abs(r["z"] - z0) for r in checked),
        "steps_per_s": n / wall,
        "timed_start_unix": timed_start_unix,
        "timed_end_unix": timed_end_unix,
        "start_late_s": start_late_s,
        "median_record_wall_s": float(np.median([r["wall_s"] for r in recs])),
        "rgb_shape": list(rgb.shape) if rgb is not None else None,
        "depth_shape": list(depth.shape) if depth is not None else None,
        "rgb_mean": float(rgb.mean()) if rgb is not None else None,
        "depth_min_m": last_extremes["min_finite_m"],
        "no_hit_px": last_extremes["no_hit_px"],
        "depth_extremes": depth_extremes,
        "vram_used_mib": used,
        "vram_total_mib": total,
        "vram_sample_unix": vram_sample_unix,
        "get_images_probe": probe_get_images(smoke),
    }
    out["pass"] = (
        out["sim_time_exact"] == m and out["frames_eq_target"] == m and out["kin_ts_eq_target"] == m
        and out["max_camera_pose_error_m"] < 0.1
        and out["z_drift_m"] < 0.5 and out["rgb_shape"] == [256, 256, 3] and out["depth_shape"] == [256, 256]
        and out["rgb_mean"] is not None and out["rgb_mean"] > 10.0
        and out["depth_min_m"] is not None and out["depth_min_m"] >= 0.5
    )
    return out


def phase_velocity(smoke: Smoke, args: argparse.Namespace) -> dict:
    start = smoke.record()
    ahead = center_depth(decode_depth(start["depth_msg"]))
    if not ahead > 15.0:
        return {"pass": False, "reason": f"BlocksMap geometry {ahead:.2f} m ahead of the start pose; not flying"}
    north, east = heading_unit(start["yaw"])
    fwd = [smoke.record(v_forward=2.0) for _ in range(10)]
    dx, dy = fwd[-1]["x"] - start["x"], fwd[-1]["y"] - start["y"]
    along, lateral = dx * north + dy * east, -dx * east + dy * north
    hold = [smoke.record() for _ in range(5)]
    up = [smoke.record(v_up=1.0) for _ in range(5)]
    climb = -(up[-1]["z"] - hold[-1]["z"])
    hold2 = [smoke.record() for _ in range(5)]
    yaws = [hold2[-1]["yaw"]] + [smoke.record(yaw_rate=0.5)["yaw"] for _ in range(10)]
    dyaw = sum(wrap_pi(b - a) for a, b in zip(yaws, yaws[1:]))
    for _ in range(5):
        smoke.record()
    out = {
        "forward_along_m": along, "forward_lateral_m": lateral, "climb_m": climb,
        "yaw_change_rad": dyaw, "yaw_rate_measured_rad_s": dyaw / 2.0, "yaw_rate_ratio": (dyaw / 2.0) / 0.5,
    }
    out["pass"] = along >= 2.0 and abs(lateral) < 1.0 and climb >= 0.5 and abs(dyaw) >= 0.3
    return out


def phase_reset(smoke: Smoke, args: argparse.Namespace) -> dict:
    o = smoke.origin
    ok = bool(smoke.drone.set_pose(make_pose(o["x"], o["y"], o["z"], 0.0), reset_kinematics=True))
    recs = [smoke.record() for _ in range(5)]
    last = recs[-1]
    pos_err = math.dist((last["x"], last["y"], last["z"]), (o["x"], o["y"], o["z"]))
    speed = math.hypot(*last["vel"])
    yaw_err = abs(wrap_pi(last["yaw"]))
    out = {"set_pose_ok": ok, "position_error_m": pos_err, "speed_m_s": speed, "yaw_error_rad": yaw_err,
           "camera_pose_vs_state_m": last["camera_pose_error_m"]}
    out["pass"] = ok and pos_err < 0.5 and speed < 0.5 and yaw_err < 0.1 and last["camera_pose_error_m"] < 0.1
    return out


def save_fixture(msg: dict, stem: str, directory: Path, extra: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = bytes(msg["data"])
    (directory / f"{stem}.bin").write_bytes(data)
    meta = {k: v for k, v in msg.items() if k != "data"}
    meta.update(extra)
    meta["data_len"] = len(data)
    (directory / f"{stem}.json").write_text(json.dumps(meta, indent=2, default=str))


def phase_spawn(smoke: Smoke, args: argparse.Namespace) -> dict:
    before = smoke.record()
    assets = smoke.world.list_assets(f"^{CUBE_ASSET}$")
    north, east = heading_unit(before["yaw"])
    px, py, pz = before["x"] + 8.0 * north, before["y"] + 8.0 * east, before["z"]
    name = smoke.world.spawn_object(PILLAR_NAME, CUBE_ASSET, make_pose(px, py, pz, before["yaw"]), PILLAR_SCALE, False)
    smoke.pillar_name, smoke.pillar_center, smoke.pillar_yaw = name, (px, py, pz), before["yaw"]
    bbox = smoke.world.get_3d_bounding_box(name, BoxAlignment.OBJECT_ORIENTED)
    # Depth, face width and the fixtures come from the cube's own opaque material, before any material change.
    after = smoke.record()
    rgb_before, rgb_after = decode_rgb(before["rgb_msg"]), decode_rgb(after["rgb_msg"])
    depth_after = decode_depth(after["depth_msg"])
    along = (px - after["x"]) * north + (py - after["y"]) * east
    expected = along - PILLAR_SCALE[0] / 2.0 - CAMERA_X_OFFSET_M
    measured = center_depth(depth_after)
    # Field of view: the 3 m face is a flat plane facing the camera, so its planar depth is constant across row 128.
    row = depth_after[128]
    face_width = contiguous_width(np.isfinite(row) & (np.abs(row - measured) < 0.2))
    expected_face_width = flat_face_width_px(PILLAR_SCALE[1] / 2.0, measured) if 0.0 < measured < math.inf else None
    hfov_consistent = expected_face_width is not None and abs(face_width - expected_face_width) <= 3.0
    # R7 (controller ruling): a 54-px-wide pillar cannot move the whole-256x256-frame RGB mean by 5.0 grey
    # levels (rgb_mean_abs_diff below is kept as informational only); the visibility gate instead measures
    # the mean absolute RGB difference over the object's own projected bounding-box patch, using the same
    # pinhole geometry already predicting expected_face_width_px to within 0.1 px. The pillar's face and the
    # camera share height (mount origin has no z offset) and the drone/pillar share no lateral offset here,
    # so the projected patch is centred on the image.
    expected_face_height = flat_face_width_px(PILLAR_SCALE[2] / 2.0, measured) if 0.0 < measured < math.inf else None
    patch_bbox_px = None
    patch_rgb_mean_abs_diff = None
    if expected_face_width is not None and expected_face_height is not None:
        cx = cy = IMAGE_W / 2.0
        x_min = max(0, int(round(cx - expected_face_width / 2.0)))
        x_max = min(IMAGE_W, int(round(cx + expected_face_width / 2.0)))
        y_min = max(0, int(round(cy - expected_face_height / 2.0)))
        y_max = min(IMAGE_W, int(round(cy + expected_face_height / 2.0)))
        if x_max > x_min and y_max > y_min:
            patch_bbox_px = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
            patch_rgb_mean_abs_diff = float(np.mean(np.abs(
                rgb_after[y_min:y_max, x_min:x_max].astype(np.int16)
                - rgb_before[y_min:y_max, x_min:x_max].astype(np.int16))))
    material_ok = bool(smoke.world.set_object_material(name, BASE_MATERIAL))
    after_material = smoke.record()
    rgb_material = decode_rgb(after_material["rgb_msg"])
    patch_diff = float(np.mean(np.abs(rgb_material[112:144, 112:144].astype(np.int16) - rgb_after[112:144, 112:144].astype(np.int16))))
    material_center_rgb = [float(v) for v in rgb_material[120:136, 120:136].reshape(-1, 3).mean(axis=0)]
    center_err = math.dist((bbox["center"]["x"], bbox["center"]["y"], bbox["center"]["z"]), (px, py, pz)) if bbox else None
    size = [bbox["size"][k] for k in ("x", "y", "z")] if bbox else None
    tx, ty = px - 10.0 * east, py + 10.0 * north
    tmp = smoke.world.spawn_object(TMP_NAME, CUBE_ASSET, make_pose(tx, ty, pz, 0.0), [1.0, 1.0, 1.0], False)
    listed = smoke.world.list_objects(f"{tmp}.*")
    moved_ok = bool(smoke.world.set_object_pose(tmp, make_pose(tx + 2.0, ty, pz - 1.0, 0.0), True))
    scaled_ok = bool(smoke.world.set_object_scale(tmp, [2.0, 2.0, 2.0]))
    tmp_translation = smoke.world.get_object_pose(tmp)["translation"]
    tmp_move_err = math.dist((tmp_translation["x"], tmp_translation["y"], tmp_translation["z"]), (tx + 2.0, ty, pz - 1.0))
    tmp_bbox = smoke.world.get_3d_bounding_box(tmp, BoxAlignment.WORLD_AXIS)
    tmp_size = [tmp_bbox["size"][k] for k in ("x", "y", "z")] if tmp_bbox else None
    # Spec §4.1's material-instance route, informational: the UMaterial class filter predicts False.
    material_instance_ok = bool(smoke.world.set_object_material(tmp, INSTANCE_MATERIAL))
    destroyed = bool(smoke.world.destroy_object(tmp))
    listed_after = smoke.world.list_objects(f"{tmp}.*")
    if args.fixtures_dir is not None:
        rgb_data = bytes(after["rgb_msg"]["data"])
        center_index = (128 * 256 + 128) * 3
        save_fixture(after["rgb_msg"], "rgb_msg", args.fixtures_dir,
                     {"center_pixel_bgr": list(rgb_data[center_index:center_index + 3])})
        depth_data = bytes(after["depth_msg"]["data"])
        inf_count = sum(1 for (v,) in struct.iter_unpack("<e", depth_data) if math.isinf(v))
        save_fixture(after["depth_msg"], "depth_msg", args.fixtures_dir,
                     {"inf_count": inf_count, "expected_center_depth_m": expected})
        Image.fromarray(rgb_after).save(args.fixtures_dir / "rgb_after_spawn.png")
        Image.fromarray(rgb_material).save(args.fixtures_dir / "rgb_after_material.png")
    out = {
        "assets_found": assets, "spawned_name": name, "bbox_center_error_m": center_err, "bbox_size_m": size,
        "rgb_mean_abs_diff": float(np.mean(np.abs(rgb_after.astype(np.int16) - rgb_before.astype(np.int16)))),
        "center_depth_m": measured, "expected_center_depth_m": expected, "depth_error_m": abs(measured - expected),
        "center_face_width_px": face_width, "expected_face_width_px": expected_face_width,
        "expected_face_height_px": expected_face_height, "hfov_consistent": hfov_consistent,
        "patch_bbox_px": patch_bbox_px, "patch_rgb_mean_abs_diff": patch_rgb_mean_abs_diff,
        "material": BASE_MATERIAL, "material_ok": material_ok, "material_patch_mean_abs_diff": patch_diff,
        "material_center_rgb": material_center_rgb,
        "material_instance": INSTANCE_MATERIAL, "material_instance_ok": material_instance_ok,
        "tmp_name": tmp, "tmp_listed": listed, "tmp_set_pose_ok": moved_ok, "tmp_move_error_m": tmp_move_err,
        "tmp_set_scale_ok": scaled_ok, "tmp_bbox_size_after_scale_m": tmp_size,
        "tmp_destroyed": destroyed, "tmp_listed_after": listed_after,
    }
    out["pass"] = (
        len(assets) > 0 and name.startswith(PILLAR_NAME) and center_err is not None and center_err < 0.1
        and size is not None and all(abs(a - b) < 0.1 for a, b in zip(size, PILLAR_SCALE))
        and patch_rgb_mean_abs_diff is not None and patch_rgb_mean_abs_diff >= 10.0
        and out["depth_error_m"] < 0.3 and hfov_consistent
        and material_ok and patch_diff > 10.0
        and len(listed) > 0 and moved_ok and tmp_move_err < 0.05
        and scaled_ok and tmp_size is not None and all(abs(v - 2.0) < 0.1 for v in tmp_size)
        and destroyed and len(listed_after) == 0
    )
    return out


def phase_collision(smoke: Smoke, args: argparse.Namespace) -> dict:
    if smoke.pillar_name is None:
        return {"pass": False, "reason": "spawn phase did not run"}
    smoke.collision_msgs.clear()
    events: list[dict] = []
    hit_step = None
    for i in range(30):
        rec = smoke.record(v_forward=2.0)
        events.extend(rec["events"])
        if rec["events"]:
            hit_step = i
            break
    tail = smoke.record()
    events.extend(tail["events"])
    names = sorted({e["object_name"] for e in events})
    topic_names = sorted({m["object_name"] for m in smoke.collision_msgs})
    out = {"hit_step": hit_step, "step_event_names": names, "topic_event_names": topic_names,
           "first_event": events[0] if events else None, "drone_at_end": [tail["x"], tail["y"], tail["z"]]}
    out["pass"] = hit_step is not None and any(smoke.pillar_name in n for n in names)
    return out


def teleport_and_record(smoke: Smoke, x: float, y: float, z: float, yaw: float, records: int = 1) -> dict:
    ok = bool(smoke.drone.set_pose(make_pose(x, y, z, yaw), reset_kinematics=True))
    recs = [smoke.record() for _ in range(records)]
    last = dict(recs[-1])
    last["set_pose_ok"] = ok
    last["all_events"] = [e for r in recs for e in r["events"]]
    return last


def phase_teleport(smoke: Smoke, args: argparse.Namespace) -> dict:
    """Research gotcha 6 live: set_pose sweeps the Unreal actor, so a teleport through an obstacle can leave the camera
    behind while kinematics jump. Gated: the backend's reset path (straight up to a safe altitude, across, down) must
    resynchronise camera and kinematics after the collision phase left the drone touching the pillar.
    Informational: camera/kinematics mismatch and events after a teleport straight through the pillar."""
    if smoke.pillar_center is None:
        return {"pass": False, "reason": "spawn phase did not run"}
    px, py, pz = smoke.pillar_center
    yaw = smoke.pillar_yaw
    north, east = heading_unit(yaw)
    now = smoke.record()
    up = teleport_and_record(smoke, now["x"], now["y"], SAFE_Z, yaw)
    across = teleport_and_record(smoke, px - 5.0 * north, py - 5.0 * east, SAFE_Z, yaw)
    front = teleport_and_record(smoke, px - 5.0 * north, py - 5.0 * east, pz, yaw, records=2)
    through = teleport_and_record(smoke, px + 5.0 * north, py + 5.0 * east, pz, yaw, records=2)
    msg = through["rgb_msg"]
    out = {
        "recovery_camera_errors_m": [up["camera_pose_error_m"], across["camera_pose_error_m"], front["camera_pose_error_m"]],
        "recovery_event_names": sorted({e["object_name"] for r in (up, across, front) for e in r["all_events"]}),
        "through_camera_error_m": through["camera_pose_error_m"],
        "through_event_names": sorted({e["object_name"] for e in through["all_events"]}),
        "through_camera_position": [msg["pos_x"], msg["pos_y"], msg["pos_z"]] if msg else None,
        "through_kinematics_position": [through["x"], through["y"], through["z"]],
        "set_pose_ok": all(r["set_pose_ok"] for r in (up, across, front, through)),
    }
    out["pass"] = out["set_pose_ok"] and all(e < 0.1 for e in out["recovery_camera_errors_m"])
    return out


PHASES = {"lockstep": phase_lockstep, "velocity": phase_velocity, "reset": phase_reset,
          "spawn": phase_spawn, "collision": phase_collision, "teleport": phase_teleport}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--scene", default="scene_autofly_m0.jsonc")
    parser.add_argument("--phases", default=",".join(PHASE_ORDER))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--start-at", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, default=None)
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--first-frame-timeout", type=float, default=120.0)
    parser.add_argument("--command-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.warmup_steps < 1:
        # R8 (controller ruling): frame 0 of a session is captured before the depth buffer has stabilised
        # (measured: no_hit_px=14, min_finite_m=0.0425, max_finite_m=0.998, vs. no_hit_px~24,690,
        # min_finite_m~1.4 on every later frame) and must be discarded, never scored as real data.
        parser.error(
            f"--warmup-steps must be >= 1, got {args.warmup_steps}: record()'s first frame is captured "
            "before the depth buffer has stabilised (measured frame 0: no_hit_px=14, min_finite_m=0.0425, "
            "max_finite_m=0.998, vs. no_hit_px~24,690, min_finite_m~1.4 on every later frame) and must be "
            "discarded, not scored"
        )
    phases = args.phases.split(",")
    unknown = set(phases) - set(PHASE_ORDER)
    if unknown:
        parser.error(f"unknown phases {sorted(unknown)}")
    ports = ports_for_instance(args.instance)
    report: dict = {"instance": args.instance, "ports": [ports.topics, ports.services], "scene": args.scene, "phases": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    route_client_log(args.out.with_suffix(".client.log"))
    client = ProjectAirSimClient(port_topics=ports.topics, port_services=ports.services)
    try:
        client.connect()
        load_start = time.monotonic()
        world = World(client, args.scene, delay_after_load_sec=0, sim_config_path=str(CONFIGS_DIR))
        report["load_scene_s"] = time.monotonic() - load_start
        report["clock_type"] = world.get_sim_clock_type()
        report["paused_on_load"] = world.is_paused()
        drone = Drone(client, world, ROBOT)
        report["camera_topics"] = sorted(drone.sensors.get(CAMERA, {}))
        drone.enable_api_control()
        drone.arm()
        smoke = Smoke(client, world, drone, args.frame_timeout, args.first_frame_timeout, args.command_timeout)
        for name in PHASE_ORDER:
            if name in phases:
                report["phases"][name] = PHASES[name](smoke, args)
                args.out.write_text(json.dumps(report, indent=2, default=str))
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
    finally:
        client.disconnect()
    report["pass"] = (
        "error" not in report and report.get("clock_type") == "steppable"
        and set(report["phases"]) == set(phases) and all(p["pass"] for p in report["phases"].values())
    )
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"pass": report["pass"], "error": report.get("error"),
                      "phases": {k: v["pass"] for k, v in report["phases"].items()}}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
