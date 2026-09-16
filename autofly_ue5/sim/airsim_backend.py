"""Project AirSim implementation of the Simulator interface (lock-step, 5 Hz records)."""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path

from autofly_ue5.frames import body_to_ned, quat_to_yaw, yaw_to_quat
from autofly_ue5.gpu import check_gpu_for_launch, gpu_memory_mib
from autofly_ue5.paths import CONFIGS_DIR, PACKAGED_BINARY
from autofly_ue5.sim.decode import decode_depth, decode_rgb
from autofly_ue5.sim.events import CollisionLog, collisions_after
from autofly_ue5.sim.process import (
    SIM_RUN_DIR,
    SimPorts,
    SimProcess,
    instance_dir,
    launch_process,
    own_running_instances,
    packaged_command,
    ports_for_instance,
    sim_environment,
    stop,
    wait_ready,
)
from autofly_ue5.sim.sync import FrameCollector
from autofly_ue5.sim.types import (
    CONTROL_DT_S,
    STEP_NS,
    ObjectNotFoundError,
    Observation,
    Pose,
    SessionNotResetError,
    dt_to_ns,
)

ROBOT = "Drone1"
CAMERA = "FrontCamera"
NO_EPISODE_NS = 2**62
CAMERA_RECHECK_S = 0.2
FRAME_KEYS = ("rgb", "depth")


@dataclass(frozen=True)
class PasApi:
    client_cls: type
    world_cls: type
    drone_cls: type
    pose_cls: object
    yaw_mode_max_dof: int


def real_api() -> PasApi:
    from projectairsim import Drone, ProjectAirSimClient, World
    from projectairsim.drone import YawControlMode
    from projectairsim.types import Pose as PasPose

    return PasApi(ProjectAirSimClient, World, Drone, PasPose, YawControlMode.MaxDegreeOfFreedom)


class StepTimingError(RuntimeError):
    """world.step() ended at a simulator time other than the requested one."""


class StaleStateError(RuntimeError):
    """Ground-truth kinematics never reached the step time."""


class CommandTimeoutError(RuntimeError):
    """A velocity command's reply did not arrive after its step (the server measured its duration from a later tick)."""


class CameraPoseError(RuntimeError):
    """The camera pose stamped in the image disagrees with kinematics (Unreal actor left behind by a set_pose sweep)."""


class ProjectAirSimSimulator:
    def __init__(
        self,
        scene_config: str = "scene_autofly_s01.jsonc",
        config_dir: Path = CONFIGS_DIR,
        binary: Path = PACKAGED_BINARY,
        run_root: Path = SIM_RUN_DIR,
        api: PasApi | None = None,
        frame_timeout_s: float = 5.0,
        first_frame_timeout_s: float = 120.0,
        ready_timeout_s: float = 900.0,
        safe_altitude_m: float = 20.0,
        settle_steps: int = 2,
        collision_grace_s: float = 0.02,
        kinematics_retries: int = 5,
        command_timeout_s: float = 10.0,
        camera_offset_m: float = 0.40,
        camera_pose_tolerance_m: float = 0.10,
    ) -> None:
        self._scene_config = scene_config
        self._config_dir = Path(config_dir)
        self._binary = Path(binary)
        self._run_root = Path(run_root)
        self._api = api
        self._frame_timeout_s = frame_timeout_s
        self._first_frame_timeout_s = first_frame_timeout_s
        self._ready_timeout_s = ready_timeout_s
        self._safe_altitude_m = safe_altitude_m
        self._settle_steps = settle_steps
        self._collision_grace_s = collision_grace_s
        self._kinematics_retries = kinematics_retries
        self._command_timeout_s = command_timeout_s
        self._camera_offset_m = camera_offset_m
        self._camera_pose_tolerance_m = camera_pose_tolerance_m
        self._frames = FrameCollector(FRAME_KEYS)
        self._collisions = CollisionLog()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = self._world = self._drone = None
        self._proc: SimProcess | None = None
        self._t_ns = 0
        self._pending: tuple[float, float, float] | None = None
        self._steps = 0
        self._frame_steps = 0
        self._collided = False
        self._episode_start_ns = 0
        self._last_obs: Observation | None = None
        # Per-session: whether reset() has run at least once on the current connection. step()/observe()
        # refuse until it has, since frame 0 of a session is corrupt (spec §7.1) and only reset()'s
        # internal steps are allowed to consume it. Cleared in connect(), set at the end of reset().
        self._reset_done = False

    @property
    def steps_taken(self) -> int:
        return self._steps

    def launch(self, map_path: str, instance: int) -> None:
        ports = ports_for_instance(instance)
        used, total = gpu_memory_mib()
        check_gpu_for_launch(used, total, own_running=len(own_running_instances(self._run_root)))
        directory = instance_dir(instance, self._run_root)
        directory.mkdir(parents=True, exist_ok=True)
        cmd = packaged_command(map_path, ports, directory / "sim.log", instance, binary=self._binary)
        self._proc = launch_process(cmd, instance, ports, sim_environment(editor_mode=False), run_root=self._run_root)
        try:
            wait_ready(self._proc, self._ready_timeout_s)
            self.connect(ports)
        except Exception:
            self.close()
            raise

    def connect(self, ports: SimPorts) -> None:
        # A new connection is a new session: a relaunched simulator process's sim time and step timing
        # both restart from ~0, so any state keyed by them, or carried over from the previous session's
        # episode, must not survive. Otherwise a session-2 collision can collide (sim_time_ns,
        # object_name)-wise with a session-1 one and be dropped as a duplicate (silent data corruption),
        # a session-2 first frame can be timed out early with the short steady-state timeout instead of
        # the long first-frame one, and _current_pose() (used by reset()) can hand back a pose left over
        # from the previous session instead of querying the new one's kinematics. _steps is deliberately
        # left alone: steps_taken is documented (protocol.py) as counted since construction, not per
        # session, and callers that care about a single session's count already take a delta (see
        # tests, live_m1.check_one_step).
        self._reset_done = False
        self._collisions.clear()
        self._frames = FrameCollector(FRAME_KEYS)
        self._frame_steps = 0
        self._last_obs = None
        self._pending = None
        self._collided = False
        self._episode_start_ns = 0
        api = self._api if self._api is not None else real_api()
        self._api = api
        self._loop = asyncio.new_event_loop()
        self._client = api.client_cls(port_topics=ports.topics, port_services=ports.services)
        self._client.connect()
        self._world = api.world_cls(self._client, self._scene_config, delay_after_load_sec=0, sim_config_path=str(self._config_dir))
        clock = self._world.get_sim_clock_type()
        if clock != "steppable":
            raise RuntimeError(f"scene clock is {clock!r}; the backend needs 'steppable'")
        self._drone = api.drone_cls(self._client, self._world, ROBOT)
        cams = self._drone.sensors[CAMERA]
        self._client.subscribe(cams["scene_camera"], self._frames.callback("rgb"))
        self._client.subscribe(cams["depth_planar_camera"], self._frames.callback("depth"))
        self._client.subscribe(self._drone.robot_info["collision_info"], self._collisions.topic_callback)
        self._drone.enable_api_control()
        self._drone.arm()
        self._t_ns = int(self._world.get_sim_time())

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.disconnect()
        finally:
            # A raising disconnect() must not leave the simulator process running: it would keep ~1.7 GiB
            # of VRAM held and make the next launch() fail permanently (process.py: "already running" /
            # "port already in use"). stop()/loop teardown run regardless of what disconnect() did.
            self._client = self._world = self._drone = None
            if self._loop is not None:
                self._loop.close()
                self._loop = None
            if self._proc is not None:
                stop(self._proc.instance, run_root=self._run_root)
                self._proc = None

    def _require_connected(self) -> None:
        if self._drone is None:
            raise RuntimeError("not connected; call launch() or connect() first")

    def _pas_pose(self, pose: Pose):
        w, qx, qy, qz = yaw_to_quat(pose.yaw)
        return self._api.pose_cls({"frame_id": "DEFAULT_FRAME",
                                   "translation": {"x": float(pose.x), "y": float(pose.y), "z": float(pose.z)},
                                   "rotation": {"w": w, "x": qx, "y": qy, "z": qz}})

    def _current_pose(self) -> Pose:
        if self._last_obs is not None:
            return self._last_obs.pose
        kin = self._drone.get_ground_truth_kinematics()
        pos, ori = kin["pose"]["position"], kin["pose"]["orientation"]
        return Pose(float(pos["x"]), float(pos["y"]), float(pos["z"]), quat_to_yaw(ori["w"], ori["x"], ori["y"], ori["z"]))

    def _teleport(self, pose: Pose) -> None:
        if not self._drone.set_pose(self._pas_pose(pose), reset_kinematics=True):
            raise RuntimeError(f"set_pose({pose}) failed")

    def reset(self, pose: Pose) -> Observation:
        self._require_connected()
        self._episode_start_ns = NO_EPISODE_NS
        current = self._current_pose()
        safe_z = -abs(self._safe_altitude_m)
        for waypoint in (Pose(current.x, current.y, safe_z, current.yaw), Pose(pose.x, pose.y, safe_z, pose.yaw)):
            self._teleport(waypoint)
            self.command_velocity(0.0, 0.0, 0.0)
            self._step_impl()  # bypasses the reset_done gate: these steps are what consumes frame 0
        self._teleport(pose)
        for _ in range(self._settle_steps):
            self.command_velocity(0.0, 0.0, 0.0)
            self._step_impl()
        self._episode_start_ns = self._t_ns
        self._collided = False
        self._last_obs = dataclasses.replace(self._last_obs, collided=False, step_collisions=())
        self._reset_done = True
        return self._last_obs

    def spawn(self, name: str, asset: str, pose: Pose, scale: tuple[float, float, float], material: str | None = None) -> str:
        self._require_connected()
        actual = self._world.spawn_object(name, asset, self._pas_pose(pose), [float(s) for s in scale], False)
        if material is not None and not self._world.set_object_material(actual, material):
            raise RuntimeError(f"set_object_material({actual!r}, {material!r}) failed")
        return actual

    def destroy(self, name: str) -> None:
        self._require_connected()
        if not self._world.destroy_object(name):
            raise ObjectNotFoundError(f"destroy_object({name!r}) found no such object")

    def command_velocity(self, v_forward: float, yaw_rate: float, v_z: float) -> None:
        self._pending = (float(v_forward), float(yaw_rate), float(v_z))

    def _kinematics_at(self, target_ns: int) -> dict:
        kin = self._drone.get_ground_truth_kinematics()
        for _ in range(self._kinematics_retries - 1):
            if int(kin["time_stamp"]) == target_ns:
                return kin
            time.sleep(0.01)
            kin = self._drone.get_ground_truth_kinematics()
        if int(kin["time_stamp"]) != target_ns:
            raise StaleStateError(f"kinematics time_stamp {kin['time_stamp']} != step time {target_ns}")
        return kin

    def step(self, dt: float = CONTROL_DT_S) -> int:
        self._require_connected()
        if not self._reset_done:
            raise SessionNotResetError(
                "step() called before reset() on this connection: frame 0 of a session is corrupt "
                "(spec §7.1) and only reset()'s own steps may consume it"
            )
        return self._step_impl(dt)

    def _step_impl(self, dt: float = CONTROL_DT_S) -> int:
        dt_ns = dt_to_ns(dt, STEP_NS)
        if self._pending is None:
            raise RuntimeError("command_velocity must be called before every step")
        v_forward, yaw_rate, v_z = self._pending
        self._pending = None
        target = self._t_ns + dt_ns
        self._frames.arm(target)
        # Duration two clock ticks short of dt: the server may read the command's start one tick after Step started.
        task = self._loop.run_until_complete(self._drone.move_by_velocity_body_frame_async(
            v_forward, 0.0, -v_z, duration=(dt_ns - 2 * STEP_NS) / 1e9,
            yaw_control_mode=self._api.yaw_mode_max_dof, yaw_is_rate=True, yaw=yaw_rate))
        self._loop.run_until_complete(asyncio.sleep(0.002))
        result = self._world.step(dt_ns)
        self._steps += 1
        self._t_ns = int(result["sim_time_ns"])
        if self._t_ns != target:
            raise StepTimingError(f"step ended at {self._t_ns} ns, expected {target} ns")
        try:
            self._loop.run_until_complete(asyncio.wait_for(task, timeout=self._command_timeout_s))
        except asyncio.TimeoutError as err:
            raise CommandTimeoutError(
                f"move command reply missing {self._command_timeout_s} s after the step to {target} ns") from err
        timeout = self._first_frame_timeout_s if self._frame_steps == 0 else self._frame_timeout_s
        frames = self._frames.wait(timeout)
        self._frame_steps += 1
        kin = self._kinematics_at(target)
        if self._collision_grace_s > 0:
            time.sleep(self._collision_grace_s)
        events = self._collisions.collect(result["robots"][ROBOT].get("events", []), target)
        new_events = collisions_after(events, self._episode_start_ns)
        self._collided = self._collided or bool(new_events)
        pos, ori = kin["pose"]["position"], kin["pose"]["orientation"]
        lin, ang = kin["twist"]["linear"], kin["twist"]["angular"]
        mount = body_to_ned(float(ori["w"]), float(ori["x"]), float(ori["y"]), float(ori["z"]), (self._camera_offset_m, 0.0, 0.0))
        rgb_msg = frames["rgb"]
        camera_error = math.dist(
            (float(rgb_msg["pos_x"]), float(rgb_msg["pos_y"]), float(rgb_msg["pos_z"])),
            (float(pos["x"]) + mount[0], float(pos["y"]) + mount[1], float(pos["z"]) + mount[2]),
        )
        if camera_error > self._camera_pose_tolerance_m and not new_events:
            # The sweep hit that stopped the actor may still be on its way as a collision_info message: look once more.
            time.sleep(CAMERA_RECHECK_S)
            new_events = collisions_after(self._collisions.collect([], target), self._episode_start_ns)
            self._collided = self._collided or bool(new_events)
        if camera_error > self._camera_pose_tolerance_m and not new_events:
            raise CameraPoseError(
                f"camera at step {target} ns is {camera_error:.3f} m from kinematics + mount offset "
                f"(tolerance {self._camera_pose_tolerance_m} m); the Unreal actor was probably stopped by a sweep")
        self._last_obs = Observation(
            rgb=decode_rgb(rgb_msg),
            depth=decode_depth(frames["depth"]),
            pose=Pose(float(pos["x"]), float(pos["y"]), float(pos["z"]), quat_to_yaw(ori["w"], ori["x"], ori["y"], ori["z"])),
            velocity_ned=(float(lin["x"]), float(lin["y"]), float(lin["z"])),
            yaw_rate=float(ang["z"]),
            sim_time_ns=target,
            collided=self._collided,
            step_collisions=new_events,
            rgb_time_ns=int(rgb_msg["time_stamp"]),
            depth_time_ns=int(frames["depth"]["time_stamp"]),
            kinematics_time_ns=int(kin["time_stamp"]),
            camera_pose_error_m=camera_error,
        )
        return target

    def observe(self) -> Observation:
        if not self._reset_done:
            raise SessionNotResetError("observe() called before reset() on this connection")
        return self._last_obs
