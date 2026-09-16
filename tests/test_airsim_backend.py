import asyncio
import math

import pytest

from autofly_ue5.frames import quat_to_yaw, yaw_to_quat
from autofly_ue5.sim.airsim_backend import (
    CameraPoseError,
    CommandTimeoutError,
    PasApi,
    ProjectAirSimSimulator,
    StaleStateError,
    StepTimingError,
)
from autofly_ue5.sim.process import SimPorts
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.sync import FrameTimeoutError
from autofly_ue5.sim.types import ObjectNotFoundError, Pose


class FakeClient:
    def __init__(self, port_topics, port_services):
        self.ports = (port_topics, port_services)
        self.subs = {}
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def subscribe(self, topic, callback):
        self.subs.setdefault(topic, []).append(callback)

    def publish(self, topic, msg):
        for callback in self.subs.get(topic, []):
            callback(topic, msg)


class FakeWorld:
    def __init__(self, client, scene_config_name, delay_after_load_sec=0, sim_config_path=""):
        self.client, self.scene, self.config_path = client, scene_config_name, sim_config_path
        self.t = 0
        self.step_calls = 0
        self.drone = None
        self.spawned = {}
        self.pending_events = []
        self.stale_kinematics = 0
        self.drop_depth = False
        self.time_skew_ns = 0
        self.camera_lag_m = 0.0  # simulated stuck Unreal actor: the camera stays this far behind kinematics

    def get_sim_clock_type(self):
        return "steppable"

    def get_sim_time(self):
        return self.t

    def step(self, dt_ns):
        self.step_calls += 1
        self.t += dt_ns + self.time_skew_ns
        self.drone.integrate(dt_ns / 1e9)
        cams = self.drone.sensors["FrontCamera"]
        offset = 0.4 - self.camera_lag_m
        cam = {"pos_x": self.drone.x + offset * math.cos(self.drone.yaw), "pos_y": self.drone.y + offset * math.sin(self.drone.yaw),
               "pos_z": self.drone.z}
        self.client.publish(cams["scene_camera"], {"time_stamp": self.t, "encoding": "BGR", "height": 4, "width": 4, "data": bytes(48), **cam})
        if not self.drop_depth:
            self.client.publish(cams["depth_planar_camera"], {"time_stamp": self.t, "encoding": "16FC1", "height": 4, "width": 4, "data": bytes(32), **cam})
        events, self.pending_events = self.pending_events, []
        return {"sim_time_ns": self.t, "robots": {"Drone1": {"state": {}, "events": events}}}

    def spawn_object(self, object_name, asset_path, object_pose, object_scale, enable_physics):
        self.spawned[object_name] = (asset_path, object_pose, object_scale, enable_physics)
        return object_name

    def set_object_material(self, object_name, material_asset_path):
        return object_name in self.spawned

    def destroy_object(self, object_name):
        return self.spawned.pop(object_name, None) is not None


class FakeDrone:
    def __init__(self, client, world, name):
        self.client, self.world, self.name = client, world, name
        world.drone = self
        self.sensors = {"FrontCamera": {"scene_camera": "/rgb", "depth_planar_camera": "/depth"}}
        self.robot_info = {"collision_info": "/collision"}
        self.x, self.y, self.z, self.yaw = 0.0, 0.0, -2.0, 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.commands, self.poses = [], []
        self.api_enabled = self.armed = False
        self.hang_reply = False

    def enable_api_control(self):
        self.api_enabled = True
        return True

    def arm(self):
        self.armed = True
        return True

    async def move_by_velocity_body_frame_async(self, v_forward, v_right, v_down, duration, yaw_control_mode, yaw_is_rate, yaw):
        self.commands.append({"v_forward": v_forward, "v_right": v_right, "v_down": v_down, "duration": duration,
                              "yaw_is_rate": yaw_is_rate, "yaw": yaw, "step_calls_at_send": self.world.step_calls})
        self.cmd = (v_forward, v_down, yaw)
        hang = self.hang_reply

        async def _reply():
            if hang:  # the server never answers, e.g. the command's start was read one tick late
                await asyncio.get_running_loop().create_future()
            return True

        return asyncio.ensure_future(_reply())

    def integrate(self, dt):
        v_forward, v_down, yaw_rate = self.cmd
        self.x += v_forward * math.cos(self.yaw) * dt
        self.y += v_forward * math.sin(self.yaw) * dt
        self.z += v_down * dt
        self.yaw += yaw_rate * dt

    def set_pose(self, pose, reset_kinematics=True):
        self.poses.append(pose)
        t, r = pose["translation"], pose["rotation"]
        self.x, self.y, self.z = t["x"], t["y"], t["z"]
        self.yaw = quat_to_yaw(r["w"], r["x"], r["y"], r["z"])
        return True

    def get_ground_truth_kinematics(self):
        stale = self.world.stale_kinematics > 0
        if stale:
            self.world.stale_kinematics -= 1
        w, qx, qy, qz = yaw_to_quat(self.yaw)
        return {"time_stamp": self.world.t - (5_000_000 if stale else 0),
                "pose": {"position": {"x": self.x, "y": self.y, "z": self.z}, "orientation": {"w": w, "x": qx, "y": qy, "z": qz}},
                "twist": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}}


API = PasApi(client_cls=FakeClient, world_cls=FakeWorld, drone_cls=FakeDrone, pose_cls=dict, yaw_mode_max_dof=0)


def make_sim(**kwargs):
    options = dict(api=API, frame_timeout_s=0.2, first_frame_timeout_s=0.2, collision_grace_s=0.0, command_timeout_s=0.2)
    options.update(kwargs)
    sim = ProjectAirSimSimulator(**options)
    sim.connect(SimPorts(8989, 8990))
    return sim


def test_connect_subscribes_arms_and_loads_config():
    sim = make_sim()
    assert isinstance(sim, Simulator)
    drone = sim._drone
    assert drone.api_enabled and drone.armed
    assert set(sim._client.subs) == {"/rgb", "/depth", "/collision"}
    assert sim._world.scene == "scene_autofly_s01.jsonc" and sim._world.config_path.endswith("/configs")


def test_step_sends_command_before_one_world_step():
    sim = make_sim()
    with pytest.raises(RuntimeError, match="command_velocity"):
        sim.step()
    sim.command_velocity(2.0, 0.5, 1.0)
    assert sim.step(0.2) == 200_000_000
    cmd = sim._drone.commands[-1]
    assert cmd["step_calls_at_send"] == 0 and sim._world.step_calls == 1 and sim.steps_taken == 1
    assert cmd["duration"] == pytest.approx(0.19) and cmd["v_down"] == -1.0 and cmd["v_right"] == 0.0
    assert cmd["yaw_is_rate"] is True and cmd["yaw"] == 0.5
    obs = sim.observe()
    assert obs.sim_time_ns == 200_000_000 and obs.rgb.shape == (4, 4, 3) and obs.depth.shape == (4, 4)
    assert obs.rgb_time_ns == obs.depth_time_ns == obs.kinematics_time_ns == 200_000_000
    assert obs.pose.x == pytest.approx(0.4) and obs.collided is False
    assert obs.camera_pose_error_m == pytest.approx(0.0, abs=1e-9)


def test_missing_command_reply_is_an_error():
    sim = make_sim()
    sim._drone.hang_reply = True
    sim.command_velocity(1.0, 0.0, 0.0)
    with pytest.raises(CommandTimeoutError):
        sim.step()


def test_camera_left_behind_is_an_error_outside_collisions():
    sim = make_sim()
    sim._world.camera_lag_m = 0.5
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(CameraPoseError):
        sim.step()


def test_camera_error_on_a_collision_step_is_recorded_not_raised():
    sim = make_sim()
    sim._world.camera_lag_m = 0.5
    sim._world.pending_events = [{"type": "collision", "sim_time_ns": 200_000_000, "object_name": "obs_0003",
                                  "impact_point": {"x": 0.4, "y": 0.0, "z": -2.0}, "normal": {"x": -1.0, "y": 0.0, "z": 0.0}}]
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    obs = sim.observe()
    assert obs.collided is True and obs.camera_pose_error_m == pytest.approx(0.5)


def test_missing_depth_frame_times_out():
    sim = make_sim()
    sim._world.drop_depth = True
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(FrameTimeoutError):
        sim.step()


def test_step_time_mismatch_is_an_error():
    sim = make_sim()
    sim._world.time_skew_ns = 5_000_000
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(StepTimingError):
        sim.step()


def test_stale_kinematics_are_retried_then_rejected():
    sim = make_sim(kinematics_retries=3)
    sim._world.stale_kinematics = 1
    sim.command_velocity(0.0, 0.0, 0.0)
    sim.step()
    sim._world.stale_kinematics = 10
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(StaleStateError):
        sim.step()


def test_collision_sets_flag_and_reset_clears_it():
    sim = make_sim(settle_steps=2)
    sim._world.pending_events = [{"type": "collision", "sim_time_ns": 200_000_000, "object_name": "StaticMeshActor_7",
                                  "impact_point": {"x": 0.4, "y": 0.0, "z": -2.0}, "normal": {"x": -1.0, "y": 0.0, "z": 0.0}}]
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    obs = sim.observe()
    assert obs.collided is True and obs.step_collisions[0].object_name == "StaticMeshActor_7"
    steps_before = sim.steps_taken
    obs = sim.reset(Pose(-31.0, -25.0, -2.0, math.pi / 2))
    assert obs.collided is False and obs.step_collisions == ()
    assert sim.steps_taken - steps_before == 2 + 2
    zs = [p["translation"]["z"] for p in sim._drone.poses]
    assert zs == [-20.0, -20.0, -2.0]
    assert obs.pose.x == pytest.approx(-31.0) and obs.pose.yaw == pytest.approx(math.pi / 2)


def test_spawn_and_destroy():
    sim = make_sim()
    name = sim.spawn("AF_Target", "SM_Target", Pose(1.0, 2.0, -0.5, 0.0), (1.0, 1.0, 1.0), "/Game/AutoFly/Materials/M_Red")
    asset, pose, scale, physics = sim._world.spawned[name]
    assert asset == "SM_Target" and pose["translation"] == {"x": 1.0, "y": 2.0, "z": -0.5}
    assert scale == [1.0, 1.0, 1.0] and physics is False
    sim.destroy(name)
    with pytest.raises(ObjectNotFoundError):
        sim.destroy(name)


def test_invalid_dt_is_rejected():
    sim = make_sim()
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        sim.step(0.203)
