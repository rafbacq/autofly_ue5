import asyncio
import math
from types import SimpleNamespace

import pytest

from autofly_ue5.frames import quat_to_yaw, yaw_to_quat
from autofly_ue5.sim import airsim_backend
from autofly_ue5.sim.airsim_backend import (
    CameraPoseError,
    CommandTimeoutError,
    PasApi,
    ProjectAirSimSimulator,
    SessionNotResetError,
    StaleStateError,
    StepTimingError,
)
from autofly_ue5.sim.process import SimPorts
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.sync import FrameCollector, FrameTimeoutError
from autofly_ue5.sim.types import ObjectNotFoundError, Pose
from autofly_ue5.sim.types import SessionNotResetError as TypesSessionNotResetError


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
        # A canopy: a box (NED metres) the Unreal actor's swept moves cannot pass through. A move whose path crosses
        # it leaves the actor where it was (kinematics move on, the camera stays with the actor), except a retrace --
        # a horizontal move against the heading, back the way the drone flew. None: every move lands.
        self.canopy = None

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
        ax, ay, az = self.drone.actor_position()
        cam = {"pos_x": ax + offset * math.cos(self.drone.yaw), "pos_y": ay + offset * math.sin(self.drone.yaw), "pos_z": az}
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

    def get_3d_bounding_box(self, object_name, box_alignment):
        # the server answers with the box of a spawned object, or nothing for an unknown name
        if object_name not in self.spawned or box_alignment != 0:
            return {}
        _, pose, scale, _ = self.spawned[object_name]
        t = pose["translation"]
        return {"center": {"x": t["x"], "y": t["y"], "z": t["z"]}, "size": {"x": scale[0], "y": scale[1], "z": scale[2]}}


class FakeDrone:
    def __init__(self, client, world, name):
        self.client, self.world, self.name = client, world, name
        world.drone = self
        self.sensors = {"FrontCamera": {"scene_camera": "/rgb", "depth_planar_camera": "/depth"}}
        self.robot_info = {"collision_info": "/collision"}
        self.x, self.y, self.z, self.yaw = 0.0, 0.0, -2.0, 0.0
        self.actor = None  # where the Unreal actor really is when a swept move left it behind; None: with the kinematics
        self.retrace_blocked = False  # a grazing crash: moves straight back are blocked at their start as well
        self.exit_normal = None  # the direction a penetrating actor may leave by (the last collision's normal)
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
        if self.actor is not None and (v_forward or v_down):
            self.actor = None  # flying moves the actor with the kinematics again (the fake does not model a stuck flight)
        self.x += v_forward * math.cos(self.yaw) * dt
        self.y += v_forward * math.sin(self.yaw) * dt
        self.z += v_down * dt
        self.yaw += yaw_rate * dt

    def actor_position(self):
        return self.actor if self.actor is not None else (self.x, self.y, self.z)

    def set_pose(self, pose, reset_kinematics=True):
        """The swept move the plugin makes (UnrealRobot.cpp MoveRobotToUnrealPose), as the live simulator showed it on
        s02 (2026-10-07). Starting inside the canopy: the actor does not move, the kinematics do (a camera-pose error).
        Hitting the canopy on the way: the actor stops at the hit, a collision event is raised, and the kinematics snap
        back to the actor (no camera error, the pose simply did not take)."""
        self.poses.append(pose)
        t, r = pose["translation"], pose["rotation"]
        start = self.actor_position()
        end = (t["x"], t["y"], t["z"])
        self.yaw = quat_to_yaw(r["w"], r["x"], r["y"], r["z"])
        blocked_at = self._sweep_blocked(start, end)
        if blocked_at is None:
            self.x, self.y, self.z = end
            self.actor = None
        elif blocked_at == 0.0:
            self.x, self.y, self.z = end
            self.actor = start
        else:
            hit = tuple(s + blocked_at * (e - s) for s, e in zip(start, end))
            self.x, self.y, self.z = hit
            self.actor = None
            self.world.pending_events.append({"type": "collision", "sim_time_ns": self.world.t + 1, "object_name": "canopy",
                                              "impact_point": {"x": hit[0], "y": hit[1], "z": hit[2]}, "normal": {"x": 0.0, "y": 0.0, "z": 1.0}})
        return True

    def _sweep_blocked(self, start, end):
        """None when the move lands; 0.0 when it starts inside the canopy; else the fraction of the way at which it hits."""
        box = self.world.canopy
        if box is None:
            return None
        dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
        horizontal = math.hypot(dx, dy)
        if abs(dz) < 1e-9 and horizontal > 0:
            # a retrace: back along the heading, through the space the drone came through
            if not self.retrace_blocked and math.cos(self.yaw) * dx + math.sin(self.yaw) * dy < -0.98 * horizontal:
                return None
        if self.exit_normal is not None:
            n = self.exit_normal
            if (n[0] * dx + n[1] * dy + n[2] * dz) > 0.98 * math.sqrt(dx * dx + dy * dy + dz * dz):
                # moving out along the normal: UE ignores the initial overlap it is leaving, and once out of that card
                # the body is free to sweep back the way it came
                self.retrace_blocked = False
                return None

        def inside(x, y, z):
            return box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"] and box["z_min"] <= z <= box["z_max"]

        if inside(*start):
            return 0.0
        for i in range(1, 41):
            f = i / 40.0
            if inside(start[0] + f * dx, start[1] + f * dy, start[2] + f * dz):
                return (i - 1) / 40.0
        return None

    def get_ground_truth_kinematics(self):
        stale = self.world.stale_kinematics > 0
        if stale:
            self.world.stale_kinematics -= 1
        w, qx, qy, qz = yaw_to_quat(self.yaw)
        return {"time_stamp": self.world.t - (5_000_000 if stale else 0),
                "pose": {"position": {"x": self.x, "y": self.y, "z": self.z}, "orientation": {"w": w, "x": qx, "y": qy, "z": qz}},
                "twist": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}}


API = PasApi(client_cls=FakeClient, world_cls=FakeWorld, drone_cls=FakeDrone, pose_cls=dict, yaw_mode_max_dof=0)


def make_sim(*, mark_reset: bool = True, **kwargs):
    """`mark_reset` pokes past the reset-before-step precondition for tests that exercise some other
    step() behaviour and don't want reset()'s own multi-step recovery sequence polluting their step
    counts/timing. Tests of the precondition itself pass `mark_reset=False`."""
    options = dict(api=API, frame_timeout_s=0.2, first_frame_timeout_s=0.2, collision_grace_s=0.0, command_timeout_s=0.2)
    options.update(kwargs)
    sim = ProjectAirSimSimulator(**options)
    sim.connect(SimPorts(8989, 8990))
    if mark_reset:
        sim._reset_done = True
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


# a canopy over NED x in [4, 10], y in [-3, 3], from 1 m to 6 m up: the drone flying east-north... along +x at 2 m altitude
# stops under it at x = 5, and a straight climb from there sweeps through leaves
CANOPY = {"x_min": 4.0, "x_max": 10.0, "y_min": -3.0, "y_max": 3.0, "z_min": -6.0, "z_max": -1.0}


def _fly_under_the_canopy(sim):
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim._world.canopy = CANOPY
    for _ in range(13):  # 13 x 0.4 m = 5.2 m: inside the box in x, still at 2 m
        sim.command_velocity(2.0, 0.0, 0.0)
        sim.step()
    assert sim._drone.x == pytest.approx(5.2) and sim._drone.actor is None


def test_reset_from_under_a_canopy_retraces_before_climbing():
    """s02, 2026-10-07: after a crash into a tree the climb to the safe altitude was blocked by the leaf cards above
    (the Unreal actor stayed, the kinematics rose: CameraPoseError 18 m, then 50 m). The AutoFly action space only
    moves body-forward, so a move straight back along the heading retraces free space; the reset tries that, farther
    each time, before giving up."""
    sim = make_sim()
    _fly_under_the_canopy(sim)
    obs = sim.reset(Pose(-20.0, 10.0, -2.0, 1.0))
    assert obs.pose == Pose(-20.0, 10.0, -2.0, 1.0) and obs.camera_pose_error_m == pytest.approx(0.0)
    assert sim._drone.actor is None
    # the first attempt was the plain climb (blocked), then a retrace of 1.5 m from where the actor really was
    poses = [(round(p["translation"]["x"], 3), round(p["translation"]["y"], 3), round(p["translation"]["z"], 3)) for p in sim._drone.poses[-5:]]
    assert poses[0] == (5.2, 0.0, -20.0), "the climb that the canopy blocked"
    assert poses[1] == (3.7, 0.0, -2.0), "1.5 m straight back along the heading, at the actor's altitude"
    assert poses[2] == (3.7, 0.0, -20.0) and poses[3] == (-20.0, 10.0, -20.0) and poses[4] == (-20.0, 10.0, -2.0)
    assert sim.last_reset_retrace_m == 1.5 and sim.last_reset_escape == ("retrace", 1.5)


def test_reset_from_below_a_canopy_notices_the_snapped_back_kinematics():
    """The other way a swept climb fails: it travels, hits the canopy above, the plugin raises a collision and the
    kinematics snap back to the actor. The camera then agrees with the kinematics, so only checking that each hop
    arrived tells the reset to retrace (s02's second smoke, 2026-10-07: ResetPoseError 38 m with no camera error)."""
    sim = make_sim()
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim._world.canopy = {**CANOPY, "z_min": -6.0, "z_max": -3.0}  # the canopy hangs from 3 to 6 m: the drone flies under it
    for _ in range(13):
        sim.command_velocity(2.0, 0.0, 0.0)
        sim.step()
    obs = sim.reset(Pose(-20.0, 10.0, -2.0, 1.0))
    assert obs.pose == Pose(-20.0, 10.0, -2.0, 1.0) and obs.collided is False
    poses = [(round(p["translation"]["x"], 3), round(p["translation"]["y"], 3), round(p["translation"]["z"], 3)) for p in sim._drone.poses[-5:]]
    assert poses[0] == (5.2, 0.0, -20.0), "the climb that hit the canopy at 3 m"
    assert poses[1][0] == pytest.approx(3.7) and poses[1][2] == pytest.approx(-3.0, abs=0.15), "back 1.5 m from where the actor stopped (just under the canopy)"
    assert poses[2][:2] == pytest.approx((3.7, 0.0)) and poses[2][2] == -20.0
    assert sim.last_reset_retrace_m == 1.5
    assert sim.last_reset_escape == ("retrace", 1.5), "the snap-back's collision event had an upward normal: a nudge up is blocked too"


def test_reset_after_a_grazing_crash_nudges_out_along_the_collision_normal():
    """A card hit edge-on has a normal across the heading: a move straight back is not 'out of' it and UE blocks it at
    its start (s02's gate, 2026-10-07: camera 8.000 m behind the kinematics after an 8 m retrace). The nudge along the
    collision's normal is the direction UE lets a penetrating actor leave by."""
    sim = make_sim()
    _fly_under_the_canopy(sim)
    # the crash: the last step reported a collision whose normal points +y (across the +x heading)
    sim._world.pending_events = [{"type": "collision", "sim_time_ns": sim._world.t + 1, "object_name": "obs_0002",
                                  "impact_point": {"x": 5.6, "y": 0.0, "z": -2.0}, "normal": {"x": 0.0, "y": 1.0, "z": 0.0}}]
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    assert sim.observe().collided
    sim._drone.retrace_blocked = True  # this canopy also stops moves straight back (the fake's grazing case)
    sim._drone.exit_normal = (0.0, 1.0, 0.0)
    before = len(sim._drone.poses)
    obs = sim.reset(Pose(-20.0, 10.0, -2.0, 1.0))
    assert obs.pose == Pose(-20.0, 10.0, -2.0, 1.0)
    poses = [(round(p["translation"]["x"], 3), round(p["translation"]["y"], 3), round(p["translation"]["z"], 3)) for p in sim._drone.poses[before:]]
    assert poses[0] == (5.6, 0.0, -20.0), "the plain climb, blocked from inside the canopy"
    assert poses[1] == (5.6, 0.5, -2.0), "0.5 m along the normal, +y, at the actor's altitude: this one lands"
    assert poses[2] == (5.6, 0.5, -20.0), "but the climb from there is still under leaves"
    # the body is out of the card it was overlapping: the retraces now sweep free space, and the second one clears the canopy
    assert sim.last_reset_escape == ("retrace", 4.0)
    assert poses[-3][:2] == pytest.approx((0.1, 2.0)) and poses[-2] == (-20.0, 10.0, -20.0) and poses[-1] == (-20.0, 10.0, -2.0)


def test_reset_gives_up_when_no_retrace_frees_the_actor():
    sim = make_sim()
    _fly_under_the_canopy(sim)
    sim._world.canopy = {**CANOPY, "x_min": -100.0, "x_max": 100.0, "y_min": -100.0, "y_max": 100.0}  # a roof over everything
    with pytest.raises(CameraPoseError):
        sim.reset(Pose(-20.0, 10.0, -2.0, 1.0))


def test_a_reset_in_the_clear_makes_the_same_calls_as_before():
    """The escape ladder changes nothing when the first climb lands: three set_pose calls, up, across, down."""
    sim = make_sim()
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    before = len(sim._drone.poses)
    sim.reset(Pose(3.0, 4.0, -2.0, 0.5))
    poses = [(p["translation"]["x"], p["translation"]["y"], p["translation"]["z"]) for p in sim._drone.poses[before:]]
    assert poses == [(0.0, 0.0, -20.0), (3.0, 4.0, -20.0), (3.0, 4.0, -2.0)] and sim.last_reset_escape == ("climb", 0.0)


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


def test_camera_desync_during_reset_always_raises():
    sim = make_sim()
    sim._world.camera_lag_m = 0.5
    # A fresh, real collision lands at every one of reset()'s four internal step targets (two
    # waypoint steps, two settle steps -- see the step accounting in
    # test_collision_sets_flag_and_reset_clears_it). Outside reset() (see
    # test_camera_error_on_a_collision_step_is_recorded_not_raised), a collision on the same step as
    # a camera desync swallows the error into Observation.camera_pose_error_m instead of raising. If
    # that swallow applied inside reset() too, all four steps would swallow their desync in turn and
    # reset() would hand back a normal-looking Observation despite the camera having been left behind
    # for the whole recovery -- exactly the silently-unrepeatable-teleport failure mode the waypoint
    # sequence exists to avoid. Publishing via the collision_info topic (not world.pending_events,
    # which a single world.step() call drains all at once) is what lets one collision line up with
    # each separate step instead of only the first.
    collision_topic = sim._drone.robot_info["collision_info"]
    for i, target_ns in enumerate((200_000_000, 400_000_000, 600_000_000, 800_000_000)):
        sim._client.publish(collision_topic, {"time_stamp": target_ns, "object_name": f"reset_obstacle_{i}",
                                              "impact_point": {"x": 0.0, "y": 0.0, "z": -20.0},
                                              "normal": {"x": -1.0, "y": 0.0, "z": 0.0}})
    with pytest.raises(CameraPoseError):
        sim.reset(Pose(-31.0, -25.0, -2.0, math.pi / 2))


def test_spawn_and_destroy():
    sim = make_sim()
    name = sim.spawn("AF_Target", "SM_Target", Pose(1.0, 2.0, -0.5, 0.0), (1.0, 1.0, 1.0), "/Game/AutoFly/Materials/M_Red")
    asset, pose, scale, physics = sim._world.spawned[name]
    assert asset == "SM_Target" and pose["translation"] == {"x": 1.0, "y": 2.0, "z": -0.5}
    assert scale == [1.0, 1.0, 1.0] and physics is False
    sim.destroy(name)
    with pytest.raises(ObjectNotFoundError):
        sim.destroy(name)


def test_object_bounds_reads_the_world_axis_box_and_misses_unknown_names():
    sim = make_sim()
    name = sim.spawn("AF_Box", "1M_Cube", Pose(1.0, 2.0, -0.5, 0.3), (2.0, 1.0, 1.0), None)
    assert sim.object_bounds(name) == ((1.0, 2.0, -0.5), (2.0, 1.0, 1.0))
    assert sim.object_bounds("nothing_here") is None


def test_invalid_dt_is_rejected():
    sim = make_sim()
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        sim.step(0.203)


def test_step_before_any_reset_raises():
    # Frame 0 of a session is corrupt (spec §7.1); only reset()'s own steps may consume it.
    sim = make_sim(mark_reset=False)
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(SessionNotResetError):
        sim.step()


def test_observe_before_any_reset_raises():
    sim = make_sim(mark_reset=False)
    with pytest.raises(SessionNotResetError):
        sim.observe()


def test_reset_allows_step_and_observe_afterward():
    sim = make_sim(mark_reset=False)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.command_velocity(0.0, 0.0, 0.0)
    sim.step()  # must not raise now that reset() has run once on this connection
    assert sim.observe() is not None


def test_close_stops_process_even_when_disconnect_raises(monkeypatch):
    # A raising disconnect() must not skip stop(), or the simulator process leaks (~1.7 GiB of VRAM) and
    # the next launch() fails permanently ("already running" / "port already in use").
    sim = make_sim()
    sim._proc = SimpleNamespace(instance=3, pid=12345)
    stopped = []
    monkeypatch.setattr(airsim_backend, "stop",
                        lambda instance, run_root=None, expected_pid=None: stopped.append((instance, expected_pid)))
    sim._client.disconnect = lambda: (_ for _ in ()).throw(RuntimeError("disconnect boom"))

    with pytest.raises(RuntimeError, match="disconnect boom"):
        sim.close()

    assert stopped == [(3, 12345)], "close() stops only the process it launched"
    assert sim._proc is None and sim._client is None and sim._world is None and sim._drone is None


def test_a_close_abandoned_past_a_relaunch_does_not_stop_the_successor(tmp_path):
    # C1 (2026-09-24 review), reproduced with real processes: the resilient wrapper gives close() a bounded wait,
    # then clears the slot and relaunches it. If the abandoned close() returns later, its stop() must not kill the
    # simulator now recorded in the same slot.
    import os
    import threading
    import time

    from autofly_ue5.sim.process import is_alive, launch_process, ports_for_instance, stop

    def sleeper():
        return launch_process(["sleep", "300"], 7, ports_for_instance(47), dict(os.environ), run_root=tmp_path)

    class HangingClient:
        def disconnect(self):
            time.sleep(1.5)

    sim = ProjectAirSimSimulator(run_root=tmp_path)
    sim._client, sim._proc = HangingClient(), sleeper()
    closer = threading.Thread(target=sim.close, daemon=True)
    closer.start()
    closer.join(0.3)
    assert closer.is_alive(), "the test needs close() still hung when the slot is relaunched"
    stop(7, grace_s=2.0, run_root=tmp_path)  # the relaunch path clears the slot ...
    successor = sleeper()  # ... and launches its replacement
    try:
        closer.join(10.0)
        assert not closer.is_alive()
        assert is_alive(successor.pid), "the late close() stopped the simulator that replaced it"
    finally:
        stop(7, grace_s=2.0, run_root=tmp_path)


def test_connect_clears_per_session_state():
    sim = make_sim()
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.command_velocity(1.0, 0.0, 0.0)
    sim.step()
    assert sim.observe() is not None

    sim.connect(SimPorts(8989, 8990))
    assert sim._reset_done is False and sim._last_obs is None and sim._pending is None
    with pytest.raises(SessionNotResetError):
        sim.observe()


def test_reconnect_does_not_dedupe_a_collision_seen_in_a_previous_session():
    def crash_once(sim):
        # After reset() (2 waypoint steps + 2 settle steps, from a fresh world at t=0), _t_ns ==
        # 800_000_000; one more step lands on exactly 1_000_000_000 -- identically in both sessions,
        # since a reconnect gives a fresh FakeWorld starting at t=0 again (as a relaunched simulator
        # process's sim time also restarts near 0).
        sim._world.pending_events = [{
            "type": "collision", "sim_time_ns": 1_000_000_000, "object_name": "StaticMeshActor_1",
            "impact_point": {"x": 0.0, "y": 0.0, "z": -2.0}, "normal": {"x": -1.0, "y": 0.0, "z": 0.0},
        }]
        sim.command_velocity(1.0, 0.0, 0.0)
        sim.step()
        return sim.observe()

    sim = make_sim()
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    first = crash_once(sim)
    assert first.collided is True

    # Simulate a simulator restart on the same object: reconnect, then reset() as any new episode would.
    sim.connect(SimPorts(8989, 8990))
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    second = crash_once(sim)
    assert second.collided is True
    assert second.step_collisions and second.step_collisions[0].object_name == "StaticMeshActor_1"


def test_reconnect_restores_the_long_first_frame_timeout(monkeypatch):
    # A relaunched simulator process is cold: its first frame takes far longer than a steady-state one,
    # which is why _frame_steps picks first_frame_timeout_s over frame_timeout_s. _frame_steps is
    # per-session, so a reconnect that left it set would time the new process's first frame out early.
    seen = []
    real_wait = FrameCollector.wait
    monkeypatch.setattr(FrameCollector, "wait", lambda self, timeout: (seen.append(timeout), real_wait(self, timeout))[1])
    sim = make_sim(frame_timeout_s=0.2, first_frame_timeout_s=9.0)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.command_velocity(0.0, 0.0, 0.0)
    sim.step()
    assert seen[0] == 9.0 and seen[-1] == 0.2  # first frame long, steady state short

    seen.clear()
    sim.connect(SimPorts(8989, 8990))
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    assert seen[0] == 9.0, "a reconnected session's first frame must get the cold-start budget again"


def test_both_simulators_raise_the_same_reset_precondition_error():
    # M2/M3 develop against FakeSimulator and run against the real backend, so `except
    # SessionNotResetError` has to work for both -- the error belongs to the interface, not to one
    # implementation (same precedent as ObjectNotFoundError).
    assert SessionNotResetError is TypesSessionNotResetError
    fake = FakeSimulator()
    fake.launch("/Game/AutoFly/Maps/S01", 0)
    fake.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(TypesSessionNotResetError):
        fake.step()
    with pytest.raises(TypesSessionNotResetError):
        fake.observe()

    real = make_sim(mark_reset=False)
    real.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(TypesSessionNotResetError):
        real.step()


# ------------------------------------------------------------------------------------------------------
# C9 (2026-09-24 review): an episode must start where it was asked to, and the drone must not teleport.
# The recorded M2 gate had 15 one-step "collisions" whose drone was >= 4 m from its start -- all right after a
# collision episode (15 of 73 such resets, 0 of 723 others). They slipped through because the camera check is
# skipped on collision steps, so these checks deliberately are not.
# ------------------------------------------------------------------------------------------------------
def _collision_event(t_ns):
    return {"type": "collision", "sim_time_ns": t_ns, "object_name": "obs_0003",
            "impact_point": {"x": 0.4, "y": 0.0, "z": -2.0}, "normal": {"x": -1.0, "y": 0.0, "z": 0.0}}


def test_reset_raises_when_the_drone_does_not_arrive_at_the_requested_pose():
    from autofly_ue5.sim.types import ResetPoseError

    sim = make_sim(mark_reset=False)
    sim._drone.set_pose = lambda pose, reset_kinematics=True: True  # accepted, never applied
    with pytest.raises(ResetPoseError):
        sim.reset(Pose(-31.0, -25.0, -2.0, 0.0))


def test_reset_to_a_yaw_of_pi_is_not_a_pose_error():
    # +pi and -pi are the same heading; a raw difference would call this a 6.28 rad error.
    sim = make_sim(mark_reset=False)
    obs = sim.reset(Pose(-31.0, -25.0, -2.0, math.pi))
    assert abs(abs(obs.pose.yaw) - math.pi) < 1e-6


def test_a_position_jump_between_steps_raises_even_on_a_collision_step():
    from autofly_ue5.sim.types import KinematicsJumpError

    sim = make_sim(mark_reset=False)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    drone, integrate = sim._drone, sim._drone.integrate
    drone.integrate = lambda dt: (integrate(dt), setattr(drone, "x", drone.x + 5.0))
    sim._world.pending_events = [_collision_event(sim._t_ns + 200_000_000)]
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(KinematicsJumpError):
        sim.step()


def test_flying_at_the_commanded_speed_limit_is_not_a_jump():
    sim = make_sim(mark_reset=False)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    for _ in range(10):
        sim.command_velocity(2.0, 0.5, 0.0)
        sim.step()
    assert sim.observe().pose.x > 3.0


def test_a_refused_set_pose_is_a_typed_error():
    from autofly_ue5.sim.types import SetPoseError

    sim = make_sim(mark_reset=False)
    sim._drone.set_pose = lambda pose, reset_kinematics=True: False
    with pytest.raises(SetPoseError):
        sim.reset(Pose(-31.0, -25.0, -2.0, 0.0))


# --------------------------------------------------------------------------------------------------------
# set_object_poses (spec §6.5, §7.1): exact allow-list, one teleporting request per name, typed errors.
# --------------------------------------------------------------------------------------------------------
class MovableWorld(FakeWorld):
    """FakeWorld plus the baked, tagged pillars of a built level and the server's SetObjectPose replies:
    WorldSimApi.cpp:745-791 throws for an unknown or immovable object, which the client raises as
    RuntimeError("ERROR code: ..."); a missing reply makes client.request() disconnect and raise
    RuntimeError("Fatal Timeout ...")."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objects = {"obs_0000": {"translation": {"x": 1.0, "y": 2.0, "z": -5.0}},
                        "obs_0010": {"translation": {"x": 5.0, "y": 2.0, "z": -5.0}}}
        self.set_pose_calls = []
        self.fail_with = None
        self.status = True

    def set_object_pose(self, object_name, object_pose, teleport):
        self.set_pose_calls.append((object_name, object_pose, teleport))
        if self.fail_with is not None:
            raise self.fail_with
        if object_name not in self.objects and object_name not in self.spawned:
            raise RuntimeError(f"ERROR code: -32603, message: SetObjectPose failed. No objects of name {object_name} "
                               f"were found in the world.")
        if self.status:
            self.objects[object_name] = object_pose
        return self.status

    def get_object_poses(self, object_names):
        nan = float("nan")
        return [self.objects.get(n, {"translation": {"x": nan, "y": nan, "z": nan}}) for n in object_names]


MOVABLE_API = PasApi(client_cls=FakeClient, world_cls=MovableWorld, drone_cls=FakeDrone, pose_cls=dict,
                     yaw_mode_max_dof=0)


def test_set_object_poses_teleports_each_allowed_name_in_order():
    sim = make_sim(api=MOVABLE_API, movable_objects={"obs_0000", "obs_0010"})
    sim.set_object_poses({"obs_0010": Pose(6.0, 3.0, -5.0, 0.0), "obs_0000": Pose(1.5, 2.5, -5.0, math.pi / 2)})
    calls = sim._world.set_pose_calls
    assert [c[0] for c in calls] == ["obs_0010", "obs_0000"] and all(c[2] is True for c in calls)
    t = calls[1][1]["translation"]
    assert (t["x"], t["y"], t["z"]) == (1.5, 2.5, -5.0)
    r = calls[1][1]["rotation"]
    assert quat_to_yaw(r["w"], r["x"], r["y"], r["z"]) == pytest.approx(math.pi / 2)


def test_set_object_poses_refuses_a_name_off_the_allow_list_before_any_request():
    from autofly_ue5.sim.types import ObjectPoseError

    # The server's lookup also matches any actor whose name merely CONTAINS the string, or a spawned object
    # ("target"): only an exact allow-list keeps a typo from moving the wrong actor.
    sim = make_sim(api=MOVABLE_API, movable_objects={"obs_0000"})
    for name in ("obs_0010", "target", "obs_000"):
        with pytest.raises(ObjectPoseError, match="allow-list"):
            sim.set_object_poses({"obs_0000": Pose(1.0, 1.0, -5.0, 0.0), name: Pose(0.0, 0.0, -5.0, 0.0)})
    assert sim._world.set_pose_calls == []
    with pytest.raises(ObjectPoseError, match="allow-list"):
        make_sim(api=MOVABLE_API).set_object_poses({"obs_0000": Pose(1.0, 1.0, -5.0, 0.0)})  # default: nothing


def test_set_object_poses_maps_server_errors_and_a_false_status_to_object_pose_error():
    from autofly_ue5.sim.types import ObjectPoseError

    sim = make_sim(api=MOVABLE_API, movable_objects={"obs_0000", "obs_0042"})
    with pytest.raises(ObjectPoseError, match="No objects of name obs_0042"):
        sim.set_object_poses({"obs_0042": Pose(0.0, 0.0, -5.0, 0.0)})
    sim._world.status = False
    with pytest.raises(ObjectPoseError, match="status"):
        sim.set_object_poses({"obs_0000": Pose(0.0, 0.0, -5.0, 0.0)})


def test_a_fatal_timeout_is_typed_and_every_later_call_asks_for_a_relaunch():
    from autofly_ue5.sim.types import SimConnectionLostError, SimRequestTimeoutError

    sim = make_sim(api=MOVABLE_API, movable_objects={"obs_0000"})
    sim._world.fail_with = RuntimeError("Fatal Timeout ocurred while processing request for method: "
                                        "/Sim/AutoFlyScene/SetObjectPose")
    with pytest.raises(SimRequestTimeoutError, match="Fatal Timeout"):
        sim.set_object_poses({"obs_0000": Pose(0.0, 0.0, -5.0, 0.0)})
    # projectairsim's client disconnected itself: nothing on this connection can work any more.
    for call in (lambda: sim.reset(Pose(0.0, 0.0, -2.0, 0.0)), lambda: sim.set_object_poses({}),
                 lambda: sim.destroy("target"), lambda: sim.step()):
        with pytest.raises(SimConnectionLostError):
            call()
    sim.connect(SimPorts(8989, 8990))  # a new connection (a relaunch) clears it
    sim.set_object_poses({"obs_0000": Pose(0.0, 0.0, -5.0, 0.0)})


def test_an_unrecognised_runtime_error_is_not_swallowed():
    sim = make_sim(api=MOVABLE_API, movable_objects={"obs_0000"})
    sim._world.fail_with = RuntimeError("something else entirely")
    with pytest.raises(RuntimeError, match="something else entirely"):
        sim.set_object_poses({"obs_0000": Pose(0.0, 0.0, -5.0, 0.0)})


def test_get_object_poses_reads_back_baked_objects_and_reports_a_missing_one_as_none():
    sim = make_sim(api=MOVABLE_API, movable_objects={"obs_0000"})
    got = sim.get_object_poses(["obs_0000", "obs_0099"])
    assert got["obs_0000"] == Pose(1.0, 2.0, -5.0, 0.0)
    assert got["obs_0099"] is None  # the server answers NaN rather than raising


# --------------------------------------------------------------------------------------------------------
# A move Unreal refuses although the pillar is movable (2026-10-02, s01d_r1 session 0: "Unable to move obs_0047" once
# in ~2 million moves, 5.5 h in, which ended the run). Read back; accept if it is already there; retry once; else a
# recoverable fault -- never the end of a run.
# --------------------------------------------------------------------------------------------------------
class RefusingWorld(MovableWorld):
    """Refuses the next `refusals` moves of a found object the way WorldSimApi.cpp:780-787 does; `move_anyway` puts the
    object where it was asked before refusing (the no-op case: Unreal saw no change)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refusals = 0
        self.move_anyway = False

    def set_object_pose(self, object_name, object_pose, teleport):
        if self.refusals > 0 and object_name in self.objects:
            self.refusals -= 1
            self.set_pose_calls.append((object_name, object_pose, teleport))
            if self.move_anyway:
                self.objects[object_name] = object_pose
            raise RuntimeError(f"ERROR code: 1.0, message: SetObjectPose failed. Unable to move object {object_name}, "
                               f"check if object state is movable!")
        return super().set_object_pose(object_name, object_pose, teleport)


REFUSING_API = PasApi(client_cls=FakeClient, world_cls=RefusingWorld, drone_cls=FakeDrone, pose_cls=dict,
                      yaw_mode_max_dof=0)


def test_a_refused_move_of_an_object_already_where_it_was_asked_is_accepted(capsys):
    sim = make_sim(api=REFUSING_API, movable_objects={"obs_0000"})
    sim._world.refusals, sim._world.move_anyway = 1, True
    sim.set_object_poses({"obs_0000": Pose(1.5, 2.0, -5.0, 0.0)})
    assert len(sim._world.set_pose_calls) == 1, "no retry needed: it is already there"
    assert "MOVE-REFUSED" in capsys.readouterr().err


def test_a_refused_move_is_retried_once_and_then_succeeds():
    sim = make_sim(api=REFUSING_API, movable_objects={"obs_0000"})
    sim._world.refusals = 1
    sim.set_object_poses({"obs_0000": Pose(3.0, 2.0, -5.0, 0.0)})
    assert len(sim._world.set_pose_calls) == 2
    assert sim.get_object_poses(["obs_0000"])["obs_0000"] == Pose(3.0, 2.0, -5.0, 0.0)


def test_a_move_refused_twice_is_a_recoverable_fault_not_the_end_of_the_run():
    from autofly_ue5.expert.faults import FAULT_ERRORS_STEP
    from autofly_ue5.sim.types import ObjectMoveRefusedError, ObjectPoseError

    sim = make_sim(api=REFUSING_API, movable_objects={"obs_0000"})
    sim._world.refusals = 2
    with pytest.raises(ObjectMoveRefusedError, match="obs_0000"):
        sim.set_object_poses({"obs_0000": Pose(3.0, 2.0, -5.0, 0.0)})
    assert issubclass(ObjectMoveRefusedError, FAULT_ERRORS_STEP)
    assert not issubclass(ObjectMoveRefusedError, ObjectPoseError), "ObjectPoseError stays fatal: a wrong level"
    # An object the server cannot find at all is still a wrong level or a bug.
    with pytest.raises(ObjectPoseError, match="No objects of name"):
        make_sim(api=REFUSING_API, movable_objects={"obs_0042"}).set_object_poses({"obs_0042": Pose(0.0, 0.0, -5.0, 0.0)})
