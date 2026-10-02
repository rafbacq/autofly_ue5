"""Value types of the simulator interface (spec §7). World frame: NED metres, yaw radians."""

from dataclasses import dataclass

import numpy as np

STEP_NS = 5_000_000
CONTROL_DT_S = 0.2


class ObjectNotFoundError(KeyError):
    """destroy() was given a name that no spawned object has."""


class SessionNotResetError(RuntimeError):
    """step()/observe() called before reset() has run on this session; frame 0 of a session is corrupt (spec §7.1).

    Lives here, not beside the backend's own errors, because it belongs to the Simulator interface rather than to one
    implementation: both ProjectAirSimSimulator and FakeSimulator raise it, so M2/M3 code can catch it portably while
    developing against the fake.
    """


class SetPoseError(RuntimeError):
    """The simulator refused a set_pose() (teleport) request."""


class ResetPoseError(RuntimeError):
    """reset() settled somewhere other than the requested start pose (C9, 2026-09-24 review: the recorded M2 gate had
    15 episodes whose drone was >= 4 m from its start, all right after a collision episode)."""


class KinematicsJumpError(RuntimeError):
    """Between two consecutive steps the drone moved farther than it physically can -- a teleport, not flight."""


class StartCollisionError(RuntimeError):
    """An episode's first observation (right after reset and spawning) already reports a collision."""


class CameraPoseError(RuntimeError):
    """The camera pose stamped in the image disagrees with kinematics (Unreal actor left behind by a set_pose sweep).

    Raised by the Project AirSim backend; defined here, beside the interface, so code that must tell it apart (the
    env's mover-collision inference, spec §6.5) can catch it without importing a backend."""


class ObjectPoseError(RuntimeError):
    """set_object_poses() named an object that is not on the allow-list, that the server could not find or could not
    move, or the server refused the request (WorldSimApi.cpp:745-791). Not recoverable: a wrong level or a bug."""


class SimRequestTimeoutError(RuntimeError):
    """A request got no reply within projectairsim's 300 s receive timeout. The client disconnects itself before
    raising (client.py:255-282), so this connection is finished: see SimConnectionLostError."""


class SimConnectionLostError(RuntimeError):
    """A call on a connection that an earlier SimRequestTimeoutError ended. Only a relaunch recovers, so the resilient
    wrapper treats it like a failed launch and relaunches at once rather than retrying reset() on a dead client."""


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class CollisionEvent:
    sim_time_ns: int
    object_name: str
    impact_point: tuple[float, float, float]
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth: np.ndarray
    pose: Pose
    velocity_ned: tuple[float, float, float]
    yaw_rate: float
    sim_time_ns: int
    collided: bool
    step_collisions: tuple[CollisionEvent, ...]
    rgb_time_ns: int
    depth_time_ns: int
    kinematics_time_ns: int
    camera_pose_error_m: float


def dt_to_ns(dt: float, step_ns: int = STEP_NS) -> int:
    dt_ns = round(dt * 1e9)
    if dt_ns <= 0 or dt_ns % step_ns != 0:
        raise ValueError(f"dt={dt} s is not a positive multiple of the {step_ns} ns clock step")
    return dt_ns
