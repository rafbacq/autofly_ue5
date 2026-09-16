"""Value types of the simulator interface (spec §7). World frame: NED metres, yaw radians."""

from dataclasses import dataclass

import numpy as np

STEP_NS = 5_000_000
CONTROL_DT_S = 0.2


class ObjectNotFoundError(KeyError):
    """destroy() was given a name that no spawned object has."""


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
