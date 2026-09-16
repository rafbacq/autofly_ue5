"""Expert observation encoding (spec §8): depth image + privileged target vector. RGB is never an input."""

from __future__ import annotations

import math

import numpy as np

from autofly_ue5.frames import wrap_pi
from autofly_ue5.sim.types import Observation, Pose

DEPTH_SIZE = 84
DEPTH_CLIP_M = 30.0
VECTOR_DIM = 8
NORM_DIST_M = 100.0   # ~ the 70x70 m scene diagonal, so distance lands in [0, 1]
NORM_DZ_M = 5.0       # the altitude band is 1-3 m; 5 m keeps dz comfortably inside [-1, 1]
NORM_V_MS = 2.0       # max commanded forward speed


def _block_min(a: np.ndarray, size: int) -> np.ndarray:
    """Downsample by block MINIMUM to `size` x `size`.

    Minimum, not mean: averaging across a depth discontinuity invents intermediate depths and can erase a
    thin near obstacle. A minimum can only ever report an obstacle as closer than it is, which is the safe
    direction for collision avoidance. The 256 -> 84 ratio is not an integer, so block boundaries come from
    linspace and np.minimum.reduceat, which handles unequal block sizes exactly.
    """
    for axis in (0, 1):
        idx = np.linspace(0, a.shape[axis], size + 1).astype(int)[:-1]
        a = np.minimum.reduceat(a, idx, axis=axis)
    return a


def encode_depth(depth: np.ndarray, size: int = DEPTH_SIZE, clip_m: float = DEPTH_CLIP_M) -> np.ndarray:
    """(H, W) metres, +inf for no-hit (spec §7.1) -> (1, size, size) float32 in [0, 1], 1.0 = clip or sky."""
    d = np.asarray(depth, dtype=np.float32)
    d = np.nan_to_num(d, nan=clip_m, posinf=clip_m, neginf=0.0)
    np.clip(d, 0.0, clip_m, out=d)
    return (_block_min(d, size) / clip_m).astype(np.float32)[None, ...]


def target_geometry(pose: Pose, target_xy_z: tuple[float, float, float]) -> tuple[float, float, float]:
    """(horizontal distance m, bearing rad in the body frame, height difference m).

    The single definition of where the target is relative to the drone: the observation and the reward both
    call this, so they cannot drift apart. `dz` is the NED difference (negative = target above the drone).
    """
    dx = target_xy_z[0] - pose.x
    dy = target_xy_z[1] - pose.y
    return math.hypot(dx, dy), wrap_pi(math.atan2(dy, dx) - pose.yaw), target_xy_z[2] - pose.z


def encode_vector(pose: Pose, velocity_ned, yaw_rate: float, target_xy_z) -> np.ndarray:
    dist, bearing, dz = target_geometry(pose, target_xy_z)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    vx, vy, vz_ned = (float(v) for v in velocity_ned)
    return np.array([
        dist / NORM_DIST_M,
        math.sin(bearing),                 # sin/cos, not the raw angle: no discontinuity at +/-pi
        math.cos(bearing),
        dz / NORM_DZ_M,
        (vx * cy + vy * sy) / NORM_V_MS,   # body forward
        (-vx * sy + vy * cy) / NORM_V_MS,  # body right
        -vz_ned / NORM_V_MS,               # positive up, matching the action convention
        float(yaw_rate),
    ], dtype=np.float32)


def encode(obs: Observation, target_xy_z) -> dict[str, np.ndarray]:
    return {"depth": encode_depth(obs.depth),
            "vector": encode_vector(obs.pose, obs.velocity_ned, obs.yaw_rate, target_xy_z)}
