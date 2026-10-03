"""Expert observation encoding (spec §8): depth image + privileged target vector, and on a dynamic scene the nearby
moving pillars. RGB is never an input."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

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


# --------------------------------------------------------------------------------------------------------
# Privileged moving-pillar input (2026-10-03, after the s01d gate failed at 0.775). A mover that yields to the drone
# stands still and looks exactly like a static pillar in depth (and in RGB: both are white), yet its contact rule fires
# 1.0 m from its surface while a static pillar needs physical contact (~0.48 m). The expert is the data source, not the
# student, so it may know which pillars move: the dataset records RGB and state[9] whatever the expert sees.
# --------------------------------------------------------------------------------------------------------
MOVER_FEATURES = 6               # present, forward, right, clearance, forward velocity, right velocity
MOVER_RANGE_M = 10.0             # surface distance at which a mover is reported: ~3 s at the fastest closing speed
MOVER_CLEARANCE_NORM_M = 5.0     # clearance to the contact boundary, normalised
MOVER_SPEED_NORM_M_S = 2.0       # the drone's own speed limit; s01d's movers reach 1.2 m/s
DYNAMIC_MOVER_SLOTS = 4          # s01d puts 8-12 movers among 80 pillars: rarely more than 3 within 10 m


def encode_movers(pose: Pose, positions, previous_positions, radii, *, contact_m: float, dt: float,
                  slots: int) -> np.ndarray:
    """(slots * MOVER_FEATURES,) float32 in [-1, 1]: per mover whose surface is within MOVER_RANGE_M, nearest
    clearance first -- present (1), body-frame forward and right offset of its centre, clearance to the contact
    boundary (surface distance - contact_m; negative inside it), and body-frame velocity over the last step (zero while
    it yields). Slots without a mover are all zero."""
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    rows = []
    for (x, y), (px, py), radius in zip(positions, previous_positions, radii):
        dx, dy = x - pose.x, y - pose.y
        surface = math.hypot(dx, dy) - radius
        if surface > MOVER_RANGE_M:
            continue
        vx, vy = (x - px) / dt, (y - py) / dt
        rows.append((surface - contact_m, [
            1.0,
            (dx * cy + dy * sy) / MOVER_RANGE_M,
            (-dx * sy + dy * cy) / MOVER_RANGE_M,
            (surface - contact_m) / MOVER_CLEARANCE_NORM_M,
            (vx * cy + vy * sy) / MOVER_SPEED_NORM_M_S,
            (-vx * sy + vy * cy) / MOVER_SPEED_NORM_M_S,
        ]))
    rows.sort(key=lambda row: row[0])
    out = np.zeros((slots, MOVER_FEATURES), dtype=np.float32)
    for i, (_clearance, features) in enumerate(rows[:slots]):
        out[i] = features
    return np.clip(out, -1.0, 1.0).reshape(-1)


def encode(obs: Observation, target_xy_z) -> dict[str, np.ndarray]:
    return {"depth": encode_depth(obs.depth),
            "vector": encode_vector(obs.pose, obs.velocity_ned, obs.yaw_rate, target_xy_z)}


# --------------------------------------------------------------------------------------------------------
# Depth stacking (spec §8, amended 2026-10-02): a depth-only policy cannot see motion in one frame.
# --------------------------------------------------------------------------------------------------------
DYNAMIC_DEPTH_FRAMES = 3  # 0.4 s of history at 5 Hz


@dataclass(frozen=True)
class ObsConfig:
    """How many depth frames the expert sees (newest last) and how they are stored. One float32 frame is what every
    M2 checkpoint was trained on; a dynamic scene stacks three as float16, which keeps a 150k replay buffer at 12.7 GB
    (three float32 frames would need 25 GB; docs/decisions/2026-10-02-dynamic-obstacles.md)."""
    depth_frames: int = 1
    depth_dtype: str = "float32"
    mover_slots: int = 0  # 0: no movers key, the observation every checkpoint before 2026-10-03 was trained on

    def __post_init__(self) -> None:
        if self.depth_frames < 1 or self.depth_dtype not in ("float32", "float16") or self.mover_slots < 0:
            raise ValueError(f"unsupported observation config {self}")

    def space(self) -> spaces.Dict:
        keys = {
            "depth": spaces.Box(0.0, 1.0, (self.depth_frames, DEPTH_SIZE, DEPTH_SIZE), dtype=np.dtype(self.depth_dtype)),
            "vector": spaces.Box(-np.inf, np.inf, (VECTOR_DIM,), dtype=np.float32),
        }
        if self.mover_slots:
            keys["movers"] = spaces.Box(-1.0, 1.0, (self.mover_slots * MOVER_FEATURES,), dtype=np.float32)
        return spaces.Dict(keys)

    def to_json(self) -> dict:
        # mover_slots only when set, so every earlier record and run identity serialises exactly as before
        return {"depth_frames": self.depth_frames, "depth_dtype": self.depth_dtype,
                **({"mover_slots": self.mover_slots} if self.mover_slots else {})}


def obs_config_for_frames(depth_frames: int) -> ObsConfig:
    """One frame stays float32 (M2's space exactly); a stack is float16."""
    return ObsConfig(depth_frames, "float32" if depth_frames == 1 else "float16")


def obs_config_for_scene(scene) -> ObsConfig:
    if scene.dynamic is None:
        return ObsConfig()
    return ObsConfig(DYNAMIC_DEPTH_FRAMES, "float16", mover_slots=DYNAMIC_MOVER_SLOTS)


def obs_config_from_space(space: spaces.Dict) -> ObsConfig:
    """The config a checkpoint was trained with, read back from its saved observation space."""
    depth = space["depth"]
    movers = space.spaces.get("movers")
    config = ObsConfig(int(depth.shape[0]), np.dtype(depth.dtype).name,
                       mover_slots=int(movers.shape[0]) // MOVER_FEATURES if movers is not None else 0)
    if config.space() != space:
        raise ValueError(f"{space} is not an AutoFly expert observation space (expected {config.space()})")
    return config


class DepthStacker:
    """The last `depth_frames` encoded depth frames, oldest first, cast to the configured dtype. reset() fills the
    whole stack with the episode's first frame."""

    def __init__(self, config: ObsConfig) -> None:
        self._config = config
        self._stack: np.ndarray | None = None

    def reset(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)[0].astype(self._config.depth_dtype)
        self._stack = np.repeat(frame[None, ...], self._config.depth_frames, axis=0)
        return self._stack.copy()

    def push(self, frame: np.ndarray) -> np.ndarray:
        if self._stack is None:
            raise RuntimeError("DepthStacker.push() before reset()")
        frame = np.asarray(frame)[0].astype(self._config.depth_dtype)
        self._stack = np.concatenate([self._stack[1:], frame[None, ...]], axis=0)
        return self._stack.copy()
