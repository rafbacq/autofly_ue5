"""Deterministic in-memory simulator with the Simulator interface, for offline tests."""

import hashlib
import math

import numpy as np

from autofly_ue5.frames import wrap_pi
from autofly_ue5.sim.types import (
    CONTROL_DT_S,
    CollisionEvent,
    ObjectNotFoundError,
    Observation,
    Pose,
    SessionNotResetError,
    dt_to_ns,
)


class FakeSimulator:
    def __init__(self, obstacles: list[tuple[float, float, float]] | None = None, image_size: int = 256,
                 reset_steps: int = 4) -> None:
        self._obstacles = list(obstacles or [])
        self._size = image_size
        self._reset_steps = reset_steps
        self._launched: tuple[str, int] | None = None
        self._pose = Pose(0.0, 0.0, -2.0, 0.0)
        self._velocity = (0.0, 0.0, 0.0)
        self._yaw_rate = 0.0
        self._t_ns = 0
        self._pending: tuple[float, float, float] | None = None
        self._steps = 0
        self._collided = False
        self._step_collisions: tuple[CollisionEvent, ...] = ()
        self._objects: dict[str, tuple[str, Pose, tuple[float, float, float], str | None]] = {}
        # Mirrors ProjectAirSimSimulator's reset-before-step precondition (spec §7.1): frame 0 of a real
        # session is corrupt, so the backend requires reset() before step()/observe() will run. The fake
        # has no such corrupt frame, but skipping reset() here would let an (M2/M3) caller bug that only
        # breaks against the real backend pass silently against the fake. Per-session: cleared in launch(),
        # also doubles as "an observation exists" for observe().
        self._reset_done = False

    @property
    def steps_taken(self) -> int:
        return self._steps

    def launch(self, map_path: str, instance: int) -> None:
        if instance < 0:
            raise ValueError("instance must be >= 0")
        self._launched = (map_path, instance)
        self._reset_done = False

    def close(self) -> None:
        self._launched = None

    def _require_launched(self) -> None:
        if self._launched is None:
            raise RuntimeError("simulator is not launched")

    def reset(self, pose: Pose) -> Observation:
        self._require_launched()
        self._t_ns += self._reset_steps * dt_to_ns(CONTROL_DT_S)
        self._steps += self._reset_steps
        self._pose = pose
        self._velocity = (0.0, 0.0, 0.0)
        self._yaw_rate = 0.0
        self._pending = None
        self._collided = False
        self._step_collisions = ()
        self._reset_done = True
        return self.observe()

    def spawn(self, name: str, asset: str, pose: Pose, scale: tuple[float, float, float], material: str | None = None) -> str:
        self._require_launched()
        unique, k = name, 1
        while unique in self._objects:
            unique = f"{name}{k}"
            k += 1
        self._objects[unique] = (asset, pose, tuple(scale), material)
        return unique

    def destroy(self, name: str) -> None:
        if name not in self._objects:
            raise ObjectNotFoundError(name)
        del self._objects[name]

    def command_velocity(self, v_forward: float, yaw_rate: float, v_z: float) -> None:
        self._pending = (float(v_forward), float(yaw_rate), float(v_z))

    def step(self, dt: float = CONTROL_DT_S) -> int:
        self._require_launched()
        if not self._reset_done:
            raise SessionNotResetError("step() called before reset() on this session")
        dt_ns = dt_to_ns(dt)
        if self._pending is None:
            raise RuntimeError("command_velocity must be called before every step")
        v_forward, yaw_rate, v_z = self._pending
        self._pending = None
        dt_s = dt_ns / 1e9
        p = self._pose
        vx, vy, vz_ned = v_forward * math.cos(p.yaw), v_forward * math.sin(p.yaw), -v_z
        new = Pose(p.x + vx * dt_s, p.y + vy * dt_s, p.z + vz_ned * dt_s, wrap_pi(p.yaw + yaw_rate * dt_s))
        self._t_ns += dt_ns
        self._steps += 1
        hits = tuple(
            CollisionEvent(self._t_ns, f"obstacle_{i}", (new.x, new.y, new.z), (0.0, 0.0, 0.0))
            for i, (ox, oy, radius) in enumerate(self._obstacles)
            if math.hypot(new.x - ox, new.y - oy) <= radius
        )
        self._pose, self._velocity, self._yaw_rate = new, (vx, vy, vz_ned), yaw_rate
        self._step_collisions = hits
        self._collided = self._collided or bool(hits)
        return self._t_ns

    def observe(self) -> Observation:
        if not self._reset_done:
            raise SessionNotResetError("no observation before reset() or step()")
        return Observation(
            rgb=self._render_rgb(),
            depth=np.full((self._size, self._size), np.inf, dtype=np.float32),
            pose=self._pose,
            velocity_ned=self._velocity,
            yaw_rate=self._yaw_rate,
            sim_time_ns=self._t_ns,
            collided=self._collided,
            step_collisions=self._step_collisions,
            rgb_time_ns=self._t_ns,
            depth_time_ns=self._t_ns,
            kinematics_time_ns=self._t_ns,
            camera_pose_error_m=0.0,
        )

    def _render_rgb(self) -> np.ndarray:
        p = self._pose
        digest = hashlib.sha256(f"{p.x:.3f},{p.y:.3f},{p.z:.3f},{p.yaw:.4f}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return rng.integers(0, 256, size=(self._size, self._size, 3), dtype=np.uint8)
