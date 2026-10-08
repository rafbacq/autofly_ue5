"""Deterministic in-memory simulator with the Simulator interface, for offline tests."""

import hashlib
import math
import re
from typing import Iterable, Mapping

import numpy as np

from autofly_ue5.frames import wrap_pi
from autofly_ue5.sim.types import (
    CONTROL_DT_S,
    CollisionEvent,
    ObjectNotFoundError,
    ObjectPoseError,
    Observation,
    Pose,
    SessionNotResetError,
    dt_to_ns,
)


class FakeSimulator:
    """`obstacles`: fixed (x, y, radius) circles. `scene_objects`: named objects a built level carries, {name: (pose,
    radius)} -- movable with set_object_poses and collided with wherever they are now, like s01's tagged pillars.
    `assets`: the runtime spawn table's short names (the cooked static meshes): spawn() refuses any other and
    list_assets() lists them, as the server does. None: the table is not modelled (any short name spawns, none is listed)."""

    def __init__(self, obstacles: list[tuple[float, float, float]] | None = None, image_size: int = 256,
                 reset_steps: int = 4, scene_objects: Mapping[str, tuple[Pose, float]] | None = None, *,
                 render_depth: bool = False, obstacle_height_m: float = 10.0, camera_offset_m: float = 0.40,
                 assets: Iterable[str] | None = None) -> None:
        self._assets = None if assets is None else tuple(sorted(set(assets)))
        self._obstacles = list(obstacles or [])
        # Opt-in planar depth of the obstacle cylinders (standing on the ground, obstacle_height_m tall) from a 90 deg
        # pinhole camera camera_offset_m ahead of the body, for the live checks' offline tests. Off by default: the
        # golden s01 test pins observations rendered without it.
        self._render_depth = render_depth
        self._obstacle_height_m = obstacle_height_m
        self._camera_offset_m = camera_offset_m
        self._scene_objects = {name: (pose, float(radius)) for name, (pose, radius) in (scene_objects or {}).items()}
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
        # Enforces the contract Simulator.spawn() has always documented (protocol.py): `asset` is a short
        # AssetRegistry name with no slashes, `material` (when given) is a full UE package path. Measured
        # live (Task 7): a caller that violates this -- e.g. passing a bare registry key like "cylinder" or
        # "orange" instead of the resolved short spawn name / ue_path -- passes silently here (a bare
        # string is a perfectly good dict key) but is rejected by the real backend's set_object_material()
        # and spawn_object() on every single call, 100% reproducibly. A fake that accepts an invalid call
        # is worse than no fake at all for exactly this reason, so it now raises instead.
        self._require_launched()
        if "/" in asset:
            raise ValueError(
                f"asset={asset!r} looks like a package path, not a short AssetRegistry mesh name "
                f'(Simulator.spawn()\'s contract, protocol.py: \'asset is the short AssetRegistry name of a '
                f"cooked static mesh (e.g. \"1M_Cube\"), not an object path'); the real backend's runtime "
                f"spawn table is keyed by short name only"
            )
        if material is not None and not material.startswith("/"):
            raise ValueError(
                f"material={material!r} is not a UE package path (Simulator.spawn()'s contract, protocol.py: "
                f"'material is a package path of a base UMaterial (e.g. \"/Game/Geometry/Materials/M_Orange\")'); "
                f"the real backend's set_object_material() rejects anything else"
            )
        if material is not None and material.rsplit("/", 1)[-1].startswith("MI_"):
            # setMaterial loads with an exact base-UMaterial class filter (WorldSimApi.cpp:1019-1033), so a material
            # instance fails live however right its path. The fake knows a material only by its path, and UE names
            # instances MI_ (every registry color_instance, every imported glTF material): a target pool's palette-
            # coloured primitive (plan 5 C5) would otherwise pass here and fail on the first live spawn.
            raise ValueError(
                f"material={material!r} is a material instance (MI_), not a base UMaterial; the real backend's "
                f"set_object_material() cannot load it (expert/episode.py _runtime_material_path)"
            )
        if self._assets is not None and asset not in self._assets:
            # WorldSimApi.cpp spawnObject: the asset maps hold only cooked assets; a name not in them raises
            raise RuntimeError(f"There were no objects with name {asset} found in the Registry")
        unique, k = name, 1
        while unique in self._objects:
            unique = f"{name}{k}"
            k += 1
        self._objects[unique] = (asset, pose, tuple(scale), material)
        return unique

    def list_assets(self, pattern: str = ".*") -> list[str]:
        """The spawn table's short names that fully match `pattern` (the server's std::regex_match)."""
        self._require_launched()
        return [a for a in (self._assets or ()) if re.fullmatch(pattern, a)]

    def destroy(self, name: str) -> None:
        if name not in self._objects:
            raise ObjectNotFoundError(name)
        del self._objects[name]

    def set_object_poses(self, poses: Mapping[str, Pose]) -> None:
        # Name by name, like the real backend (one SetObjectPose request each): an unknown name raises after the
        # names before it have moved, so a caller that assumes all-or-nothing fails here too.
        self._require_launched()
        for name, pose in poses.items():
            if name not in self._scene_objects:
                raise ObjectPoseError(f"SetObjectPose failed. No objects of name {name} were found in the world.")
            self._scene_objects[name] = (pose, self._scene_objects[name][1])

    def scene_object_poses(self) -> dict[str, Pose]:
        """Where every scene object is now (tests only; the real backend's equivalent is get_object_poses)."""
        return {name: pose for name, (pose, _radius) in self._scene_objects.items()}

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
        ) + tuple(
            CollisionEvent(self._t_ns, name, (new.x, new.y, new.z), (0.0, 0.0, 0.0))
            for name, (pose, radius) in self._scene_objects.items()
            if math.hypot(new.x - pose.x, new.y - pose.y) <= radius and pose.z < 0.0  # z >= 0: below the ground
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
            depth=self._render_depth_image() if self._render_depth else np.full((self._size, self._size), np.inf, dtype=np.float32),
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

    def _render_depth_image(self) -> np.ndarray:
        """Planar depth of every obstacle cylinder: for the pixel (row, col) the ray (1, a, b) in the camera's frame
        (a right, b down, both over the focal length n/2 of a 90 deg field), so the ray parameter is the planar depth."""
        p = self._pose
        n = self._size
        f = n / 2.0
        a = (np.arange(n) + 0.5 - n / 2.0) / f
        b = (np.arange(n) + 0.5 - n / 2.0) / f
        cos_y, sin_y = math.cos(p.yaw), math.sin(p.yaw)
        dx = cos_y - a[None, :] * sin_y  # forward + a * right, right = (-sin, cos)
        dy = sin_y + a[None, :] * cos_y
        dz = np.broadcast_to(b[:, None], (n, n))
        dx = np.broadcast_to(dx, (n, n))
        dy = np.broadcast_to(dy, (n, n))
        ox, oy, oz = p.x + self._camera_offset_m * cos_y, p.y + self._camera_offset_m * sin_y, p.z
        depth = np.full((n, n), np.inf)
        for cx, cy, r in self._obstacles:
            fx, fy = ox - cx, oy - cy
            qa = dx * dx + dy * dy
            qb = fx * dx + fy * dy
            qc = fx * fx + fy * fy - r * r
            disc = qb * qb - qa * qc
            ok = disc >= 0
            t = np.where(ok, (-qb - np.sqrt(np.where(ok, disc, 0.0))) / np.where(qa > 0, qa, 1.0), np.inf)
            z_hit = oz + t * dz
            hit = ok & (t > 0) & (z_hit <= 0.0) & (z_hit >= -self._obstacle_height_m)
            depth = np.where(hit & (t < depth), t, depth)
        return depth.astype(np.float32)

    def object_bounds(self, name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        if name.startswith("obstacle_") and name[len("obstacle_"):].isdigit() and int(name[len("obstacle_"):]) < len(self._obstacles):
            x, y, r = self._obstacles[int(name[len("obstacle_"):])]
            h = self._obstacle_height_m
            return ((x, y, -h / 2.0), (2 * r, 2 * r, h))
        if name in self._scene_objects:
            pose, r = self._scene_objects[name]
            return ((pose.x, pose.y, pose.z), (2 * r, 2 * r, 2 * r))
        if name in self._objects:
            _, pose, scale, _ = self._objects[name]
            return ((pose.x, pose.y, pose.z), tuple(scale))
        return None

    def _render_rgb(self) -> np.ndarray:
        p = self._pose
        digest = hashlib.sha256(f"{p.x:.3f},{p.y:.3f},{p.z:.3f},{p.yaw:.4f}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return rng.integers(0, 256, size=(self._size, self._size, 3), dtype=np.uint8)
