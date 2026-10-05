"""One episode's moving pillars, as AutoFlyEnv drives them through a simulator (spec §6.5).

`MoverController` holds the movers' clocks and the last few frames, and turns the pure rules of
`autofly_ue5.scenes.motion` into poses for `Simulator.set_object_poses`. `park_poses` and `home_poses` are the two
places a displaced pillar can be sent between episodes: 50 m under its own home (out of the reset's up-across-down
path; the drone may have ended on a vacated home spot) and home.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from autofly_ue5.expert.obs import encode_movers
from autofly_ue5.scenes.model import Instance
from autofly_ue5.scenes.motion import (
    VIEW_HISTORY_FRAMES,
    MoverRoute,
    first_contact,
    in_view,
    surface_gaps,
    yield_clocks,
)
from autofly_ue5.sim.types import Pose

PARK_DEPTH_M = 50.0  # below the ground slab (NED z grows downward); pillars are at most 12 m tall
FORWARD_SPEED_MAX_M_S = 2.0  # the action space's forward limit (spec §8)


def home_poses(instances: dict[str, Instance], tags) -> dict[str, Pose]:
    return {tag: Pose(instances[tag].x, instances[tag].y, instances[tag].z_center, instances[tag].yaw)
            for tag in sorted(tags)}


def park_poses(instances: dict[str, Instance], tags) -> dict[str, Pose]:
    return {tag: Pose(instances[tag].x, instances[tag].y, instances[tag].z_center + PARK_DEPTH_M, instances[tag].yaw)
            for tag in sorted(tags)}


class MoverController:
    def __init__(self, routes: tuple[MoverRoute, ...], *, contact_m: float, yield_margin_m: float, max_speed_m_s: float,
                 dt: float) -> None:
        self.routes = routes
        self.contact_m = contact_m
        self.yield_distance_m = contact_m + yield_margin_m
        self.dt = dt
        # The farthest a mover's surface can be before a step for the contact rule to fire during it: the drone flies
        # at most FORWARD_SPEED_MAX_M_S and the mover at most max_speed_m_s for one dt. A backend fault this close is
        # scored as a mover collision (spec §6.5, "defensive inference").
        self.inference_radius_m = contact_m + (FORWARD_SPEED_MAX_M_S + max_speed_m_s) * dt
        self.taus = [0.0] * len(routes)
        self.positions = [route.position(0.0) for route in routes]
        self.previous_positions = list(self.positions)  # before the last advance(): the observation's velocity
        self.moving = [False] * len(routes)  # whether each mover's clock advanced on the last advance() (else it yielded)
        self._frames: deque = deque(maxlen=VIEW_HISTORY_FRAMES)

    def initial_poses(self) -> dict[str, Pose]:
        return {r.tag: Pose(x, y, r.z, r.yaw) for r, (x, y) in zip(self.routes, self.positions)}

    def advance(self, drone_xy: tuple[float, float]) -> dict[str, Pose]:
        """Advance every clock that may advance (yield rule) and return the poses of the movers that moved."""
        new = yield_clocks(self.routes, self.taus, drone_xy, self.dt, yield_distance_m=self.yield_distance_m)
        self.moving = [tau != old for tau, old in zip(new, self.taus)]
        self.previous_positions = list(self.positions)
        moved = {}
        for i, (route, old, tau) in enumerate(zip(self.routes, self.taus, new)):
            if tau != old:
                self.positions[i] = route.position(tau)
                moved[route.tag] = Pose(*self.positions[i], route.z, route.yaw)
        self.taus = new
        return moved

    def contact(self, before_xy: tuple[float, float], after_xy: tuple[float, float]) -> tuple[int, float] | None:
        return first_contact(self.routes, self.positions, before_xy, after_xy, self.contact_m)

    def gaps(self, xy: tuple[float, float]) -> list[float]:
        """Every mover's surface distance from a drone at `xy`."""
        return surface_gaps(self.routes, self.positions, xy)

    def nearest_gap(self, xy: tuple[float, float]) -> float:
        gaps = self.gaps(xy)
        return min(gaps) if gaps else float("inf")

    def observation(self, pose: Pose, slots: int) -> np.ndarray:
        """The expert's privileged mover input for a drone at `pose` (`obs.encode_movers`)."""
        return encode_movers(pose, self.positions, self.previous_positions,
                             [route.footprint.radius_m for route in self.routes], contact_m=self.contact_m, dt=self.dt,
                             slots=slots)

    def record_frame(self, pose: Pose) -> None:
        """What one observation showed: the drone's pose and where every mover stood."""
        self._frames.append(((pose.x, pose.y, pose.yaw), list(self.positions)))

    def seen_recently(self, index: int) -> bool:
        """Whether mover `index` was inside the camera's field of view in any of the last frames recorded."""
        return any(in_view(drone, positions[index]) for drone, positions in self._frames)

    def info(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.positions]
