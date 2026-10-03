"""AutoFly's state[9] as this project writes it (spec §10.1; docs/decisions/2026-10-03-m3-a0-and-collection.md).

The release documents none of its nine fields. Spec §3.2 established state[0], [3] and [6..8] from the two real
episodes on this host. scripts/decode_state.py tested hypotheses for the rest, adopting one only at |r| >= 0.95 on both
episodes. Only state[2] passed: it tracks altitude (|r| 0.986 and 0.996; 0.985 z - 1.470 and 0.987 z - 1.477). The
others take the best-fitting definition and are marked `not_autofly_verified` in every record's dataset card.
"""

from __future__ import annotations

import math

import numpy as np

from autofly_ue5.expert.obs import target_geometry
from autofly_ue5.sim.types import Observation, Pose

STATE2_OFFSET_M = 1.47  # both real episodes' fitted intercept (1.470, 1.477); its meaning is unknown

STATE_FIELDS = (
    {"index": 0, "name": "distance_to_target_m", "status": "spec_3_2",
     "definition": "horizontal distance from the drone to the target's centre, m"},
    {"index": 1, "name": "bearing_to_target_rad", "status": "not_autofly_verified",
     "definition": "bearing of the target relative to the drone's heading, rad in [-pi, pi], positive to the right "
                   "(|r| 0.92 / 0.88 with a motion-based estimate on the real episodes)"},
    {"index": 2, "name": "altitude_minus_1_47_m", "status": "adopted",
     "definition": "altitude above ground minus 1.47 m (|r| 0.986 / 0.996; fitted 0.985 z - 1.470, 0.987 z - 1.477)"},
    {"index": 3, "name": "speed_m_s", "status": "spec_3_2",
     "definition": "speed: the norm of the velocity vector, m/s"},
    {"index": 4, "name": "vertical_velocity_m_s", "status": "not_autofly_verified",
     "definition": "vertical velocity, up positive, m/s (|r| 0.77 / 0.60)"},
    {"index": 5, "name": "yaw_rate_rad_s", "status": "not_autofly_verified",
     "definition": "yaw rate, rad/s, positive clockwise seen from above (|r| 0.54 / 0.76)"},
    {"index": 6, "name": "x_from_start_m", "status": "spec_3_2",
     "definition": "x (north) relative to the episode's start, m"},
    {"index": 7, "name": "y_from_start_m", "status": "spec_3_2",
     "definition": "y (east) relative to the episode's start, m"},
    {"index": 8, "name": "altitude_m", "status": "spec_3_2", "definition": "altitude above ground, m (z up)"},
)


def autofly_state(obs: Observation, start: Pose, target_xy_z) -> np.ndarray:
    """state[9] for one observation of an episode that started at `start` with its target at `target_xy_z` (NED)."""
    distance, bearing, _dz = target_geometry(obs.pose, target_xy_z)
    vx, vy, vz_ned = (float(v) for v in obs.velocity_ned)
    altitude = -obs.pose.z
    return np.array([
        distance,
        bearing,
        altitude - STATE2_OFFSET_M,
        math.sqrt(vx * vx + vy * vy + vz_ned * vz_ned),
        -vz_ned,
        float(obs.yaw_rate),
        obs.pose.x - start.x,
        obs.pose.y - start.y,
        altitude,
    ], dtype=np.float32)
