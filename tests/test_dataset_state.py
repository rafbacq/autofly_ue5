"""autofly_ue5/dataset/state.py: AutoFly's state[9] as M3 writes it (docs/decisions/2026-10-03-m3-a0-and-collection.md),
checked against hand-computed poses."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autofly_ue5.sim.types import Observation, Pose


def _obs(pose: Pose, velocity=(0.0, 0.0, 0.0), yaw_rate=0.0) -> Observation:
    z = np.zeros((2, 2), np.float32)
    return Observation(rgb=np.zeros((2, 2, 3), np.uint8), depth=z, pose=pose, velocity_ned=velocity, yaw_rate=yaw_rate,
                       sim_time_ns=0, collided=False, step_collisions=(), rgb_time_ns=0, depth_time_ns=0,
                       kinematics_time_ns=0, camera_pose_error_m=0.0)


def test_each_field_is_computed_as_documented():
    from autofly_ue5.dataset.state import STATE2_OFFSET_M, autofly_state

    start = Pose(-30.0, 2.0, -2.0, 0.0)
    # 10 m east and 3 m north of the start, heading east (yaw +90 deg in NED), climbing at 0.5 m/s, turning at 0.2 rad/s.
    pose = Pose(-27.0, 12.0, -2.4, math.pi / 2)
    target = (-27.0, 32.0, -1.0)  # straight ahead, 20 m
    s = autofly_state(_obs(pose, velocity=(0.0, 1.9, -0.5), yaw_rate=0.2), start, target)
    assert s.dtype == np.float32 and s.shape == (9,)
    assert s[0] == pytest.approx(20.0)
    assert s[1] == pytest.approx(0.0, abs=1e-6)
    assert s[2] == pytest.approx(2.4 - STATE2_OFFSET_M) and STATE2_OFFSET_M == 1.47
    assert s[3] == pytest.approx(math.hypot(1.9, 0.5))
    assert s[4] == pytest.approx(0.5), "vertical velocity is up-positive"
    assert s[5] == pytest.approx(0.2)
    assert (s[6], s[7], s[8]) == pytest.approx((3.0, 10.0, 2.4))


def test_the_bearing_is_positive_when_the_target_is_to_the_right():
    from autofly_ue5.dataset.state import autofly_state

    start = Pose(0.0, 0.0, -2.0, 0.0)  # facing north (+x)
    right = autofly_state(_obs(start), start, (10.0, 10.0, -1.0))  # north-east: to the right in NED
    left = autofly_state(_obs(start), start, (10.0, -10.0, -1.0))
    assert right[1] == pytest.approx(math.pi / 4) and left[1] == pytest.approx(-math.pi / 4)


def test_every_field_states_its_meaning_and_whether_autofly_confirms_it():
    from autofly_ue5.dataset.state import STATE_FIELDS

    assert len(STATE_FIELDS) == 9
    status = {f["index"]: f["status"] for f in STATE_FIELDS}
    assert {i for i, s in status.items() if s == "not_autofly_verified"} == {1, 4, 5}
    assert status[2] == "adopted" and all(f["definition"] for f in STATE_FIELDS)
