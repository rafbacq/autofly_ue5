import math

import pytest

from autofly_ue5.frames import (
    body_to_ned,
    heading_unit,
    ned_to_ue_cm,
    ned_yaw_deg_to_ue_yaw_deg,
    quat_to_yaw,
    ue_cm_to_ned,
    wrap_pi,
    yaw_to_quat,
    z_up_from_ned,
)


def test_ned_to_ue_cm_flips_z_and_scales():
    assert ned_to_ue_cm(1.0, -2.0, -3.0) == (100.0, -200.0, 300.0)


def test_ue_cm_round_trip():
    assert ue_cm_to_ned(*ned_to_ue_cm(12.5, -7.25, -1.75)) == pytest.approx((12.5, -7.25, -1.75))


def test_ground_top_is_ue_zero():
    assert ned_to_ue_cm(0.0, 0.0, 0.0) == (0.0, 0.0, -0.0)


def test_yaw_is_identical_in_ue():
    assert ned_yaw_deg_to_ue_yaw_deg(37.5) == 37.5


def test_yaw_quaternion_round_trip():
    w, x, y, z = yaw_to_quat(math.pi / 2)
    assert (w, x, y, z) == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
    assert quat_to_yaw(w, x, y, z) == pytest.approx(math.pi / 2)
    assert quat_to_yaw(*yaw_to_quat(-2.5)) == pytest.approx(-2.5)


def test_wrap_pi():
    assert wrap_pi(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_pi(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
    assert wrap_pi(0.3) == pytest.approx(0.3)


def test_autofly_z_up():
    assert z_up_from_ned(-2.0) == 2.0


def test_heading_unit_east():
    assert heading_unit(math.pi / 2) == pytest.approx((0.0, 1.0))


def test_body_to_ned_yaw_and_nose_up_pitch():
    assert body_to_ned(*yaw_to_quat(math.pi / 2), (0.4, 0.0, 0.0)) == pytest.approx((0.0, 0.4, 0.0), abs=1e-12)
    half = math.radians(30.0) / 2  # 30 deg nose-up pitch about body y: a forward point moves up (negative z in NED)
    forward = body_to_ned(math.cos(half), 0.0, math.sin(half), 0.0, (0.4, 0.0, 0.0))
    assert forward == pytest.approx((0.4 * math.cos(math.radians(30.0)), 0.0, -0.4 * math.sin(math.radians(30.0))))
