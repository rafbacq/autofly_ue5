import math

import numpy as np
import pytest

from autofly_ue5.scenes.model import Bounds
from autofly_ue5.sim.types import Pose
from autofly_ue5.validate.geometry import (
    Pillar,
    choose_depth_probes,
    contiguous_width,
    depth_agreement,
    expected_center_depth,
    mean_abs_diff,
    pillar_width_px,
    pillars_from_layout_json,
    ray_circle_distance,
    speed_along_heading,
    tracking_error,
    yaw_rates_from_yaws,
)

BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)


def test_ray_circle_distance():
    assert ray_circle_distance(0.0, 0.0, 0.0, 5.0, 0.0, 1.0) == pytest.approx(4.0)
    assert ray_circle_distance(0.0, 0.0, math.pi / 2, 5.0, 0.0, 1.0) is None
    assert ray_circle_distance(0.0, 0.0, 0.0, -5.0, 0.0, 1.0) is None
    assert ray_circle_distance(5.0, 0.0, 0.0, 5.0, 0.0, 1.0) == pytest.approx(1.0)


def test_expected_center_depth_subtracts_camera_offset():
    pillar = Pillar("obs_0000", 5.0, 0.0, 0.5, 10.0)
    assert expected_center_depth(Pose(0.0, 0.0, -2.0, 0.0), pillar) == pytest.approx(4.1)


def test_probe_on_a_lone_pillar_matches_requested_distance():
    probes = choose_depth_probes([Pillar("obs_0000", 0.0, 0.0, 0.5, 10.0)], BOUNDS, distances=(3.0, 6.0))
    assert [p.distance_m for p in probes] == [3.0, 6.0]
    for probe in probes:
        assert probe.expected_depth_m == pytest.approx(probe.distance_m)
        assert probe.pose.z == -2.0


def test_probe_avoids_an_occluded_bearing():
    target = Pillar("obs_0000", 0.0, 0.0, 0.5, 10.0)
    blocker = Pillar("obs_0001", -2.0, 0.0, 0.3, 10.0)  # sits on the first (yaw 0) approach
    probe = choose_depth_probes([target, blocker], BOUNDS, distances=(3.0,))[0]
    assert probe.tag == "obs_0000" and probe.pose.yaw != 0.0
    assert expected_center_depth(probe.pose, target) == pytest.approx(3.0)


def test_no_probe_possible_raises():
    with pytest.raises(ValueError, match="12.0"):
        choose_depth_probes([Pillar("obs_0000", 0.0, 0.0, 0.5, 10.0)], Bounds(-5.0, 5.0, -5.0, 5.0), distances=(12.0,))


def test_tracking_error():
    result = tracking_error([2.0] * 5, [0.0, 1.0, 1.8, 1.8, 1.8], settle=2)
    assert result["mean_abs_error"] == pytest.approx(0.2) and result["relative_error"] == pytest.approx(0.1)
    assert result["samples"] == 3
    with pytest.raises(ValueError):
        tracking_error([1.0], [1.0, 2.0], settle=0)


def test_yaw_rates_wrap_across_pi():
    assert yaw_rates_from_yaws([3.1, -3.1], 0.2)[0] == pytest.approx((2 * math.pi - 6.2) / 0.2)


def test_speed_and_image_difference():
    assert speed_along_heading((0.0, 2.0, 0.0), math.pi / 2) == pytest.approx(2.0)
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = np.full((2, 2, 3), 12, dtype=np.uint8)
    assert mean_abs_diff(a, b) == pytest.approx(12.0) and mean_abs_diff(b, a) == pytest.approx(12.0)


def test_pillar_width_and_contiguous_width():
    assert pillar_width_px(3.5, 0.5) == pytest.approx(256 * 0.5 / math.sqrt(3.5**2 - 0.5**2))  # about 36.95 px
    assert pillar_width_px(3.5, 0.5, hfov_deg=60.0) == pytest.approx(pillar_width_px(3.5, 0.5) / math.tan(math.radians(30.0)))
    row = np.array([False, True, True, True, False, True])
    assert contiguous_width(row, 2) == 3 and contiguous_width(row, 5) == 1 and contiguous_width(row, 0) == 0


def test_depth_agreement():
    a = np.array([[1.0, 2.0], [np.inf, 40.0]], dtype=np.float32)
    b = np.array([[1.05, 2.0], [np.inf, np.inf]], dtype=np.float32)
    result = depth_agreement(a, b)
    assert result["mask_agreement"] == 1.0 and result["pixels_compared"] == 2
    assert result["median_relative_diff"] == pytest.approx(0.025, abs=1e-6)
    c = np.array([[np.inf, 2.0], [np.inf, 40.0]], dtype=np.float32)
    assert depth_agreement(a, c)["mask_agreement"] == 0.75
    far = np.full((2, 2), np.inf, dtype=np.float32)
    assert depth_agreement(far, far) == {"mask_agreement": 1.0, "pixels_compared": 0, "median_relative_diff": None}


def test_pillars_from_layout_json():
    data = {"layout": {"instances": [{"tag": "obs_0000", "x": 1.0, "y": 2.0, "radius_m": 0.5, "height_m": 9.0,
                                      "asset": "cylinder", "z_center": -4.5, "yaw": 0.0, "scale": [1, 1, 9], "material": "white"}]}}
    assert pillars_from_layout_json(data) == [Pillar("obs_0000", 1.0, 2.0, 0.5, 9.0)]
