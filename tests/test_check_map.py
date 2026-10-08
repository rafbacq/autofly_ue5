import math

import pytest

from autofly_ue5.scenes.model import load_registry
from autofly_ue5.validate.obstacles import bbox_error, obstacle_from_instance

REGISTRY = load_registry()
PILLAR = obstacle_from_instance({"tag": "obs_0000", "asset": "cylinder", "x": -24.1, "y": 3.2, "z_center": -5.0, "yaw": 0.0,
                                 "scale": [1.0, 1.0, 10.0], "material": "white", "radius_m": 0.5, "height_m": 10.0}, REGISTRY)


def test_matching_bbox():
    result = bbox_error(PILLAR, ((-24.1, 3.2, -5.0), (1.0, 1.0, 10.0)))
    assert result["found"] is True
    assert result["center_error_m"] == pytest.approx(0.0) and result["size_error_m"] == pytest.approx(0.0)


def test_offset_bbox_reports_errors():
    result = bbox_error(PILLAR, ((-24.0, 3.2, -5.0), (1.0, 1.2, 10.0)))
    assert result["center_error_m"] == pytest.approx(0.1) and result["size_error_m"] == pytest.approx(0.2)


def test_missing_actor():
    assert bbox_error(PILLAR, None) == {"found": False}


def test_a_yawed_crate_is_judged_against_ue_s_axis_aligned_box():
    # UE reports the axis-aligned box of the rotated 1.2 m cube: 1.2 * (cos 30 + sin 30) = 1.639 m across, not 1.2 m
    crate = obstacle_from_instance({"tag": "obs_0001", "asset": "crate", "x": 2.0, "y": -3.0, "z_center": -0.6, "yaw": math.radians(30),
                                    "scale": [1.2, 1.2, 1.2], "material": "wood", "radius_m": 0.8485, "height_m": 1.2}, REGISTRY)
    across = 1.2 * (math.cos(math.radians(30)) + math.sin(math.radians(30)))
    ok = bbox_error(crate, ((2.0, -3.0, -0.6), (across, across, 1.2)))
    assert ok["center_error_m"] == pytest.approx(0.0, abs=1e-9) and ok["size_error_m"] == pytest.approx(0.0, abs=1e-9)
    naive = bbox_error(crate, ((2.0, -3.0, -0.6), (1.2, 1.2, 1.2)))
    assert naive["size_error_m"] == pytest.approx(across - 1.2)
