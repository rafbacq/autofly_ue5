import pytest

from autofly_ue5.sim.check_map import bbox_error

INSTANCE = {"tag": "obs_0000", "x": -24.1, "y": 3.2, "z_center": -5.0, "radius_m": 0.5, "height_m": 10.0}


def test_matching_bbox():
    bbox = {"center": {"x": -24.1, "y": 3.2, "z": -5.0}, "size": {"x": 1.0, "y": 1.0, "z": 10.0}}
    result = bbox_error(INSTANCE, bbox)
    assert result["found"] is True
    assert result["center_error_m"] == pytest.approx(0.0) and result["size_error_m"] == pytest.approx(0.0)


def test_offset_bbox_reports_errors():
    bbox = {"center": {"x": -24.0, "y": 3.2, "z": -5.0}, "size": {"x": 1.0, "y": 1.2, "z": 10.0}}
    result = bbox_error(INSTANCE, bbox)
    assert result["center_error_m"] == pytest.approx(0.1) and result["size_error_m"] == pytest.approx(0.2)


def test_missing_actor():
    assert bbox_error(INSTANCE, {}) == {"found": False}
