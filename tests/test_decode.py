import numpy as np
import pytest

from autofly_ue5.sim.decode import ImageFormatError, decode_depth, decode_rgb


def _rgb_msg(rgb: np.ndarray, as_list: bool = False) -> dict:
    data = np.ascontiguousarray(rgb[:, :, ::-1]).tobytes()
    return {"encoding": "BGR", "height": rgb.shape[0], "width": rgb.shape[1], "data": list(data) if as_list else data}


def test_decode_rgb_reverses_bgr():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[10, 20, 30], [40, 50, 60], [70, 80, 90]]], dtype=np.uint8)
    out = decode_rgb(_rgb_msg(rgb))
    assert out.dtype == np.uint8 and out.shape == (2, 3, 3)
    assert np.array_equal(out, rgb)


def test_decode_rgb_accepts_int_list():
    rgb = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(decode_rgb(_rgb_msg(rgb, as_list=True)), rgb)


def test_decode_depth_half_float_metres_with_inf():
    depth = np.array([[1.5, np.inf], [0.25, 30.0]], dtype="<f2")
    msg = {"encoding": "16FC1", "height": 2, "width": 2, "data": depth.tobytes()}
    out = decode_depth(msg)
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(1.5) and np.isinf(out[0, 1])
    assert out[1, 0] == pytest.approx(0.25) and out[1, 1] == pytest.approx(30.0)


def test_decode_depth_maps_exact_zero_to_no_hit():
    """R5 (controller ruling): this UE5.7.4 + Project AirSim build's DepthPlanar encodes no-hit (sky) as
    exact 0.0, not +inf. Evidence: a captured 256x256 M0 smoke-test frame (runs/m0/fixtures/depth_msg.json
    and .bin) had 20,145 exact-zero pixels, every one of them in the upper half of the image (rows 0-127,
    i.e. sky), while the frame's own reported inf_count was 0 and its finite range was 1.484-254.5 m. This
    fixture-shaped array reproduces that pattern in miniature: a block of exact zeros (sky), a real +inf
    (still passed through unchanged), and finite hits spanning that observed 1.484-254.5 m range.
    """
    depth = np.array([[0.0, 0.0, np.inf], [1.484, 254.5, 0.0]], dtype="<f2")
    msg = {"encoding": "16FC1", "height": 2, "width": 3, "data": depth.tobytes()}
    out = decode_depth(msg)
    assert np.isinf(out[0, 0]) and np.isinf(out[0, 1]) and np.isinf(out[0, 2]) and np.isinf(out[1, 2])
    assert out[1, 0] == pytest.approx(1.484, rel=1e-3)
    assert out[1, 1] == pytest.approx(254.5)


def test_wrong_encoding_is_rejected():
    with pytest.raises(ImageFormatError, match="PNG"):
        decode_rgb({"encoding": "PNG", "height": 1, "width": 1, "data": b"\x00\x00\x00"})
    with pytest.raises(ImageFormatError, match="BGR"):
        decode_depth({"encoding": "BGR", "height": 1, "width": 1, "data": b"\x00\x00"})


def test_size_mismatch_is_rejected():
    with pytest.raises(ImageFormatError, match="bytes"):
        decode_rgb({"encoding": "BGR", "height": 2, "width": 2, "data": b"\x00" * 11})
