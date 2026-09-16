import json

import numpy as np

from autofly_ue5.paths import FIXTURES_DIR
from autofly_ue5.sim.decode import decode_depth, decode_rgb

PAS = FIXTURES_DIR / "pas"


def _load(stem: str) -> dict:
    msg = json.loads((PAS / f"{stem}.json").read_text())
    msg["data"] = (PAS / f"{stem}.bin").read_bytes()
    return msg


def test_real_rgb_message_decodes_to_256_rgb():
    msg = _load("rgb_msg")
    assert msg["encoding"] == "BGR" and msg["data_len"] == 256 * 256 * 3
    img = decode_rgb(msg)
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    b, g, r = msg["center_pixel_bgr"]
    assert img[128, 128].tolist() == [r, g, b]


def test_real_rgb_message_as_int_list_matches_bytes():
    msg = _load("rgb_msg")
    as_list = dict(msg, data=list(msg["data"]))
    assert np.array_equal(decode_rgb(as_list), decode_rgb(msg))


def test_real_depth_message_is_metres_with_inf_sky():
    msg = _load("depth_msg")
    assert msg["encoding"] == "16FC1" and msg["data_len"] == 256 * 256 * 2
    depth = decode_depth(msg)
    assert depth.shape == (256, 256) and depth.dtype == np.float32
    # msg["inf_count"] is the raw wire count of +inf half-floats, taken before decoding; this
    # build's no-hit sentinel is exact 0.0 (R5), which decode_depth maps to +inf, so the
    # post-decode inf count is raw_zeros + raw_infs, not msg["inf_count"] alone (ruling R10).
    raw = np.frombuffer(msg["data"], dtype=np.float16)
    raw_zeros = int((raw == 0).sum())
    raw_infs = int(np.isinf(raw).sum())
    assert raw_infs == msg["inf_count"]
    assert int(np.isinf(depth).sum()) == raw_zeros + raw_infs
    center = float(np.median(depth[124:132, 124:132]))
    assert abs(center - msg["expected_center_depth_m"]) < 0.3
    assert float(np.nanmin(depth)) >= 0.5
