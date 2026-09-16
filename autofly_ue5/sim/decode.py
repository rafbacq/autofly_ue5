"""Decode Project AirSim image messages (camera compress=false).

RGB (image type 0): encoding "BGR", 3 uint8 per pixel in B, G, R order, rows top to bottom.
Depth (image types 1/2): encoding "16FC1", one little-endian IEEE half float per pixel, metres, +inf = no hit.

R5 (controller ruling, M0 smoke test): on this UE5.7.4 + Project AirSim build, DepthPlanar's raw wire
encoding is exact 0.0 for no-hit (sky), not +inf. Evidence: a captured 256x256 M0 smoke-test frame
(runs/m0/fixtures/depth_msg.json and .bin) had 20,145 exact-zero pixels, every one of them in the upper
half of the image (rows 0-127, i.e. sky), while the frame's own raw inf_count was 0 and its finite range
was 1.484-254.5 m. decode_depth() canonicalises exact 0.0 to +inf so the project's no-hit convention stays
+inf everywhere downstream; the raw 0.0-encoded values remain recoverable from the captured .bin fixtures.
"""

import numpy as np


class ImageFormatError(ValueError):
    """The message encoding or payload size is not what the decoder expects."""


def _as_uint8(data) -> np.ndarray:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    return np.asarray(data, dtype=np.uint8)


def decode_rgb(msg: dict) -> np.ndarray:
    if msg["encoding"] != "BGR":
        raise ImageFormatError(f"expected encoding BGR, got {msg['encoding']}")
    height, width = int(msg["height"]), int(msg["width"])
    raw = _as_uint8(msg["data"])
    if raw.size != height * width * 3:
        raise ImageFormatError(f"expected {height * width * 3} bytes for {height}x{width} BGR, got {raw.size}")
    return np.ascontiguousarray(raw.reshape(height, width, 3)[:, :, ::-1])


def decode_depth(msg: dict) -> np.ndarray:
    if msg["encoding"] != "16FC1":
        raise ImageFormatError(f"expected encoding 16FC1, got {msg['encoding']}")
    height, width = int(msg["height"]), int(msg["width"])
    raw = _as_uint8(msg["data"])
    if raw.size != height * width * 2:
        raise ImageFormatError(f"expected {height * width * 2} bytes for {height}x{width} 16FC1, got {raw.size}")
    depth = raw.view("<f2").reshape(height, width).astype(np.float32)
    depth[depth == 0.0] = np.inf  # this build's no-hit sentinel is exact 0.0, not +inf; see module docstring (R5).
    return depth
