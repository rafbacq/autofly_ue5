"""Decode Project AirSim image messages (camera compress=false).

RGB (image type 0): encoding "BGR", 3 uint8 per pixel in B, G, R order, rows top to bottom.
Depth (image types 1/2): encoding "16FC1", one little-endian IEEE half float per pixel, metres, +inf = no hit.
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
    return raw.view("<f2").reshape(height, width).astype(np.float32)
