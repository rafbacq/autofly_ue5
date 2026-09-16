import math

import numpy as np
import pytest

from autofly_ue5.sim.types import Pose


def test_no_hit_depth_becomes_the_clip_distance():
    from autofly_ue5.expert.obs import DEPTH_CLIP_M, encode_depth

    d = np.full((256, 256), np.inf, dtype=np.float32)
    out = encode_depth(d)
    assert out.shape == (1, 84, 84) and out.dtype == np.float32
    assert np.allclose(out, 1.0), "sky (+inf) must read as maximally far, not as zero"


def test_downsample_keeps_the_nearest_obstacle_not_the_average():
    # A single very near pixel in an otherwise empty frame must survive downsampling: averaging would
    # dilute it to invisibility, and a thin near obstacle is exactly what must not be missed.
    from autofly_ue5.expert.obs import DEPTH_CLIP_M, encode_depth

    d = np.full((256, 256), np.inf, dtype=np.float32)
    d[130, 130] = 1.5
    out = encode_depth(d)[0]
    assert out.min() == pytest.approx(1.5 / DEPTH_CLIP_M, abs=1e-6)
    assert (out < 1.0).sum() == 1, "exactly one output cell should carry the near pixel"


def test_depth_beyond_the_clip_saturates_and_shape_is_channel_first():
    from autofly_ue5.expert.obs import encode_depth

    d = np.full((256, 256), 120.0, dtype=np.float32)
    assert np.allclose(encode_depth(d), 1.0)


def test_target_geometry_is_in_the_body_frame():
    from autofly_ue5.expert.obs import target_geometry

    # Facing +x (yaw 0), target 10 m ahead and 2 m to the left (+y is right in NED, so left is -y).
    dist, bearing, dz = target_geometry(Pose(0.0, 0.0, -2.0, 0.0), (10.0, 0.0, -2.0))
    assert dist == pytest.approx(10.0) and bearing == pytest.approx(0.0) and dz == pytest.approx(0.0)

    # Same target, but the drone is yawed 90 deg: the target is now 90 deg to its right -> bearing -pi/2.
    dist, bearing, _ = target_geometry(Pose(0.0, 0.0, -2.0, math.pi / 2), (10.0, 0.0, -2.0))
    assert dist == pytest.approx(10.0) and bearing == pytest.approx(-math.pi / 2)


def test_bearing_wraps_instead_of_jumping():
    from autofly_ue5.expert.obs import target_geometry

    _, b, _ = target_geometry(Pose(0.0, 0.0, -2.0, -math.pi + 0.01), (-10.0, 0.0, -2.0))
    assert abs(b) < 0.02, "a target dead ahead must read as ~0 bearing at any yaw"


def test_vector_is_finite_normalised_and_the_right_width():
    from autofly_ue5.expert.obs import VECTOR_DIM, encode_vector

    v = encode_vector(Pose(-30.0, 0.0, -2.0, 0.0), (1.5, 0.0, -0.1), 0.2, (30.0, 0.0, -1.5))
    assert v.shape == (VECTOR_DIM,) and v.dtype == np.float32
    assert np.all(np.isfinite(v))
    assert abs(v[0]) <= 1.5, "distance must be normalised, not raw metres"
    assert v[1] ** 2 + v[2] ** 2 == pytest.approx(1.0), "bearing is carried as sin/cos"


def test_encode_matches_its_parts():
    from autofly_ue5.expert.obs import encode, encode_depth, encode_vector
    from autofly_ue5.sim.fake import FakeSimulator

    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    o = sim.reset(Pose(-30.0, 0.0, -2.0, 0.0))
    target = (30.0, 0.0, -2.0)
    enc = encode(o, target)
    assert set(enc) == {"depth", "vector"}
    assert np.array_equal(enc["depth"], encode_depth(o.depth))
    assert np.array_equal(enc["vector"], encode_vector(o.pose, o.velocity_ned, o.yaw_rate, target))
