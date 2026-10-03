"""The expert's privileged moving-pillar input (2026-10-03, after the s01d gate failed at 0.775).

A moving pillar that yields to the drone stands still and looks exactly like a static one in depth (and in RGB: both
are white), yet its contact rule fires at 1.0 m from its surface while a static pillar needs physical contact. So the
expert is told, per nearby mover: where it is in the body frame, how far it is from the contact boundary, and how it
moves. Static scenes and every existing checkpoint keep their observation exactly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from autofly_ue5.sim.types import Pose

R = 0.5          # mover radius
CONTACT = 1.0    # contact_m
DT = 0.2


def _encode(pose, positions, previous=None, radii=None, slots=4):
    from autofly_ue5.expert.obs import encode_movers

    previous = positions if previous is None else previous
    radii = [R] * len(positions) if radii is None else radii
    return encode_movers(pose, positions, previous, radii, contact_m=CONTACT, dt=DT, slots=slots)


def _slot(v, i):
    from autofly_ue5.expert.obs import MOVER_FEATURES

    return v[i * MOVER_FEATURES:(i + 1) * MOVER_FEATURES]


def test_a_mover_straight_ahead_is_forward_with_its_clearance_to_the_contact_boundary():
    from autofly_ue5.expert.obs import MOVER_CLEARANCE_NORM_M, MOVER_FEATURES, MOVER_RANGE_M

    v = _encode(Pose(0.0, 0.0, -2.0, 0.0), [(5.0, 0.0)])
    assert v.shape == (4 * MOVER_FEATURES,) and v.dtype == np.float32
    present, fwd, right, clearance, v_fwd, v_right = _slot(v, 0)
    assert present == 1.0
    assert fwd == pytest.approx(5.0 / MOVER_RANGE_M) and right == pytest.approx(0.0, abs=1e-7)
    assert clearance == pytest.approx((5.0 - R - CONTACT) / MOVER_CLEARANCE_NORM_M)
    assert (v_fwd, v_right) == (0.0, 0.0), "a mover that did not move"
    assert not np.any(v[MOVER_FEATURES:]), "empty slots are zero"


def test_the_body_frame_follows_the_heading():
    from autofly_ue5.expert.obs import MOVER_RANGE_M

    east = Pose(0.0, 0.0, -2.0, math.pi / 2)  # NED: yaw +90 deg faces east (+y)
    _p, fwd, right, *_ = _slot(_encode(east, [(0.0, 4.0)]), 0)
    assert fwd == pytest.approx(4.0 / MOVER_RANGE_M) and right == pytest.approx(0.0, abs=1e-6)
    _p, fwd, right, *_ = _slot(_encode(east, [(4.0, 0.0)]), 0)  # north of a drone facing east: on its left
    assert fwd == pytest.approx(0.0, abs=1e-6) and right == pytest.approx(-4.0 / MOVER_RANGE_M)


def test_velocity_is_the_last_step_s_displacement_in_the_body_frame():
    from autofly_ue5.expert.obs import MOVER_SPEED_NORM_M_S

    *_, v_fwd, v_right = _slot(_encode(Pose(0.0, 0.0, -2.0, 0.0), [(5.0, 0.2)], previous=[(5.0, 0.0)]), 0)
    assert v_fwd == pytest.approx(0.0, abs=1e-6) and v_right == pytest.approx(1.0 / MOVER_SPEED_NORM_M_S)


def test_slots_hold_the_nearest_movers_first_and_ignore_those_out_of_range():
    from autofly_ue5.expert.obs import MOVER_RANGE_M

    pose = Pose(0.0, 0.0, -2.0, 0.0)
    positions = [(9.0, 0.0), (3.0, 0.0), (0.0, 30.0), (6.0, 0.0), (0.0, -4.5), (MOVER_RANGE_M + R + 0.5, 0.0)]
    v = _encode(pose, positions, radii=[R, R, R, 1.0, R, R])
    clearances = [_slot(v, i)[3] for i in range(4)]
    assert [_slot(v, i)[0] for i in range(4)] == [1.0, 1.0, 1.0, 1.0]
    assert clearances == sorted(clearances), "sorted by clearance, nearest first"
    assert _slot(v, 0)[1] == pytest.approx(0.3), "the mover 3 m ahead fills slot 0"
    v2 = _encode(pose, [(0.0, 30.0), (MOVER_RANGE_M + R + 0.5, 0.0)])
    assert not np.any(v2), "every mover beyond range: all slots empty"


def test_inside_the_contact_zone_the_clearance_is_negative_and_everything_stays_in_range():
    v = _encode(Pose(0.0, 0.0, -2.0, 0.0), [(1.2, 0.0)], previous=[(0.0, 50.0)])
    assert _slot(v, 0)[3] < 0.0
    assert np.all(np.abs(v) <= 1.0) and np.all(np.isfinite(v))


def test_no_movers_is_all_zero():
    assert not np.any(_encode(Pose(0.0, 0.0, -2.0, 0.0), []))


# ---------------------------------------------------------------------------------------------------------------
# The observation config: a dynamic scene adds the movers key; everything else is unchanged.
# ---------------------------------------------------------------------------------------------------------------
def test_the_config_adds_a_movers_key_only_when_asked():
    from autofly_ue5.expert.obs import MOVER_FEATURES, ObsConfig, obs_config_from_space

    plain = ObsConfig(3, "float16")
    assert set(plain.space().spaces) == {"depth", "vector"} and plain.to_json() == {"depth_frames": 3, "depth_dtype": "float16"}
    movers = ObsConfig(3, "float16", mover_slots=4)
    space = movers.space()
    assert set(space.spaces) == {"depth", "vector", "movers"}
    assert space["movers"].shape == (4 * MOVER_FEATURES,) and space["movers"].dtype == np.float32
    assert float(space["movers"].low.min()) == -1.0 and float(space["movers"].high.max()) == 1.0
    assert movers.to_json() == {"depth_frames": 3, "depth_dtype": "float16", "mover_slots": 4}
    assert obs_config_from_space(space) == movers and obs_config_from_space(plain.space()) == plain
    with pytest.raises(ValueError):
        ObsConfig(3, "float16", mover_slots=-1)


def test_a_dynamic_scene_defaults_to_the_mover_input_and_a_static_one_does_not():
    from tests.test_dynamic_env import dynamic_scene
    from tests.test_expert_episode import scene_and_layout

    from autofly_ue5.expert.obs import DYNAMIC_MOVER_SLOTS, ObsConfig, obs_config_for_scene

    assert obs_config_for_scene(dynamic_scene()[0]) == ObsConfig(3, "float16", mover_slots=DYNAMIC_MOVER_SLOTS)
    assert obs_config_for_scene(scene_and_layout()[0]) == ObsConfig()


# ---------------------------------------------------------------------------------------------------------------
# The network: an optional branch, so every existing checkpoint keeps its exact parameters.
# ---------------------------------------------------------------------------------------------------------------
def _extractor(config):
    from autofly_ue5.expert.features import DepthVectorExtractor

    return DepthVectorExtractor(config.space(), features_dim=256)


def test_without_a_movers_key_the_extractor_s_parameters_are_unchanged():
    from autofly_ue5.expert.obs import ObsConfig

    names = {n: tuple(p.shape) for n, p in _extractor(ObsConfig(3, "float16")).state_dict().items()}
    assert names == {"cnn.0.weight": (32, 3, 8, 8), "cnn.0.bias": (32,), "cnn.2.weight": (64, 32, 4, 4),
                     "cnn.2.bias": (64,), "cnn.4.weight": (64, 64, 3, 3), "cnn.4.bias": (64,),
                     "mlp.0.weight": (64, 8), "mlp.0.bias": (64,), "head.0.weight": (256, 3136 + 64),
                     "head.0.bias": (256,)}


def test_with_a_movers_key_the_extractor_gains_a_mover_branch():
    import torch

    from autofly_ue5.expert.obs import MOVER_FEATURES, ObsConfig

    ex = _extractor(ObsConfig(3, "float16", mover_slots=4))
    shapes = {n: tuple(p.shape) for n, p in ex.state_dict().items()}
    assert shapes["mover_mlp.0.weight"] == (64, 4 * MOVER_FEATURES) and shapes["head.0.weight"] == (256, 3136 + 64 + 64)
    out = ex({"depth": torch.zeros(2, 3, 84, 84), "vector": torch.zeros(2, 8), "movers": torch.zeros(2, 4 * MOVER_FEATURES)})
    assert out.shape == (2, 256)


R1_BEST = Path(__file__).resolve().parents[1] / "runs" / "expert" / "s01d_r1" / "best" / "best_model.zip"


@pytest.mark.skipif(not R1_BEST.is_file(), reason="needs the s01d_r1 checkpoint on this host")
def test_the_s01d_r1_checkpoint_still_loads_and_acts():
    from stable_baselines3 import SAC

    from autofly_ue5.expert.obs import ObsConfig

    model = SAC.load(R1_BEST, device="cpu")
    space = ObsConfig(3, "float16").space()
    assert model.observation_space == space
    action, _ = model.predict(space.sample(), deterministic=True)
    assert action.shape == (3,)


# ---------------------------------------------------------------------------------------------------------------
# The env: the movers key follows the controller; a checkpoint without it still gets its old observation.
# ---------------------------------------------------------------------------------------------------------------
def test_a_dynamic_env_observes_its_movers_and_passes_the_checker():
    from tests.test_dynamic_env import ZERO, make_dynamic_env

    from autofly_ue5.expert.obs import MOVER_FEATURES, MOVER_RANGE_M

    env = make_dynamic_env()
    check_env(env, skip_render_check=True)
    for seed in range(1_000_040, 1_000_044):
        obs, info = env.reset(seed=seed)
        for _ in range(40):
            assert env.observation_space.contains(obs)
            pose = env.unwrapped._last_pose
            near = [m for m, r in zip(info["movers"], (rt.footprint.radius_m for rt in env.unwrapped._movers.routes))
                    if math.dist(m, (pose.x, pose.y)) - r <= MOVER_RANGE_M]
            presents = obs["movers"][0::MOVER_FEATURES]
            assert int(presents.sum()) == min(len(near), len(presents)), f"seed {seed}: present flags"
            obs, _r, terminated, truncated, info = env.step(ZERO)
            if terminated or truncated:
                break


def test_a_checkpoint_without_the_mover_input_keeps_its_observation_on_s01d():
    from tests.test_dynamic_env import make_dynamic_env

    from autofly_ue5.expert.obs import ObsConfig

    env = make_dynamic_env(obs_config=ObsConfig(3, "float16"))
    obs, _info = env.reset(seed=1_000_050)
    assert set(obs) == {"depth", "vector"}
