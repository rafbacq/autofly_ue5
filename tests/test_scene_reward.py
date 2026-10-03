"""A scene may set its own reward coefficients (2026-10-03, after the s01d gate failed).

s01d_r1's late policy learned to dive out of the altitude band: leaving the bounds cost 5, a collision 10, so bailing
out was the cheaper ending when a collision looked likely (final.zip: 58 of 62 out-of-bounds gate episodes were
altitude_low, at a median of 79 steps). s01d now prices both failures alike. Static s01 keeps M2's reward exactly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.model import SceneFileError, load_scene_file
from tests.test_scene_resolver import _dynamic_variant


def _load(tmp_path, **changes):
    path = tmp_path / "s01d_x.json"
    path.write_text(json.dumps(_dynamic_variant(**changes)))
    return load_scene_file(path)


def test_a_reward_block_overrides_only_the_coefficients_it_names(tmp_path):
    from autofly_ue5.expert.reward import RewardConfig, reward_config_for_scene

    scene = _load(tmp_path, reward={"r_bounds": 10.0, "k_t": 0.02})
    assert scene.reward == (("k_t", 0.02), ("r_bounds", 10.0))
    cfg = reward_config_for_scene(scene)
    assert cfg.r_bounds == 10.0 and cfg.k_t == 0.02
    assert cfg.r_collision == RewardConfig().r_collision and cfg.step_limit == RewardConfig().step_limit


@pytest.mark.parametrize("block", [{"r_bound": 10.0}, {"step_limit": 500}, {"r_bounds": -1.0}, {"r_bounds": "10"}, {}])
def test_a_bad_reward_block_is_refused(tmp_path, block):
    with pytest.raises(SceneFileError):
        _load(tmp_path, reward=block)


def test_s01d_prices_leaving_the_bounds_like_a_collision_and_s01_keeps_m2_s_reward():
    from autofly_ue5.expert.reward import RewardConfig, reward_config_for_scene

    s01d = load_scene_file(SCENES_DIR / "s01d_moving_pillars.json")
    assert s01d.reward == (("r_bounds", 10.0),)
    assert reward_config_for_scene(s01d).r_bounds == RewardConfig().r_collision
    s01 = load_scene_file(SCENES_DIR / "s01_white_pillars.json")
    assert s01.reward == () and reward_config_for_scene(s01) == RewardConfig()


def _climb_out(env):
    env.reset(seed=1_000_070)
    for _ in range(300):
        _obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if terminated or truncated:
            return reward, info
    raise AssertionError("never left the band")


def test_leaving_the_band_costs_ten_on_s01d_and_five_on_s01():
    from tests.test_dynamic_env import make_dynamic_env
    from tests.test_expert_episode import scene_and_layout

    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.sim.fake import FakeSimulator

    reward, info = _climb_out(make_dynamic_env())
    assert info["oob_kind"] == "altitude_high" and -10.2 < reward < -9.8
    scene, layout = scene_and_layout()
    reward, info = _climb_out(AutoFlyEnv(scene, layout, FakeSimulator, map_path="/Game/AutoFly/Maps/S01", instance=0))
    assert info["oob_kind"] == "altitude_high" and -5.2 < reward < -4.8


def test_the_run_identity_and_the_gate_record_carry_the_effective_reward(tmp_path):
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.train import run_identity
    from autofly_ue5.scenes.resolve import resolve_scene

    identity = run_identity(resolve_scene("s01d"), ObsConfig(3, "float16", mover_slots=4))
    assert identity["reward_config"]["r_bounds"] == 10.0 and identity["reward_config"]["r_collision"] == 10.0
    assert identity["reward_config"]["altitude_band_m"] == [1.0, 3.0], "JSON-normalised, as sessions.json stores it"
    static = run_identity(resolve_scene("s01"), ObsConfig())
    assert static["reward_config"]["r_bounds"] == 5.0
