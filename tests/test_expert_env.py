import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from autofly_ue5.sim.fake import FakeSimulator
from tests.test_expert_episode import scene_and_layout


def make_env(**kw):
    from autofly_ue5.expert.env import AutoFlyEnv

    scene, layout = scene_and_layout()
    return AutoFlyEnv(scene, layout, FakeSimulator, map_path="/Game/AutoFly/Maps/S01", instance=0, **kw)


def test_passes_the_gymnasium_api_checker():
    check_env(make_env(), skip_render_check=True)


def test_action_space_is_exactly_the_spec_action():
    env = make_env()
    assert np.allclose(env.action_space.low, [0.0, -1.0, -1.0])
    assert np.allclose(env.action_space.high, [2.0, 1.0, 1.0])


def test_reset_returns_an_observation_inside_the_space():
    env = make_env()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert info["outcome"] == "running" and info["steps"] == 0


def test_an_episode_terminates_and_reports_its_outcome():
    env = make_env()
    env.reset(seed=2)
    for _ in range(400):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        if terminated or truncated:
            break
    assert terminated or truncated, "an episode must end within the 300-step limit"
    assert info["outcome"] in {"success", "collision", "out_of_bounds", "timeout"}
    assert isinstance(info["is_success"], bool)


def test_flying_straight_at_a_target_succeeds():
    # The fake integrates the commanded velocity exactly, so a policy that points at the target and flies
    # must succeed. If this fails, the reward/geometry wiring disagrees with the action convention.
    import math

    from autofly_ue5.expert.obs import target_geometry

    env = make_env()
    env.reset(seed=5)
    outcome = None
    for _ in range(300):
        dist, bearing, _ = target_geometry(env._sim.observe().pose, env._setup.target_xy_z)
        yaw_rate = float(np.clip(bearing * 2.0, -1.0, 1.0))
        v = 2.0 if abs(bearing) < 0.3 else 0.3
        _, _, terminated, truncated, info = env.step(np.array([v, yaw_rate, 0.0], dtype=np.float32))
        if terminated or truncated:
            outcome = info["outcome"]
            break
    assert outcome == "success", f"a straight-line pilot ended in {outcome}"


def test_reset_clears_the_previous_episode_objects():
    env = make_env()
    env.reset(seed=1)
    first = set(env._sim._objects) if hasattr(env._sim, "_objects") else None
    env.reset(seed=2)
    second = set(env._sim._objects) if hasattr(env._sim, "_objects") else None
    if first is not None:
        assert len(second) == len(first), "objects accumulated across episodes"


def test_two_envs_with_different_seed_bases_diverge():
    a, b = make_env(seed_base=0), make_env(seed_base=1000)
    oa, _ = a.reset(seed=None)
    ob, _ = b.reset(seed=None)
    assert not np.array_equal(oa["vector"], ob["vector"]), "vectorised envs must not fly identical episodes"


def test_close_is_idempotent():
    env = make_env()
    env.reset(seed=1)
    env.close()
    env.close()
