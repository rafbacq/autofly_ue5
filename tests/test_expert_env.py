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
    # The real claim is "objects do not accumulate across episodes", not "two different episodes spawn
    # the same number of objects": the distractor count is uniform in 3-5 per episode (see
    # N_DISTRACTORS_DEFAULT), so two different seeds legitimately produce different object counts. It is
    # also not "the two episodes' spawned names are disjoint" -- apply_setup always spawns under the same
    # literal base names ("target", "distractor_0", ...), so a correctly-cleaned-up episode gets the SAME
    # names back, not different ones. The actual signature of a leak is FakeSimulator.spawn()
    # uniquifying a name on collision (appending a numeric suffix, e.g. "target1") because a same-named
    # object from a previous episode is still alive when the next one spawns -- so assert the returned
    # names stay clean across many resets, and that the live count never exceeds one episode's worth.
    from autofly_ue5.expert.episode import N_DISTRACTORS_DEFAULT

    def is_clean(spawned: tuple[str, ...]) -> bool:
        expected = {"target", *(f"distractor_{i}" for i in range(len(spawned) - 1))}
        return set(spawned) == expected

    env = make_env()
    for seed in range(10):
        env.reset(seed=seed)
        assert is_clean(env._spawned), (
            f"a spawned name was uniquified -- a previous episode's object is still alive: {env._spawned}"
        )
        if hasattr(env._sim, "_objects"):
            assert len(env._sim._objects) <= 1 + N_DISTRACTORS_DEFAULT[1], "objects accumulated across episodes"


def test_reset_with_the_same_seed_twice_gives_the_identical_episode():
    # The Gymnasium contract check_env enforces: reset(seed=K) is a deterministic function of K alone.
    # This is also why folding seed into a call counter (the earlier, wrong design) was rejected -- that
    # would make reset(seed=K) depend on how many resets came before it.
    env = make_env()
    obs_1, _ = env.reset(seed=42)
    obs_2, _ = env.reset(seed=42)
    assert np.array_equal(obs_1["vector"], obs_2["vector"]), "reset(seed=K) must be reproducible"


def test_distinct_seeds_give_distinct_episodes():
    # Task 9's M2 gate draws >= 200 evaluation episodes as reset(seed=EVAL_SEED_BASE + i) for i in
    # range(200): if distinct seeds collapsed onto the same handful of episodes (or just one), the gate
    # would silently re-run one episode over and over and report a meaningless 0% or 100% success rate.
    env = make_env()
    targets = {tuple(np.round(env.reset(seed=seed)[0]["vector"], 6)) for seed in range(12)}
    assert len(targets) > 1, "distinct seeds must not all produce the same episode"


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
