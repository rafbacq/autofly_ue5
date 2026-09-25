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


def test_two_envs_with_the_same_seed_base_fly_identical_streams():
    # This documents a real hazard rather than fixing one: under SubprocVecEnv (Task 8), N workers built
    # with the SAME seed_base (the default, if a caller forgets to stagger it) fly byte-identical episode
    # sequences -- an N-fold diversity loss that looks exactly like "SAC plateaued", not like "the workers
    # aren't actually diverse". Task 8 must pass a distinct seed_base per worker; this test pins the
    # current (unchanged) default behaviour so the hazard lives in code, not folklore.
    a, b = make_env(), make_env()
    oa, _ = a.reset(seed=None)
    ob, _ = b.reset(seed=None)
    assert np.array_equal(oa["vector"], ob["vector"]), "two envs sharing a seed_base must fly identical streams"


def test_close_is_idempotent():
    env = make_env()
    env.reset(seed=1)
    env.close()
    env.close()


def test_v_z_positive_is_up_and_negative_is_down():
    # v_z was the one action axis with no direct sign-guard test: test_flying_straight_at_a_target_succeeds
    # already catches an inverted v_forward or yaw_rate (see the sign-flip experiment in the report), but
    # nothing exercised v_z directly. An inverted v_z flies the agent into the floor and ends every episode
    # out_of_bounds -- indistinguishable from "SAC failed to learn" after the GPU hours are already spent.
    # One step is enough: the start altitude is always drawn well inside the hard [1, 3] m band (see
    # START_ALTITUDE_MARGIN_M in episode.py), so a single +-0.2 m step can never itself leave the band.
    up = make_env()
    up.reset(seed=11)
    start_altitude = -up._sim.observe().pose.z
    obs, _, terminated, truncated, _ = up.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert not (terminated or truncated)
    assert -up._sim.observe().pose.z > start_altitude, "v_z=+1 must increase altitude (positive up)"
    assert obs["vector"][6] > 0, "the vector obs's vertical-velocity component must be positive for v_z=+1"

    down = make_env()
    down.reset(seed=11)
    start_altitude = -down._sim.observe().pose.z
    obs, _, terminated, truncated, _ = down.step(np.array([0.0, 0.0, -1.0], dtype=np.float32))
    assert not (terminated or truncated)
    assert -down._sim.observe().pose.z < start_altitude, "v_z=-1 must decrease altitude"
    assert obs["vector"][6] < 0, "the vector obs's vertical-velocity component must be negative for v_z=-1"


def test_reset_observation_is_rendered_after_spawning_the_episode_objects():
    # FakeSimulator's depth is a constant +inf regardless of what is spawned (see fake.py), so the fake
    # cannot be used to show the RETURNED DEPTH reflects the spawn -- asserting that would be vacuous.
    # Assert instead, through the simulator's own step counter, that reset() issues one extra render step
    # AFTER apply_setup has run: a bare sim.reset() advances the clock by its own settle sweep only
    # (reset_steps clock steps); AutoFlyEnv.reset() must advance it by exactly one step more, since it
    # re-renders once after spawning so the observation it returns actually reflects the completed scene
    # (needed because M3's dataset collector treats this env's first frame as a0).
    from autofly_ue5.sim.types import Pose

    bare = FakeSimulator()
    bare.launch("/Game/AutoFly/Maps/S01", 0)
    bare.reset(Pose(0.0, 0.0, -2.0, 0.0))
    steps_from_bare_reset_alone = bare.steps_taken

    env = make_env()
    env.reset(seed=13)
    assert env._sim.steps_taken == steps_from_bare_reset_alone + 1, (
        "reset() must issue one extra step after apply_setup so the returned observation is rendered "
        "post-spawn, not just after the simulator's own reset settle sweep"
    )
    assert env._step_index == 0, "the post-spawn re-render step must not count against the episode's step budget"


def test_step_reward_uses_the_previous_steps_distance_not_the_episode_start():
    # A regression net for a real mutation: deleting `self._prev_dist = dist` after a step leaves every
    # other test in this file green while silently inflating every subsequent step's reward (the progress
    # term keeps being measured against the *episode's starting* distance instead of the previous step's).
    from autofly_ue5.expert.obs import target_geometry
    from autofly_ue5.expert.reward import evaluate

    env = make_env()
    env.reset(seed=21)
    action = np.array([0.4, 0.0, 0.0], dtype=np.float32)

    _, _, terminated, truncated, _ = env.step(action)
    assert not (terminated or truncated), "test needs two non-terminal steps; pick a different seed if this fires"
    dist_1, _, _ = target_geometry(env._sim.observe().pose, env._setup.target_xy_z)

    _, reward_2, terminated, truncated, _ = env.step(action)
    assert not (terminated or truncated)
    obs_after = env._sim.observe()
    dist_2, bearing_2, _ = target_geometry(obs_after.pose, env._setup.target_xy_z)
    bounds = env._layout.bounds
    in_bounds_2 = bounds.x_min <= obs_after.pose.x <= bounds.x_max and bounds.y_min <= obs_after.pose.y <= bounds.y_max

    expected = evaluate(
        prev_dist_m=dist_1, dist_m=dist_2, bearing_rad=bearing_2, altitude_m=-obs_after.pose.z,
        in_bounds=in_bounds_2, collided=obs_after.collided, step_index=2, cfg=env._cfg,
    )
    assert reward_2 == pytest.approx(expected.reward), (
        "step 2's reward must be computed against step 1's distance, not the episode's starting distance"
    )


def test_truncates_exactly_at_max_episode_steps():
    # A regression net for a real mutation: deleting `self._step_index += 1` leaves every other test in
    # this file green while the episode never truncates at all (step_index stays 0 forever, so
    # classify()'s step_index >= cfg.step_limit check never fires).
    env = make_env(max_episode_steps=5)
    env.reset(seed=17)
    zero_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(1, 5):
        _, _, terminated, truncated, info = env.step(zero_action)
        assert not (terminated or truncated), f"must not end before step 5: ended at step {i}"
    _, _, terminated, truncated, info = env.step(zero_action)
    assert truncated and not terminated
    assert info["outcome"] == "timeout" and info["steps"] == 5


def test_close_detaches_the_simulator_before_closing_it():
    # C1: a relaunch abandons a hung close() and installs a new simulator; when the old close() finally returns it
    # must not detach the replacement. Detaching first means the late close() has nothing of the new one to touch.
    env = make_env()
    env.reset(seed=1)
    sim = env._sim
    seen = []
    real_close = sim.close
    sim.close = lambda: (seen.append(env._sim), real_close())
    env.close()
    assert seen == [None]


def test_a_failed_launch_is_not_kept():
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.sim.process import SimExitedError

    class ExitsOnLaunch(FakeSimulator):
        def launch(self, map_path, instance):
            raise SimExitedError("simulator exited before its ports opened")

    sims = [ExitsOnLaunch(), FakeSimulator()]
    scene, layout = scene_and_layout()
    env = AutoFlyEnv(scene, layout, lambda: sims.pop(0), map_path="/Game/AutoFly/Maps/S01", instance=0)
    with pytest.raises(SimExitedError):
        env.reset(seed=1)
    assert env._sim is None, "a simulator that never launched must not be reused by the next reset()"
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
