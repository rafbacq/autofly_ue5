"""C4 (2026-09-24 review): every consumer of episodes owns a disjoint seed range, and a training run's directory
belongs to one run -- a fresh run cannot mix with an old one, a resumed session cannot replay an earlier one."""

import json

import pytest

from autofly_ue5.sim.fake import FakeSimulator
from tests.test_expert_episode import scene_and_layout


def _record_seeds(monkeypatch) -> list[int]:
    import autofly_ue5.expert.env as env_module

    seen: list[int] = []
    real = env_module.sample_setup

    def recording(scene, layout, rng):
        seen.append(int(rng.bit_generator.seed_seq.entropy))
        return real(scene, layout, rng)

    monkeypatch.setattr(env_module, "sample_setup", recording)
    return seen


def _vec(seed_base: int):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from autofly_ue5.expert.env import AutoFlyEnv

    scene, layout = scene_and_layout()
    return DummyVecEnv([lambda: AutoFlyEnv(scene, layout, FakeSimulator, map_path="/x", instance=0, seed_base=seed_base,
                                           max_episode_steps=3)])


def test_every_seed_range_is_disjoint():
    from autofly_ue5.expert.seeds import (
        EVAL_CALLBACK_SEED_BASE,
        EVAL_SEED_BASE,
        MAX_SESSIONS,
        SESSION_SEED_STRIDE,
        WORKER_SEED_STRIDE,
        session_seed_base,
    )

    ranges = [(session_seed_base(rank, s), session_seed_base(rank, s) + SESSION_SEED_STRIDE)
              for rank in range(64) for s in range(MAX_SESSIONS)]
    from autofly_ue5.expert.seeds import COLLECTION_SEED_BASE, PROBE_SEED_BASE

    ranges += [(EVAL_SEED_BASE, EVAL_SEED_BASE + WORKER_SEED_STRIDE),
               (EVAL_CALLBACK_SEED_BASE, EVAL_CALLBACK_SEED_BASE + WORKER_SEED_STRIDE),
               (PROBE_SEED_BASE, PROBE_SEED_BASE + WORKER_SEED_STRIDE),
               (COLLECTION_SEED_BASE, COLLECTION_SEED_BASE + 100 * WORKER_SEED_STRIDE)]  # M3/M5 collection
    ranges.sort()
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(ranges, ranges[1:]))
    # SB3 seeds a vec env's first reset explicitly with seed+rank (seed=0 here): those must not be any worker's.
    assert ranges[0][0] >= 64


def test_session_seed_base_refuses_to_overflow_into_the_next_worker():
    from autofly_ue5.expert.seeds import MAX_SESSIONS, session_seed_base

    session_seed_base(0, MAX_SESSIONS - 1)
    with pytest.raises(ValueError, match="session"):
        session_seed_base(0, MAX_SESSIONS)


def test_sac_seed_0_does_not_fly_the_first_episode_twice(monkeypatch):
    # Reproduced in the review: SAC(seed=0) reset worker 0 with an explicit seed 0, and worker 0's own counter then
    # drew default_rng(0 + 0) again -- seeds [0, 0, 1].
    from autofly_ue5.expert.seeds import worker_seed_base
    from autofly_ue5.expert.train import build_model

    seen = _record_seeds(monkeypatch)
    model = build_model(_vec(worker_seed_base(0)), device="cpu", buffer_size=100, learning_starts=1000, seed=0, verbose=0)
    model.learn(total_timesteps=10)
    assert len(seen) >= 3 and len(set(seen)) == len(seen), seen


def test_a_fresh_run_refuses_a_run_root_that_already_holds_a_run(tmp_path):
    from autofly_ue5.expert.train import prepare_run_root

    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "rl_model_200000_steps.zip").write_text("old run")
    with pytest.raises(RuntimeError, match="already holds"):
        prepare_run_root(tmp_path, resume=False, reward_version="v2", seed=0)


def test_sessions_are_numbered_and_recorded(tmp_path):
    from autofly_ue5.expert.train import prepare_run_root

    assert prepare_run_root(tmp_path, resume=False, reward_version="v2", seed=0) == 0
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "rl_model_10000_steps.zip").write_text("model")
    (tmp_path / "checkpoints" / "rl_model_replay_buffer_10000_steps.pkl").write_text("buffer")
    assert prepare_run_root(tmp_path, resume=True, reward_version="v2", seed=0) == 1
    sessions = json.loads((tmp_path / "sessions.json").read_text())["sessions"]
    assert [s["index"] for s in sessions] == [0, 1] and all(s["reward_version"] == "v2" for s in sessions)


@pytest.mark.parametrize("recorded", [None, "v1"])
def test_resume_refuses_a_run_trained_under_another_reward_version(tmp_path, recorded):
    # A replay buffer holds rewards: continuing it under a changed reward mixes two objectives. A run without a
    # sessions.json predates this record (and its buffer holds fault rows that can no longer be told apart).
    from autofly_ue5.expert.train import prepare_run_root

    if recorded is not None:
        (tmp_path / "sessions.json").write_text(json.dumps({"sessions": [{"index": 0, "reward_version": recorded}]}))
    with pytest.raises(RuntimeError, match="reward"):
        prepare_run_root(tmp_path, resume=True, reward_version="v2", seed=0)


def test_a_resumed_session_reseeds_its_first_episodes(tmp_path, monkeypatch):
    # SAC.load re-seeds the env with the saved seed, so a resumed session would start with session 0's episodes.
    from stable_baselines3 import SAC

    from autofly_ue5.expert.seeds import SESSION_SEED_STRIDE
    from autofly_ue5.expert.train import build_model, reseed_resumed_model

    model = build_model(_vec(1_000_000), device="cpu", buffer_size=100, learning_starts=1000, seed=0, verbose=0)
    model.save(tmp_path / "m.zip")
    seen = _record_seeds(monkeypatch)
    loaded = SAC.load(tmp_path / "m.zip", env=_vec(1_000_000 + SESSION_SEED_STRIDE), device="cpu")
    reseed_resumed_model(loaded, seed=0, session=1)
    loaded.learn(total_timesteps=2, reset_num_timesteps=False)
    assert seen[0] == SESSION_SEED_STRIDE, seen


# ------------------------------------------------------------------------------------------------------
# Final review #1: claiming a run directory must not strand it. On this shared GPU host the likeliest failures
# (foreign job, bad --scene-config, launch failure) happen before the first checkpoint.
# ------------------------------------------------------------------------------------------------------
def test_a_fresh_start_that_never_checkpointed_can_be_started_again(tmp_path):
    from autofly_ue5.expert.train import prepare_run_root

    assert prepare_run_root(tmp_path, resume=False, reward_version="v2", seed=0) == 0
    # ... that session died before its first checkpoint; nothing in the directory is worth protecting
    assert prepare_run_root(tmp_path, resume=False, reward_version="v2", seed=0) == 0
    assert len(json.loads((tmp_path / "sessions.json").read_text())["sessions"]) == 1


def test_resume_without_a_checkpoint_refuses_before_recording_a_session(tmp_path):
    from autofly_ue5.expert.train import prepare_run_root

    prepare_run_root(tmp_path, resume=False, reward_version="v2", seed=0)
    with pytest.raises(RuntimeError, match="checkpoint"):
        prepare_run_root(tmp_path, resume=True, reward_version="v2", seed=0)
    assert len(json.loads((tmp_path / "sessions.json").read_text())["sessions"]) == 1


def test_a_legacy_run_directory_is_refused_without_suggesting_resume(tmp_path):
    from autofly_ue5.expert.train import prepare_run_root

    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "rl_model_200000_steps.zip").write_text("run 1")
    with pytest.raises(RuntimeError) as err:
        prepare_run_root(tmp_path, resume=False, reward_version="v2", seed=0)
    assert "--resume" not in str(err.value) and "--run-root" in str(err.value)


@pytest.mark.parametrize("bad", [["--scene", "s99"], ["--scene-config", "no_such_config.jsonc"]])
def test_a_bad_scene_or_scene_config_fails_before_claiming_the_run_root(tmp_path, bad, monkeypatch):
    from autofly_ue5.expert.train import main
    from autofly_ue5.sim import airsim_backend

    def no_launch(self, map_path, instance):  # this test must never start a real simulator, whatever main() does
        raise AssertionError("main() tried to launch a simulator instead of refusing up front")

    monkeypatch.setattr(airsim_backend.ProjectAirSimSimulator, "launch", no_launch)

    code = main([*bad, "--run-root", str(tmp_path / "run"), "--out", str(tmp_path / "out.json"), "--device", "cpu"])
    assert code == 2
    assert not (tmp_path / "run" / "sessions.json").exists()
    assert not (tmp_path / "out.json").exists()


def test_a_checkpointed_run_resumes_with_its_filtering_buffer_gradient_steps_and_fresh_seeds(tmp_path, monkeypatch):
    # The whole offline resume path (final review recommendation): CheckpointCallback with the replay buffer, then
    # SAC.load + load_replay_buffer + the class check + re-seeding + learn(reset_num_timesteps=False).
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    from autofly_ue5.expert.resilient import FaultFilteringDictReplayBuffer
    from autofly_ue5.expert.seeds import SESSION_SEED_STRIDE, session_seed_base
    from autofly_ue5.expert.train import build_model, newest_checkpoint, replay_buffer_for, reseed_resumed_model

    model = build_model(_vec(session_seed_base(0, 0)), device="cpu", buffer_size=100, learning_starts=1000, seed=0, verbose=0)
    model.learn(total_timesteps=6, callback=CheckpointCallback(save_freq=6, save_path=str(tmp_path), save_replay_buffer=True))
    checkpoint = newest_checkpoint(tmp_path)
    assert checkpoint is not None and replay_buffer_for(checkpoint).is_file()

    seen = _record_seeds(monkeypatch)
    resumed = SAC.load(checkpoint, env=_vec(session_seed_base(0, 1)), device="cpu", gradient_steps=-1)
    resumed.load_replay_buffer(replay_buffer_for(checkpoint))
    assert isinstance(resumed.replay_buffer, FaultFilteringDictReplayBuffer)
    # 5, not 6: SB3 runs callbacks (the checkpoint) before storing the step's transition.
    assert resumed.replay_buffer.pos == 5 and resumed.gradient_steps == -1
    reseed_resumed_model(resumed, seed=0, session=1)
    resumed.learn(total_timesteps=4, reset_num_timesteps=False)
    assert resumed.num_timesteps == 10
    assert seen[0] == SESSION_SEED_STRIDE and seen[1] == session_seed_base(0, 1), seen
