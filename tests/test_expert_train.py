"""Offline tests for autofly_ue5.expert.train: config, seeding, resume logic, and the resilience wrapper
(hazard #4) against FakeSimulator (spec §8, Task 8). No GPU or live simulator required."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pynng.exceptions import ConnectionReset as NngConnectionReset
from pynng.exceptions import NNGException
from pynng.exceptions import Timeout as NngTimeout

from autofly_ue5.sim.airsim_backend import CameraPoseError, CommandTimeoutError, StaleStateError, StepTimingError
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import CONTROL_DT_S
from tests.test_expert_episode import scene_and_layout

# All five step-time hazards ResilientAutoFlyEnv must recover from -- the spec's four documented siblings
# plus the raw NNG transport timeout found live during this task's own shakedown (train.py's docstring has
# the full story). Shared here so both parametrized tests below stay in sync with train.py's own set.
ALL_STEP_FAULTS = [CameraPoseError, StepTimingError, StaleStateError, CommandTimeoutError, NngTimeout, NngConnectionReset]
# pynng's own errno for each exception it raises (pynng.exceptions.EXCEPTION_MAP).
_NNG_ERRNO = {NngTimeout: 5, NngConnectionReset: 19}


# ------------------------------------------------------------------------------------------------------
# The brief's own tests
# ------------------------------------------------------------------------------------------------------
def test_target_entropy_is_minus_three():
    from autofly_ue5.expert.train import build_model
    from tests.test_expert_env import make_env

    m = build_model(make_env(), device="cpu", buffer_size=200, learning_starts=10)
    assert float(m.target_entropy) == -3.0


def test_seed_bases_are_disjoint_across_workers():
    from autofly_ue5.expert.train import worker_seed_base

    bases = [worker_seed_base(i) for i in range(8)]
    assert len(set(bases)) == 8
    # Each worker owns a range wide enough that two workers cannot collide within a run.
    assert min(b2 - b1 for b1, b2 in zip(sorted(bases), sorted(bases)[1:])) >= 1_000_000


def test_evaluation_seed_base_is_disjoint_from_every_training_worker():
    from autofly_ue5.expert.train import EVAL_SEED_BASE, worker_seed_base

    assert all(EVAL_SEED_BASE > worker_seed_base(i) + 1_000_000 for i in range(64))


def test_resume_picks_the_newest_checkpoint(tmp_path):
    from autofly_ue5.expert.train import newest_checkpoint

    (tmp_path / "rl_model_10000_steps.zip").write_text("a")
    (tmp_path / "rl_model_90000_steps.zip").write_text("b")
    (tmp_path / "rl_model_200000_steps.zip").write_text("c")
    assert newest_checkpoint(tmp_path).name == "rl_model_200000_steps.zip", "sort by step count, not by string"


def test_a_short_run_against_the_fake_learns_without_crashing(tmp_path):
    from autofly_ue5.expert.train import build_model
    from tests.test_expert_env import make_env

    m = build_model(make_env(), device="cpu", buffer_size=500, learning_starts=20, batch_size=8)
    m.learn(total_timesteps=60)
    assert m.num_timesteps >= 60


# ------------------------------------------------------------------------------------------------------
# Extra coverage: resume plumbing details the brief's tests don't pin down
# ------------------------------------------------------------------------------------------------------
def test_newest_checkpoint_ignores_replay_buffer_files(tmp_path):
    from autofly_ue5.expert.train import newest_checkpoint

    (tmp_path / "rl_model_10000_steps.zip").write_text("a")
    (tmp_path / "rl_model_replay_buffer_200000_steps.pkl").write_text("b")
    assert newest_checkpoint(tmp_path).name == "rl_model_10000_steps.zip"


def test_newest_checkpoint_with_no_checkpoints_is_none(tmp_path):
    from autofly_ue5.expert.train import newest_checkpoint

    assert newest_checkpoint(tmp_path) is None


def test_replay_buffer_for_derives_the_matching_pickle_name():
    from autofly_ue5.expert.train import replay_buffer_for

    got = replay_buffer_for(Path("/x/rl_model_90000_steps.zip"))
    assert got.name == "rl_model_replay_buffer_90000_steps.pkl"


def test_replay_buffer_for_rejects_a_non_checkpoint_name():
    from autofly_ue5.expert.train import replay_buffer_for

    with pytest.raises(ValueError):
        replay_buffer_for(Path("/x/not_a_checkpoint.zip"))


def test_scene_and_layout_finds_s01():
    from autofly_ue5.expert.train import scene_and_layout as train_scene_and_layout

    scene, layout = train_scene_and_layout("s01")
    assert scene.id == "s01"
    assert layout.bounds.width == 70.0 and layout.bounds.height == 70.0


def test_scene_and_layout_raises_a_clear_error_for_an_unknown_scene():
    from autofly_ue5.expert.train import scene_and_layout as train_scene_and_layout

    with pytest.raises(FileNotFoundError):
        train_scene_and_layout("s99")


# ------------------------------------------------------------------------------------------------------
# Hazard #4: ResilientAutoFlyEnv's fault handling, exercised with an injectable fake backend.
# ------------------------------------------------------------------------------------------------------
class _FlakyFakeSimulator(FakeSimulator):
    """A FakeSimulator that raises `error` on chosen (1-indexed) call numbers of reset()/step(), then
    behaves normally. The five backend hazards this trainer must survive only ever occur against the real
    ProjectAirSimSimulator (or, for NngTimeout, the third-party client library underneath it); this stands
    in for one without needing a live GPU/simulator."""

    def __init__(self, *args, fail_on_reset_calls=(), fail_on_step_calls=(), error=CameraPoseError, **kwargs):
        super().__init__(*args, **kwargs)
        self._fail_on_reset_calls = set(fail_on_reset_calls)
        self._fail_on_step_calls = set(fail_on_step_calls)
        self._error = error
        self._reset_call_count = 0
        self._step_call_count = 0
        self.close_count = 0

    def _raise(self, msg: str):
        # pynng's NNGException subclasses (unlike our own error types) require an `errno` argument.
        if issubclass(self._error, NNGException):
            raise self._error(msg, _NNG_ERRNO[self._error])
        raise self._error(msg)

    def close(self):
        self.close_count += 1
        super().close()

    def reset(self, pose):
        self._reset_call_count += 1
        if self._reset_call_count in self._fail_on_reset_calls:
            self._raise(f"injected reset failure on call {self._reset_call_count}")
        return super().reset(pose)

    def step(self, dt=CONTROL_DT_S):
        self._step_call_count += 1
        if self._step_call_count in self._fail_on_step_calls:
            self._raise(f"injected step failure on call {self._step_call_count}")
        return super().step(dt)


def _sequenced_factory(configs: list[dict]):
    """A sim_factory that returns a fresh _FlakyFakeSimulator per call, cycling through `configs` (holding
    on the last one once exhausted) -- models a relaunch producing a genuinely fresh connection/process."""
    calls = {"n": 0}

    def factory():
        cfg = configs[min(calls["n"], len(configs) - 1)]
        calls["n"] += 1
        return _FlakyFakeSimulator(**cfg)

    factory.call_count = lambda: calls["n"]
    return factory


def _make_resilient(sim_factory, **wrapper_kwargs):
    import tempfile

    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.train import ResilientAutoFlyEnv

    # A relaunch stops its own slot's recorded simulator; point that bookkeeping at a scratch directory so no test
    # ever reads or signals the real runs/sim (a live training run's simulators live there).
    wrapper_kwargs.setdefault("sim_root", Path(tempfile.mkdtemp(prefix="autofly_sim_")))
    scene, layout = scene_and_layout()
    base = AutoFlyEnv(scene, layout, sim_factory, map_path="/Game/AutoFly/Maps/S01", instance=0)
    return ResilientAutoFlyEnv(base, instance=0, **wrapper_kwargs)


@pytest.mark.parametrize("error", ALL_STEP_FAULTS)
def test_resilient_env_retries_reset_in_place_and_recovers(error):
    # Deterministic, offline proof that each of the four backend hazards is actually handled -- not
    # something we wait to observe by luck during a live run (Task 7 measured them as "roughly one per a
    # few hundred to ~1000 steps", which is not a guarantee any given window hits all four).
    factory = _sequenced_factory([{"fail_on_reset_calls": (1,), "error": error}])
    env = _make_resilient(factory, max_reset_attempts=3, max_relaunch_attempts=1)

    obs, info = env.reset(seed=1)

    assert env.observation_space.contains(obs)
    assert env.fault_counts[error.__name__] == 1
    assert env.recovered_counts[error.__name__] == 1
    assert env.relaunch_count == 0, "one in-place retry should be enough; no relaunch needed"
    # Every other known fault type must still show an explicit 0, not be silently absent.
    from autofly_ue5.expert.train import KNOWN_FAULT_NAMES

    for other in KNOWN_FAULT_NAMES:
        if other != error.__name__:
            assert env.fault_counts[other] == 0


def test_fault_and_recovery_events_are_logged_not_just_counted(capsys):
    # Coordinator review (Task 8): a live shakedown's aggregate "10 faults / 5 recovered" could not be
    # distinguished from "half the retries just don't work" vs. "one relaunch's follow-up attempt happened
    # to hit an unrelated crash" without a per-event log. Every catch, every successful recovery, and every
    # relaunch must be individually greppable from stderr afterward, not just reflected in the final counts.
    factory = _sequenced_factory([
        {"fail_on_reset_calls": set(range(1, 10)), "error": CameraPoseError},  # exhausts round 0 entirely
        {},  # round 1's fresh connection succeeds immediately
    ])
    env = _make_resilient(factory, max_reset_attempts=2, max_relaunch_attempts=1)

    env.reset(seed=1)
    err = capsys.readouterr().err

    assert err.count("FAULT instance 0: caught CameraPoseError during reset()") == 2
    assert "RELAUNCH instance 0: relaunching (this will be relaunch #1)" in err
    assert "RECOVERED instance 0: reset() succeeded after 2 fault(s)" in err
    # The exception's own message (a real CameraPoseError's carries "X.XXX m from kinematics ...") must be
    # in the log line itself, not just the class name -- coordinator review: "the exception carries the
    # actual error in metres but the wrapper logs only its own line", so the magnitude was ungreppable.
    assert "injected reset failure on call 1" in err


def test_step_fault_log_line_includes_the_exceptions_own_message(capsys):
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.train import ResilientAutoFlyEnv

    scene, layout = scene_and_layout()
    sim = _FlakyFakeSimulator(fail_on_step_calls=(2,), error=CameraPoseError)
    base = AutoFlyEnv(scene, layout, lambda: sim, map_path="/Game/AutoFly/Maps/S01", instance=0)
    env = ResilientAutoFlyEnv(base, instance=0)

    env.reset(seed=1)
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    err = capsys.readouterr().err

    assert "injected step failure on call 2" in err


def test_resilient_env_retries_reset_when_episode_setup_raises(monkeypatch):
    # EpisodeSetupError is raised by sample_setup() (episode.py) before any Simulator call happens, so it
    # cannot be injected through a Simulator double the way the other four can; monkeypatch the name
    # AutoFlyEnv.reset() actually calls (env.py's own imported binding) instead. This does not edit any
    # closed file -- it is a per-test monkeypatch, undone automatically at teardown.
    import autofly_ue5.expert.env as env_module
    from autofly_ue5.expert.episode import EpisodeSetupError

    real_sample_setup = env_module.sample_setup
    calls = {"n": 0}

    def flaky_sample_setup(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise EpisodeSetupError("injected: rejection sampling exhausted")
        return real_sample_setup(*args, **kwargs)

    monkeypatch.setattr(env_module, "sample_setup", flaky_sample_setup)

    env = _make_resilient(FakeSimulator, max_reset_attempts=3, max_relaunch_attempts=1)
    obs, info = env.reset(seed=1)

    assert env.observation_space.contains(obs)
    assert env.fault_counts["EpisodeSetupError"] == 1
    assert env.recovered_counts["EpisodeSetupError"] == 1
    assert env.relaunch_count == 0
    assert calls["n"] == 2, "the retry must actually call sample_setup() again, not just swallow the error"


@pytest.mark.parametrize("error", ALL_STEP_FAULTS)
def test_resilient_env_step_fault_truncates_and_recovers(error):
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.train import ResilientAutoFlyEnv

    scene, layout = scene_and_layout()
    # AutoFlyEnv.reset() itself issues one extra fake step() (the post-spawn re-render); fail the SECOND
    # step() call overall so reset() succeeds cleanly and the injected fault lands on step() as intended.
    sim = _FlakyFakeSimulator(fail_on_step_calls=(2,), error=error)
    base = AutoFlyEnv(scene, layout, lambda: sim, map_path="/Game/AutoFly/Maps/S01", instance=0)
    env = ResilientAutoFlyEnv(base, instance=0)

    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert not terminated and truncated, "a fault must end the episode by truncation, not termination"
    assert reward == 0.0
    assert env.observation_space.contains(obs)
    assert info["sim_fault"] == error.__name__
    assert info["is_success"] is False
    assert env.fault_counts[error.__name__] == 1
    assert env.recovered_counts[error.__name__] == 1
    assert env.relaunch_count == 0

    # The wrapper must actually be usable afterward -- prove recovery, don't just assert it was attempted.
    obs2, reward2, terminated2, truncated2, info2 = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.isfinite(reward2)
    assert info2["outcome"] == "running"


def test_resilient_env_relaunches_after_exhausting_in_place_retries():
    # The first connection fails every reset() attempt (all max_reset_attempts of them); the second
    # (post-relaunch) connection works normally.
    factory = _sequenced_factory([
        {"fail_on_reset_calls": set(range(1, 10)), "error": CameraPoseError},
        {},
    ])
    env = _make_resilient(factory, max_reset_attempts=2, max_relaunch_attempts=2)

    obs, info = env.reset(seed=1)

    assert env.observation_space.contains(obs)
    assert env.relaunch_count == 1
    assert env.fault_counts["CameraPoseError"] == 2, "both in-place attempts against the broken connection must be counted"
    assert env.recovered_counts["CameraPoseError"] == 2, "the eventual success recovers every prior fault, not just the last"


def test_resilient_env_gives_up_after_exhausting_every_relaunch_attempt():
    factory = _sequenced_factory([{"fail_on_reset_calls": set(range(1, 100)), "error": CameraPoseError}])
    env = _make_resilient(factory, max_reset_attempts=2, max_relaunch_attempts=2)

    with pytest.raises(RuntimeError, match="giving up"):
        env.reset(seed=1)

    assert env.relaunch_count == 2, "must relaunch exactly max_relaunch_attempts times before giving up"


def test_resilient_env_relaunch_closes_the_old_connection():
    instances: list[_FlakyFakeSimulator] = []

    def factory():
        sim = _FlakyFakeSimulator(fail_on_reset_calls=({1} if not instances else set()), error=CameraPoseError)
        instances.append(sim)
        return sim

    env = _make_resilient(factory, max_reset_attempts=1, max_relaunch_attempts=1)
    env.reset(seed=1)

    assert len(instances) == 2, "a relaunch must build a brand new Simulator, not reuse the broken one"
    assert instances[0].close_count == 1, "the broken connection must actually be closed during relaunch"


def test_get_fault_summary_and_combine_fault_summaries():
    from autofly_ue5.expert.train import KNOWN_FAULT_NAMES, combine_fault_summaries

    factory = _sequenced_factory([{"fail_on_reset_calls": (1,), "error": CameraPoseError}])
    env = _make_resilient(factory, max_reset_attempts=3, max_relaunch_attempts=1)
    env.reset(seed=1)

    zero_except_camera_pose = {name: 0 for name in KNOWN_FAULT_NAMES}
    summary = env.get_fault_summary()
    assert summary == {
        "instance": 0,
        "fault_counts": {**zero_except_camera_pose, "CameraPoseError": 1},
        "recovered_counts": {**zero_except_camera_pose, "CameraPoseError": 1},
        "relaunch_count": 0,
    }

    combined = combine_fault_summaries([
        {"fault_counts": {"CameraPoseError": 2}, "recovered_counts": {"CameraPoseError": 2}, "relaunch_count": 0},
        {"fault_counts": {"CameraPoseError": 1, "StepTimingError": 1}, "recovered_counts": {"CameraPoseError": 1}, "relaunch_count": 1},
    ])
    assert combined["fault_counts"] == {**zero_except_camera_pose, "CameraPoseError": 3, "StepTimingError": 1}
    assert combined["recovered_counts"] == {**zero_except_camera_pose, "CameraPoseError": 3}
    assert combined["relaunch_count"] == 1


def test_fault_counters_are_explicit_zeros_when_nothing_ever_faults():
    # The coordinator's own concern: a 12-hour run that never hits any fault must be distinguishable, in
    # the run record, from one whose counting is silently broken -- so the zeros must be present keys, not
    # an empty {} that could mean either.
    from autofly_ue5.expert.train import KNOWN_FAULT_NAMES, combine_fault_summaries

    env = _make_resilient(FakeSimulator)
    env.reset(seed=1)

    summary = env.get_fault_summary()
    assert summary["fault_counts"] == {name: 0 for name in KNOWN_FAULT_NAMES}
    assert summary["recovered_counts"] == {name: 0 for name in KNOWN_FAULT_NAMES}

    combined = combine_fault_summaries([])  # e.g. an early failure before any env was ever built
    assert combined["fault_counts"] == {name: 0 for name in KNOWN_FAULT_NAMES}
    assert combined["recovered_counts"] == {name: 0 for name in KNOWN_FAULT_NAMES}
    assert combined["relaunch_count"] == 0


# ------------------------------------------------------------------------------------------------------
# Checkpoint retention: keep every model .zip, prune all but the newest DEFAULT_KEEP_REPLAY_BUFFERS
# replay-buffer pickles.
# ------------------------------------------------------------------------------------------------------
def _touch(path: Path, size: int = 1) -> Path:
    path.write_bytes(b"x" * size)
    return path


def test_prune_old_replay_buffers_keeps_only_the_newest_two(tmp_path):
    from autofly_ue5.expert.train import prune_old_replay_buffers

    for n in (1000, 2000, 3000, 4000):
        _touch(tmp_path / f"rl_model_replay_buffer_{n}_steps.pkl")
        _touch(tmp_path / f"rl_model_{n}_steps.zip")  # model checkpoints must never be touched

    deleted = prune_old_replay_buffers(tmp_path, keep=2)

    remaining_buffers = sorted(p.name for p in tmp_path.glob("rl_model_replay_buffer_*_steps.pkl"))
    assert remaining_buffers == ["rl_model_replay_buffer_3000_steps.pkl", "rl_model_replay_buffer_4000_steps.pkl"]
    assert {p.name for p in deleted} == {"rl_model_replay_buffer_1000_steps.pkl", "rl_model_replay_buffer_2000_steps.pkl"}
    remaining_models = sorted(p.name for p in tmp_path.glob("rl_model_*_steps.zip"))
    assert remaining_models == [f"rl_model_{n}_steps.zip" for n in (1000, 2000, 3000, 4000)], "model checkpoints must be kept forever"


def test_prune_old_replay_buffers_is_a_noop_with_at_most_keep_files(tmp_path):
    from autofly_ue5.expert.train import prune_old_replay_buffers

    _touch(tmp_path / "rl_model_replay_buffer_1000_steps.pkl")
    assert prune_old_replay_buffers(tmp_path, keep=2) == []
    assert (tmp_path / "rl_model_replay_buffer_1000_steps.pkl").exists()


def test_prune_old_replay_buffers_with_a_missing_directory_is_a_noop(tmp_path):
    from autofly_ue5.expert.train import prune_old_replay_buffers

    assert prune_old_replay_buffers(tmp_path / "does_not_exist", keep=2) == []


def test_prune_old_replay_buffers_callback_fires_on_the_save_cadence(tmp_path):
    from autofly_ue5.expert.train import PruneOldReplayBuffersCallback

    for n in (10, 20, 30):
        _touch(tmp_path / f"rl_model_replay_buffer_{n}_steps.pkl")

    cb = PruneOldReplayBuffersCallback(save_freq=5, checkpoints_dir=tmp_path, keep=1)
    cb.n_calls = 4
    cb._on_step()  # not a multiple of save_freq: must not prune yet
    assert len(list(tmp_path.glob("rl_model_replay_buffer_*_steps.pkl"))) == 3

    cb.n_calls = 5
    cb._on_step()  # a multiple of save_freq: must prune down to `keep`
    remaining = list(tmp_path.glob("rl_model_replay_buffer_*_steps.pkl"))
    assert len(remaining) == 1 and remaining[0].name == "rl_model_replay_buffer_30_steps.pkl"


# ------------------------------------------------------------------------------------------------------
# make_vec_env: n=1 must be a DummyVecEnv (no IPC to hang on); n>1 a SubprocVecEnv with staggered launch.
# ------------------------------------------------------------------------------------------------------
def test_make_vec_env_n1_is_a_dummy_vec_env_and_works_end_to_end(tmp_path):
    from autofly_ue5.expert.train import make_vec_env, worker_seed_base

    scene, layout = scene_and_layout()
    vec_env = make_vec_env(
        scene, layout, 1, map_path="/Game/AutoFly/Maps/S01", monitor_dir=tmp_path,
        seed_base_fn=worker_seed_base, sim_factory=FakeSimulator, sim_root=tmp_path / "sim",
    )
    try:
        assert isinstance(vec_env, DummyVecEnv)
        obs = vec_env.reset()
        assert set(obs.keys()) == {"depth", "vector"}
        actions = np.stack([vec_env.action_space.sample()])
        obs, rewards, dones, infos = vec_env.step(actions)
        assert "is_success" in infos[0]
        summaries = vec_env.env_method("get_fault_summary")
        assert summaries[0]["relaunch_count"] == 0
    finally:
        vec_env.close()


def test_make_vec_env_n_greater_than_1_is_a_subproc_vec_env(tmp_path):
    from autofly_ue5.expert.train import make_vec_env, worker_seed_base

    scene, layout = scene_and_layout()
    vec_env = make_vec_env(
        scene, layout, 2, map_path="/Game/AutoFly/Maps/S01", monitor_dir=tmp_path,
        seed_base_fn=worker_seed_base, sim_factory=FakeSimulator, sim_root=tmp_path / "sim",
    )
    try:
        assert isinstance(vec_env, SubprocVecEnv)
        obs = vec_env.reset()
        assert obs["vector"].shape[0] == 2
    finally:
        vec_env.close()


def test_two_workers_via_make_vec_env_get_disjoint_seed_bases(tmp_path):
    # Regression net for hazard #1 inside make_vec_env itself, not just worker_seed_base() in isolation.
    from autofly_ue5.expert.train import make_vec_env, worker_seed_base

    scene, layout = scene_and_layout()
    vec_env = make_vec_env(
        scene, layout, 2, map_path="/Game/AutoFly/Maps/S01", monitor_dir=tmp_path,
        seed_base_fn=worker_seed_base, sim_factory=FakeSimulator, sim_root=tmp_path / "sim",
    )
    try:
        obs = vec_env.reset()
        assert not np.array_equal(obs["vector"][0], obs["vector"][1]), "vectorised workers must not fly identical episodes"
    finally:
        vec_env.close()


# ------------------------------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------------------------------
def test_stop_on_wall_clock_returns_false_once_the_budget_elapses():
    from autofly_ue5.expert.train import StopOnWallClock

    cb = StopOnWallClock(hours=1e-9, verbose=0)
    time.sleep(0.01)
    assert cb._on_step() is False


def test_stop_on_wall_clock_returns_true_before_the_budget():
    from autofly_ue5.expert.train import StopOnWallClock

    cb = StopOnWallClock(hours=1.0, verbose=0)
    assert cb._on_step() is True


def test_outcome_histogram_tallies_only_completed_episodes():
    from autofly_ue5.expert.train import OutcomeHistogramCallback

    cb = OutcomeHistogramCallback()
    cb.locals = {"infos": [{"outcome": "success"}, {"outcome": "running"}], "dones": [True, False]}
    cb._on_step()
    cb.locals = {"infos": [{"outcome": "collision"}], "dones": [True]}
    cb._on_step()

    assert cb.histogram == {"success": 1, "collision": 1}


def test_outcome_histogram_buckets_a_fault_truncation_under_its_own_hazard_not_running():
    # Live finding (Task 8 shakedown3): a ResilientAutoFlyEnv fault-truncation's info["outcome"] is the
    # FRESH episode's own "running", not the truncated episode's real fate -- info["sim_fault"] is. Without
    # preferring it, every fault-truncated episode is indistinguishable from a genuine "running" bucket.
    from autofly_ue5.expert.train import OutcomeHistogramCallback

    cb = OutcomeHistogramCallback()
    cb.locals = {
        "infos": [{"outcome": "running", "sim_fault": "CameraPoseError"}, {"outcome": "success"}],
        "dones": [True, True],
    }
    cb._on_step()

    assert cb.histogram == {"CameraPoseError": 1, "success": 1}


# ------------------------------------------------------------------------------------------------------
# C1 (2026-09-24 review): a relaunch or teardown stops only its own simulator, and a worker never hangs.
# ------------------------------------------------------------------------------------------------------
class _ProcessBackedFake(_FlakyFakeSimulator):
    """A fake whose launch() starts a real, recorded `sleep` process in its slot under `root`, and whose close()
    stops it the way the real backend does -- so slot bookkeeping is exercised against real processes."""

    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)
        self._root = root
        self.proc = None

    def launch(self, map_path, instance):
        import os

        from autofly_ue5.sim.process import launch_process, ports_for_instance

        super().launch(map_path, instance)
        self.proc = launch_process(["sleep", "300"], instance, ports_for_instance(60 + instance), dict(os.environ),
                                   run_root=self._root)

    def close(self):
        from autofly_ue5.sim.process import stop

        super().close()
        if self.proc is not None:
            stop(self.proc.instance, grace_s=2.0, run_root=self._root, expected_pid=self.proc.pid)


def _wrap(factory, instance, sim_root, **kwargs):
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.resilient import ResilientAutoFlyEnv

    scene, layout = scene_and_layout()
    base = AutoFlyEnv(scene, layout, factory, map_path="/Game/AutoFly/Maps/S01", instance=instance)
    return ResilientAutoFlyEnv(base, instance=instance, sim_root=sim_root, **kwargs)


def test_a_relaunch_stops_only_its_own_slot_never_a_siblings_simulator(tmp_path):
    # The Task 8 run's feedback loop, in miniature: the training (inst0) and eval (inst1) simulators killed each
    # other at every evaluation because a relaunch swept every simulator on the host.
    from autofly_ue5.sim.process import is_alive, stop

    sibling = _wrap(lambda: _ProcessBackedFake(tmp_path), 2, tmp_path)
    sibling.reset(seed=1)
    sibling_pid = sibling.unwrapped._sim.proc.pid
    broken = _wrap(lambda: _ProcessBackedFake(tmp_path, fail_on_reset_calls=set(range(1, 100))), 1, tmp_path,
                   max_reset_attempts=1, max_relaunch_attempts=1)
    try:
        with pytest.raises(RuntimeError, match="giving up"):
            broken.reset(seed=1)
        assert broken.relaunch_count == 1
        assert is_alive(sibling_pid), "a relaunch in slot 1 stopped the simulator in slot 2"
    finally:
        for inst in (1, 2):
            stop(inst, grace_s=2.0, run_root=tmp_path)


def test_a_close_that_returns_after_the_relaunch_leaves_the_new_simulator_in_place(tmp_path):
    import threading
    import time

    release = threading.Event()

    class SlowToClose(_FlakyFakeSimulator):
        def close(self):
            release.wait(5.0)
            super().close()

    sims = [SlowToClose(fail_on_reset_calls=set(range(1, 100))), _FlakyFakeSimulator()]
    env = _make_resilient(lambda: sims.pop(0), max_reset_attempts=1, max_relaunch_attempts=1, close_timeout_s=0.2)
    env.reset(seed=1)  # round 0 fails; the relaunch abandons the hung close() after 0.2 s; round 1 succeeds
    replacement = env.unwrapped._sim
    release.set()
    time.sleep(0.3)  # let the abandoned close() finish
    assert env.unwrapped._sim is replacement, "the late close() detached the simulator that replaced it"
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_a_launch_failure_moves_straight_to_the_next_relaunch_round():
    from autofly_ue5.sim.process import SimReadyTimeout

    launches = {"n": 0}

    class ReadyOnThirdLaunch(FakeSimulator):
        def launch(self, map_path, instance):
            launches["n"] += 1
            if launches["n"] <= 2:
                raise SimReadyTimeout("ports never opened")
            super().launch(map_path, instance)

    env = _make_resilient(ReadyOnThirdLaunch, max_reset_attempts=5, max_relaunch_attempts=3)
    obs, _ = env.reset(seed=1)

    assert env.observation_space.contains(obs)
    assert launches["n"] == 3, "a failed launch costs one attempt per relaunch round, not max_reset_attempts of them"
    assert env.fault_counts["SimReadyTimeout"] == 2 and env.recovered_counts["SimReadyTimeout"] == 2
    assert env.relaunch_count == 2


def test_worker_mode_exits_the_process_on_an_error_it_cannot_recover(tmp_path):
    # In a SubprocVecEnv worker an uncaught exception does not end the process: projectairsim's non-daemon thread
    # blocks interpreter shutdown, the worker hangs, and so does the training process waiting on its pipe.
    import os
    import subprocess
    import sys
    import textwrap

    from autofly_ue5.paths import ROOT

    code = textwrap.dedent(f"""
        import sys, threading
        sys.path.insert(0, {str(ROOT)!r})
        threading.Thread(target=threading.Event().wait, daemon=False).start()  # like projectairsim's client thread
        from autofly_ue5.expert.env import AutoFlyEnv
        from autofly_ue5.expert.resilient import ResilientAutoFlyEnv
        from autofly_ue5.sim.fake import FakeSimulator
        from tests.test_expert_episode import scene_and_layout

        class Busy(FakeSimulator):
            def launch(self, map_path, instance):
                raise RuntimeError("GPU busy: not a recoverable fault")

        scene, layout = scene_and_layout()
        env = ResilientAutoFlyEnv(AutoFlyEnv(scene, layout, Busy, map_path="/x", instance=0), instance=0,
                                  sim_root={str(tmp_path)!r}, worker_mode=True)
        env.reset(seed=1)
    """)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=env, cwd=ROOT)
    assert result.returncode == 1
    assert "GPU busy" in result.stderr


def test_teardown_stops_only_the_slots_it_is_given(tmp_path):
    import os

    from autofly_ue5.expert.vec import teardown
    from autofly_ue5.sim.process import is_alive, launch_process, ports_for_instance, stop

    a = launch_process(["sleep", "300"], 1, ports_for_instance(71), dict(os.environ), run_root=tmp_path)
    b = launch_process(["sleep", "300"], 2, ports_for_instance(72), dict(os.environ), run_root=tmp_path)
    try:
        results = teardown(None, [1], tmp_path)
        assert [r["instance"] for r in results] == [1]
        assert not is_alive(a.pid) and is_alive(b.pid)
    finally:
        stop(2, grace_s=2.0, run_root=tmp_path)


def test_make_vec_env_routes_client_logs_under_its_sim_root(tmp_path):
    from autofly_ue5.expert.vec import make_vec_env

    scene, layout = scene_and_layout()
    vec_env = make_vec_env(scene, layout, 1, map_path="/x", monitor_dir=tmp_path / "mon", sim_factory=FakeSimulator,
                           sim_root=tmp_path / "sim")
    try:
        assert (tmp_path / "sim" / "inst0" / "client.log").is_file()
    finally:
        vec_env.close()

