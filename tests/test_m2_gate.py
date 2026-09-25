"""Offline tests for the M2 exit gate (Task 9, spec Sec8/Sec9.5): report arithmetic, seed disjointness,
threshold logic, and the fault-vs-policy-outcome separation -- all against `FakeSimulator` with a scripted
dummy policy, never a trained model or a live simulator (no GPU required)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autofly_ue5.sim.airsim_backend import CameraPoseError
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import CONTROL_DT_S
from tests.test_expert_episode import scene_and_layout
from tests.test_expert_train import _FlakyFakeSimulator

MAP_PATH = "/Game/AutoFly/Maps/S01"


# ------------------------------------------------------------------------------------------------------
# The brief's own tests, verbatim.
# ------------------------------------------------------------------------------------------------------
def test_rates_sum_to_one():
    from autofly_ue5.expert.evaluate import EvalReport

    r = EvalReport.from_outcomes(["success"] * 190 + ["collision"] * 5 + ["timeout"] * 4 + ["out_of_bounds"])
    assert r.success_rate == 0.95
    assert r.success_rate + r.collision_rate + r.timeout_rate + r.out_of_bounds_rate == 1.0
    assert r.n_episodes == 200


def test_gate_needs_both_the_rate_and_the_episode_count():
    from scripts.m2_gate import gate_passes

    assert gate_passes(success_rate=0.96, n_episodes=200, faults_ok=True) is True
    assert gate_passes(success_rate=0.94, n_episodes=200, faults_ok=True) is False
    assert gate_passes(success_rate=0.99, n_episodes=150, faults_ok=True) is False, "needs >= 200 episodes"
    assert gate_passes(success_rate=0.99, n_episodes=200, faults_ok=False) is False


def test_an_unsuccessful_run_is_reported_as_failing_not_rounded_up():
    from scripts.m2_gate import gate_passes

    assert gate_passes(success_rate=0.9499, n_episodes=200, faults_ok=True) is False


# ------------------------------------------------------------------------------------------------------
# Extra coverage: EvalReport
# ------------------------------------------------------------------------------------------------------
def test_gate_passes_at_the_exact_boundary():
    from scripts.m2_gate import gate_passes

    assert gate_passes(success_rate=0.95, n_episodes=200, faults_ok=True) is True
    assert gate_passes(success_rate=0.95, n_episodes=199, faults_ok=True) is False


def test_from_outcomes_with_no_episodes_reports_zeros_not_a_crash():
    # A partial run that failed on its very first episode still needs a valid (if empty) EvalReport --
    # see EvaluationInterrupted below.
    from autofly_ue5.expert.evaluate import EvalReport

    r = EvalReport.from_outcomes([])
    assert r.n_episodes == 0
    assert r.success_rate == 0.0 and r.collision_rate == 0.0


def test_fault_counts_default_to_explicit_zeros_not_omitted():
    # Task 9 brief, gate item 6: "emit them as 0 rather than omitting them ... so a clean run is
    # distinguishable from broken counting."
    from autofly_ue5.expert.evaluate import EvalReport
    from autofly_ue5.expert.train import KNOWN_FAULT_NAMES

    r = EvalReport.from_outcomes(["success"] * 5)
    assert r.fault_counts == {name: 0 for name in KNOWN_FAULT_NAMES}
    assert r.episodes_retried == 0


def test_fault_summary_delta_isolates_one_combinations_share_of_a_reused_wrapper():
    from autofly_ue5.expert.evaluate import fault_summary_delta
    from autofly_ue5.expert.train import KNOWN_FAULT_NAMES

    zeros = {name: 0 for name in KNOWN_FAULT_NAMES}
    before = {"fault_counts": {**zeros, "CameraPoseError": 3}, "recovered_counts": {**zeros, "CameraPoseError": 3}, "relaunch_count": 1}
    after = {"fault_counts": {**zeros, "CameraPoseError": 5, "Timeout": 1}, "recovered_counts": {**zeros, "CameraPoseError": 5, "Timeout": 1}, "relaunch_count": 1}

    delta = fault_summary_delta(before, after)

    assert delta["fault_counts"] == {**zeros, "CameraPoseError": 2, "Timeout": 1}
    assert delta["recovered_counts"] == {**zeros, "CameraPoseError": 2, "Timeout": 1}
    assert delta["relaunch_count"] == 0


# ------------------------------------------------------------------------------------------------------
# A scripted dummy policy (matches SB3's `model.predict(obs, deterministic=...) -> (action, state)`),
# steering directly at the target from the privileged vector observation. test_expert_env.py's own
# test_flying_straight_at_a_target_succeeds already proves this exact strategy reaches SUCCESS against
# FakeSimulator; reused here as "a scripted dummy policy", per the brief, not a trained model.
# ------------------------------------------------------------------------------------------------------
class _StraightAtTargetModel:
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        vector = observation["vector"]
        bearing = math.atan2(float(vector[1]), float(vector[2]))
        yaw_rate = float(np.clip(bearing * 2.0, -1.0, 1.0))
        v = 2.0 if abs(bearing) < 0.3 else 0.3
        return np.array([v, yaw_rate, 0.0], dtype=np.float32), None


def _make_resilient_env(sim_factory, **wrapper_kwargs):
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.train import ResilientAutoFlyEnv

    scene, layout = scene_and_layout()
    base = AutoFlyEnv(scene, layout, sim_factory, map_path=MAP_PATH, instance=0)
    return ResilientAutoFlyEnv(base, instance=0, **wrapper_kwargs)


# ------------------------------------------------------------------------------------------------------
# evaluate_policy_episodes end to end against FakeSimulator.
# ------------------------------------------------------------------------------------------------------
def test_evaluate_policy_episodes_draws_the_requested_seed_stream():
    from autofly_ue5.expert.evaluate import evaluate_policy_episodes

    env = _make_resilient_env(FakeSimulator)
    seed_base = 100_000_000

    report = evaluate_policy_episodes(_StraightAtTargetModel(), env, n_episodes=5, seed_base=seed_base)

    assert report.n_episodes == 5
    assert [e["seed"] for e in report.per_episode] == list(range(seed_base, seed_base + 5))
    assert report.success_rate == 1.0, "a straight-line pilot must succeed against FakeSimulator every time"
    assert report.episodes_retried == 0
    assert all(e["return"] > 10.0 for e in report.per_episode), "a success earns the +10 bonus on top of progress"
    assert report.mean_return == pytest.approx(sum(e["return"] for e in report.per_episode) / 5)


def test_evaluate_policy_episodes_accepts_a_deterministic_flag():
    from autofly_ue5.expert.evaluate import evaluate_policy_episodes

    env = _make_resilient_env(FakeSimulator)
    report = evaluate_policy_episodes(_StraightAtTargetModel(), env, n_episodes=2, seed_base=1, deterministic=False)
    assert report.n_episodes == 2


def test_a_mid_episode_backend_fault_is_retried_on_the_same_seed_not_scored_as_a_failure():
    # Task 9 brief, gate item 6: a simulator hiccup must never be counted as a policy failure. Call #1 on
    # this sim is AutoFlyEnv.reset()'s own internal post-spawn re-render step (see env.py); call #2 is the
    # FIRST real policy-driven step -- exactly the pattern test_expert_train.py's own
    # test_resilient_env_step_fault_truncates_and_recovers uses to land the fault mid-episode, not mid-reset.
    from autofly_ue5.expert.evaluate import evaluate_policy_episodes

    factory = lambda: _FlakyFakeSimulator(fail_on_step_calls=(2,), error=CameraPoseError)
    env = _make_resilient_env(factory)

    report = evaluate_policy_episodes(_StraightAtTargetModel(), env, n_episodes=2, seed_base=42)

    assert report.n_episodes == 2, "a backend fault must not shrink the requested episode count"
    assert report.success_rate == 1.0, "the faulted attempt must be replayed, not scored as a failure"
    assert report.episodes_retried >= 1
    assert report.fault_counts["CameraPoseError"] >= 1
    assert [e["seed"] for e in report.per_episode] == [42, 43], "the retried episode keeps its own seed"


# ------------------------------------------------------------------------------------------------------
# EvaluationInterrupted: a genuinely broken backend must not be reported as a completed (or worse,
# passing) evaluation -- it must raise, carrying whatever DID complete, per gate item 7.
# ------------------------------------------------------------------------------------------------------
class _AlwaysFaultingSimulator(FakeSimulator):
    """Every step() raises -- models a backend so broken that ResilientAutoFlyEnv's own bounded retries
    (not just this module's episode-level ones) are exhausted, including inside reset()'s own post-spawn
    re-render step (env.py), so even reset() itself ultimately raises."""

    def step(self, dt: float = CONTROL_DT_S) -> int:
        raise CameraPoseError("permanently broken")


def test_a_catastrophically_broken_backend_raises_evaluation_interrupted_with_a_partial_report():
    from autofly_ue5.expert.evaluate import EvaluationInterrupted, evaluate_policy_episodes

    env = _make_resilient_env(_AlwaysFaultingSimulator, max_reset_attempts=1, max_relaunch_attempts=0)

    with pytest.raises(EvaluationInterrupted) as excinfo:
        evaluate_policy_episodes(_StraightAtTargetModel(), env, n_episodes=3, seed_base=7)

    err = excinfo.value
    assert err.partial_report.n_episodes == 0, "nothing completed -- the very first episode never finished"
    assert isinstance(err.cause, RuntimeError)
    assert "giving up" in str(err.cause)


def test_exhausting_episode_level_fault_retries_also_raises_evaluation_interrupted():
    # A different failure mode from the one above: ResilientAutoFlyEnv itself DOES recover each time (a
    # step fault truncates, then reset() succeeds), but every replay of the SAME seed faults again, so this
    # module's own max_fault_retries_per_episode bound (not the wrapper's) is what must fire.
    from autofly_ue5.expert.evaluate import EvaluationInterrupted, evaluate_policy_episodes

    # Fails on every even-numbered step() call -- guarantees the very first real policy step of every
    # single attempt faults (call #2, #4, #6, ... -- see the mid-episode fault test above for why call #2
    # is the first real step), while each attempt's own reset()-render step (an odd call) succeeds, so this
    # never escalates to a relaunch. (The wrapper no longer resets by itself after a fault: the harness's
    # next attempt is the only reset.)
    class _EveryOtherStepFaults(FakeSimulator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._n = 0

        def step(self, dt: float = CONTROL_DT_S) -> int:
            self._n += 1
            if self._n % 2 == 0:
                raise CameraPoseError("flaky")
            return super().step(dt)

    env = _make_resilient_env(_EveryOtherStepFaults)

    with pytest.raises(EvaluationInterrupted) as excinfo:
        evaluate_policy_episodes(
            _StraightAtTargetModel(), env, n_episodes=1, seed_base=99, max_fault_retries_per_episode=2,
        )

    err = excinfo.value
    assert err.partial_report.n_episodes == 0
    assert isinstance(err.cause, RuntimeError)
    assert "backend-fault retries" in str(err.cause)


# ------------------------------------------------------------------------------------------------------
# scripts/m2_gate.py's own pure helpers.
# ------------------------------------------------------------------------------------------------------
def test_parse_model_args_defaults_to_both_task_8_checkpoints():
    from scripts.m2_gate import parse_model_args

    models = parse_model_args(None, "s01")
    assert list(models) == ["best_model", "final"]
    assert models["best_model"].name == "best_model.zip" and "best" in models["best_model"].parts
    assert models["final"].name == "final.zip"


def test_parse_model_args_accepts_explicit_name_equals_path():
    from scripts.m2_gate import parse_model_args

    models = parse_model_args(["a=/x/a.zip", "b=/x/b.zip"], "s01")
    assert models == {"a": __import__("pathlib").Path("/x/a.zip"), "b": __import__("pathlib").Path("/x/b.zip")}


def test_parse_model_args_rejects_a_spec_without_equals():
    from scripts.m2_gate import parse_model_args

    with pytest.raises(ValueError):
        parse_model_args(["not_a_valid_spec"], "s01")


def test_combo_order_runs_every_checkpoints_deterministic_condition_before_any_stochastic_one():
    from pathlib import Path

    from scripts.m2_gate import combo_order

    models = {"best_model": Path("best.zip"), "final": Path("final.zip")}
    order = combo_order(models, ["deterministic", "stochastic"])

    assert order == [
        ("best_model", "deterministic"),
        ("final", "deterministic"),
        ("best_model", "stochastic"),
        ("final", "stochastic"),
    ], "priority order (a)(b)(c)(d) from the Task 9 brief"


def test_combo_order_respects_a_restricted_condition_list():
    from pathlib import Path

    from scripts.m2_gate import combo_order

    models = {"best_model": Path("best.zip")}
    assert combo_order(models, ["stochastic"]) == [("best_model", "stochastic")]


def test_load_throughput_projection_reads_task_7s_own_gate_file():
    from scripts.m2_gate import _load_throughput_projection

    projection = _load_throughput_projection()
    assert projection is not None
    assert projection["chosen_n"] == 1
    assert projection["measured_env_steps_per_s_total"] is not None
    assert projection["projection"]["assumed_n_scenes"] == 10


def test_load_throughput_projection_with_a_missing_file_is_none(tmp_path):
    from scripts.m2_gate import _load_throughput_projection

    assert _load_throughput_projection(tmp_path / "does_not_exist.json") is None


def test_parse_model_args_default_paths_follow_the_run_root():
    from pathlib import Path

    from scripts.m2_gate import parse_model_args

    models = parse_model_args(None, "s01", run_root=Path("/runs/expert/s01_r2"))
    assert models == {"best_model": Path("/runs/expert/s01_r2/best/best_model.zip"),
                      "final": Path("/runs/expert/s01_r2/final.zip")}
