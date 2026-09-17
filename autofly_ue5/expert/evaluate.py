"""Evaluation harness for a trained expert (spec Sec8/Sec9.5, Task 9's M2 gate).

Draws episodes from `reset(seed=seed_base + i)` for `i in range(n_episodes)` -- a fixed, reproducible
evaluation stream, disjoint from every training worker's own stream (`autofly_ue5.expert.train.
EVAL_SEED_BASE` / `worker_seed_base`; see that module's own disjointness test) -- and scores each with
`model.predict(obs, deterministic=...)`.

Backend faults, never policy failures (Task 9 brief, gate item 6): reuses `autofly_ue5.expert.train.
ResilientAutoFlyEnv` -- not a second retry wrapper -- for the five recoverable hazards Task 8 measured
live (`CameraPoseError`, `StepTimingError`, `StaleStateError`, `CommandTimeoutError`,
`pynng.exceptions.Timeout`). A fault DURING `reset()` is already retried transparently by that wrapper
with the SAME seed, so it never reaches this module. A fault mid-`step()` is different: the wrapper's own
step() contract truncates the CURRENT episode and internally calls `reset(seed=None)` to recover (see its
docstring) -- which means the observation handed back belongs to a fresh, wrongly-seeded episode, not the
one this harness asked for. This module detects that (`info["sim_fault"]` is set) and discards the whole
attempt, replaying the SAME seed from `env.reset(seed=seed)` again, so a simulator hiccup is never counted
as a success/collision/out_of_bounds/timeout outcome and every eval seed still gets a genuine policy
outcome. `episodes_retried` and `fault_counts` report exactly how often that happened.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Reused so a clean eval run's fault_counts still show an explicit 0 for every known hazard (not just the
# ones actually seen) -- exactly the same reasoning as train.py's own KNOWN_FAULT_NAMES-seeded counters, so
# "no faults happened" is distinguishable from "counting was silently broken" here too.
from autofly_ue5.expert.train import KNOWN_FAULT_NAMES

OUTCOME_KEYS = ("success", "collision", "timeout", "out_of_bounds")

DEFAULT_MAX_STEPS_PER_EPISODE = 1_000  # generous vs. AutoFlyEnv's own 300-step limit (spec Sec8): a
# defensive bound against a runaway episode, not a target any real episode should ever reach.
DEFAULT_MAX_FAULT_RETRIES_PER_EPISODE = 20  # Task 8 measured faults at roughly 1 per few hundred to ~1000
# steps; 20 in a row on ONE seed means the backend itself is broken, not bad luck -- raise, don't loop.


@dataclass(frozen=True)
class EvalReport:
    """The result of evaluating a policy over one or more episodes.

    `success_rate + collision_rate + timeout_rate + out_of_bounds_rate == 1.0` whenever `n_episodes > 0`
    (every completed episode lands in exactly one of `OUTCOME_KEYS`, per `reward.classify`'s exhaustive
    branches) -- `test_rates_sum_to_one` in tests/test_m2_gate.py pins this.
    """

    n_episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    out_of_bounds_rate: float
    mean_steps: float | None
    mean_final_distance_m: float | None
    per_episode: list[dict[str, Any]]
    episodes_retried: int = 0
    fault_counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in KNOWN_FAULT_NAMES})

    @classmethod
    def from_outcomes(
        cls,
        outcomes: list[str],
        *,
        per_episode: list[dict[str, Any]] | None = None,
        episodes_retried: int = 0,
        fault_counts: dict[str, int] | None = None,
    ) -> "EvalReport":
        n = len(outcomes)
        counts = Counter(outcomes)
        if per_episode is None:
            per_episode = [{"outcome": o} for o in outcomes]
        steps = [e["steps"] for e in per_episode if e.get("steps") is not None]
        dists = [e["final_distance_m"] for e in per_episode if e.get("final_distance_m") is not None]
        merged_faults = {name: 0 for name in KNOWN_FAULT_NAMES}
        merged_faults.update(fault_counts or {})
        return cls(
            n_episodes=n,
            success_rate=(counts.get("success", 0) / n) if n else 0.0,
            collision_rate=(counts.get("collision", 0) / n) if n else 0.0,
            timeout_rate=(counts.get("timeout", 0) / n) if n else 0.0,
            out_of_bounds_rate=(counts.get("out_of_bounds", 0) / n) if n else 0.0,
            mean_steps=(sum(steps) / len(steps)) if steps else None,
            mean_final_distance_m=(sum(dists) / len(dists)) if dists else None,
            per_episode=per_episode,
            episodes_retried=episodes_retried,
            fault_counts=merged_faults,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "timeout_rate": self.timeout_rate,
            "out_of_bounds_rate": self.out_of_bounds_rate,
            "mean_steps": self.mean_steps,
            "mean_final_distance_m": self.mean_final_distance_m,
            "episodes_retried": self.episodes_retried,
            "fault_counts": dict(self.fault_counts),
            "per_episode": self.per_episode,
        }


class EvaluationInterrupted(RuntimeError):
    """Raised when the episode loop cannot continue (the backend gave up beyond what `ResilientAutoFlyEnv`
    itself can recover from) -- carries whatever `EvalReport` could be built from the episodes that DID
    complete before that happened, so a caller can record a partial run honestly (Task 9 brief, gate item
    7) instead of losing every episode already collected."""

    def __init__(self, message: str, *, partial_report: EvalReport, cause: BaseException) -> None:
        super().__init__(message)
        self.partial_report = partial_report
        self.cause = cause


def _run_one_episode(model, env, seed: int, *, deterministic: bool, max_steps: int) -> dict[str, Any]:
    """One attempt at `seed`. Returns `{"fault": <name>}` if a backend hazard truncated it before a real
    outcome, or the completed episode's own record otherwise."""
    obs, info = env.reset(seed=seed)
    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        fault = info.get("sim_fault")
        if fault:
            return {"fault": fault}
        if terminated or truncated:
            return {
                "seed": seed,
                "outcome": info.get("outcome", "unknown"),
                "steps": info.get("steps", step),
                "final_distance_m": info.get("final_distance_m"),
                "is_success": bool(info.get("is_success", False)),
            }
    # AutoFlyEnv's own step_limit (default 300, spec Sec8) always truncates well before this bound;
    # reaching it means something is not honouring that contract -- report it, don't spin forever.
    return {
        "seed": seed,
        "outcome": "exceeded_max_steps",
        "steps": max_steps,
        "final_distance_m": info.get("final_distance_m"),
        "is_success": False,
    }


def _run_one_episode_with_retry(
    model, env, seed: int, *, deterministic: bool, max_steps: int, max_fault_retries: int,
) -> dict[str, Any]:
    fault_counts: Counter[str] = Counter()
    for _attempt in range(max_fault_retries + 1):
        result = _run_one_episode(model, env, seed, deterministic=deterministic, max_steps=max_steps)
        if "fault" in result:
            fault_counts[result["fault"]] += 1
            continue
        return {**result, "retries": sum(fault_counts.values()), "episode_fault_counts": dict(fault_counts)}
    raise RuntimeError(
        f"seed={seed} did not produce a real outcome after {max_fault_retries} backend-fault retries "
        f"(fault_counts={dict(fault_counts)}) -- this is a broken backend, not bad luck; it must be "
        f"investigated rather than silently reporting a policy outcome that never actually happened"
    )


def evaluate_policy_episodes(
    model,
    env,
    n_episodes: int,
    seed_base: int,
    *,
    deterministic: bool = True,
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE,
    max_fault_retries_per_episode: int = DEFAULT_MAX_FAULT_RETRIES_PER_EPISODE,
) -> EvalReport:
    """Runs `n_episodes` episodes at `reset(seed=seed_base + i)` for `i in range(n_episodes)`, scoring each
    with `model.predict(obs, deterministic=deterministic)`.

    `env` must be a `ResilientAutoFlyEnv` (directly, or wrapped further e.g. by `Monitor`) so a backend
    hazard mid-episode is retried here on the SAME seed instead of ending the run, hanging it, or -- worse
    -- being scored as a policy failure. Raises `EvaluationInterrupted` (carrying however many episodes DID
    complete, as a genuine `EvalReport`) if the backend itself gives up beyond what that wrapper can
    recover from -- see its own `RuntimeError("... giving up ...")`.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be > 0, got {n_episodes}")
    outcomes: list[str] = []
    per_episode: list[dict[str, Any]] = []
    fault_counts: Counter[str] = Counter()
    episodes_retried = 0

    for i in range(n_episodes):
        seed = seed_base + i
        try:
            record = _run_one_episode_with_retry(
                model, env, seed, deterministic=deterministic,
                max_steps=max_steps_per_episode, max_fault_retries=max_fault_retries_per_episode,
            )
        except Exception as err:
            partial = EvalReport.from_outcomes(
                outcomes, per_episode=per_episode, episodes_retried=episodes_retried, fault_counts=dict(fault_counts),
            )
            raise EvaluationInterrupted(
                f"evaluation stopped after {len(outcomes)}/{n_episodes} episode(s) (seed={seed} failed): "
                f"{type(err).__name__}: {err}",
                partial_report=partial, cause=err,
            ) from err
        outcomes.append(record["outcome"])
        episodes_retried += record["retries"]
        for name, cnt in record["episode_fault_counts"].items():
            fault_counts[name] += cnt
        per_episode.append({k: v for k, v in record.items() if k != "episode_fault_counts"})

    return EvalReport.from_outcomes(
        outcomes, per_episode=per_episode, episodes_retried=episodes_retried, fault_counts=dict(fault_counts),
    )


def fault_summary_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """The portion of `ResilientAutoFlyEnv.get_fault_summary()` accumulated strictly between two snapshots
    of the SAME (long-lived, reused-across-conditions) wrapper instance -- so each (checkpoint, condition)
    combination in the gate can report its OWN backend-fault counts, not the whole run's cumulative total."""
    fc_after, fc_before = after.get("fault_counts", {}), before.get("fault_counts", {})
    rc_after, rc_before = after.get("recovered_counts", {}), before.get("recovered_counts", {})
    names = {name: 0 for name in KNOWN_FAULT_NAMES}
    return {
        "fault_counts": {**names, **{k: fc_after.get(k, 0) - fc_before.get(k, 0) for k in fc_after}},
        "recovered_counts": {**names, **{k: rc_after.get(k, 0) - rc_before.get(k, 0) for k in rc_after}},
        "relaunch_count": after.get("relaunch_count", 0) - before.get("relaunch_count", 0),
    }
