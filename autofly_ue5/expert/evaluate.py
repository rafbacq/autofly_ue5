"""Evaluation harness for a trained expert (spec Sec8/Sec9.5, Task 9's M2 gate).

Draws episodes from `reset(seed=seed_base + i)` for `i in range(n_episodes)` -- a fixed, reproducible
evaluation stream, disjoint from every training worker's and the training-time evaluation's own streams
(`autofly_ue5.expert.seeds`; see its disjointness test) -- and scores each with
`model.predict(obs, deterministic=...)`.

Backend faults, never policy failures (Task 9 brief, gate item 6): reuses `autofly_ue5.expert.resilient.
ResilientAutoFlyEnv` -- not a second retry wrapper -- for the recoverable hazards in `autofly_ue5.expert.faults`
(`CameraPoseError`, `StepTimingError`, `StaleStateError`, `CommandTimeoutError`, `pynng.exceptions.Timeout`, the
C9 start/teleport checks, ...). A fault DURING `reset()` is already retried transparently by that wrapper
with the SAME seed, so it never reaches this module. A fault mid-`step()` truncates the current episode on its
last real observation with `info["sim_fault"]` set (see the wrapper's docstring); this module discards the whole
attempt and replays the SAME seed from `env.reset(seed=seed)`, so a simulator hiccup is never counted as a
success/collision/out_of_bounds/timeout outcome and every eval seed still gets a genuine policy outcome.
`episodes_retried` and `fault_counts` report exactly how often that happened.

`FaultAwareEvalCallback` runs this same harness periodically during training, replacing SB3's `EvalCallback`,
which scored faulted episodes as policy outcomes and drew a new slice of the eval env's seed stream at every
evaluation (walking through the M2 gate's own episodes in the 2026-09-17 run).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Reused so a clean eval run's fault_counts still show an explicit 0 for every known hazard (not just the
# ones actually seen) -- exactly the same reasoning as train.py's own KNOWN_FAULT_NAMES-seeded counters, so
# "no faults happened" is distinguishable from "counting was silently broken" here too.
from autofly_ue5.expert.faults import KNOWN_FAULT_NAMES

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
    mean_return: float | None = None
    # What the collisions hit (spec §6.5): "sim", "mover" or "mover_inferred"; empty without collisions.
    collision_sources: dict[str, int] = field(default_factory=dict)

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
        returns = [e["return"] for e in per_episode if e.get("return") is not None]
        merged_faults = {name: 0 for name in KNOWN_FAULT_NAMES}
        merged_faults.update(fault_counts or {})
        sources = Counter(e.get("collision_source") or "sim" for e in per_episode if e.get("outcome") == "collision")
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
            mean_return=(sum(returns) / len(returns)) if returns else None,
            collision_sources=dict(sources),
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
            "mean_return": self.mean_return,
            "episodes_retried": self.episodes_retried,
            "fault_counts": dict(self.fault_counts),
            "collision_sources": dict(self.collision_sources),
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
    episode_return = 0.0
    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        fault = info.get("sim_fault")
        if fault:
            return {"fault": fault}
        episode_return += float(reward)
        if terminated or truncated:
            return {
                "seed": seed,
                "outcome": info.get("outcome", "unknown"),
                "steps": info.get("steps", step),
                "final_distance_m": info.get("final_distance_m"),
                "is_success": bool(info.get("is_success", False)),
                "return": episode_return,
                "final_pose": info.get("pose"),
                "final_bearing_deg": info.get("bearing_deg"),
                "oob_kind": info.get("oob_kind"),
                # Moving obstacles (spec §6.5): what a collision hit, how many movers flew, whether the one hit was seen.
                "collision_source": info.get("collision_source"),
                "n_movers": info.get("n_movers", 0),
                "mover_in_view": info.get("mover_in_view"),
                **({"inferred_from": info["inferred_from"]} if info.get("inferred_from") else {}),
            }
    # AutoFlyEnv's own step_limit (default 300, spec Sec8) always truncates well before this bound;
    # reaching it means something is not honouring that contract -- report it, don't spin forever.
    return {
        "seed": seed,
        "outcome": "exceeded_max_steps",
        "steps": max_steps,
        "final_distance_m": info.get("final_distance_m"),
        "is_success": False,
        "return": episode_return,
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


class FaultAwareEvalCallback(BaseCallback):
    """Periodic evaluation during training, through `evaluate_policy_episodes` -- SB3's `EvalCallback` replaced.

    Every `eval_freq` timesteps (counted in `num_timesteps`, so a resumed run keeps its cadence) it plays the SAME
    `n_eval_episodes` episodes, `reset(seed=seed_base + i)`, so successive evaluations are comparable and a faulted
    episode is replayed rather than scored. It writes SB3's `evaluations.npz` keys (`timesteps`, `results`,
    `ep_lengths`, `successes`, one equal-length row per evaluation) and saves `best_model.zip` on a new best mean
    return -- SB3's own selection criterion. On resume it reloads the npz, so the history and the best value
    survive. An evaluation the backend cannot finish is logged and skipped; it never ends a 12-hour run.
    """

    def __init__(self, eval_env, *, n_eval_episodes: int, eval_freq: int, seed_base: int, best_model_save_path: Path,
                 log_path: Path, deterministic: bool = True, verbose: int = 0) -> None:
        super().__init__(verbose)
        if eval_freq <= 0 or n_eval_episodes <= 0:
            raise ValueError(f"eval_freq and n_eval_episodes must be > 0, got {eval_freq}, {n_eval_episodes}")
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes
        self._eval_freq = eval_freq
        self._seed_base = seed_base
        self._best_model_save_path = Path(best_model_save_path)
        self._log_path = Path(log_path)
        self._deterministic = deterministic
        self.evaluations_timesteps: list[int] = []
        self.evaluations_results: list[list[float]] = []
        self.evaluations_length: list[list[int]] = []
        self.evaluations_successes: list[list[bool]] = []
        self.best_mean_reward = -np.inf
        self.interrupted_evaluations = 0
        self._next_eval = eval_freq

    def _init_callback(self) -> None:
        self._best_model_save_path.mkdir(parents=True, exist_ok=True)
        self._log_path.mkdir(parents=True, exist_ok=True)
        npz = self._log_path / "evaluations.npz"
        if npz.is_file():
            data = np.load(npz)
            self.evaluations_timesteps = [int(t) for t in data["timesteps"]]
            self.evaluations_results = [[float(r) for r in row] for row in data["results"]]
            self.evaluations_length = [[int(n) for n in row] for row in data["ep_lengths"]]
            self.evaluations_successes = [[bool(x) for x in row] for row in data["successes"]]
            recorded = {len(row) for row in self.evaluations_results}
            if recorded and recorded != {self._n_eval_episodes}:
                raise ValueError(
                    f"{npz} holds evaluations of {sorted(recorded)} episodes but this run asks for {self._n_eval_episodes}; "
                    f"resume with the same --eval-episodes (rows of different lengths cannot share one npz)"
                )
            if self.evaluations_results:
                self.best_mean_reward = max(float(np.mean(row)) for row in self.evaluations_results)
        self._next_eval = (self.model.num_timesteps // self._eval_freq + 1) * self._eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval = (self.num_timesteps // self._eval_freq + 1) * self._eval_freq
        try:
            report = evaluate_policy_episodes(self.model, self._eval_env, self._n_eval_episodes, self._seed_base,
                                              deterministic=self._deterministic)
        except EvaluationInterrupted as err:
            if not isinstance(err.cause, RuntimeError):
                # The backend's failures all arrive as RuntimeError (the wrapper giving up, the per-episode retry
                # bound). Anything else is a bug in the evaluation path: skipping it silently would disable model
                # selection for the whole run.
                raise err.cause from err
            self.interrupted_evaluations += 1
            print(f"WARNING: evaluation at {self.num_timesteps} timesteps could not finish and is skipped: {err}",
                  file=sys.stderr)
            self.logger.record("eval/interrupted_evaluations", self.interrupted_evaluations)
            return True
        returns = [float(e["return"]) for e in report.per_episode]
        lengths = [int(e["steps"]) for e in report.per_episode]
        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_results.append(returns)
        self.evaluations_length.append(lengths)
        self.evaluations_successes.append([bool(e["is_success"]) for e in report.per_episode])
        np.savez(self._log_path / "evaluations.npz", timesteps=self.evaluations_timesteps,
                 results=self.evaluations_results, ep_lengths=self.evaluations_length,
                 successes=self.evaluations_successes)
        mean_reward = float(np.mean(returns))
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", float(np.mean(lengths)))
        self.logger.record("eval/success_rate", report.success_rate)
        self.logger.record("eval/episodes_retried", report.episodes_retried)
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)
        if self.verbose:
            print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f}, "
                  f"success_rate={report.success_rate:.2f}, episodes_retried={report.episodes_retried}")
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(self._best_model_save_path / "best_model")
            if self.verbose:
                print("New best mean reward!")
        return True

