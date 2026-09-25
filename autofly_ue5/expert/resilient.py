"""`ResilientAutoFlyEnv`: keeps a documented, recoverable simulator hazard from ending (or hanging) a training or
evaluation run. See `autofly_ue5/expert/train.py`'s module docstring for the full design rationale and the live
measurements behind it.
"""

from __future__ import annotations

import sys
import threading
from collections import Counter

import gymnasium as gym

from autofly_ue5.expert.env import AutoFlyEnv
from autofly_ue5.expert.faults import FAULT_ERRORS_RESET, FAULT_ERRORS_STEP, KNOWN_FAULT_NAMES
from autofly_ue5.sim.process import sweep_stale_instances

DEFAULT_MAX_RESET_ATTEMPTS = 5
DEFAULT_MAX_RELAUNCH_ATTEMPTS = 3
DEFAULT_CLOSE_TIMEOUT_S = 20.0  # matches autofly_ue5.expert.vec.VEC_ENV_CLOSE_TIMEOUT_S's philosophy.


def _bounded_close(env: AutoFlyEnv, timeout_s: float) -> bool:
    """Best-effort, bounded close(). Task 7 measured live that a hung backend call can block a caller
    forever (the projectairsim client can leave a background thread a caller ends up waiting on); a
    relaunch must never block on it. Returns whether close() actually finished in time -- callers must
    sweep_stale_instances() regardless, since that (not this thread) is what frees the real Unreal process
    if close() did not return."""
    finished = threading.Event()

    def _do_close() -> None:
        try:
            env.close()
        except Exception:
            pass  # a raising close() is still a close attempt; sweep_stale_instances() cleans up either way
        finally:
            finished.set()

    threading.Thread(target=_do_close, daemon=True).start()
    return finished.wait(timeout_s)


class ResilientAutoFlyEnv(gym.Wrapper):
    """Wraps one AutoFlyEnv so a documented-common simulator hazard truncates the current episode and
    retries reset() instead of raising -- an uncaught raise here would (Task 7, live) hang the SB3 worker
    running this env, not just crash it, wedging the whole training run. See the module docstring for the
    full design.

    `step()`'s contract on a fault: end the episode by truncation (terminated=False, truncated=True,
    reward=0.0) rather than pretending the same episode continues -- the action that triggered the fault
    was never actually confirmed applied, so returning a `reset()`-fresh observation as if it were the
    consequence of that action would corrupt the RL problem far worse than one lost transition does.
    """

    def __init__(
        self,
        env: AutoFlyEnv,
        instance: int,
        *,
        max_reset_attempts: int = DEFAULT_MAX_RESET_ATTEMPTS,
        max_relaunch_attempts: int = DEFAULT_MAX_RELAUNCH_ATTEMPTS,
        close_timeout_s: float = DEFAULT_CLOSE_TIMEOUT_S,
    ) -> None:
        super().__init__(env)
        self._instance = instance
        self._max_reset_attempts = max_reset_attempts
        self._max_relaunch_attempts = max_relaunch_attempts
        self._close_timeout_s = close_timeout_s
        self.fault_counts: Counter[str] = Counter({name: 0 for name in KNOWN_FAULT_NAMES})
        self.recovered_counts: Counter[str] = Counter({name: 0 for name in KNOWN_FAULT_NAMES})
        self.relaunch_count = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        return self._reset_with_retry(seed=seed, options=options)

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except FAULT_ERRORS_STEP as err:
            name = type(err).__name__
            self.fault_counts[name] += 1
            print(
                f"FAULT instance {self._instance}: caught {name} during step() (occurrence "
                f"#{self.fault_counts[name]} this instance): {err}; truncating the episode and recovering via reset()",
                file=sys.stderr,
            )
            obs, info = self._reset_with_retry(seed=None, options=None)
            self.recovered_counts[name] += 1
            info = {**info, "sim_fault": name}
            return obs, 0.0, False, True, info
        return obs, reward, terminated, truncated, info

    def _reset_with_retry(self, *, seed: int | None, options: dict | None):
        # Every occurrence and every recovery is logged (not just counted) so a run's exact fault sequence
        # -- which single attempts recovered on the very next try vs. which needed a full relaunch -- is
        # reconstructable from the log alone afterward, rather than left to be inferred from the final
        # aggregate counts (Task 8 coordinator review: a live shakedown's 10 faults / 5 recovered could not
        # otherwise be distinguished from "half the retries just don't work" vs. "one relaunch's follow-up
        # attempt happened to hit a separate, unrelated crash").
        faults_so_far: list[str] = []  # accumulated across every round: a later relaunch still "recovers"
        # every fault an earlier round hit, because the overall call went on to succeed regardless.
        for relaunch_round in range(self._max_relaunch_attempts + 1):
            for attempt in range(self._max_reset_attempts):
                try:
                    obs, info = self.env.reset(seed=seed, options=options)
                except FAULT_ERRORS_RESET as err:
                    name = type(err).__name__
                    self.fault_counts[name] += 1
                    faults_so_far.append(name)
                    print(
                        f"FAULT instance {self._instance}: caught {name} during reset() (occurrence "
                        f"#{self.fault_counts[name]} this instance, in-place attempt {attempt + 1}/"
                        f"{self._max_reset_attempts}, relaunch round {relaunch_round}/{self._max_relaunch_attempts}): "
                        f"{err}; retrying",
                        file=sys.stderr,
                    )
                    continue
                for name in faults_so_far:
                    self.recovered_counts[name] += 1
                if faults_so_far:
                    print(
                        f"RECOVERED instance {self._instance}: reset() succeeded after {len(faults_so_far)} "
                        f"fault(s) ({faults_so_far}) and {self.relaunch_count} relaunch(es) this call",
                        file=sys.stderr,
                    )
                return obs, info
            if relaunch_round < self._max_relaunch_attempts:
                self._relaunch()
        raise RuntimeError(
            f"instance {self._instance}: reset() did not recover after {self._max_reset_attempts} "
            f"in-place retries x {self._max_relaunch_attempts + 1} relaunch attempts -- giving up "
            f"(fault_counts={dict(self.fault_counts)}); losing this worker, not the process, is the "
            f"correct failure mode here, but this should be investigated -- it means the simulator itself "
            f"could not be relaunched cleanly {self._max_relaunch_attempts} times in a row"
        )

    def _relaunch(self) -> None:
        print(f"RELAUNCH instance {self._instance}: relaunching (this will be relaunch #{self.relaunch_count + 1})", file=sys.stderr)
        finished = _bounded_close(self.env, self._close_timeout_s)
        if not finished:
            print(
                f"WARNING: instance {self._instance}: close() did not return within {self._close_timeout_s}s "
                f"during relaunch; abandoning it and sweeping stale instances directly",
                file=sys.stderr,
            )
        # AutoFlyEnv.close() (autofly_ue5/expert/env.py, closed for editing) has no try/finally around its
        # own `self._sim = None`: if close() raised, or is still hung in the background thread above,
        # self._sim may still reference a broken connection. Force it to None here (an attribute poke, not
        # a file edit -- the existing test suite already treats AutoFlyEnv's private state this way, e.g.
        # tests/test_expert_env.py's env._sim/_spawned/_setup) so the next reset() unconditionally goes
        # through _ensure_launched() and builds a brand new Simulator + process, instead of reusing a
        # zombie one that would raise "not connected" -- a RuntimeError this wrapper does not know how to
        # treat as recoverable, unlike the faults above.
        self.env._sim = None
        sweep_stale_instances()
        self.relaunch_count += 1

    def get_fault_summary(self) -> dict:
        """Picklable/env_method-friendly summary for the run record -- works whether this wrapper lives in
        the calling process (DummyVecEnv, n=1) or a SubprocVecEnv worker (n>1), since VecEnv.env_method()
        dispatches through Gym's Wrapper.__getattr__ delegation to whatever object actually defines it."""
        return {
            "instance": self._instance,
            "fault_counts": dict(self.fault_counts),
            "recovered_counts": dict(self.recovered_counts),
            "relaunch_count": self.relaunch_count,
        }
