"""`ResilientAutoFlyEnv`: keeps a documented, recoverable simulator hazard from ending (or hanging) a training or
evaluation run. See `autofly_ue5/expert/train.py`'s module docstring for the full design rationale and the live
measurements behind it.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from collections import Counter
from pathlib import Path

import gymnasium as gym

from autofly_ue5.expert.faults import FAULT_ERRORS_LAUNCH, FAULT_ERRORS_RESET, FAULT_ERRORS_STEP, KNOWN_FAULT_NAMES
from autofly_ue5.sim.process import SIM_RUN_DIR, stop_instance

DEFAULT_MAX_RESET_ATTEMPTS = 5
DEFAULT_MAX_RELAUNCH_ATTEMPTS = 3
DEFAULT_CLOSE_TIMEOUT_S = 20.0  # matches autofly_ue5.expert.vec.VEC_ENV_CLOSE_TIMEOUT_S's philosophy.


def _bounded_close(closable, timeout_s: float) -> bool:
    """Best-effort, bounded close(). Task 7 measured live that a hung backend call can block a caller
    forever (the projectairsim client can leave a background thread a caller ends up waiting on); a
    relaunch must never block on it. Returns whether close() actually finished in time -- callers must
    stop the slot's process regardless, since that (not this thread) is what frees the real Unreal process
    if close() did not return."""
    finished = threading.Event()

    def _do_close() -> None:
        try:
            closable.close()
        except Exception:
            pass  # a raising close() is still a close attempt; the caller stops the slot either way
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

    `sim_root`: where this slot's simulator is recorded (`autofly_ue5.sim.process`); a relaunch stops that one slot
    and nothing else. `worker_mode` (SubprocVecEnv workers): any exception that would leave reset()/step() ends the
    process instead, because a worker that raises does not exit -- projectairsim's non-daemon thread blocks its
    interpreter shutdown, and the training process then waits on its pipe forever.
    """

    def __init__(
        self,
        env: gym.Env,
        instance: int,
        *,
        max_reset_attempts: int = DEFAULT_MAX_RESET_ATTEMPTS,
        max_relaunch_attempts: int = DEFAULT_MAX_RELAUNCH_ATTEMPTS,
        close_timeout_s: float = DEFAULT_CLOSE_TIMEOUT_S,
        sim_root: Path = SIM_RUN_DIR,
        worker_mode: bool = False,
    ) -> None:
        super().__init__(env)
        self._instance = instance
        self._max_reset_attempts = max_reset_attempts
        self._max_relaunch_attempts = max_relaunch_attempts
        self._close_timeout_s = close_timeout_s
        self._sim_root = Path(sim_root)
        self._worker_mode = worker_mode
        self.fault_counts: Counter[str] = Counter({name: 0 for name in KNOWN_FAULT_NAMES})
        self.recovered_counts: Counter[str] = Counter({name: 0 for name in KNOWN_FAULT_NAMES})
        self.relaunch_count = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        try:
            return self._reset_with_retry(seed=seed, options=options)
        except BaseException:
            self._exit_if_worker()
            raise

    def step(self, action):
        try:
            return self._step(action)
        except BaseException:
            self._exit_if_worker()
            raise

    def _exit_if_worker(self) -> None:
        if self._worker_mode:
            traceback.print_exc()
            print(f"FATAL instance {self._instance}: unrecoverable error in a vec-env worker; exiting the worker "
                  f"so the training process sees EOF instead of hanging", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

    def _step(self, action):
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
                    launch_failed = isinstance(err, FAULT_ERRORS_LAUNCH)
                    print(
                        f"FAULT instance {self._instance}: caught {name} during reset() (occurrence "
                        f"#{self.fault_counts[name]} this instance, in-place attempt {attempt + 1}/"
                        f"{self._max_reset_attempts}, relaunch round {relaunch_round}/{self._max_relaunch_attempts}): "
                        f"{err}; {'going to the next relaunch round' if launch_failed else 'retrying'}",
                        file=sys.stderr,
                    )
                    if launch_failed:
                        break  # the simulator never came up: another reset() on this slot cannot help
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
            f"(fault_counts={dict(self.fault_counts)}). The simulator could not be brought back "
            f"{self._max_relaunch_attempts} times in a row; this ends the run (or, in a vec-env worker, the "
            f"worker) and must be investigated"
        )

    def _relaunch(self) -> None:
        print(f"RELAUNCH instance {self._instance}: relaunching (this will be relaunch #{self.relaunch_count + 1})", file=sys.stderr)
        # Detach the old simulator before closing it, and close only it: if the bounded close() is abandoned and
        # returns later, it can neither detach nor (via stop(expected_pid=...)) stop the replacement. Acts on the
        # unwrapped AutoFlyEnv so a Monitor between us and it is never closed (that would end its CSV for good).
        base = self.env.unwrapped
        old_sim, base._sim = base._sim, None
        if old_sim is not None and not _bounded_close(old_sim, self._close_timeout_s):
            print(
                f"WARNING: instance {self._instance}: close() did not return within {self._close_timeout_s}s "
                f"during relaunch; abandoning it and stopping this slot's process directly",
                file=sys.stderr,
            )
        # This slot only: the global sweep this replaced also stopped every sibling's simulator (C1).
        result = stop_instance(self._instance, self._sim_root)
        print(f"RELAUNCH instance {self._instance}: stopped its own slot: {result}", file=sys.stderr)
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
