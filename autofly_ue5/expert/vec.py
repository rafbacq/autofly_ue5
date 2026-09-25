"""Vectorised AutoFlyEnv construction and GPU-safe teardown, shared by training, the M2 gate and the throughput
measurement (moved here from `train.py` and `scripts/measure_instances.py` so the package never imports `scripts/`).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from autofly_ue5.expert.env import AutoFlyEnv
from autofly_ue5.expert.resilient import ResilientAutoFlyEnv
from autofly_ue5.expert.seeds import worker_seed_base
from autofly_ue5.scenes.model import Layout, SceneFile
from autofly_ue5.sim.airsim_backend import ProjectAirSimSimulator
from autofly_ue5.sim.process import (
    SIM_RUN_DIR,
    RunOwner,
    current_run_owner,
    instance_dir,
    route_client_log,
    set_run_owner,
    stop_instances,
)

# SB3's SubprocVecEnv.close() joins each worker process; if a worker died on an uncaught exception (e.g. an
# AutoFlyEnv.reset() failure -- measured live: a spawn() material error) that join can hang far longer than any
# teardown has a right to. A bounded background thread protects the one thing that actually matters here --
# getting the real Unreal process stopped -- from an indefinitely stuck Python teardown.
VEC_ENV_CLOSE_TIMEOUT_S = 20.0
# How long to wait for one worker's staggered reset() to reply before treating it as hung. Measured live: an
# uncaught exception inside a worker's reset() does NOT make the worker process exit -- the projectairsim client
# leaves a non-daemon background thread running, so the crashed worker blocks forever in interpreter shutdown, its
# pipe never reaches EOF, and a plain env_method() call (which does an unbounded remote.recv()) hangs forever right
# along with it, never reaching the caller's own try/finally. Polling with a timeout is the only way an
# orchestrator can guarantee it reaches teardown. The bound covers a worker's own worst case: every relaunch round
# (DEFAULT_MAX_RELAUNCH_ATTEMPTS + 1 = 4) spending a full 300 s ready timeout, plus margin -- measured launches take
# single-digit seconds, so reaching it means the worker is hung, not slow.
LAUNCH_REPLY_TIMEOUT_S = 1500.0


def make_vec_env(
    scene: SceneFile,
    layout: Layout,
    n: int,
    *,
    map_path: str,
    monitor_dir: Path,
    seed_base_fn: Callable[[int], int] = worker_seed_base,
    instance_offset: int = 0,
    sim_factory: Callable[[], object] = ProjectAirSimSimulator,
    sim_root: Path = SIM_RUN_DIR,
    owner: RunOwner | None = None,
    launch: bool = True,
) -> VecEnv:
    """n `AutoFlyEnv`s, each `ResilientAutoFlyEnv(Monitor(AutoFlyEnv(...)))`, vectorised.

    `Monitor` sits inside the resilient wrapper, so it only ever sees real episodes: a backend fault raises
    through it (no done, no `info["episode"]`, no CSV row) and its `allow_early_resets` drops the partial episode
    on the next reset. With Monitor outermost (before 2026-09-24), every faulted episode's partial return reached
    `rollout/ep_rew_mean`. `info_keywords=("is_success",)` because a plain Monitor does not copy `is_success` into
    its episode record on its own. SB3 still finds `info["episode"]` on real episode ends: the resilient wrapper
    passes those infos through unchanged.

    n == 1 uses DummyVecEnv: in-process, no IPC to hang on (run 1 trained at n=1; its throughput record's
    chosen_n=1 was a crash, see scripts/measure_instances.py). n > 1 uses SubprocVecEnv, staggering each worker's first reset() (hazard #2: two
    simulators' first reset() at once race check_gpu_for_launch) via a bounded-timeout call rather than SB3's
    unbounded env_method().

    `sim_root`: where each slot's simulator is recorded (and its client log written). `owner`: the run the simulators
    belong to -- the calling process by default; SubprocVecEnv workers adopt it so a slot is never recorded as owned
    by a worker (see `autofly_ue5.sim.process.set_run_owner`). Workers (n > 1) run their wrapper in `worker_mode`.
    `launch=False` skips the staggered first reset, for a caller that launches the workers itself (the throughput
    measurement checks VRAM between launches).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    monitor_dir.mkdir(parents=True, exist_ok=True)
    owner = owner if owner is not None else current_run_owner()

    def _env_fn(rank: int) -> Callable[[], gym.Env]:
        instance = instance_offset + rank

        def _make() -> gym.Env:
            set_run_owner(owner)
            route_client_log(instance_dir(instance, sim_root) / "client.log")
            base = AutoFlyEnv(scene, layout, sim_factory, map_path=map_path, instance=instance, seed_base=seed_base_fn(rank))
            monitored = Monitor(base, filename=str(monitor_dir / f"{instance}.monitor.csv"), info_keywords=("is_success",))
            return ResilientAutoFlyEnv(monitored, instance=instance, sim_root=sim_root, worker_mode=n > 1)

        return _make

    if n == 1:
        return DummyVecEnv([_env_fn(0)])

    vec_env = SubprocVecEnv([_env_fn(i) for i in range(n)])
    if launch:
        try:
            for i in range(n):
                call_reset_with_timeout(vec_env, i)  # staggered, bounded -- not SB3's unbounded env_method()
        except BaseException:
            # The caller never gets this vec env, so nobody else can stop what already launched: the healthy workers
            # (daemon processes that outlive an os._exit caller) and their simulators, holding VRAM.
            teardown(vec_env, [instance_offset + i for i in range(n)], sim_root)
            raise
    return vec_env


def teardown(vec_env: SubprocVecEnv | None, instances, sim_root: Path = SIM_RUN_DIR) -> list[dict]:
    """Close every worker (which closes its Simulator, which stops its process), then stop the listed slots
    directly as a belt-and-braces check -- so a failed run never leaves its processes holding VRAM. Only the
    listed slots: another run's simulators are never touched (C1).

    vec_env.close() runs in a background thread with a hard timeout: measured live, it can hang
    indefinitely joining a worker that already died on an uncaught exception. Whether or not that thread
    finishes, stop_instances() below stops the actual Unreal processes directly via our own pidfile
    bookkeeping (autofly_ue5.sim.process), independent of SB3's cooperative shutdown.
    """
    if vec_env is not None:
        close_error: list[BaseException] = []

        def _do_close() -> None:
            try:
                vec_env.close()
            except Exception as err:  # a dead worker's pipe can raise on close(); still sweep below
                close_error.append(err)

        closer = threading.Thread(target=_do_close, daemon=True)
        closer.start()
        closer.join(VEC_ENV_CLOSE_TIMEOUT_S)
        if closer.is_alive():
            print(f"WARNING: vec_env.close() did not return within {VEC_ENV_CLOSE_TIMEOUT_S}s; "
                  f"abandoning it and force-killing worker processes directly", file=sys.stderr)
        elif close_error:
            err = close_error[0]
            print(f"WARNING: vec_env.close() raised {type(err).__name__}: {err}", file=sys.stderr)
        # Belt-and-braces regardless of the above: a worker that crashed without exiting (measured live --
        # see LAUNCH_REPLY_TIMEOUT_S) will never respond to close() and must be killed directly, or it
        # leaks a process every attempt. SIGKILL is unconditional, unlike anything cooperative.
        for i, proc in enumerate(getattr(vec_env, "processes", [])):
            try:
                if proc.is_alive():
                    print(f"WARNING: worker {i} (pid {proc.pid}) still alive after close(); killing it", file=sys.stderr)
                    proc.kill()
                    proc.join(5.0)
            except Exception as err:
                print(f"WARNING: could not kill worker {i}: {type(err).__name__}: {err}", file=sys.stderr)
    return stop_instances(list(instances), sim_root)


def call_method_with_timeout(vec_env: SubprocVecEnv, index: int, method: str, timeout_s: float):
    """Like `vec_env.env_method(method, indices=[index])[0]`, but bounded.

    SB3's env_method() does an unbounded `remote.recv()`. That is safe only if a crashed worker always
    exits promptly; measured live, it does not (see LAUNCH_REPLY_TIMEOUT_S's comment), so this polls with a
    timeout instead, specifically so a hung worker cannot prevent the orchestrator from reaching its own
    teardown path.
    """
    remote = vec_env.remotes[index]
    remote.send(("env_method", (method, (), {})))
    if not remote.poll(timeout_s):
        raise TimeoutError(
            f"worker {index} did not reply to {method}() within {timeout_s}s -- it likely crashed without "
            f"exiting (check the job log for a worker traceback) and must be torn down forcibly"
        )
    return remote.recv()


def call_reset_with_timeout(vec_env: SubprocVecEnv, index: int, timeout_s: float = LAUNCH_REPLY_TIMEOUT_S) -> None:
    """A bounded `reset()` of one worker (its simulator launches lazily inside it)."""
    call_method_with_timeout(vec_env, index, "reset", timeout_s)


def collect_fault_summaries(vec_env: VecEnv, timeout_s: float = VEC_ENV_CLOSE_TIMEOUT_S) -> tuple[list[dict], list[int]]:
    """Every reachable worker's `get_fault_summary()`, one worker at a time and bounded, plus the indices of the
    workers that could not answer. SB3's env_method() is all-or-nothing: one dead worker lost every summary, and the
    record then showed explicit zeros -- "no faults" -- for faults that did happen."""
    summaries: list[dict] = []
    missing: list[int] = []
    for index in range(vec_env.num_envs):
        try:
            if isinstance(vec_env, SubprocVecEnv):
                summaries.append(call_method_with_timeout(vec_env, index, "get_fault_summary", timeout_s))
            else:
                summaries.append(vec_env.env_method("get_fault_summary", indices=[index])[0])
        except Exception as err:
            print(f"WARNING: worker {index} gave no fault summary ({type(err).__name__}: {err})", file=sys.stderr)
            missing.append(index)
    return summaries, missing

