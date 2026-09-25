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
from autofly_ue5.sim.process import instance_dir, route_client_log, sweep_stale_instances

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
# orchestrator can guarantee it reaches teardown. 300s is generous next to every launch measured so far (single
# digits to tens of seconds).
LAUNCH_REPLY_TIMEOUT_S = 300.0


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
) -> VecEnv:
    """n `AutoFlyEnv`s, each `Monitor(ResilientAutoFlyEnv(AutoFlyEnv(...)))`, vectorised.

    `Monitor` is outermost (what the VecEnv actually calls) so SB3 can find `info["episode"]` and
    `rollout/ep_rew_mean`/`ep_len_mean` get logged; `info_keywords=("is_success",)` because a plain Monitor
    does not copy `is_success` into its episode record on its own. `ResilientAutoFlyEnv` sits directly
    around the raw env so the fault it catches never has to cross Monitor's step() at all.

    n == 1 uses DummyVecEnv: in-process, no IPC to hang on, which is what this project's Task 7 measurement
    chose (chosen_n=1). n > 1 uses SubprocVecEnv, staggering each worker's first reset() (hazard #2: two
    simulators' first reset() at once race check_gpu_for_launch) via a bounded-timeout call rather than SB3's
    unbounded env_method().
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    monitor_dir.mkdir(parents=True, exist_ok=True)

    def _env_fn(rank: int) -> Callable[[], gym.Env]:
        instance = instance_offset + rank

        def _make() -> gym.Env:
            route_client_log(instance_dir(instance) / "client.log")
            base = AutoFlyEnv(scene, layout, sim_factory, map_path=map_path, instance=instance, seed_base=seed_base_fn(rank))
            resilient = ResilientAutoFlyEnv(base, instance=instance)
            return Monitor(resilient, filename=str(monitor_dir / f"{instance}.monitor.csv"), info_keywords=("is_success",))

        return _make

    if n == 1:
        return DummyVecEnv([_env_fn(0)])

    vec_env = SubprocVecEnv([_env_fn(i) for i in range(n)])
    for i in range(n):
        call_reset_with_timeout(vec_env, i)  # staggered, bounded -- not SB3's unbounded env_method()
    return vec_env


def teardown(vec_env: SubprocVecEnv | None) -> list[dict]:
    """Close every worker (which closes its Simulator, which stops its process), then sweep stale
    instances as a belt-and-braces check -- so a failed run never leaves processes holding VRAM.

    vec_env.close() runs in a background thread with a hard timeout: measured live, it can hang
    indefinitely joining a worker that already died on an uncaught exception. Whether or not that thread
    finishes, sweep_stale_instances() below stops the actual Unreal process directly via our own pidfile
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
    return sweep_stale_instances()


def call_reset_with_timeout(vec_env: SubprocVecEnv, index: int, timeout_s: float = LAUNCH_REPLY_TIMEOUT_S) -> None:
    """Like `vec_env.env_method("reset", indices=[index])`, but bounded.

    SB3's env_method() does an unbounded `remote.recv()`. That is safe only if a crashed worker always
    exits promptly; measured live, it does not (see LAUNCH_REPLY_TIMEOUT_S's comment), so this polls with a
    timeout instead, specifically so a hung worker cannot prevent the orchestrator from reaching its own
    teardown path.
    """
    remote = vec_env.remotes[index]
    remote.send(("env_method", ("reset", (), {})))
    if not remote.poll(timeout_s):
        raise TimeoutError(
            f"worker {index} did not reply to reset() within {timeout_s}s -- it likely crashed without "
            f"exiting (check the job log for a worker traceback) and must be torn down forcibly"
        )
    remote.recv()
