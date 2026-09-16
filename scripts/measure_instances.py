"""Instance-scaling throughput measurement (spec §8's throughput gate, Task 7).

Measures environment steps/s (one `command_velocity` + one `sim.step(0.2)` + one `observe()` + the
observation encoding, i.e. one `AutoFlyEnv.step()`) for N concurrent `AutoFlyEnv` workers, N in
(1, 2, 4, 6), on the packaged s01 simulator. Random actions only -- this measures the simulator, not a
policy. Stops climbing N the moment VRAM would exceed the budget or total throughput falls versus the
previous N (see `should_stop`).

Safety, in order:
  1. Sweep stale instances (a crashed earlier run can leave Unreal children holding VRAM).
  2. Stagger the workers' first reset(): each `AutoFlyEnv` launches its simulator lazily inside its first
     reset(), so N workers resetting at once would all read the same pre-launch VRAM figure and could
     collectively exhaust the GPU. This script launches instances one at a time via `env_method("reset",
     indices=[i])` against a single index of a `SubprocVecEnv`, checking VRAM headroom before each.
  3. Tear down every instance of the current N on any failure path (try/finally).
  4. Sample VRAM only inside the window where all N instances are actively stepping.

env -u PYTHONPATH .venv/bin/python scripts/measure_instances.py --out docs/gates/m2_instances.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

from autofly_ue5.expert.env import AutoFlyEnv
from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import ROOT, RUNS_DIR
from autofly_ue5.scenes.model import Bounds, Instance, Layout, SceneFile, load_scene_file
from autofly_ue5.sim.airsim_backend import ProjectAirSimSimulator
from autofly_ue5.sim.process import instance_dir, own_running_instances, route_client_log, stop
from autofly_ue5.validate.engine_check import boot_id, count_device_lost, xid_count

MAP_PATH = "/Game/AutoFly/Maps/S01"
CANDIDATE_NS = (1, 2, 4, 6)
WARMUP_S = 30.0
TIMED_S = 180.0
VRAM_BUDGET_MIB = 20_000
# M1's measured figure for one packaged instance while capturing; used only as a conservative pre-launch
# guard between stagger steps, not as the recorded measurement (that comes from live nvidia-smi samples).
EST_PER_INSTANCE_MIB = 1_882
VRAM_SAMPLE_INTERVAL_S = 15.0
WORKER_SEED_STRIDE = 1_000_000
MAX_ATTEMPTS_PER_N = 2
VRAM_DRAIN_TIMEOUT_S = 30.0
VRAM_DRAIN_MARGIN_MIB = 400
# SB3's SubprocVecEnv.close() joins each worker process; if a worker died on an uncaught exception (e.g.
# an AutoFlyEnv.reset() failure -- measured live: a spawn() material error) that join can hang far longer
# than any teardown has a right to. A bounded background thread protects the one thing that actually
# matters here -- getting the real Unreal process stopped -- from an indefinitely stuck Python teardown.
VEC_ENV_CLOSE_TIMEOUT_S = 20.0
# How long to wait for one worker's staggered reset() to reply before treating it as hung. Measured live:
# an uncaught exception inside a worker's reset() does NOT make the worker process exit -- the
# projectairsim client leaves a non-daemon background thread running, so the crashed worker blocks
# forever in interpreter shutdown, its pipe never reaches EOF, and a plain env_method() call (which does
# an unbounded remote.recv()) hangs forever right along with it, never reaching this script's own
# try/finally. Polling with a timeout is the only way this orchestrator can guarantee it reaches
# teardown. 300s is generous next to every launch measured so far (single digits to tens of seconds).
LAUNCH_REPLY_TIMEOUT_S = 300.0


def project_cost(steps_per_s_total: float, steps_needed: int, n_scenes: int) -> dict:
    """Wall-clock cost of training `n_scenes` experts at `steps_needed` env steps each, given a measured
    aggregate throughput. Spec §8/§12: "project the cost of all agents before M5"."""
    if steps_per_s_total <= 0:
        raise ValueError(f"steps_per_s_total must be > 0, got {steps_per_s_total}")
    if steps_needed <= 0:
        raise ValueError(f"steps_needed must be > 0, got {steps_needed}")
    if n_scenes <= 0:
        raise ValueError(f"n_scenes must be > 0, got {n_scenes}")
    hours_per_scene = steps_needed / steps_per_s_total / 3600.0
    hours_all_scenes = hours_per_scene * n_scenes
    return {
        "hours_per_scene": hours_per_scene,
        "hours_all_scenes": hours_all_scenes,
        "days_all_scenes": hours_all_scenes / 24.0,
    }


def should_stop(vram_total_mib: int, prev_total: float | None, this_total: float) -> tuple[bool, str]:
    """Whether to stop climbing N, and why. Two independent triggers (either one stops the climb):

    (a) this N's VRAM sample would exceed the machine budget -- never push toward OOM;
    (b) total throughput fell versus the previous N -- more instances bought nothing, or made it worse.

    `prev_total=None` (the first N measured) skips trigger (b): there is nothing to compare against yet.
    """
    if vram_total_mib > VRAM_BUDGET_MIB:
        return True, f"vram_mib {vram_total_mib} exceeds the {VRAM_BUDGET_MIB} MiB budget"
    if prev_total is not None and this_total < prev_total:
        return True, f"total throughput fell: {this_total:.3f} < previous {prev_total:.3f} steps/s"
    return False, ""


def choose_best_n(per_n: dict[str, dict]) -> int | None:
    """The N with the highest total throughput among entries that stayed inside the VRAM budget."""
    in_budget = {int(n): rec for n, rec in per_n.items() if rec.get("vram_mib", VRAM_BUDGET_MIB + 1) <= VRAM_BUDGET_MIB}
    if not in_budget:
        return None
    return max(in_budget, key=lambda n: in_budget[n]["env_steps_per_s_total"])


def scene_and_layout() -> tuple[SceneFile, Layout]:
    """Mirrors tests/test_expert_episode.py's scene_and_layout() helper (not imported: that module is
    reviewed/closed test code, this is a live driver script)."""
    scene = load_scene_file(ROOT / "scenes" / "s01_white_pillars.json")
    raw = json.loads((RUNS_DIR / "levels" / "s01.layout.json").read_text())["layout"]
    b = Bounds(**raw["bounds"])
    inst = tuple(Instance(**i) for i in raw["instances"])
    return scene, Layout(scene_id=raw["scene_id"], seed=raw["seed"], bounds=b, instances=inst)


def make_env_fn(scene: SceneFile, layout: Layout, instance: int, seed_base: int) -> Callable[[], AutoFlyEnv]:
    """Returns a picklable, zero-arg factory for one AutoFlyEnv, to run inside a SubprocVecEnv worker.

    Constructing the env here does NOT launch the simulator -- AutoFlyEnv launches lazily inside its first
    reset() (see module docstring point 2). Each instance gets its own client log path and a distinct
    seed_base (sharing one across workers would fly byte-identical episode streams -- test_expert_env.py's
    test_two_envs_with_the_same_seed_base_fly_identical_streams documents this hazard).
    """

    def _make() -> AutoFlyEnv:
        route_client_log(instance_dir(instance) / "client.log")
        return AutoFlyEnv(
            scene, layout, ProjectAirSimSimulator, map_path=MAP_PATH, instance=instance, seed_base=seed_base
        )

    return _make


def sweep_stale_instances() -> list[dict]:
    """Stop any simulator this project owns that is still recorded as running. A crashed earlier run can
    leave Unreal children holding VRAM, and launch_process() then refuses with "already running" or "port
    already in use"."""
    swept = []
    for sp in own_running_instances():
        result = stop(sp.instance)
        swept.append({"instance": sp.instance, "pid": sp.pid, "result": result})
    return swept


def wait_for_vram_drop(baseline_mib: int, margin_mib: int = VRAM_DRAIN_MARGIN_MIB, timeout_s: float = VRAM_DRAIN_TIMEOUT_S) -> int:
    """Poll until VRAM has drained back near `baseline_mib` (a torn-down UE process can take a moment to
    release its allocation) or `timeout_s` elapses; returns the last-seen used_mib either way."""
    deadline = time.monotonic() + timeout_s
    used, _ = gpu_memory_mib()
    while used > baseline_mib + margin_mib and time.monotonic() < deadline:
        time.sleep(1.0)
        used, _ = gpu_memory_mib()
    return used


def teardown(vec_env: SubprocVecEnv | None) -> list[dict]:
    """Close every worker (which closes its Simulator, which stops its process), then sweep stale
    instances as a belt-and-braces check -- so a failed measurement never leaves processes holding VRAM.

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


def _call_reset_with_timeout(vec_env: SubprocVecEnv, index: int, timeout_s: float = LAUNCH_REPLY_TIMEOUT_S) -> None:
    """Like `vec_env.env_method("reset", indices=[index])`, but bounded.

    SB3's env_method() does an unbounded `remote.recv()`. That is safe only if a crashed worker always
    exits promptly; measured live, it does not (see LAUNCH_REPLY_TIMEOUT_S's comment), so this script
    polls with a timeout instead, specifically so a hung worker cannot prevent this orchestrator from
    reaching its own teardown path.
    """
    remote = vec_env.remotes[index]
    remote.send(("env_method", ("reset", (), {})))
    if not remote.poll(timeout_s):
        raise TimeoutError(
            f"worker {index} did not reply to reset() within {timeout_s}s -- it likely crashed without "
            f"exiting (check the job log for a worker traceback) and must be torn down forcibly"
        )
    remote.recv()


def _random_actions(vec_env: SubprocVecEnv, n: int) -> np.ndarray:
    return np.stack([vec_env.action_space.sample() for _ in range(n)])


def _step_for(vec_env: SubprocVecEnv, n: int, duration_s: float) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        vec_env.step_async(_random_actions(vec_env, n))
        vec_env.step_wait()


def measure_n(
    n: int, scene: SceneFile, layout: Layout, warmup_s: float = WARMUP_S, timed_s: float = TIMED_S,
    launch_reply_timeout_s: float = LAUNCH_REPLY_TIMEOUT_S,
) -> dict:
    """Launch n AutoFlyEnv workers (staggered, VRAM-checked between each), warm up, then measure random-
    action throughput over a fixed wall-clock window. Always tears down its own workers before returning
    or raising."""
    env_fns = [make_env_fn(scene, layout, i, seed_base=(i + 1) * WORKER_SEED_STRIDE) for i in range(n)]
    vec_env: SubprocVecEnv | None = None
    per_instance_launch_s: list[float] = []
    try:
        vec_env = SubprocVecEnv(env_fns)
        launch_start = time.monotonic()
        for i in range(n):
            used, _total = gpu_memory_mib()
            projected = used + EST_PER_INSTANCE_MIB
            if projected > VRAM_BUDGET_MIB:
                raise RuntimeError(
                    f"launching instance {i} of {n} would push VRAM to an estimated {projected} MiB "
                    f"(currently {used} MiB with {i} instance(s) up), over the {VRAM_BUDGET_MIB} MiB budget"
                )
            t0 = time.monotonic()
            # blocks (with a bound) until this ONE worker's launch+reset completes
            _call_reset_with_timeout(vec_env, i, launch_reply_timeout_s)
            per_instance_launch_s.append(time.monotonic() - t0)
        launch_s = time.monotonic() - launch_start

        _step_for(vec_env, n, warmup_s)  # untimed: shader warm-up, first-episode settle

        vram_samples: list[int] = []
        next_sample = time.monotonic()
        total_steps = 0
        episodes_completed = 0
        window_start = time.monotonic()
        window_end = window_start + timed_s
        while time.monotonic() < window_end:
            vec_env.step_async(_random_actions(vec_env, n))
            _obs, _rews, dones, _infos = vec_env.step_wait()
            total_steps += n
            episodes_completed += int(np.sum(dones))
            now = time.monotonic()
            if now >= next_sample:
                vram_samples.append(gpu_memory_mib()[0])
                next_sample = now + VRAM_SAMPLE_INTERVAL_S
        elapsed_s = time.monotonic() - window_start
        if not vram_samples:  # window shorter than one sample interval: still sample once, inside the window
            vram_samples.append(gpu_memory_mib()[0])

        total_per_s = total_steps / elapsed_s
        return {
            "env_steps_per_s_total": total_per_s,
            "env_steps_per_s_per_instance": total_per_s / n,
            "vram_mib": max(vram_samples),
            "vram_samples_mib": vram_samples,
            "launch_s": launch_s,
            "per_instance_launch_s": per_instance_launch_s,
            "episodes_completed": episodes_completed,
            "total_env_steps": total_steps,
            "warmup_s": warmup_s,
            "timed_window_s": elapsed_s,
        }
    finally:
        stale = teardown(vec_env)
        if stale:
            print(f"WARNING: swept stale instances after n={n}: {stale}", file=sys.stderr)


def run(
    out_path: Path, candidate_ns: tuple[int, ...] = CANDIDATE_NS, warmup_s: float = WARMUP_S, timed_s: float = TIMED_S,
    launch_reply_timeout_s: float = LAUNCH_REPLY_TIMEOUT_S,
) -> dict:
    scene, layout = scene_and_layout()
    baseline_used_mib, gpu_total_mib = gpu_memory_mib()
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")

    initial_sweep = sweep_stale_instances()
    if initial_sweep:
        print(f"swept stale instances before starting: {initial_sweep}", file=sys.stderr)
        baseline_used_mib = wait_for_vram_drop(baseline_used_mib)

    per_n: dict[str, dict] = {}
    attempted_ns: list[int] = []
    failures: dict[str, dict] = {}
    stop_reason = "measured every candidate N without hitting a stop condition"
    prev_total: float | None = None

    for n in candidate_ns:
        attempted_ns.append(n)
        xid_before = xid_count(run_started)
        boot_before = boot_id()
        record: dict | None = None
        last_error: str | None = None
        attempts = 0
        while attempts < MAX_ATTEMPTS_PER_N and record is None:
            attempts += 1
            wait_for_vram_drop(baseline_used_mib)
            try:
                record = measure_n(
                    n, scene, layout, warmup_s=warmup_s, timed_s=timed_s, launch_reply_timeout_s=launch_reply_timeout_s
                )
            except Exception as err:
                last_error = f"{type(err).__name__}: {err}"
                print(f"n={n} attempt {attempts}/{MAX_ATTEMPTS_PER_N} failed: {last_error}", file=sys.stderr)

        if record is None:
            failures[str(n)] = {"attempts": attempts, "error": last_error}
            stop_reason = f"n={n} failed to launch/measure after {attempts} attempt(s): {last_error}"
            break

        logs = [instance_dir(i) / "sim.log" for i in range(n) if (instance_dir(i) / "sim.log").is_file()]
        xid_after = xid_count(run_started)
        boot_after = boot_id()
        device_lost = count_device_lost(logs)
        record["faults"] = {
            "xid_delta": xid_after - xid_before,
            "boot_changed": boot_after != boot_before,
            "device_lost": device_lost,
        }
        record["faults_ok"] = (xid_after == xid_before) and (boot_after == boot_before) and sum(device_lost.values()) == 0
        per_n[str(n)] = record

        stop_climbing, reason = should_stop(
            vram_total_mib=record["vram_mib"], prev_total=prev_total, this_total=record["env_steps_per_s_total"]
        )
        prev_total = record["env_steps_per_s_total"]
        if stop_climbing:
            stop_reason = f"stopped after n={n}: {reason}"
            break

    chosen_n = choose_best_n(per_n)
    projection = None
    projection_note = (
        "D7 (plan2 decisions): training runs on a wall-clock budget, not a fixed step count, so the "
        "steps-needed figure below is an assumption for costing purposes, not a measured requirement. "
        "1,000,000 env steps/scene is used as a round, order-of-magnitude planning figure for a SAC agent "
        "converging on a single ~70x70 m crossing task; n_scenes=10 is the train split (s01-s10, spec §6)."
    )
    if chosen_n is not None:
        projection = project_cost(
            steps_per_s_total=per_n[str(chosen_n)]["env_steps_per_s_total"], steps_needed=1_000_000, n_scenes=10
        )
        projection["assumed_steps_needed_per_scene"] = 1_000_000
        projection["assumed_n_scenes"] = 10
        projection["note"] = projection_note

    gate = {
        "description": "Task 7: instance-scaling throughput measurement with an RL env in the loop (spec §8/§12).",
        "map": MAP_PATH,
        "candidate_ns": list(candidate_ns),
        "attempted_ns": attempted_ns,
        "warmup_s": warmup_s,
        "timed_window_target_s": timed_s,
        "vram_budget_mib": VRAM_BUDGET_MIB,
        "gpu_total_mib": gpu_total_mib,
        "baseline_vram_mib": baseline_used_mib,
        "per_n": per_n,
        "failures": failures,
        "chosen_n": chosen_n,
        "stop_reason": stop_reason,
        "projection": projection,
        "run_started": run_started,
        "run_finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if per_n:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(gate, indent=2) + "\n")
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "gates" / "m2_instances.json")
    parser.add_argument("--candidate-ns", type=int, nargs="+", default=list(CANDIDATE_NS))
    parser.add_argument("--warmup-s", type=float, default=WARMUP_S)
    parser.add_argument("--timed-s", type=float, default=TIMED_S)
    parser.add_argument("--launch-reply-timeout-s", type=float, default=LAUNCH_REPLY_TIMEOUT_S)
    args = parser.parse_args(argv)
    gate = run(
        args.out, candidate_ns=tuple(args.candidate_ns), warmup_s=args.warmup_s, timed_s=args.timed_s,
        launch_reply_timeout_s=args.launch_reply_timeout_s,
    )
    if not gate["per_n"]:
        print(f"no N could be measured safely: {gate['stop_reason']}", file=sys.stderr)
        return 1
    print(json.dumps(
        {"chosen_n": gate["chosen_n"], "stop_reason": gate["stop_reason"],
         "per_n_summary": {n: {"env_steps_per_s_total": r["env_steps_per_s_total"], "vram_mib": r["vram_mib"],
                               "faults_ok": r["faults_ok"]} for n, r in gate["per_n"].items()}},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
