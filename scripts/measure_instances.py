"""Instance-scaling throughput measurement (spec §8's throughput gate, Task 7).

Measures environment steps/s (one `command_velocity` + one `sim.step(0.2)` + one `observe()` + the
observation encoding, i.e. one `AutoFlyEnv.step()`) for N concurrent workers, N in (1, 2, 4, 6), on the
packaged s01 simulator. The workers are training's own (`make_vec_env`: resilient wrappers, DummyVecEnv at
N=1, SubprocVecEnv above), with the scene config given by --scene-config (the clock rate lives there). Random
actions only -- this measures the simulator, not a policy. Stops climbing N the moment VRAM would exceed the
budget or total throughput falls versus the previous N (see `should_stop`).

Safety, in order:
  1. Stop orphaned instances (a crashed earlier run can leave Unreal children holding VRAM) -- never a live run's.
  2. Stagger the workers' first reset(): each `AutoFlyEnv` launches its simulator lazily inside its first
     reset(), so N workers resetting at once would all read the same pre-launch VRAM figure and could
     collectively exhaust the GPU. This script launches instances one at a time, bounded, checking VRAM
     headroom before each.
  3. Tear down every instance of the current N on any failure path (try/finally), and only those.
  4. Sample VRAM only inside the window where all N instances are actively stepping.

env -u PYTHONPATH .venv/bin/python scripts/measure_instances.py --scene-config scene_autofly_s01_fast.jsonc \
    --out docs/gates/m2_instances.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from autofly_ue5.expert.faults import combine_fault_summaries
from autofly_ue5.expert.seeds import WORKER_SEED_STRIDE  # noqa: F401  (kept for callers of this script)
from autofly_ue5.expert.vec import (  # noqa: F401  (moved from this script; names kept for its callers)
    LAUNCH_REPLY_TIMEOUT_S,
    VEC_ENV_CLOSE_TIMEOUT_S,
    call_method_with_timeout,
    call_reset_with_timeout,
    collect_fault_summaries,
    make_vec_env,
    teardown,
)
from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import ROOT, RUNS_DIR
from autofly_ue5.scenes.model import Bounds, Instance, Layout, SceneFile, load_scene_file
from autofly_ue5.sim.airsim_backend import ProjectAirSimSimulator, scene_config_factory, scene_config_record
from autofly_ue5.sim.process import SIM_RUN_DIR, instance_dir, sweep_orphaned_instances
from autofly_ue5.validate.engine_check import audit_engine_faults, boot_id, xid_count

MAP_PATH = "/Game/AutoFly/Maps/S01"
CANDIDATE_NS = (1, 2, 4, 6)
WARMUP_S = 30.0
TIMED_S = 180.0
VRAM_BUDGET_MIB = 20_000
# M1's measured figure for one packaged instance while capturing; used only as a conservative pre-launch
# guard between stagger steps, not as the recorded measurement (that comes from live nvidia-smi samples).
EST_PER_INSTANCE_MIB = 1_882
VRAM_SAMPLE_INTERVAL_S = 15.0
MAX_ATTEMPTS_PER_N = 2
VRAM_DRAIN_TIMEOUT_S = 30.0
VRAM_DRAIN_MARGIN_MIB = 400
# How long to wait for one worker's reply to a single step() before treating it as hung. Measured live:
# the same non-daemon-thread hang that afflicted the staggered launch also happens mid-measurement, not
# just at launch -- a worker's step() auto-reset-on-done can hit a real, documented-but-rare simulator
# hazard (CameraPoseError: "the Unreal actor was probably stopped by a sweep", spec Sec7.1) that leaves it
# hung the same way a spawn crash did. A single step is milliseconds in the successful case (M0: ~135 ms
# at 7.43 steps/s), so 30s is already two orders of magnitude of margin, not a tight bound.
STEP_REPLY_TIMEOUT_S = 30.0


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


def wait_for_vram_drop(baseline_mib: int, margin_mib: int = VRAM_DRAIN_MARGIN_MIB, timeout_s: float = VRAM_DRAIN_TIMEOUT_S) -> int:
    """Poll until VRAM has drained back near `baseline_mib` (a torn-down UE process can take a moment to
    release its allocation) or `timeout_s` elapses; returns the last-seen used_mib either way."""
    deadline = time.monotonic() + timeout_s
    used, _ = gpu_memory_mib()
    while used > baseline_mib + margin_mib and time.monotonic() < deadline:
        time.sleep(1.0)
        used, _ = gpu_memory_mib()
    return used


_call_reset_with_timeout = call_reset_with_timeout  # the name this script used before the move


def _random_actions(vec_env: VecEnv, n: int) -> np.ndarray:
    return np.stack([vec_env.action_space.sample() for _ in range(n)])


def _launch(vec_env: VecEnv, index: int, timeout_s: float) -> None:
    """Launch one worker's simulator (inside its first reset). In-process for n=1 -- the same DummyVecEnv training
    uses -- and bounded across the pipe for SubprocVecEnv workers."""
    if isinstance(vec_env, SubprocVecEnv):
        call_reset_with_timeout(vec_env, index, timeout_s)
    else:
        vec_env.env_method("reset", indices=[index])


def _step_all(vec_env: VecEnv, actions: np.ndarray, timeout_s: float = STEP_REPLY_TIMEOUT_S) -> tuple[list[bool], int]:
    """Advance every worker by one step, bounded; returns each worker's `done` flag and how many of those dones
    were backend-fault truncations (resilient workers end a faulted episode instead of raising).

    Deliberately bypasses SB3's `step_async()`/`step_wait()` (together, equivalent to this): `step_wait()`
    does an unbounded `remote.recv()` per worker, which is exactly the hazard `_call_reset_with_timeout`
    exists for, except here it can strike mid-measurement rather than only at launch -- measured live, a
    worker's step() auto-reset-on-done hit a real CameraPoseError (a documented, expected-to-be-rare
    hazard, not a code defect) and hung the whole orchestrator in step_wait() for the rest of the run,
    holding the real Unreal process's VRAM the entire time. This script only needs `done` (for
    episodes_completed) and the step count, never the observation/reward/info SB3 normally stacks, so
    there is no need to reconstruct step_wait()'s full return shape.
    """
    if not isinstance(vec_env, SubprocVecEnv):
        _obs, _rewards, done_array, infos = vec_env.step(actions)
        return [bool(d) for d in done_array], sum(1 for info in infos if info.get("sim_fault"))
    remotes = vec_env.remotes
    for remote, action in zip(remotes, actions, strict=True):
        remote.send(("step", action))
    deadline = time.monotonic() + timeout_s
    dones: list[bool] = []
    faults = 0
    for i, remote in enumerate(remotes):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not remote.poll(remaining):
            raise TimeoutError(
                f"worker {i} did not reply to step() within {timeout_s}s -- it likely crashed without "
                f"exiting (check the job log for a worker traceback) and must be torn down forcibly"
            )
        _obs, _reward, done, info, _reset_info = remote.recv()
        dones.append(bool(done))
        faults += int(bool(info.get("sim_fault")))
    return dones, faults


def _step_for(vec_env: VecEnv, n: int, duration_s: float, timeout_s: float = STEP_REPLY_TIMEOUT_S) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        _step_all(vec_env, _random_actions(vec_env, n), timeout_s)


def measure_n(
    n: int, scene: SceneFile, layout: Layout, warmup_s: float = WARMUP_S, timed_s: float = TIMED_S,
    launch_reply_timeout_s: float = LAUNCH_REPLY_TIMEOUT_S, step_reply_timeout_s: float = STEP_REPLY_TIMEOUT_S,
    *, sim_factory: Callable[[], object] = ProjectAirSimSimulator, vram_reader: Callable[[], tuple[int, int]] = gpu_memory_mib,
    sim_root: Path = SIM_RUN_DIR, monitor_dir: Path | None = None, map_path: str = MAP_PATH,
) -> dict:
    """Launch n workers (staggered, VRAM-checked between each), warm up, then measure random-action throughput over
    a fixed wall-clock window. Always tears down its own workers before returning or raising.

    The workers are exactly training's (`make_vec_env`: resilient, DummyVecEnv at n=1, SubprocVecEnv above). The
    2026-09-16 measurement used bare AutoFlyEnvs, so its n=2 run died on an unrecovered CameraPoseError hang and
    chosen_n=1 recorded a crash, not a throughput comparison."""
    monitor_dir = monitor_dir or RUNS_DIR / "m2" / "measure_monitor" / f"n{n}"
    vec_env: VecEnv | None = None
    per_instance_launch_s: list[float] = []
    try:
        vec_env = make_vec_env(scene, layout, n, map_path=map_path, monitor_dir=monitor_dir, sim_factory=sim_factory,
                               sim_root=sim_root, launch=False)
        launch_start = time.monotonic()
        for i in range(n):
            used, _total = vram_reader()
            projected = used + EST_PER_INSTANCE_MIB
            if projected > VRAM_BUDGET_MIB:
                raise RuntimeError(
                    f"launching instance {i} of {n} would push VRAM to an estimated {projected} MiB "
                    f"(currently {used} MiB with {i} instance(s) up), over the {VRAM_BUDGET_MIB} MiB budget"
                )
            t0 = time.monotonic()
            _launch(vec_env, i, launch_reply_timeout_s)  # this ONE worker's launch + first reset, bounded
            per_instance_launch_s.append(time.monotonic() - t0)
        launch_s = time.monotonic() - launch_start

        _step_for(vec_env, n, warmup_s, step_reply_timeout_s)  # untimed: shader warm-up, first-episode settle

        vram_samples: list[int] = []
        next_sample = time.monotonic()
        total_steps = 0
        episodes_completed = 0
        fault_truncations = 0
        window_start = time.monotonic()
        window_end = window_start + timed_s
        while time.monotonic() < window_end:
            dones, faults = _step_all(vec_env, _random_actions(vec_env, n), step_reply_timeout_s)
            total_steps += n
            episodes_completed += sum(dones) - faults
            fault_truncations += faults
            now = time.monotonic()
            if now >= next_sample:
                vram_samples.append(vram_reader()[0])
                next_sample = now + VRAM_SAMPLE_INTERVAL_S
        elapsed_s = time.monotonic() - window_start
        if not vram_samples:  # window shorter than one sample interval: still sample once, inside the window
            vram_samples.append(vram_reader()[0])

        total_per_s = total_steps / elapsed_s
        summaries, missing = collect_fault_summaries(vec_env, step_reply_timeout_s)
        return {
            "env_steps_per_s_total": total_per_s,
            "env_steps_per_s_per_instance": total_per_s / n,
            "vram_mib": max(vram_samples),
            "vram_samples_mib": vram_samples,
            "launch_s": launch_s,
            "per_instance_launch_s": per_instance_launch_s,
            "episodes_completed": episodes_completed,
            "fault_truncations": fault_truncations,
            "backend_faults": combine_fault_summaries(summaries),
            "backend_faults_missing_workers": missing,
            "total_env_steps": total_steps,
            "warmup_s": warmup_s,
            "timed_window_s": elapsed_s,
        }
    finally:
        stopped = teardown(vec_env, range(n), sim_root)
        print(f"n={n}: stopped its own slots: {stopped}", file=sys.stderr)


def run(
    out_path: Path, candidate_ns: tuple[int, ...] = CANDIDATE_NS, warmup_s: float = WARMUP_S, timed_s: float = TIMED_S,
    launch_reply_timeout_s: float = LAUNCH_REPLY_TIMEOUT_S, step_reply_timeout_s: float = STEP_REPLY_TIMEOUT_S,
    scene_config: str = "scene_autofly_s01.jsonc",
) -> dict:
    sim_factory = scene_config_factory(scene_config)
    scene, layout = scene_and_layout()
    baseline_used_mib, gpu_total_mib = gpu_memory_mib()
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")

    initial_sweep = sweep_orphaned_instances()
    if initial_sweep:
        print(f"swept orphaned instances before starting: {initial_sweep}", file=sys.stderr)
        baseline_used_mib = wait_for_vram_drop(baseline_used_mib)

    per_n: dict[str, dict] = {}
    attempted_ns: list[int] = []
    failures: dict[str, dict] = {}
    stop_reason = "measured every candidate N without hitting a stop condition"
    prev_total: float | None = None

    for n in candidate_ns:
        attempted_ns.append(n)
        n_start_epoch = time.time()
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
                    n, scene, layout, warmup_s=warmup_s, timed_s=timed_s, launch_reply_timeout_s=launch_reply_timeout_s,
                    step_reply_timeout_s=step_reply_timeout_s, sim_factory=sim_factory,
                )
            except Exception as err:
                last_error = f"{type(err).__name__}: {err}"
                print(f"n={n} attempt {attempts}/{MAX_ATTEMPTS_PER_N} failed: {last_error}", file=sys.stderr)

        if record is None:
            failures[str(n)] = {"attempts": attempts, "error": last_error}
            stop_reason = f"n={n} failed to launch/measure after {attempts} attempt(s): {last_error}"
            break

        record["faults"] = audit_engine_faults(since=run_started, since_epoch=n_start_epoch, xid_before=xid_before,
                                               boot_before=boot_before, log_dirs=[instance_dir(i) for i in range(n)])
        record["faults_ok"] = record["faults"].pop("ok")
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
        "scene_config": scene_config_record(scene_config),
        "resilient": True,  # training's own workers (make_vec_env); the 2026-09-16 record used bare AutoFlyEnvs
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
    parser.add_argument("--step-reply-timeout-s", type=float, default=STEP_REPLY_TIMEOUT_S)
    parser.add_argument("--scene-config", default="scene_autofly_s01.jsonc",
                        help="e.g. scene_autofly_s01_fast.jsonc (1 ms clock); recorded with its sha256")
    args = parser.parse_args(argv)
    gate = run(
        args.out, candidate_ns=tuple(args.candidate_ns), warmup_s=args.warmup_s, timed_s=args.timed_s,
        launch_reply_timeout_s=args.launch_reply_timeout_s, step_reply_timeout_s=args.step_reply_timeout_s,
        scene_config=args.scene_config,
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
    _code = main()
    # Not sys.exit(): an abandoned close() can leave projectairsim's non-daemon thread blocking interpreter
    # shutdown forever (see expert/train.py). Everything durable is already written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
