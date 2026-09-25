"""SAC training on a packaged AutoFly scene (spec §8, Task 8).

`AutoFlyEnv` (closed for editing: `autofly_ue5/expert/env.py`) deliberately propagates four backend
hazards -- `CameraPoseError`, `StepTimingError`, `StaleStateError`, `CommandTimeoutError` (all raised from
`autofly_ue5/sim/airsim_backend.py`'s lock-stepped `_step_impl`) and `EpisodeSetupError` (raised only from
`reset()`'s call into `episode.apply_setup`). Task 7 measured `CameraPoseError` striking roughly once per a
few hundred to ~1000 environment steps under random actions -- common, not rare, especially early in SAC
training when the policy IS effectively random. Spec §7.1 documents it as a genuine, expected simulator
hazard (the Unreal actor a `set_pose` sweep left behind), not a code defect.

**A sixth hazard, found live during this task's own 30-minute shakedown, not in the spec's list**:
`pynng.exceptions.Timeout` -- a raw NNG transport-level timeout -- propagated uncaught straight out of the
third-party `projectairsim` client library (not through any of `airsim_backend.py`'s own named exceptions)
and crashed the run at ~3000 steps. Same character as `CommandTimeoutError` (a transient communication
timeout a fresh reset recovers from, not a logic error), so it is treated identically here -- see
`FAULT_ERRORS_STEP` -- rather than left to end the run. The broader `pynng.exceptions.NNGException` is
deliberately *not* caught: some of its other subclasses (`NotSupported`, `InvalidOperation`, ...) indicate
real configuration/protocol bugs that retrying cannot fix and should fail loudly, not be silently retried
forever on a 12-hour budget.

That crash also surfaced a second, independent finding: the top-level training process itself (this
script, not a `SubprocVecEnv` worker -- Task 7's chosen_n=1 means the real backend's connections live
in-process here) hung after printing its own traceback, never exiting, for the exact reason Task 7 already
documented for worker processes: the projectairsim client leaves a non-daemon thread alive, and CPython's
interpreter-shutdown sequence blocks forever joining it. `main()` therefore force-exits the process
(`os._exit`) after writing the gate record, rather than trusting a plain `return`/`sys.exit()` to actually
terminate it -- otherwise a 12-hour run's *own* process could hang forever exactly like a worker once did.

That shakedown's *second* attempt (after the two fixes above) surfaced a third, more fundamental bug, in
`autofly_ue5/gpu.py` (unclosed for this one fix; every other file in the closed list stayed closed):
`check_gpu_for_launch`'s "no simulator of ours is running, but the GPU looks busy -> assume a foreign job"
rule was written in M0, when nothing but the packaged simulator itself ever touched the GPU. Once a
training process shares the GPU with the simulator instance it relaunches, that rule is a false premise --
`used_mib` legitimately includes several GiB of *this training process's own* torch/CUDA context, which
`own_running == 0` (correctly reporting no simulator instance running, mid-relaunch) does nothing to
exclude. Measured live: `CameraPoseError` fired 10 times in 2,592 steps (~1/259, well above Task 7's
"few hundred to ~1000" random-action estimate) and one of the resulting relaunches correctly tore its
simulator down, then immediately failed to bring a new one up -- `check_gpu_for_launch` saw the training
process's own ~2.6 GiB and concluded another job owned the GPU, ending the run. Fixed: `own_process_gpu_mib()`
sums `nvidia-smi --query-compute-apps`' per-PID figures for this process's own tree (self + descendants,
via one `ps` snapshot) and `check_gpu_for_launch` subtracts it before judging idleness -- a genuinely
foreign process (outside our tree) still blocks a launch exactly as before. The separate free-headroom
check is deliberately left unadjusted: a new simulator instance still cannot use VRAM this process is
genuinely holding, whoever it belongs to.

Early SB3 versions' `SubprocVecEnv` propagate an uncaught worker exception straight out of the worker
process -- but Task 7 measured, live, that this does not cleanly kill the worker either: the projectairsim
client leaves a non-daemon background thread running, so CPython's interpreter-shutdown sequence
(`threading._shutdown()`) blocks forever joining it, the worker's pipe never reaches EOF, and any caller
doing a blind `remote.recv()` (SB3's own `step_wait()`/`env_method()`) hangs right along with it -- forever,
with no exception ever raised. `docs/gates/m2_instances.json` records this happening live, twice.

Two lines of defence, matching the two places that hazard can bite:

1. `ResilientAutoFlyEnv` (below) wraps the raw `AutoFlyEnv` *inside* whatever process runs it (in-process
   for the n=1 `DummyVecEnv` this project actually runs, per Task 7's `chosen_n=1`; inside an SB3
   `SubprocVecEnv` worker for n>1). It catches the fault classes above around both `reset()` and `step()`
   and retries `reset()` -- never the same `step()` -- since a `CameraPoseError` mid-step means the episode
   itself cannot continue (the drone's true position is unknown to be right), only that the SIMULATOR can
   recover from it, which spec §7.1 and Task 7's own measurement agree it almost always does on a fresh
   reset. This means the exception now NEVER reaches SB3's worker loop in the first place -- there is
   nothing for `step_wait()`/`env_method()`'s unbounded `recv()` to hang on, because the worker process
   never tries to exit abnormally. Every occurrence and every recovery is counted (see `fault_counts`/
   `recovered_counts`) so the run record reports how often this actually fired, not how often the spec
   guessed it would.
2. If in-place retries are exhausted (the connection itself, not just one sweep, looks broken),
   `ResilientAutoFlyEnv._relaunch()` tears the `Simulator` down and builds a brand new one at the same
   `instance` slot, *before* raising again. Tearing down reuses the exact bounded-wait/force-teardown
   machinery `scripts/measure_instances.py` already wrote and verified live (`sweep_stale_instances`) rather
   than reinventing it: a bounded background-thread join around `close()` (the same pattern as that
   script's `teardown()`, applied here to one `AutoFlyEnv` instead of a whole `SubprocVecEnv`, since a
   relaunch operates on a single worker, not the whole vec env), then an unconditional
   `sweep_stale_instances()` sweep, which stops the real OS-level Unreal process via this project's own
   pidfile bookkeeping independent of whether the Python-level `close()` call ever returned. Only after
   retries are exhausted does a relaunch happen -- "losing a worker must never lose the run" -- and only
   after every relaunch attempt is exhausted does this finally raise, ending the run loudly rather than
   hanging it or silently discarding data.

`make_vec_env` reuses `scripts/measure_instances.py`'s `_call_reset_with_timeout` for n>1's staggered first
launch (hazard #2: two `AutoFlyEnv`s hitting `check_gpu_for_launch` at once could together exceed the GPU
budget) and its `teardown` for closing the whole vec env in `main()`'s `finally` -- again, not reinvented.
`_step_all`'s specific technique (bypassing SB3's `step_async`/`step_wait()` with a bounded
`remote.poll()`) is Task 7's OWN measurement loop stepping manually; SB3's `SAC.learn()` owns its rollout
loop internally and this trainer does not step vec envs by hand, so there is no call site for it here --
line 1 above (never letting the exception leave the worker) is what makes that hazard moot for `.learn()`
itself.

Configuration values are D7/spec §8 fixed points, not tuned here: `MultiInputPolicy` + `POLICY_KWARGS`
(Task 6), `buffer_size=150_000` (~8.5 GiB dict-replay-buffer arithmetic, spec'd exactly so it is not raised
without redoing that sum), `learning_starts=5_000`, `batch_size=256`, `gamma=0.99`, `tau=0.005`,
`learning_rate=3e-4`. `optimize_memory_usage` is asserted unsupported for `DictReplayBuffer` by SB3 2.9
itself (`assert not optimize_memory_usage`, `stable_baselines3/common/buffers.py`), confirmed live in this
venv before writing this module -- it stays off, at the spec'd 150k buffer.

That same ~8.5 GiB-per-buffer arithmetic means `CheckpointCallback(save_replay_buffer=True)` -- which has no
retention policy of its own -- would otherwise let a 12-hour run accumulate on the order of 200 GiB of
replay-buffer pickles. `PruneOldReplayBuffersCallback`/`prune_old_replay_buffers` keep every model `.zip`
(cheap, the whole training history) but only the newest `DEFAULT_KEEP_REPLAY_BUFFERS` replay buffers (one
to resume from, one as a fallback if the newest was mid-write when a crash landed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
import time
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

from autofly_ue5.expert.env import AutoFlyEnv  # noqa: F401  (re-exported for callers of this module)
from autofly_ue5.expert.faults import (  # noqa: F401  (re-exported: moved from this module)
    FAULT_ERRORS_RESET,
    FAULT_ERRORS_STEP,
    KNOWN_FAULT_NAMES,
    combine_fault_summaries,
)
from autofly_ue5.expert.features import POLICY_KWARGS
from autofly_ue5.expert.resilient import (  # noqa: F401  (re-exported: moved from this module)
    DEFAULT_CLOSE_TIMEOUT_S,
    DEFAULT_MAX_RELAUNCH_ATTEMPTS,
    DEFAULT_MAX_RESET_ATTEMPTS,
    ResilientAutoFlyEnv,
    _bounded_close,
)
from autofly_ue5.expert.seeds import EVAL_SEED_BASE, WORKER_SEED_STRIDE, worker_seed_base  # noqa: F401
from autofly_ue5.expert.vec import call_reset_with_timeout, make_vec_env, teardown  # noqa: F401
from autofly_ue5.paths import ROOT, RUNS_DIR, SCENES_DIR
from autofly_ue5.scenes.model import Bounds, Instance, Layout, SceneFile, load_scene_file
from autofly_ue5.sim.process import instance_dir, sweep_stale_instances
from autofly_ue5.validate.engine_check import boot_id, count_device_lost, xid_count


# --------------------------------------------------------------------------------------------------------
# Fixed configuration (spec §8 / D7). See module docstring for the buffer-size arithmetic and why
# optimize_memory_usage stays off.
# --------------------------------------------------------------------------------------------------------
BUFFER_SIZE = 150_000
LEARNING_STARTS = 5_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4
TARGET_ENTROPY_EXPECTED = -3.0  # spec §8: asserted against SB3's "auto" default, never set directly.
CHECKPOINT_FREQ = 10_000
EVAL_FREQ = 25_000
EVAL_EPISODES = 20
DEFAULT_TOTAL_TIMESTEPS = 5_000_000  # an upper safety cap; D7 says the real budget is wall-clock (--hours).


def scene_and_layout(scene: str) -> tuple[SceneFile, Layout]:
    """Load a scene's SceneFile + its built Layout by scene id, e.g. "s01".

    Generalises `scripts/measure_instances.py`'s own scene_and_layout() (which hardcodes s01) so this
    trainer works for whichever scene s01-s10 is asked for, once that scene's level has been built.
    """
    matches = sorted(SCENES_DIR.glob(f"{scene}_*.json"))
    if not matches:
        raise FileNotFoundError(f"no scene file matching scenes/{scene}_*.json for scene {scene!r}")
    scene_file = load_scene_file(matches[0])
    layout_path = RUNS_DIR / "levels" / f"{scene}.layout.json"
    if not layout_path.is_file():
        raise FileNotFoundError(f"{layout_path} does not exist -- build this scene's level before training")
    raw = json.loads(layout_path.read_text())["layout"]
    b = Bounds(**raw["bounds"])
    inst = tuple(Instance(**i) for i in raw["instances"])
    return scene_file, Layout(scene_id=raw["scene_id"], seed=raw["seed"], bounds=b, instances=inst)



# --------------------------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------------------------
def build_model(
    env,
    *,
    device: str = "cuda",
    buffer_size: int = BUFFER_SIZE,
    learning_starts: int = LEARNING_STARTS,
    batch_size: int = BATCH_SIZE,
    gamma: float = GAMMA,
    tau: float = TAU,
    learning_rate: float = LEARNING_RATE,
    tensorboard_log: str | None = None,
    seed: int | None = None,
    verbose: int = 1,
) -> SAC:
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=POLICY_KWARGS,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        train_freq=1,
        gradient_steps=1,
        gamma=gamma,
        tau=tau,
        learning_rate=learning_rate,
        device=device,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=verbose,
    )
    assert float(model.target_entropy) == TARGET_ENTROPY_EXPECTED, (
        f"SB3's 'auto' target_entropy default for a 3-D action space changed: expected "
        f"{TARGET_ENTROPY_EXPECTED} (spec §8), got {model.target_entropy} -- investigate and set it "
        f"explicitly rather than silently trusting a new SB3 default"
    )
    return model


# --------------------------------------------------------------------------------------------------------
# Resume support
# --------------------------------------------------------------------------------------------------------
_CHECKPOINT_RE = re.compile(r"^rl_model_(\d+)_steps\.zip$")


def newest_checkpoint(directory: Path) -> Path | None:
    """The checkpoint with the highest step count in `directory`, or None if there is none.

    Sorts by the parsed step count, not by filename string: "rl_model_90000_steps.zip" must sort after
    "rl_model_200000_steps.zip" is wrong lexicographically ("9" > "2"), which would silently resume from a
    stale model while reporting success.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = [(int(m.group(1)), p) for p in directory.iterdir() if (m := _CHECKPOINT_RE.match(p.name))]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def replay_buffer_for(checkpoint: Path) -> Path:
    """The replay-buffer pickle CheckpointCallback(save_replay_buffer=True) writes alongside `checkpoint`."""
    m = _CHECKPOINT_RE.match(Path(checkpoint).name)
    if not m:
        raise ValueError(f"{checkpoint} does not look like a checkpoint (rl_model_<N>_steps.zip)")
    return checkpoint.with_name(f"rl_model_replay_buffer_{m.group(1)}_steps.pkl")


# --------------------------------------------------------------------------------------------------------
# Checkpoint retention: SB3's CheckpointCallback has no retention policy at all. Each replay-buffer pickle
# is ~8.5 GiB (buffer_size=150_000's own arithmetic, see the module docstring); over a 12-hour run at the
# production checkpoint_freq=10_000 that is dozens of them -- on the order of 200 GiB -- for a resume path
# that only ever needs the newest one. Model checkpoints (.zip) are a few tens of MB each and are the run's
# whole training history, so those are kept forever; only replay buffers are pruned.
# --------------------------------------------------------------------------------------------------------
_REPLAY_BUFFER_RE = re.compile(r"^rl_model_replay_buffer_(\d+)_steps\.pkl$")
DEFAULT_KEEP_REPLAY_BUFFERS = 2  # one to resume from, one as a fallback if the newest was mid-write on a crash


def prune_old_replay_buffers(checkpoints_dir: Path, keep: int = DEFAULT_KEEP_REPLAY_BUFFERS) -> list[Path]:
    """Deletes every `rl_model_replay_buffer_<N>_steps.pkl` under `checkpoints_dir` except the `keep` ones
    with the highest step count. Never touches model checkpoints (`rl_model_<N>_steps.zip`). Returns the
    paths actually deleted."""
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.is_dir() or keep < 0:
        return []
    candidates = sorted(
        ((int(m.group(1)), p) for p in checkpoints_dir.iterdir() if (m := _REPLAY_BUFFER_RE.match(p.name))),
        key=lambda t: t[0],
        reverse=True,
    )
    to_delete = [p for _, p in candidates[keep:]]
    for p in to_delete:
        p.unlink()
    return to_delete


class PruneOldReplayBuffersCallback(BaseCallback):
    """Runs prune_old_replay_buffers() on the same cadence CheckpointCallback saves on, immediately after
    it (callbacks in a CallbackList run in list order within one _on_step(), so the just-written replay
    buffer is already on disk when this checks). Placed after CheckpointCallback in main()'s CallbackList."""

    def __init__(self, save_freq: int, checkpoints_dir: Path, keep: int = DEFAULT_KEEP_REPLAY_BUFFERS, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._save_freq = save_freq
        self._checkpoints_dir = Path(checkpoints_dir)
        self._keep = keep

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq == 0:
            deleted = prune_old_replay_buffers(self._checkpoints_dir, keep=self._keep)
            if deleted and self.verbose:
                print(f"pruned {len(deleted)} old replay buffer(s): {[p.name for p in deleted]}")
        return True


# --------------------------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------------------------
class StopOnWallClock(BaseCallback):
    """D7's wall-clock training budget. SB3 ships StopTrainingOnMaxEpisodes/...OnNoModelImprovement/
    ...OnRewardThreshold but nothing time-based; this is the "StopTrainingOnMaxTime-style callback" the
    brief calls for. The deadline is wall-clock from construction, checked every env step."""

    def __init__(self, hours: float, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._deadline = time.monotonic() + hours * 3600.0

    def _on_step(self) -> bool:
        if time.monotonic() >= self._deadline:
            if self.verbose:
                print(f"StopOnWallClock: wall-clock budget reached at {self.num_timesteps} timesteps; stopping")
            return False
        return True


class OutcomeHistogramCallback(BaseCallback):
    """Tallies the outcome at every completed episode (terminated or truncated), across training --
    "if success_rate is flat at ~0 after 100k steps ... check the outcome histogram" (brief).

    A `ResilientAutoFlyEnv` fault-truncation's `info["outcome"]` is the FRESH episode's own initial value
    ("running" -- see `AutoFlyEnv._info`), not a real outcome of the episode that just got cut short; that
    episode's actual fate is `info["sim_fault"]` (the error class name). Live finding (Task 8 shakedown3, a
    45-fault run): without preferring `sim_fault` when present, every fault-truncated episode was tallied
    as "running", indistinguishable from a policy/reward-shaping bucket rather than a backend hazard -- this
    keeps backend faults out of the same buckets as genuine policy outcomes (success/collision/out_of_bounds
    /timeout), mirroring the same separation Task 9's own gate now requires.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.histogram: Counter[str] = Counter()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                self.histogram[info.get("sim_fault") or info.get("outcome", "unknown")] += 1
        return True


def read_eval_results(run_root: Path) -> dict:
    """Reads EvalCallback's own evaluations.npz (written under `<run_root>/eval_logs/`), if any."""
    npz_path = run_root / "eval_logs" / "evaluations.npz"
    result = {
        "n_evaluations": 0,
        "timesteps": [],
        "final_mean_reward": None,
        "best_mean_reward": None,
        "final_success_rate": None,
        "best_success_rate": None,
    }
    if not npz_path.is_file():
        return result
    data = np.load(npz_path, allow_pickle=True)
    result["timesteps"] = [int(t) for t in data["timesteps"].tolist()]
    result["n_evaluations"] = len(result["timesteps"])
    if "results" in data.files:
        means = [float(np.mean(r)) for r in data["results"]]
        if means:
            result["final_mean_reward"] = means[-1]
            result["best_mean_reward"] = max(means)
    if "successes" in data.files:
        rates = [float(np.mean(s)) for s in data["successes"]]
        if rates:
            result["final_success_rate"] = rates[-1]
            result["best_success_rate"] = max(rates)
    return result


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="s01")
    p.add_argument("--instances", type=int, default=1)
    p.add_argument("--hours", type=float, default=None, help="wall-clock training budget; omit for none (rely on --total-timesteps alone)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "gates" / "m2_train.json")
    p.add_argument("--map-path", default=None, help='default: "/Game/AutoFly/Maps/<SCENE upper-cased>"')
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    p.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    p.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    p.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    run_root = RUNS_DIR / "expert" / args.scene
    checkpoints_dir = run_root / "checkpoints"
    best_dir = run_root / "best"
    tb_dir = run_root / "tensorboard"
    monitor_dir = run_root / "monitor"
    eval_monitor_dir = monitor_dir / "eval"
    for d in (checkpoints_dir, best_dir, tb_dir, monitor_dir, eval_monitor_dir):
        d.mkdir(parents=True, exist_ok=True)

    map_path = args.map_path or f"/Game/AutoFly/Maps/{args.scene.upper()}"
    scene_file, layout = scene_and_layout(args.scene)

    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    xid_before = xid_count(run_started)
    boot_before = boot_id()

    swept = sweep_stale_instances()  # hazard #3: a crashed earlier run can leave Unreal children holding VRAM
    if swept:
        print(f"swept stale instances before starting: {swept}", file=sys.stderr)

    t_wall_start = time.monotonic()
    train_env: VecEnv | None = None
    eval_env: VecEnv | None = None
    model: SAC | None = None
    num_timesteps_at_start = 0  # a resumed run's env_steps_per_s must reflect only steps taken THIS session
    resume_path: Path | None = None
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    outcome_cb = OutcomeHistogramCallback()
    status = "failed"
    error_message: str | None = None

    try:
        train_env = make_vec_env(
            scene_file, layout, args.instances, map_path=map_path, monitor_dir=monitor_dir,
            seed_base_fn=worker_seed_base, instance_offset=0,
        )
        # A separate simulator instance (own ports), one slot past the training workers, so EvalCallback
        # can run concurrently with training without colliding on ports with any training worker.
        eval_env = make_vec_env(
            scene_file, layout, 1, map_path=map_path, monitor_dir=eval_monitor_dir,
            seed_base_fn=lambda _rank: EVAL_SEED_BASE, instance_offset=args.instances,
        )

        if args.resume:
            resume_path = newest_checkpoint(checkpoints_dir)
            if resume_path is None:
                raise RuntimeError(f"--resume given but no checkpoint found under {checkpoints_dir}")
            buffer_path = replay_buffer_for(resume_path)
            if not buffer_path.is_file():
                raise RuntimeError(
                    f"--resume given but {buffer_path} (the replay buffer for {resume_path.name}) is "
                    f"missing -- a SAC resume without its replay buffer restarts exploration from scratch, "
                    f"which this trainer refuses to do silently"
                )
            model = SAC.load(resume_path, env=train_env, device=args.device, tensorboard_log=str(tb_dir))
            model.load_replay_buffer(buffer_path)
            num_timesteps_at_start = int(model.num_timesteps)
            print(f"resumed from {resume_path} (num_timesteps={model.num_timesteps}) with replay buffer {buffer_path}")
        else:
            model = build_model(
                train_env, device=args.device, buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                batch_size=args.batch_size, tensorboard_log=str(tb_dir), seed=args.seed,
            )

        checkpoint_save_freq = max(args.checkpoint_freq // args.instances, 1)
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_save_freq, save_path=str(checkpoints_dir),
            name_prefix="rl_model", save_replay_buffer=True,
        )
        # Immediately after CheckpointCallback in the list (same _on_step(), same save_freq): keeps every
        # model .zip forever but only the newest DEFAULT_KEEP_REPLAY_BUFFERS replay buffers -- see the
        # prune_old_replay_buffers()/PruneOldReplayBuffersCallback docstring for why.
        prune_cb = PruneOldReplayBuffersCallback(save_freq=checkpoint_save_freq, checkpoints_dir=checkpoints_dir, verbose=1)
        eval_cb = EvalCallback(
            eval_env, n_eval_episodes=args.eval_episodes, eval_freq=max(args.eval_freq // args.instances, 1),
            best_model_save_path=str(best_dir), log_path=str(run_root / "eval_logs"), deterministic=True,
        )
        callbacks: list[BaseCallback] = [checkpoint_cb, prune_cb, eval_cb, outcome_cb]
        if args.hours is not None:
            callbacks.append(StopOnWallClock(args.hours))

        model.learn(
            total_timesteps=args.total_timesteps, callback=CallbackList(callbacks),
            tb_log_name=args.scene, reset_num_timesteps=not args.resume,
        )

        checkpoint_path = run_root / "final.zip"
        model.save(checkpoint_path)
        checkpoint_sha256 = sha256_of(checkpoint_path)
        status = "ok"
    except Exception as err:
        error_message = f"{type(err).__name__}: {err}"
        print(f"training failed: {error_message}", file=sys.stderr)
        traceback.print_exc()  # the one-line summary above is not enough to diagnose an unclassified fault
    finally:
        # Every step here is itself wrapped: a failure while cleaning up after a failure must still reach
        # the gate-record write below (docs/gates/m2_train.json must describe what happened even when
        # teardown itself hits a surprise, e.g. the live TOCTOU race sweep_stale_instances() now guards
        # against) rather than crashing main() before it can write anything.
        fault_summaries: list[dict] = []
        for env in (train_env, eval_env):
            if env is None:
                continue
            try:
                fault_summaries.extend(env.env_method("get_fault_summary"))
            except Exception as err:
                print(f"WARNING: could not collect a fault summary: {type(err).__name__}: {err}", file=sys.stderr)
            try:
                teardown(env)  # bounded close + force-kill + sweep_stale_instances -- reused, not reinvented
            except Exception as err:
                print(f"WARNING: teardown() raised {type(err).__name__}: {err}; continuing cleanup", file=sys.stderr)
                traceback.print_exc()
        try:
            sweep_stale_instances()
        except Exception as err:
            print(f"WARNING: final sweep_stale_instances() raised {type(err).__name__}: {err}", file=sys.stderr)

    wall_s = time.monotonic() - t_wall_start
    num_timesteps = int(model.num_timesteps) if model is not None else 0
    num_timesteps_this_session = num_timesteps - num_timesteps_at_start

    logs = [instance_dir(i) / "sim.log" for i in range(args.instances + 1) if (instance_dir(i) / "sim.log").is_file()]
    xid_after = xid_count(run_started)
    boot_after = boot_id()
    device_lost = count_device_lost(logs)
    faults_ok = (xid_after == xid_before) and (boot_after == boot_before) and sum(device_lost.values()) == 0

    gate = {
        "description": f"Task 8: SAC training on scene {args.scene} (spec §8).",
        "scene": args.scene,
        "map": map_path,
        "instances": args.instances,
        "status": status,
        "error": error_message,
        "resume": args.resume,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "config": {
            "policy": "MultiInputPolicy",
            "policy_kwargs": "autofly_ue5.expert.features.POLICY_KWARGS",
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "train_freq": 1,
            "gradient_steps": 1,
            "gamma": GAMMA,
            "tau": TAU,
            "learning_rate": LEARNING_RATE,
            "device": args.device,
            "target_entropy_expected": TARGET_ENTROPY_EXPECTED,
            "checkpoint_freq": args.checkpoint_freq,
            "keep_replay_buffers": DEFAULT_KEEP_REPLAY_BUFFERS,
            "eval_freq": args.eval_freq,
            "eval_episodes": args.eval_episodes,
            "hours_budget": args.hours,
            "total_timesteps_cap": args.total_timesteps,
            "seed": args.seed,
        },
        "num_timesteps": num_timesteps,
        "num_timesteps_at_start": num_timesteps_at_start,
        "num_timesteps_this_session": num_timesteps_this_session,
        "wall_clock_s": wall_s,
        # Steps taken THIS session only -- a resumed run's num_timesteps includes steps from a previous
        # session that took no wall-clock time in this one, which would otherwise inflate this figure.
        "env_steps_per_s": (num_timesteps_this_session / wall_s) if wall_s > 0 else None,
        "eval": read_eval_results(run_root),
        "outcome_histogram": dict(outcome_cb.histogram),
        "backend_faults": combine_fault_summaries(fault_summaries),
        "engine_faults": {
            "xid_delta": xid_after - xid_before,
            "boot_changed": boot_after != boot_before,
            "device_lost": device_lost,
        },
        "faults_ok": faults_ok,
        "checkpoint": {
            "path": str(checkpoint_path) if checkpoint_path is not None else None,
            "sha256": checkpoint_sha256,
        },
        "run_started": run_started,
        "run_finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"status": status, "num_timesteps": num_timesteps, "wall_clock_s": wall_s, "faults_ok": faults_ok}, indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    _code = main()
    # Not sys.exit(): Task 7 measured live (and this task's own shakedown reproduced, in this very
    # process, not just a SubprocVecEnv worker) that the projectairsim client can leave a non-daemon
    # thread alive, which blocks CPython's interpreter-shutdown sequence forever -- the process prints its
    # own traceback/summary and then never actually exits, so run_job.sh's wrapper never sees an exit code
    # and the job looks permanently "still running". Everything that must be durable (the gate JSON) is
    # already written by main() above; os._exit() skips the thread-join step entirely and guarantees this
    # process actually terminates.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
