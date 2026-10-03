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
script, not a `SubprocVecEnv` worker -- at n=1 the real backend's connections live in-process here) hung after printing its own traceback, never exiting, for the exact reason Task 7 already
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
with no exception ever raised. Task 7's record (`docs/gates/archive/2026-09-17-m2-run1/m2_instances.json`)
shows this happening live, twice.

Two lines of defence, matching the two places that hazard can bite:

1. `ResilientAutoFlyEnv` (below) wraps the raw `AutoFlyEnv` *inside* whatever process runs it (in-process
   for an n=1 `DummyVecEnv`; inside an SB3
   `SubprocVecEnv` worker for n>1). It catches the fault classes above around both `reset()` and `step()`.
   A fault in `reset()` is retried in place. A fault mid-`step()` ends the episode by truncation on its last
   real observation -- never retrying the same `step()`, since a `CameraPoseError` mid-step means the episode
   itself cannot continue (the drone's true position is unknown to be right), only that the SIMULATOR can
   recover from it on a fresh reset, which spec §7.1 and Task 7's own measurement agree it almost always
   does. That transition is not data: `FaultFilteringDictReplayBuffer` drops it, `FaultAwareEvalCallback`
   replays the episode, and neither Monitor nor `rollout/success_rate` counts it (C2, 2026-09-24 review --
   before it, the wrapper reset internally and SAC stored the next episode's start as this step's outcome).
   This means the exception NEVER reaches SB3's worker loop in the first place -- there is
   nothing for `step_wait()`/`env_method()`'s unbounded `recv()` to hang on, because the worker process
   never tries to exit abnormally. Every occurrence and every recovery is counted (see `fault_counts`/
   `recovered_counts`) so the run record reports how often this actually fired, not how often the spec
   guessed it would.
2. If in-place retries are exhausted (the connection itself, not just one sweep, looks broken),
   `ResilientAutoFlyEnv._relaunch()` tears the `Simulator` down and builds a brand new one at the same
   `instance` slot, *before* raising again. Tearing down is a bounded background-thread join around the old
   simulator's `close()` (the same pattern as `autofly_ue5.expert.vec.teardown()`, applied to one simulator),
   then `stop_instance()` on this slot alone, which stops the real OS-level Unreal process via this project's
   own pidfile bookkeeping independent of whether the Python-level `close()` call ever returned. (Until the
   2026-09-24 review this was a sweep of EVERY simulator on the host, so a relaunch in the training slot also
   killed the eval simulator and vice versa: Task 8's run recorded that feedback loop as all 77 of its Timeout
   faults and 14 of its 23 relaunches.) Only after
   retries are exhausted does a relaunch happen -- "losing a worker must never lose the run" -- and only
   after every relaunch attempt is exhausted does this finally raise, ending the run loudly rather than
   hanging it or silently discarding data.

`make_vec_env` (`autofly_ue5.expert.vec`) uses `call_reset_with_timeout` for n>1's staggered first launch
(hazard #2: two `AutoFlyEnv`s hitting `check_gpu_for_launch` at once could together exceed the GPU budget), and
`teardown` closes each vec env in `main()`'s `finally`, stopping only this run's own slots.
`_step_all`'s specific technique (bypassing SB3's `step_async`/`step_wait()` with a bounded
`remote.poll()`) is Task 7's OWN measurement loop stepping manually; SB3's `SAC.learn()` owns its rollout
loop internally and this trainer does not step vec envs by hand, so there is no call site for it here --
line 1 above (never letting the exception leave the worker) is what makes that hazard moot for `.learn()`
itself.

Configuration values are D7/spec §8 fixed points, not tuned here: `MultiInputPolicy` + `POLICY_KWARGS`
(Task 6), `buffer_size=150_000` (8.5 GB of dict replay buffer with one float32 depth frame, 12.7 GB with a dynamic
scene's three float16 frames -- `replay_buffer_bytes`; checked against the host by `host_preflight` before a run
starts), `learning_starts=5_000`, `batch_size=256`, `gamma=0.99`, `tau=0.005`,
`learning_rate=3e-4`. `optimize_memory_usage` is asserted unsupported for `DictReplayBuffer` by SB3 2.9
itself (`assert not optimize_memory_usage`, `stable_baselines3/common/buffers.py`), confirmed live in this
venv before writing this module -- it stays off, at the spec'd 150k buffer.

That same per-buffer size means `CheckpointCallback(save_replay_buffer=True)` -- which has no
retention policy of its own -- would otherwise let a 12-hour run accumulate on the order of 200 GB of
replay-buffer pickles. `PruneOldReplayBuffersCallback`/`prune_old_replay_buffers` keep every model `.zip`
(cheap, the whole training history) but only the newest `DEFAULT_KEEP_REPLAY_BUFFERS` replay buffers (one
to resume from, one as a fallback if the newest was mid-write when a crash landed).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

from autofly_ue5.expert.env import AutoFlyEnv  # noqa: F401  (re-exported for callers of this module)
from autofly_ue5.expert.env import action_space
from autofly_ue5.expert.faults import (  # noqa: F401  (re-exported: moved from this module)
    FAULT_ERRORS_RESET,
    FAULT_ERRORS_STEP,
    KNOWN_FAULT_NAMES,
    combine_fault_summaries,
)
from autofly_ue5.evidence import default_evidence_path, record_destination, refuse_existing_evidence
from autofly_ue5.expert.evaluate import FaultAwareEvalCallback
from autofly_ue5.expert.features import POLICY_KWARGS
from autofly_ue5.expert.obs import (
    DEPTH_SIZE,
    MOVER_FEATURES,
    VECTOR_DIM,
    ObsConfig,
    obs_config_for_frames,
    obs_config_for_scene,
    obs_config_from_space,
)
from autofly_ue5.expert.resilient import (  # noqa: F401  (re-exported: moved from this module)
    DEFAULT_CLOSE_TIMEOUT_S,
    DEFAULT_MAX_RELAUNCH_ATTEMPTS,
    DEFAULT_MAX_RESET_ATTEMPTS,
    FaultFilteringDictReplayBuffer,
    ResilientAutoFlyEnv,
    _bounded_close,
)
from autofly_ue5.expert.reward import REWARD_VERSION, reward_config_for_scene
from autofly_ue5.expert.seeds import (  # noqa: F401  (EVAL_SEED_BASE etc. re-exported for callers)
    EVAL_CALLBACK_SEED_BASE,
    EVAL_SEED_BASE,
    SESSION_SEED_STRIDE,
    WORKER_SEED_STRIDE,
    session_seed_base,
    worker_seed_base,
)
from autofly_ue5.expert.vec import call_reset_with_timeout, collect_fault_summaries, make_vec_env, teardown  # noqa: F401
from autofly_ue5.expert.warmstart import PolicyWarmupSAC, warm_start
from autofly_ue5.expert.warmstart import plan as warm_plan
from autofly_ue5.paths import RUNS_DIR
from autofly_ue5.scenes.model import Layout, SceneFile
from autofly_ue5.scenes.resolve import ResolvedScene, resolve_scene
from autofly_ue5.sim.airsim_backend import scene_config_factory, scene_config_record
from autofly_ue5.sim.process import SIM_RUN_DIR, instance_dir, stop_instances, sweep_orphaned_instances
from autofly_ue5.validate.engine_check import audit_engine_faults, boot_id, xid_count


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
    """A scene's SceneFile and the Layout it flies, by scene id (e.g. "s01", or "s01d", which flies s01's layout).
    Kept for its callers; `autofly_ue5.scenes.resolve.resolve_scene` also gives the map and default config."""
    resolved = resolve_scene(scene)
    return resolved.scene, resolved.layout



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
    algorithm: type[SAC] = SAC,
) -> SAC:
    model = algorithm(
        "MultiInputPolicy",
        env,
        policy_kwargs=POLICY_KWARGS,
        replay_buffer_class=FaultFilteringDictReplayBuffer,  # backend-fault transitions are never stored (C2)
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        train_freq=1,
        # -1: as many gradient steps as transitions collected (C3). At n=1 this is exactly 1, as before; with n
        # workers, gradient_steps=1 would have done ONE update per n transitions.
        gradient_steps=-1,
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
# Run isolation: one run directory holds one run, and each resumed session draws fresh episodes.
# --------------------------------------------------------------------------------------------------------
SESSIONS_FILE = "sessions.json"


# What every run recorded before 2026-10-02 (no "identity" in its sessions.json) trained on: static s01 with one float32
# depth frame. Such a run may be resumed only under that observation.
LEGACY_IDENTITY = {"obs_config": {"depth_frames": 1, "depth_dtype": "float32"}, "dynamic": None}


def default_train_record(run_root: Path, scene: str, *, resume: bool) -> Path:
    """The default --out: docs/gates/<milestone>_train.json for a run's first session, and
    <milestone>_train_session<k>.json for resumed session k -- each session's record is evidence of its own, so a
    resume must not need (or be refused for) the first session's file. Read-only: the session is claimed later."""
    sessions_path = Path(run_root) / SESSIONS_FILE
    session = len(json.loads(sessions_path.read_text())["sessions"]) if resume and sessions_path.is_file() else 0
    return default_evidence_path(scene, "train" if session == 0 else f"train_session{session}")


def run_identity(resolved: ResolvedScene, obs_config: ObsConfig, scene_config: str | None = None) -> dict:
    """What a run's replay buffer and checkpoints are tied to: resuming under anything else would mix two tasks, two
    observation shapes or two simulator clocks (the scene config holds the clock rate) in one buffer."""
    dynamic = resolved.scene.dynamic
    return {
        "scene_config": scene_config_record(scene_config) if scene_config is not None else None,
        "scene": resolved.scene.id,
        "scene_sha256": resolved.scene.sha256,
        "base_scene": resolved.base_id,
        "base_layout_sha256": resolved.layout_sha256,
        "obs_config": obs_config.to_json(),
        "dynamic": json.loads(json.dumps(dataclasses.asdict(dynamic))) if dynamic is not None else None,
        # the coefficients the run trains on (spec §8's defaults plus the scene's own; JSON-normalised like the rest)
        "reward_config": json.loads(json.dumps(dataclasses.asdict(reward_config_for_scene(resolved.scene)))),
    }


def _identity_mismatch(recorded: dict | None, identity: dict) -> list[str]:
    if recorded is None:  # a run from before identities were recorded
        return [k for k, v in LEGACY_IDENTITY.items() if identity.get(k) != v]
    return [k for k in sorted(set(recorded) | set(identity)) if recorded.get(k) != identity.get(k)]


def prepare_run_root(run_root: Path, *, resume: bool, reward_version: str, seed: int, identity: dict | None = None,
                     warm_start: dict | None = None) -> int:
    """Claim `run_root` for this training session and return the session index (0 for a fresh run).

    A fresh run refuses a directory that already holds a run's results: its old checkpoints would otherwise sit next
    to the new ones, and a later --resume picks the highest step count -- possibly the OLD run's. A directory whose
    earlier sessions produced no checkpoint and no final.zip holds nothing to protect, so a fresh run may reuse it
    (the likeliest early failures on this shared host -- a foreign GPU job, a failed launch -- happen before the
    first checkpoint). A resume refuses a run recorded under another reward version (its replay buffer holds the other
    objective's rewards), one with no sessions record at all (it predates this record, and its buffer still holds
    unfilterable backend-fault rows), and one with no checkpoint + replay buffer to continue from -- all before a
    session is recorded. With an `identity` (`run_identity`), a resume also refuses a run recorded under another scene,
    scene file, layout, observation config or motion parameters (a run from before identities counts as
    `LEGACY_IDENTITY`); the identity is recorded with the session.
    """
    run_root = Path(run_root)
    sessions_path = run_root / SESSIONS_FILE
    has_results = (run_root / "final.zip").is_file() or newest_checkpoint(run_root / "checkpoints") is not None
    if resume:
        if not sessions_path.is_file():
            raise RuntimeError(
                f"--resume: {sessions_path} is missing, so the run's reward version is unknown -- it predates run "
                f"records and cannot be resumed safely; start a fresh run with a new --run-root"
            )
        sessions = json.loads(sessions_path.read_text())["sessions"]
        recorded = sessions[-1].get("reward_version") if sessions else None
        if recorded != reward_version:
            raise RuntimeError(
                f"--resume: {run_root} was trained under reward version {recorded!r}, this code computes "
                f"{reward_version!r}; resuming would mix two objectives in one replay buffer -- start a fresh run"
            )
        if identity is not None:
            mismatch = _identity_mismatch(sessions[-1].get("identity"), identity)
            if mismatch:
                raise RuntimeError(
                    f"--resume: {run_root} was trained on a different {', '.join(mismatch)} "
                    f"(recorded {sessions[-1].get('identity', LEGACY_IDENTITY)}, now {identity}); its replay buffer and "
                    f"checkpoints belong to that task -- start a fresh run with a new --run-root")
        checkpoint = newest_checkpoint(run_root / "checkpoints")
        if checkpoint is None or not replay_buffer_for(checkpoint).is_file():
            raise RuntimeError(
                f"--resume: {run_root} has no checkpoint with its replay buffer to continue from (the previous session "
                f"ended before its first one); start it fresh instead -- the same command without --resume"
            )
        session = len(sessions)
    else:
        if has_results:
            hint = ("pass --resume to continue it, or a new --run-root for a fresh run" if sessions_path.is_file() else
                    f"it predates run records (no {SESSIONS_FILE}) and cannot be resumed; use a new --run-root")
            raise RuntimeError(f"{run_root} already holds a training run's checkpoints or final.zip; {hint}")
        sessions = []
        session = 0
    session_seed_base(0, session)  # raises before any work if this session would overflow its seed slice
    run_root.mkdir(parents=True, exist_ok=True)
    entry = {"index": session, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "resume": resume,
             "reward_version": reward_version, "seed": seed}
    if identity is not None:
        entry["identity"] = identity
    if warm_start is not None:
        entry["warm_start"] = warm_start  # the checkpoint session 0 started from (expert.warmstart)
    sessions.append(entry)
    sessions_path.write_text(json.dumps({"sessions": sessions}, indent=2) + "\n")
    return session


def reseed_resumed_model(model: SAC, *, seed: int, session: int) -> None:
    """SAC.load re-seeds the env with the saved seed, so a resumed session's first resets would replay session 0's
    first episodes. Session k's explicit seeds are seed + k * SESSION_SEED_STRIDE (+ rank), below every counter range."""
    model.set_random_seed(seed + session * SESSION_SEED_STRIDE)


# --------------------------------------------------------------------------------------------------------
# Host preflight: the replay buffer must fit in RAM next to the simulators, and its pickles on disk.
# --------------------------------------------------------------------------------------------------------
# Left for everything else a run holds in RAM: up to 5 packaged simulators, the SubprocVecEnv workers (each imports
# torch) and the trainer itself. Not measured per process; generous on a 61 GB host whose run 2 used an 8.5 GB buffer.
RAM_HEADROOM_BYTES = 16 * 2**30
DISK_HEADROOM_BYTES = 10 * 2**30


def warm_start_source(path: Path | None, obs_config: ObsConfig, *, resume: bool) -> dict | None:
    """Check a --warm-start checkpoint before anything is claimed: a fresh run only, a readable SB3 zip, and the same
    depth stack as this run (its mover input may be absent: expert.warmstart adds the branch). Its path and sha256."""
    if path is None:
        return None
    if resume:
        raise ValueError("--warm-start starts a fresh run from a checkpoint; it cannot be combined with --resume")
    if not Path(path).is_file():
        raise FileNotFoundError(f"--warm-start: no checkpoint at {path}")
    from stable_baselines3.common.save_util import load_from_zip_file

    from stable_baselines3.sac.policies import MultiInputPolicy

    data, params, _variables = load_from_zip_file(path, device="cpu", load_data=True)
    source = obs_config_from_space(data["observation_space"])
    # The new network, built offline on the CPU, against the same mapping warm_start() applies: any parameter it
    # cannot take (another depth stack, slot count or network) is refused here, before a run root or a simulator.
    policy = MultiInputPolicy(obs_config.space(), action_space(), lambda _progress: LEARNING_RATE, **POLICY_KWARGS)
    try:
        warm_plan({name: tuple(t.shape) for name, t in policy.state_dict().items()}, params["policy"])
    except ValueError as err:
        raise ValueError(f"--warm-start: {path} ({source.to_json()}) cannot seed this run ({obs_config.to_json()}): "
                         f"{err}") from err
    return {"path": str(path), "sha256": sha256_of(Path(path)), "source_obs_config": source.to_json()}


def observation_config(scene, *, depth_frames: int | None = None, mover_slots: int | None = None) -> ObsConfig:
    """The scene's default observation (obs.obs_config_for_scene) with each command-line override applied to its own
    field only. A static scene has no movers to report, so it refuses a mover input."""
    config = obs_config_for_scene(scene)
    if depth_frames is not None:
        frames = obs_config_for_frames(depth_frames)
        config = dataclasses.replace(config, depth_frames=frames.depth_frames, depth_dtype=frames.depth_dtype)
    if mover_slots is not None:
        if mover_slots and scene.dynamic is None:
            raise ValueError(f"scene {scene.id} is static: it has no movers to report (--mover-slots {mover_slots})")
        config = dataclasses.replace(config, mover_slots=mover_slots)
    return config


def replay_buffer_bytes(buffer_size: int, obs_config: ObsConfig) -> int:
    """SB3's DictReplayBuffer for this observation: obs and next_obs (depth + vector + movers) per transition, plus the
    action, reward, done and timeout columns. 150k transitions: 8.47 GB for one float32 frame, 12.70 GB for three
    float16, and 29 MB more for s01d's mover input."""
    depth = obs_config.depth_frames * DEPTH_SIZE * DEPTH_SIZE * np.dtype(obs_config.depth_dtype).itemsize
    vector = VECTOR_DIM * 4
    movers = obs_config.mover_slots * MOVER_FEATURES * 4
    return int(buffer_size) * (2 * (depth + vector + movers) + 3 * 4 + 4 + 4 + 4)


def available_ram_bytes() -> int:
    """MemAvailable from /proc/meminfo (the kernel's estimate of what can be allocated without swapping)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable line")


def host_preflight(*, buffer_bytes: int, run_root: Path) -> dict:
    """Refuse a run whose replay buffer will not fit: in RAM (plus RAM_HEADROOM_BYTES for the simulators and workers),
    or on disk as the DEFAULT_KEEP_REPLAY_BUFFERS pickles kept plus the one being written. Returns what it measured,
    for the run record."""
    existing = Path(run_root)
    while not existing.exists():
        existing = existing.parent
    report = {
        "replay_buffer_bytes": int(buffer_bytes),
        "available_ram_bytes": available_ram_bytes(),
        "ram_needed_bytes": int(buffer_bytes) + RAM_HEADROOM_BYTES,
        "free_disk_bytes": shutil.disk_usage(existing).free,
        "disk_needed_bytes": (DEFAULT_KEEP_REPLAY_BUFFERS + 1) * int(buffer_bytes) + DISK_HEADROOM_BYTES,
    }
    problems = []
    if report["available_ram_bytes"] < report["ram_needed_bytes"]:
        problems.append(f"RAM: {report['available_ram_bytes'] / 1e9:.1f} GB available, {report['ram_needed_bytes'] / 1e9:.1f} "
                        f"GB needed (a {buffer_bytes / 1e9:.1f} GB replay buffer plus {RAM_HEADROOM_BYTES / 2**30:.0f} GiB)")
    if report["free_disk_bytes"] < report["disk_needed_bytes"]:
        problems.append(f"disk: {report['free_disk_bytes'] / 1e9:.1f} GB free under {existing}, "
                        f"{report['disk_needed_bytes'] / 1e9:.1f} GB needed for the replay-buffer pickles")
    if problems:
        raise RuntimeError("host preflight failed: " + "; ".join(problems) + " -- lower --buffer-size or free memory")
    report["ok"] = True
    return report


# --------------------------------------------------------------------------------------------------------
# Checkpoint retention: SB3's CheckpointCallback has no retention policy at all. Each replay-buffer pickle
# is 8.5-12.7 GB (replay_buffer_bytes; see the module docstring); over a 12-hour run at the
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

    A `ResilientAutoFlyEnv` fault-truncation is tallied under `info["sim_fault"]` (the error class name), not
    under an outcome. Live finding (Task 8 shakedown3, a 45-fault run): when the wrapper still reset internally,
    such an episode's `info["outcome"]` was the fresh episode's "running", indistinguishable from a
    policy/reward-shaping bucket -- this keeps backend faults out of the same buckets as genuine policy outcomes
    (success/collision/out_of_bounds/timeout), mirroring the same separation Task 9's own gate requires.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.histogram: Counter[str] = Counter()
        # What each collision hit (spec §6.5): "sim" (a static pillar, or physical contact), "mover" (the d_col rule),
        # "mover_inferred" (a backend fault right next to a mover, scored as a collision).
        self.collision_sources: Counter[str] = Counter()
        # The last OUTCOME_WINDOW real episodes, for TensorBoard's outcomes/* curves (watchers, runbook-m2d step 5).
        self._recent: deque[str] = deque(maxlen=self.OUTCOME_WINDOW)
        self._fault_episodes = 0

    OUTCOME_WINDOW = 100

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            self.histogram[info.get("sim_fault") or info.get("outcome", "unknown")] += 1
            if info.get("sim_fault"):
                self._fault_episodes += 1
                if getattr(self, "model", None) is not None:
                    self.logger.record("outcomes/sim_fault_episodes", self._fault_episodes)
                continue
            label = info.get("outcome", "unknown")
            if label == "collision":
                source = info.get("collision_source") or "sim"
                self.collision_sources[source] += 1
                label = f"collision_{source}"
            self._recent.append(label)
            if getattr(self, "model", None) is not None:  # SB3 attaches the model (and its logger) in init_callback
                self._log()
        return True

    def _log(self) -> None:
        n = len(self._recent)
        counts = Counter(self._recent)
        for outcome in ("success", "out_of_bounds", "timeout"):
            self.logger.record(f"outcomes/{outcome}", counts[outcome] / n)
        self.logger.record("outcomes/collision", sum(v for k, v in counts.items() if k.startswith("collision_")) / n)
        for source in ("sim", "mover", "mover_inferred"):
            self.logger.record(f"outcomes/collision_{source}", counts[f"collision_{source}"] / n)
        self.logger.record("outcomes/sim_fault_episodes", self._fault_episodes)


def read_eval_results(run_root: Path) -> dict:
    """Reads the evaluation callback's evaluations.npz (SB3's format, under `<run_root>/eval_logs/`), if any."""
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
    p.add_argument("--out", type=Path, default=None,
                   help="the run record; default docs/gates/<milestone>_train.json for a scene with a milestone (s01: "
                        "m2, s01d: m2d). A committed record is never written over")
    p.add_argument("--map-path", default=None, help="default: the map of the level the scene flies (s01d: S01)")
    p.add_argument("--scene-config", default=None,
                   help="Project AirSim scene config in configs/ (default scene_autofly_<level>.jsonc); e.g. "
                        "scene_autofly_s01_fast.jsonc for the 1 ms clock once M1 has passed on it")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0, help=f"0 <= seed < {SESSION_SEED_STRIDE - 64} (keeps SB3's explicit reset seeds out of every counter range)")
    p.add_argument("--run-root", type=Path, default=None,
                   help="where checkpoints, best/, final.zip and logs go; default runs/expert/<scene>. A fresh run "
                        "refuses a directory that already holds one")
    p.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    p.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    p.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    p.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--depth-frames", type=int, default=None,
                   help="depth frames the expert sees (default: 1 for a static scene, 3 for a dynamic one; >1 is float16)")
    p.add_argument("--warm-start", type=Path, default=None,
                   help="a fresh run starts from this checkpoint's weights (expert.warmstart): same network, plus any new "
                        "input branch starting at zero; its warm-up acts with that policy. Not with --resume")
    p.add_argument("--actor-freeze-updates", type=int, default=0,
                   help="with --warm-start: gradient updates during which only the critic learns (expert.warmstart)")
    p.add_argument("--mover-slots", type=int, default=None,
                   help="nearby moving pillars the expert is told about (default: 4 on a dynamic scene, 0 on a static "
                        "one; 0 reproduces s01d_r1's observation)")
    p.add_argument("--sim-root", type=Path, default=SIM_RUN_DIR,
                   help="where this run's simulators are recorded (tests point it at a scratch directory)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not 0 <= args.seed < SESSION_SEED_STRIDE - 64:
        print(f"--seed must be in [0, {SESSION_SEED_STRIDE - 64}); got {args.seed}", file=sys.stderr)
        return 2
    run_root = args.run_root or RUNS_DIR / "expert" / args.scene
    sim_root = Path(args.sim_root)
    # Everything that can be checked without a simulator is checked before the run directory is claimed.
    try:
        out_path = args.out if args.out is not None else default_train_record(run_root, args.scene, resume=args.resume)
        refuse_existing_evidence(out_path)
        resolved = resolve_scene(args.scene)
        scene_file, layout = resolved.scene, resolved.layout
        map_path = args.map_path or resolved.map_path
        scene_config = args.scene_config or resolved.default_scene_config
        scene_config_record(scene_config)  # the config file must exist; its hash goes in the record
        obs_config = observation_config(scene_file, depth_frames=args.depth_frames, mover_slots=args.mover_slots)
        warm_source = warm_start_source(args.warm_start, obs_config, resume=args.resume)
        if args.actor_freeze_updates and warm_source is None:
            raise ValueError("--actor-freeze-updates is a warm start's critic warm-up: it needs --warm-start")
        identity = run_identity(resolved, obs_config, scene_config)
        host = host_preflight(buffer_bytes=replay_buffer_bytes(args.buffer_size, obs_config), run_root=run_root)
        session = prepare_run_root(run_root, resume=args.resume, reward_version=REWARD_VERSION, seed=args.seed,
                                   identity=identity, warm_start=warm_source)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as err:
        print(f"refusing to start: {err}", file=sys.stderr)
        return 2
    sim_factory = scene_config_factory(scene_config, resolved.movable_objects, run_root=sim_root)
    checkpoints_dir = run_root / "checkpoints"
    best_dir = run_root / "best"
    tb_dir = run_root / "tensorboard"
    # Per session: Monitor truncates an existing <instance>.monitor.csv, which would erase a resumed run's history.
    monitor_dir = run_root / "monitor" / f"session{session}"
    eval_monitor_dir = monitor_dir / "eval"
    for d in (checkpoints_dir, best_dir, tb_dir, monitor_dir, eval_monitor_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    run_start_epoch = time.time()
    xid_before = xid_count(run_started)
    boot_before = boot_id()

    train_slots = list(range(args.instances))
    eval_slot = args.instances

    t_wall_start = time.monotonic()
    train_env: VecEnv | None = None
    eval_env: VecEnv | None = None
    model: SAC | None = None
    num_timesteps_at_start = 0  # a resumed run's env_steps_per_s must reflect only steps taken THIS session
    resume_path: Path | None = None
    warm_record: dict | None = None
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    outcome_cb = OutcomeHistogramCallback()
    eval_cb: FaultAwareEvalCallback | None = None
    status = "failed"
    error_message: str | None = None

    try:
        # Hazard #3: a crashed earlier run can leave Unreal children holding VRAM. Only orphans -- a concurrently
        # running job's simulators are not ours to stop.
        swept = sweep_orphaned_instances(sim_root)
        if swept:
            print(f"swept orphaned instances before starting: {swept}", file=sys.stderr)
        train_env = make_vec_env(
            scene_file, layout, args.instances, map_path=map_path, monitor_dir=monitor_dir,
            seed_base_fn=lambda rank: session_seed_base(rank, session), instance_offset=0, sim_factory=sim_factory,
            sim_root=sim_root, obs_config=obs_config,
        )
        # A separate simulator instance (own ports), one slot past the training workers, so evaluation
        # can run concurrently with training without colliding on ports with any training worker. Its own seed
        # range, disjoint from the M2 gate's EVAL_SEED_BASE episodes.
        eval_env = make_vec_env(
            scene_file, layout, 1, map_path=map_path, monitor_dir=eval_monitor_dir,
            seed_base_fn=lambda _rank: EVAL_CALLBACK_SEED_BASE, instance_offset=args.instances, sim_factory=sim_factory,
            sim_root=sim_root, obs_config=obs_config,
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
            # gradient_steps=-1 explicitly: a checkpoint saved before C3 restores gradient_steps=1.
            model = SAC.load(resume_path, env=train_env, device=args.device, tensorboard_log=str(tb_dir), gradient_steps=-1)
            model.load_replay_buffer(buffer_path)
            if not isinstance(model.replay_buffer, FaultFilteringDictReplayBuffer):
                raise RuntimeError(
                    f"{buffer_path} holds a {type(model.replay_buffer).__name__}, not a FaultFilteringDictReplayBuffer: "
                    f"it predates backend-fault filtering and may contain fabricated fault transitions"
                )
            reseed_resumed_model(model, seed=args.seed, session=session)
            num_timesteps_at_start = int(model.num_timesteps)
            print(f"resumed from {resume_path} (num_timesteps={model.num_timesteps}) with replay buffer {buffer_path}")
        else:
            model = build_model(
                train_env, device=args.device, buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                batch_size=args.batch_size, tensorboard_log=str(tb_dir), seed=args.seed,
                algorithm=PolicyWarmupSAC if warm_source is not None else SAC,
            )
            if warm_source is not None:
                warm_record = {**warm_source, **warm_start(model, Path(warm_source["path"]))}
                model.actor_freeze_updates = args.actor_freeze_updates
                print(f"warm-started from {warm_source['path']}: {len(warm_record['widened'])} heads widened, "
                      f"{len(warm_record['fresh'])} new tensors, {warm_record['copied']} copied")

        checkpoint_save_freq = max(args.checkpoint_freq // args.instances, 1)
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_save_freq, save_path=str(checkpoints_dir),
            name_prefix="rl_model", save_replay_buffer=True,
        )
        # Immediately after CheckpointCallback in the list (same _on_step(), same save_freq): keeps every
        # model .zip forever but only the newest DEFAULT_KEEP_REPLAY_BUFFERS replay buffers -- see the
        # prune_old_replay_buffers()/PruneOldReplayBuffersCallback docstring for why.
        prune_cb = PruneOldReplayBuffersCallback(save_freq=checkpoint_save_freq, checkpoints_dir=checkpoints_dir, verbose=1)
        # The same args.eval_episodes episodes every evaluation (seeds EVAL_CALLBACK_SEED_BASE + i), faults replayed,
        # every args.eval_freq timesteps; best_model.zip on a new best mean return (SB3's criterion).
        eval_cb = FaultAwareEvalCallback(
            eval_env.envs[0], n_eval_episodes=args.eval_episodes, eval_freq=args.eval_freq,
            seed_base=EVAL_CALLBACK_SEED_BASE, best_model_save_path=best_dir, log_path=run_root / "eval_logs",
            deterministic=True, verbose=1,
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
        # teardown itself hits a surprise) rather than crashing main() before it can write anything.
        fault_summaries: list[dict] = []
        fault_summaries_missing: list[int] = []  # slots whose counts could not be read: the totals exclude them
        for env, slots in ((train_env, train_slots), (eval_env, [eval_slot])):
            if env is None:
                continue
            collected, missing = collect_fault_summaries(env)
            fault_summaries.extend(collected)
            fault_summaries_missing.extend(slots[i] for i in missing)
            try:
                teardown(env, slots, sim_root)  # bounded close + force-kill + stop this run's own slots
            except Exception as err:
                print(f"WARNING: teardown() raised {type(err).__name__}: {err}; continuing cleanup", file=sys.stderr)
                traceback.print_exc()
        try:
            stop_instances(train_slots + [eval_slot], sim_root)  # belt-and-braces: every slot this run used, nothing else
        except Exception as err:
            print(f"WARNING: final stop_instances() raised {type(err).__name__}: {err}", file=sys.stderr)

    wall_s = time.monotonic() - t_wall_start
    num_timesteps = int(model.num_timesteps) if model is not None else 0
    num_timesteps_this_session = num_timesteps - num_timesteps_at_start

    # Every log each slot wrote during the run -- the live sim.log AND the sim-backup-*.log each relaunch rotated it
    # to (Task 8's record scanned only the final session's sim.log, missing 24 in-run logs) -- and a readable journal.
    # Every part of the record that reads something the run left behind is guarded: on 2026-10-03 a malformed fault
    # summary crashed this assembly and s01d_r1's 5.5 h session left no record at all.
    record_errors: dict[str, str] = {}

    def guarded(key: str, compute, fallback):
        try:
            return compute()
        except Exception as err:
            record_errors[key] = f"{type(err).__name__}: {err}"
            traceback.print_exc()
            return fallback

    engine_faults = guarded(
        "engine_faults",
        lambda: audit_engine_faults(since=run_started, since_epoch=run_start_epoch, xid_before=xid_before,
                                    boot_before=boot_before,
                                    log_dirs=[instance_dir(i, sim_root) for i in train_slots + [eval_slot]]),
        {"ok": False})
    faults_ok = engine_faults.pop("ok", False)

    gate = {
        "description": f"SAC training on scene {args.scene} (spec §8; Task 8 of plan 2, §6.5 for a dynamic scene).",
        "scene": args.scene,
        "map": map_path,
        "identity": identity,
        "obs_config": obs_config.to_json(),
        "host": host,
        "instances": args.instances,
        "status": status,
        "error": error_message,
        "resume": args.resume,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "warm_start": warm_record,
        "run_root": str(run_root),
        "scene_config": scene_config_record(scene_config),
        "session": session,
        "reward_version": REWARD_VERSION,
        "seed_bases": [session_seed_base(rank, session) for rank in train_slots],
        "eval_callback_seed_base": EVAL_CALLBACK_SEED_BASE,
        "config": {
            "policy": "MultiInputPolicy",
            "policy_kwargs": "autofly_ue5.expert.features.POLICY_KWARGS",
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "actor_freeze_updates": args.actor_freeze_updates,
            "batch_size": args.batch_size,
            # What the model actually trained with, not what this file intends (a resumed model restores its own).
            "train_freq": str(model.train_freq) if model is not None else None,
            "gradient_steps": model.gradient_steps if model is not None else None,
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
        "eval": guarded("eval", lambda: read_eval_results(run_root), None),
        "outcome_histogram": dict(outcome_cb.histogram),
        "collision_sources": dict(outcome_cb.collision_sources),
        # Backend-fault transitions dropped from the replay buffer (C2); with n workers a dropped row costs n. Counted
        # by the buffer itself, so across every session of a resumed run (backend_faults is this session's).
        "dropped_fault_rows": int(getattr(getattr(model, "replay_buffer", None), "dropped_fault_rows", 0)),
        "dropped_fault_transitions": int(getattr(getattr(model, "replay_buffer", None), "dropped_transitions", 0)),
        "interrupted_evaluations": eval_cb.interrupted_evaluations if eval_cb is not None else 0,
        "backend_faults": guarded("backend_faults", lambda: combine_fault_summaries(fault_summaries), None),
        "backend_faults_missing_slots": fault_summaries_missing,
        "engine_faults": engine_faults,
        "faults_ok": faults_ok,
        "checkpoint": {
            "path": str(checkpoint_path) if checkpoint_path is not None else None,
            "sha256": checkpoint_sha256,
        },
        "run_started": run_started,
        "run_finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "record_errors": record_errors,
    }
    destination = record_destination(out_path, did_work=num_timesteps_this_session > 0)
    if destination != out_path:
        print(f"no step was trained: this record is not evidence and goes to {destination}, not {out_path}",
              file=sys.stderr)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(gate, indent=2) + "\n")
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
