# M2 SAC Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A SAC agent that flies scene s01 from a random start pose to a random target with ≥ 95 % success over 200 fresh episodes, plus the measured throughput numbers needed to project the cost of the remaining nine experts.

**Architecture:** A pure-function core (observation encoding, reward, episode sampling) that is fully testable offline against `FakeSimulator`, wrapped by one Gymnasium environment per simulator instance, trained by stable-baselines3 SAC over a `SubprocVecEnv`. Tasks 1–6 need no GPU and no simulator; Tasks 7–9 are live.

**Tech Stack:** Python 3.12, stable-baselines3 2.9 (SAC), torch 2.14 (CUDA 13), gymnasium 1.3, numpy 1.26.4, the existing `autofly_ue5.sim` interface, Project AirSim 1.0.2 on UE 5.7.4.

**Spec:** `docs/superpowers/specs/2026-09-15-autofly-ue5-dataset-design.md` — §8 (SAC expert), §9 (episode protocol), §7.1 (measured simulator contract), §6 (scenes).

**Predecessor:** `docs/superpowers/plans/2026-09-15-autofly-ue5-plan1-platform-and-scene-s01.md` (M0 + M1, complete). Its decision record is `docs/decisions/2026-09-15-plan1-rulings.md`.

## Global Constraints

Copied from the spec and from M0/M1's measured contract. Every task's requirements implicitly include this section.

- **Python is always run as `env -u PYTHONPATH .venv/bin/python`** from ROOT (`/home/jk_edge/research_uav/autofly_ue5`). A stray `PYTHONPATH` breaks imports. Tests: `env -u PYTHONPATH .venv/bin/python -m pytest -q`.
- **Packages are installed only with `uv pip install --python .venv/bin/python`.** Never `pip install` into system Python, never create a second virtualenv. `numpy` must stay at **1.26.4** — `projectairsim` 1.0.2 is built against it.
- **Action space is exactly `[v_forward ∈ [0, 2] m/s, yaw_rate ∈ [−1, 1] rad/s, v_z ∈ [−1, 1] m/s]`** (spec §8), one command per **0.2 s** step, `v_z` positive **up**.
- **Observation is depth-only plus a privileged target vector** (spec §8). RGB is never an expert input. The target's *appearance* is never an expert input.
- **Depth arrives with `+inf` for no-hit** (spec §7.1, ruling R5). Every consumer must handle `+inf` explicitly; never assume finite.
- **A simulator session must `reset()` before its first `step()`** (spec §7.1) — frame 0 of a session is corrupt. Both `ProjectAirSimSimulator` and `FakeSimulator` raise `autofly_ue5.sim.types.SessionNotResetError` otherwise.
- **World frame is NED metres, yaw radians, +z down.** Altitude above ground = `−z`. The `Pose` and `Observation` types in `autofly_ue5/sim/types.py` are not to be changed.
- **Success is `≤ 5 m` horizontal distance and `≤ 15°`** (spec §8); step limit **300** (60 s simulated).
- **Reward coefficients start at `k_p = 1, k_h = 0.1, k_t = 0.01, R_s = 10, R_c = 10, R_b = 5`** and are tuned on s01 only (spec §8).
- **Rendering is reproducible to a few grey levels, never bit-exact** (spec §7.1). Never gate anything on exact pixel equality.
- **Nothing outside ROOT is written or deleted** (ruling R3). Long jobs go through `scripts/run_job.sh`.
- **Every committed evidence file under `docs/gates/` is a record of a real run.** Never hand-edit one, never regenerate one to make a gate pass.
- Commit messages end with: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

## Decisions taken while planning

These are decisions the paper and the spec do not settle. The user's standing instruction is to pick a reasonable function and continue, recording the choice.

- **D1 — Target and distractors are built from the one obstacle asset available.** `assets/registry.json` currently holds exactly two assets (`cylinder`, `grid` ground); the 60-target pool is M4 work. This does **not** block M2: the expert's observation is depth plus a privileged target vector, so it never sees what the target *looks like*. M2 spawns the target and 3–5 distractors as `cylinder` instances with distinct materials, which keeps the physical scene and the depth image honest. M4 swaps real assets in by name only. *Cost if wrong:* none to the expert; M3's RGB frames would be visually monotonous, which is M4's problem to fix before any dataset is used for training a student.
- **D2 — Bearing enters the observation as `sin`/`cos`, not as a raw angle.** A raw bearing has a discontinuity at ±π that a network has to waste capacity learning around. *Cost if wrong:* one extra input dimension.
- **D3 — Depth is downsampled by block-minimum, not by area averaging.** Averaging across a depth discontinuity invents intermediate depths and can erase a thin near obstacle; a minimum is the conservative choice for obstacle avoidance and can only ever report an obstacle as *closer* than it is. The 256→84 ratio is not an integer, so the block boundaries come from `np.linspace(0, 256, 85).astype(int)` and `np.minimum.reduceat`, which is exact and needs no OpenCV. *Cost if wrong:* the agent sees obstacles slightly fattened, which biases it toward caution.
- **D4 — `+inf` depth maps to the clip distance (30 m), i.e. 1.0 after normalisation.** Sky and no-hit read as "maximally far", which is what they are.
- **D5 — Start and target are sampled on *opposite* edges.** `band_mask` measures distance to the nearest edge, so the start band (2–6 m) and target band (0–3 m) are rings, not sides. An episode that started and ended on the same edge would be a metre-long flight. M2 samples a start edge uniformly from the four, then places the target on the opposite one, which makes every episode a ~65 m crossing — matching AutoFly's scenes and the 300-step limit (65 m at 2 m/s = 33 s = 163 steps). *Cost if wrong:* the expert never learns short episodes; M3 can add them if the dataset needs them.
- **D6 — No curriculum in the first training run.** The paper reports none. If the run stalls below 95 %, the levers, in order, are: shrink the start–target distance and grow it as success rises; raise `k_p`; lengthen the step limit. *Cost if wrong:* one wasted training run, which Task 7's throughput numbers will have already priced.
- **D7 — Training runs on a wall-clock budget with checkpoint-and-resume, not to a fixed step count.** The step count needed is unknown and the throughput is measured only at Task 7. *Cost if wrong:* none; resume makes the budget a soft stop.
- **D8 — The evaluation episodes are drawn from a seed stream disjoint from training's.** A shared stream would let an "unseen" evaluation episode be one the agent trained on.

## File Structure

```
autofly_ue5/expert/
  __init__.py
  obs.py          # depth + vector encoding; pure functions (Task 2)
  reward.py       # reward and termination; pure functions (Task 3)
  episode.py      # episode setup sampling from a scene + layout (Task 4)
  env.py          # gymnasium Env over the Simulator protocol (Task 5)
  features.py     # SB3 CNN+MLP feature extractor (Task 6)
  train.py        # SAC training entry point (Task 8)
  evaluate.py     # evaluation over N fresh episodes (Task 9)
scripts/
  measure_instances.py   # instance-scaling throughput probe (Task 7)
  m2_gate.py             # the M2 exit gate (Task 9)
tests/
  test_expert_obs.py, test_expert_reward.py, test_expert_episode.py,
  test_expert_env.py, test_expert_features.py, test_m2_gate.py
docs/gates/
  m2_env_manifest.json, m2_instances.json, m2_train.json, m2_gate.json
```

Responsibilities: `obs`/`reward`/`episode` are pure and hold every number the spec fixes; `env` owns the simulator lifecycle and nothing else; `features` is the only file that imports torch outside the training scripts. M3 will reuse `episode.py` unchanged for the collector, so it takes a `Simulator` and a seed, never a training-specific object.

---

### Task 1: RL dependencies and a recorded environment manifest

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/make_env_manifest.py`
- Create: `docs/gates/m2_env_manifest.json`
- Test: `tests/test_env_manifest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `torch`, `stable_baselines3`, `gymnasium` importable from `.venv`; `scripts/make_env_manifest.py` writing a manifest with `{python, packages: {name: version}, cuda: {available, device_name, driver, torch_cuda}, numpy_pinned}`.

The packages were installed during planning with
`uv pip install --python .venv/bin/python torch stable-baselines3 gymnasium tensorboard`
(resolved: torch 2.14.0, stable-baselines3 2.9.0, gymnasium 1.3.0, tensorboard 2.21.0, 39 packages added, **numpy untouched at 1.26.4**). Verify that state rather than assuming it.

- [ ] **Step 1: Verify the install and that numpy did not move**

```bash
cd /home/jk_edge/research_uav/autofly_ue5
env -u PYTHONPATH .venv/bin/python -c "
import numpy, torch, gymnasium, stable_baselines3 as sb3
print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('gymnasium', gymnasium.__version__); print('sb3', sb3.__version__)
import projectairsim; print('projectairsim OK')
"
```
Expected: `numpy 1.26.4`, `torch.cuda.is_available()` **True**, device `NVIDIA GeForce RTX 4090`, and `projectairsim OK` (proving the install did not break the simulator client). If numpy is not 1.26.4, stop and report — that is a BLOCKED condition, not something to work around.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_env_manifest.py
import json
from pathlib import Path

from autofly_ue5.paths import ROOT


def test_manifest_records_versions_cuda_and_the_numpy_pin(tmp_path):
    from scripts.make_env_manifest import build_manifest

    m = build_manifest()
    assert m["packages"]["numpy"] == "1.26.4", "projectairsim 1.0.2 is built against numpy 1.26.4"
    for pkg in ("torch", "stable-baselines3", "gymnasium"):
        assert m["packages"][pkg], f"{pkg} missing from the manifest"
    assert m["cuda"]["available"] is True
    assert "4090" in m["cuda"]["device_name"]
    assert m["python"].startswith("3.12")


def test_manifest_is_json_serialisable_and_committed():
    from scripts.make_env_manifest import build_manifest

    json.dumps(build_manifest())  # must not raise
    committed = ROOT / "docs" / "gates" / "m2_env_manifest.json"
    assert committed.exists(), "run scripts/make_env_manifest.py and commit its output"
```

- [ ] **Step 3: Run it and watch it fail**

`env -u PYTHONPATH .venv/bin/python -m pytest tests/test_env_manifest.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.make_env_manifest'`.

- [ ] **Step 4: Write the manifest script**

```python
#!/usr/bin/env python3
"""Record the RL environment this milestone's numbers were produced on (spec §8, M2).

Written once at M2 start and re-run whenever a package moves. The numpy version is called out
separately because projectairsim 1.0.2 is built against 1.26.4 and a silent upgrade would break the
simulator client long after the fact.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ("numpy", "torch", "stable-baselines3", "gymnasium", "tensorboard", "projectairsim", "opencv-python")


def _driver_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_manifest() -> dict:
    import torch

    available = bool(torch.cuda.is_available())
    return {
        "description": "The Python environment M2's throughput and training numbers were measured on.",
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {p: md.version(p) for p in PACKAGES},
        "cuda": {
            "available": available,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "torch_cuda": torch.version.cuda,
            "driver": _driver_version(),
        },
        "numpy_pinned": "1.26.4 — projectairsim 1.0.2 is built against it; do not upgrade",
    }


def main() -> int:
    from autofly_ue5.paths import ROOT

    out = ROOT / "docs" / "gates" / "m2_env_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_manifest(), indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Generate the manifest and run the tests**

```bash
env -u PYTHONPATH .venv/bin/python scripts/make_env_manifest.py
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_env_manifest.py -q
```
Expected: both tests PASS.

- [ ] **Step 6: Declare the dependencies in pyproject.toml**

Add to `[project]`, recording why numpy is pinned:

```toml
dependencies = [
  "numpy==1.26.4",          # projectairsim 1.0.2 is built against this; do not upgrade
  "torch>=2.14,<3",
  "stable-baselines3>=2.9,<3",
  "gymnasium>=1.3,<2",
  "tensorboard>=2.21,<3",
]
```
Do **not** run `uv sync` or `uv pip install -e .` after this — the venv is already correct and a resolve could move packages. This block is documentation of the pinned state.

- [ ] **Step 7: Run the full suite and commit**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q   # expect 148 + 2 new = 150 passed
git add pyproject.toml scripts/make_env_manifest.py tests/test_env_manifest.py docs/gates/m2_env_manifest.json
git commit -m "$(cat <<'MSG'
Add the RL dependencies and record the environment M2 is measured on

torch/SB3/gymnasium/tensorboard installed with uv into the existing .venv. numpy stays at 1.26.4:
projectairsim 1.0.2 is built against it, and the manifest asserts the pin so a later upgrade fails a
test rather than breaking the simulator client silently.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: Observation encoding

**Files:**
- Create: `autofly_ue5/expert/__init__.py` (empty), `autofly_ue5/expert/obs.py`
- Test: `tests/test_expert_obs.py`

**Interfaces:**
- Consumes: `autofly_ue5.sim.types.Pose`, `Observation`; `autofly_ue5.frames.wrap_pi`.
- Produces:
  - `DEPTH_SIZE = 84`, `DEPTH_CLIP_M = 30.0`, `VECTOR_DIM = 8`, `NORM_DIST_M = 100.0`, `NORM_DZ_M = 5.0`
  - `encode_depth(depth: np.ndarray) -> np.ndarray` → `(1, 84, 84) float32` in `[0, 1]`
  - `encode_vector(pose: Pose, velocity_ned, yaw_rate: float, target_xy_z: tuple[float, float, float]) -> np.ndarray` → `(8,) float32`
  - `encode(obs: Observation, target_xy_z) -> dict[str, np.ndarray]` → `{"depth": ..., "vector": ...}`
  - `target_geometry(pose: Pose, target_xy_z) -> tuple[float, float, float]` → `(dist_xy_m, bearing_rad, dz_m)`, reused by `reward.py`

`target_geometry` is the single definition of "where is the target relative to me"; the reward and the observation must never compute it twice, or they can disagree.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_expert_obs.py
import math

import numpy as np
import pytest

from autofly_ue5.sim.types import Pose


def test_no_hit_depth_becomes_the_clip_distance():
    from autofly_ue5.expert.obs import DEPTH_CLIP_M, encode_depth

    d = np.full((256, 256), np.inf, dtype=np.float32)
    out = encode_depth(d)
    assert out.shape == (1, 84, 84) and out.dtype == np.float32
    assert np.allclose(out, 1.0), "sky (+inf) must read as maximally far, not as zero"


def test_downsample_keeps_the_nearest_obstacle_not_the_average():
    # A single very near pixel in an otherwise empty frame must survive downsampling: averaging would
    # dilute it to invisibility, and a thin near obstacle is exactly what must not be missed.
    from autofly_ue5.expert.obs import DEPTH_CLIP_M, encode_depth

    d = np.full((256, 256), np.inf, dtype=np.float32)
    d[130, 130] = 1.5
    out = encode_depth(d)[0]
    assert out.min() == pytest.approx(1.5 / DEPTH_CLIP_M, abs=1e-6)
    assert (out < 1.0).sum() == 1, "exactly one output cell should carry the near pixel"


def test_depth_beyond_the_clip_saturates_and_shape_is_channel_first():
    from autofly_ue5.expert.obs import encode_depth

    d = np.full((256, 256), 120.0, dtype=np.float32)
    assert np.allclose(encode_depth(d), 1.0)


def test_target_geometry_is_in_the_body_frame():
    from autofly_ue5.expert.obs import target_geometry

    # Facing +x (yaw 0), target 10 m ahead and 2 m to the left (+y is right in NED, so left is -y).
    dist, bearing, dz = target_geometry(Pose(0.0, 0.0, -2.0, 0.0), (10.0, 0.0, -2.0))
    assert dist == pytest.approx(10.0) and bearing == pytest.approx(0.0) and dz == pytest.approx(0.0)

    # Same target, but the drone is yawed 90 deg, so it faces East while the target is due North: the
    # target is now 90 deg to its LEFT -> bearing -pi/2. Negative bearing is left, positive is right.
    dist, bearing, _ = target_geometry(Pose(0.0, 0.0, -2.0, math.pi / 2), (10.0, 0.0, -2.0))
    assert dist == pytest.approx(10.0) and bearing == pytest.approx(-math.pi / 2)


def test_bearing_wraps_instead_of_jumping():
    from autofly_ue5.expert.obs import target_geometry

    _, b, _ = target_geometry(Pose(0.0, 0.0, -2.0, -math.pi + 0.01), (-10.0, 0.0, -2.0))
    assert abs(b) < 0.02, "a target dead ahead must read as ~0 bearing at any yaw"


def test_vector_is_finite_normalised_and_the_right_width():
    from autofly_ue5.expert.obs import VECTOR_DIM, encode_vector

    v = encode_vector(Pose(-30.0, 0.0, -2.0, 0.0), (1.5, 0.0, -0.1), 0.2, (30.0, 0.0, -1.5))
    assert v.shape == (VECTOR_DIM,) and v.dtype == np.float32
    assert np.all(np.isfinite(v))
    assert abs(v[0]) <= 1.5, "distance must be normalised, not raw metres"
    assert v[1] ** 2 + v[2] ** 2 == pytest.approx(1.0), "bearing is carried as sin/cos"


def test_encode_matches_its_parts():
    from autofly_ue5.expert.obs import encode, encode_depth, encode_vector
    from autofly_ue5.sim.fake import FakeSimulator

    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    o = sim.reset(Pose(-30.0, 0.0, -2.0, 0.0))
    target = (30.0, 0.0, -2.0)
    enc = encode(o, target)
    assert set(enc) == {"depth", "vector"}
    assert np.array_equal(enc["depth"], encode_depth(o.depth))
    assert np.array_equal(enc["vector"], encode_vector(o.pose, o.velocity_ned, o.yaw_rate, target))
```

- [ ] **Step 2: Run them and watch them fail**

`env -u PYTHONPATH .venv/bin/python -m pytest tests/test_expert_obs.py -q`
Expected: `ModuleNotFoundError: No module named 'autofly_ue5.expert'`.

- [ ] **Step 3: Implement**

```python
"""Expert observation encoding (spec §8): depth image + privileged target vector. RGB is never an input."""

from __future__ import annotations

import math

import numpy as np

from autofly_ue5.frames import wrap_pi
from autofly_ue5.sim.types import Observation, Pose

DEPTH_SIZE = 84
DEPTH_CLIP_M = 30.0
VECTOR_DIM = 8
NORM_DIST_M = 100.0   # ~ the 70x70 m scene diagonal, so distance lands in [0, 1]
NORM_DZ_M = 5.0       # the altitude band is 1-3 m; 5 m keeps dz comfortably inside [-1, 1]
NORM_V_MS = 2.0       # max commanded forward speed


def _block_min(a: np.ndarray, size: int) -> np.ndarray:
    """Downsample by block MINIMUM to `size` x `size`.

    Minimum, not mean: averaging across a depth discontinuity invents intermediate depths and can erase a
    thin near obstacle. A minimum can only ever report an obstacle as closer than it is, which is the safe
    direction for collision avoidance. The 256 -> 84 ratio is not an integer, so block boundaries come from
    linspace and np.minimum.reduceat, which handles unequal block sizes exactly.
    """
    for axis in (0, 1):
        idx = np.linspace(0, a.shape[axis], size + 1).astype(int)[:-1]
        a = np.minimum.reduceat(a, idx, axis=axis)
    return a


def encode_depth(depth: np.ndarray, size: int = DEPTH_SIZE, clip_m: float = DEPTH_CLIP_M) -> np.ndarray:
    """(H, W) metres, +inf for no-hit (spec §7.1) -> (1, size, size) float32 in [0, 1], 1.0 = clip or sky."""
    d = np.asarray(depth, dtype=np.float32)
    d = np.nan_to_num(d, nan=clip_m, posinf=clip_m, neginf=0.0)
    np.clip(d, 0.0, clip_m, out=d)
    return (_block_min(d, size) / clip_m).astype(np.float32)[None, ...]


def target_geometry(pose: Pose, target_xy_z: tuple[float, float, float]) -> tuple[float, float, float]:
    """(horizontal distance m, bearing rad in the body frame, height difference m).

    The single definition of where the target is relative to the drone: the observation and the reward both
    call this, so they cannot drift apart. `dz` is the NED difference (negative = target above the drone).
    """
    dx = target_xy_z[0] - pose.x
    dy = target_xy_z[1] - pose.y
    return math.hypot(dx, dy), wrap_pi(math.atan2(dy, dx) - pose.yaw), target_xy_z[2] - pose.z


def encode_vector(pose: Pose, velocity_ned, yaw_rate: float, target_xy_z) -> np.ndarray:
    dist, bearing, dz = target_geometry(pose, target_xy_z)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    vx, vy, vz_ned = (float(v) for v in velocity_ned)
    return np.array([
        dist / NORM_DIST_M,
        math.sin(bearing),                 # sin/cos, not the raw angle: no discontinuity at +/-pi
        math.cos(bearing),
        dz / NORM_DZ_M,
        (vx * cy + vy * sy) / NORM_V_MS,   # body forward
        (-vx * sy + vy * cy) / NORM_V_MS,  # body right
        -vz_ned / NORM_V_MS,               # positive up, matching the action convention
        float(yaw_rate),
    ], dtype=np.float32)


def encode(obs: Observation, target_xy_z) -> dict[str, np.ndarray]:
    return {"depth": encode_depth(obs.depth),
            "vector": encode_vector(obs.pose, obs.velocity_ned, obs.yaw_rate, target_xy_z)}
```

- [ ] **Step 4: Run the tests**

`env -u PYTHONPATH .venv/bin/python -m pytest tests/test_expert_obs.py -q` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add autofly_ue5/expert/__init__.py autofly_ue5/expert/obs.py tests/test_expert_obs.py
git commit -m "$(cat <<'MSG'
Encode the expert's observation: block-min depth plus a privileged target vector

Depth downsamples by block minimum rather than area average, so a thin near obstacle survives instead of
being diluted; +inf (sky/no-hit, spec 7.1) maps to the 30 m clip. Bearing enters as sin/cos to avoid the
discontinuity at +/-pi. target_geometry() is the one definition of the target's relative position, shared
with the reward so the two cannot disagree.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: Reward and termination

**Files:**
- Create: `autofly_ue5/expert/reward.py`
- Test: `tests/test_expert_reward.py`

**Interfaces:**
- Consumes: `autofly_ue5.expert.obs.target_geometry`.
- Produces:
  - `RewardConfig` frozen dataclass with the spec's coefficients as defaults.
  - `Outcome` enum: `RUNNING`, `SUCCESS`, `COLLISION`, `OUT_OF_BOUNDS`, `TIMEOUT`.
  - `StepResult` frozen dataclass: `reward: float`, `outcome: Outcome`, `terminated: bool`, `truncated: bool`.
  - `classify(dist_m, bearing_rad, altitude_m, in_bounds, collided, step_index, cfg) -> Outcome`
  - `step_reward(prev_dist_m, dist_m, bearing_rad, outcome, cfg) -> float`
  - `evaluate(...) -> StepResult` combining both.

Termination semantics for Gymnasium: `SUCCESS`/`COLLISION`/`OUT_OF_BOUNDS` are `terminated=True` (the episode genuinely ended); `TIMEOUT` is `truncated=True` (a time limit, not an end state) — SB3 bootstraps the value function correctly only if these are distinguished.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_expert_reward.py
import math

import pytest


def cfg():
    from autofly_ue5.expert.reward import RewardConfig
    return RewardConfig()


def test_defaults_are_the_spec_coefficients():
    c = cfg()
    assert (c.k_p, c.k_h, c.k_t) == (1.0, 0.1, 0.01)
    assert (c.r_success, c.r_collision, c.r_bounds) == (10.0, 10.0, 5.0)
    assert c.success_radius_m == 5.0 and c.success_yaw_deg == 15.0
    assert c.step_limit == 300 and c.align_radius_m == 10.0


def test_success_needs_both_distance_and_heading():
    from autofly_ue5.expert.reward import Outcome, classify

    near_and_aligned = dict(dist_m=4.0, bearing_rad=math.radians(10), altitude_m=2.0,
                            in_bounds=True, collided=False, step_index=5, cfg=cfg())
    assert classify(**near_and_aligned) is Outcome.SUCCESS
    assert classify(**{**near_and_aligned, "bearing_rad": math.radians(20)}) is Outcome.RUNNING
    assert classify(**{**near_and_aligned, "dist_m": 5.5}) is Outcome.RUNNING


def test_collision_outranks_success():
    from autofly_ue5.expert.reward import Outcome, classify

    assert classify(dist_m=1.0, bearing_rad=0.0, altitude_m=2.0, in_bounds=True,
                    collided=True, step_index=5, cfg=cfg()) is Outcome.COLLISION


def test_leaving_the_altitude_band_or_the_bounds_ends_the_episode():
    from autofly_ue5.expert.reward import Outcome, classify

    base = dict(dist_m=30.0, bearing_rad=0.0, in_bounds=True, collided=False, step_index=5, cfg=cfg())
    assert classify(**{**base, "altitude_m": 0.2}) is Outcome.OUT_OF_BOUNDS
    assert classify(**{**base, "altitude_m": 9.0}) is Outcome.OUT_OF_BOUNDS
    assert classify(**{**base, "altitude_m": 2.0, "in_bounds": False}) is Outcome.OUT_OF_BOUNDS


def test_timeout_only_at_the_step_limit():
    from autofly_ue5.expert.reward import Outcome, classify

    base = dict(dist_m=30.0, bearing_rad=0.0, altitude_m=2.0, in_bounds=True, collided=False, cfg=cfg())
    assert classify(**base, step_index=299) is Outcome.RUNNING
    assert classify(**base, step_index=300) is Outcome.TIMEOUT


def test_progress_dominates_the_time_penalty():
    # Closing 0.4 m in a step (2 m/s for 0.2 s) must beat standing still, or the agent learns to hover.
    from autofly_ue5.expert.reward import Outcome, step_reward

    moving = step_reward(30.0, 29.6, 0.0, Outcome.RUNNING, cfg())
    still = step_reward(30.0, 30.0, 0.0, Outcome.RUNNING, cfg())
    assert moving > still and still < 0.0


def test_retreating_is_punished():
    from autofly_ue5.expert.reward import Outcome, step_reward

    assert step_reward(30.0, 30.4, 0.0, Outcome.RUNNING, cfg()) < 0.0


def test_alignment_bonus_applies_only_inside_10_m():
    from autofly_ue5.expert.reward import Outcome, step_reward

    c = cfg()
    near = step_reward(9.0, 9.0, 0.0, Outcome.RUNNING, c)
    far = step_reward(29.0, 29.0, 0.0, Outcome.RUNNING, c)
    assert near - far == pytest.approx(c.k_h, abs=1e-9)


def test_terminal_rewards_have_the_spec_magnitudes():
    from autofly_ue5.expert.reward import Outcome, step_reward

    c = cfg()
    assert step_reward(6.0, 4.0, 0.0, Outcome.SUCCESS, c) > c.r_success
    assert step_reward(30.0, 30.0, 0.0, Outcome.COLLISION, c) < -c.r_collision
    assert step_reward(30.0, 30.0, 0.0, Outcome.OUT_OF_BOUNDS, c) < -c.r_bounds


def test_evaluate_maps_outcomes_to_gymnasium_flags():
    from autofly_ue5.expert.reward import Outcome, evaluate

    common = dict(prev_dist_m=30.0, dist_m=29.6, bearing_rad=0.0, altitude_m=2.0,
                  in_bounds=True, collided=False, cfg=cfg())
    running = evaluate(**common, step_index=5)
    assert running.outcome is Outcome.RUNNING and not running.terminated and not running.truncated

    timeout = evaluate(**common, step_index=300)
    assert timeout.truncated is True and timeout.terminated is False, "a time limit truncates, not terminates"

    crash = evaluate(**{**common, "collided": True}, step_index=5)
    assert crash.terminated is True and crash.truncated is False
```

- [ ] **Step 2: Run them and watch them fail**

Expected: `ModuleNotFoundError: No module named 'autofly_ue5.expert.reward'`.

- [ ] **Step 3: Implement**

```python
"""Expert reward and episode termination (spec §8). Pure functions: no simulator, no randomness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Outcome(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    COLLISION = "collision"
    OUT_OF_BOUNDS = "out_of_bounds"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class RewardConfig:
    """Spec §8's coefficients. Tuned on s01 only; the same values are then used for every scene."""
    k_p: float = 1.0            # progress, per metre closed
    k_h: float = 0.1            # alignment bonus inside align_radius_m
    k_t: float = 0.01           # time penalty, per step
    r_success: float = 10.0
    r_collision: float = 10.0
    r_bounds: float = 5.0
    success_radius_m: float = 5.0
    success_yaw_deg: float = 15.0
    align_radius_m: float = 10.0
    step_limit: int = 300
    altitude_band_m: tuple[float, float] = (1.0, 3.0)
    altitude_margin_m: float = 1.0   # the band is the *target* band; leaving it by more than this ends it


@dataclass(frozen=True)
class StepResult:
    reward: float
    outcome: Outcome
    terminated: bool
    truncated: bool


def classify(*, dist_m: float, bearing_rad: float, altitude_m: float, in_bounds: bool,
             collided: bool, step_index: int, cfg: RewardConfig) -> Outcome:
    """Order matters: a crash on the step that would otherwise have succeeded is still a crash."""
    if collided:
        return Outcome.COLLISION
    low, high = cfg.altitude_band_m
    if not in_bounds or not (low - cfg.altitude_margin_m <= altitude_m <= high + cfg.altitude_margin_m):
        return Outcome.OUT_OF_BOUNDS
    if dist_m <= cfg.success_radius_m and abs(bearing_rad) <= math.radians(cfg.success_yaw_deg):
        return Outcome.SUCCESS
    if step_index >= cfg.step_limit:
        return Outcome.TIMEOUT
    return Outcome.RUNNING


def step_reward(prev_dist_m: float, dist_m: float, bearing_rad: float,
                outcome: Outcome, cfg: RewardConfig) -> float:
    r = cfg.k_p * (prev_dist_m - dist_m) - cfg.k_t
    if dist_m <= cfg.align_radius_m:
        r += cfg.k_h * math.cos(bearing_rad)
    if outcome is Outcome.SUCCESS:
        r += cfg.r_success
    elif outcome is Outcome.COLLISION:
        r -= cfg.r_collision
    elif outcome is Outcome.OUT_OF_BOUNDS:
        r -= cfg.r_bounds
    return float(r)


def evaluate(*, prev_dist_m: float, dist_m: float, bearing_rad: float, altitude_m: float,
             in_bounds: bool, collided: bool, step_index: int, cfg: RewardConfig) -> StepResult:
    outcome = classify(dist_m=dist_m, bearing_rad=bearing_rad, altitude_m=altitude_m, in_bounds=in_bounds,
                       collided=collided, step_index=step_index, cfg=cfg)
    # Gymnasium distinguishes the two: SB3 bootstraps the value of a truncated episode but not a
    # terminated one. Calling a timeout "terminated" would teach the agent that time running out is as
    # bad as crashing.
    return StepResult(reward=step_reward(prev_dist_m, dist_m, bearing_rad, outcome, cfg),
                      outcome=outcome,
                      terminated=outcome in (Outcome.SUCCESS, Outcome.COLLISION, Outcome.OUT_OF_BOUNDS),
                      truncated=outcome is Outcome.TIMEOUT)
```

- [ ] **Step 4: Run the tests** → all PASS.

- [ ] **Step 5: Commit**

```bash
git add autofly_ue5/expert/reward.py tests/test_expert_reward.py
git commit -m "$(cat <<'MSG'
Add the expert's reward and termination rules with the spec's coefficients

Pure functions, no simulator. Success needs both the 5 m radius and the 15 deg heading; a collision
outranks a success on the same step. Timeout truncates while success/collision/bounds terminate, because
SB3 bootstraps the value of a truncated episode and calling a time limit "terminated" would teach the
agent that running out of time is as bad as crashing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: Episode setup sampling

**Files:**
- Create: `autofly_ue5/expert/episode.py`
- Test: `tests/test_expert_episode.py`

**Interfaces:**
- Consumes: `autofly_ue5.scenes.model` (`SceneFile`, `Layout`, `Bounds`, `Instance`), `autofly_ue5.scenes.reachability` (`occupancy`, `cell_centers`), `autofly_ue5.sim.protocol.Simulator`, `autofly_ue5.sim.types.Pose`.
- Produces:
  - `EDGES = ("x_min", "x_max", "y_min", "y_max")`, `OPPOSITE = {"x_min": "x_max", ...}`
  - `INSTRUCTION_TEMPLATES` — the two verbatim strings from spec §9.3, **including the original misspelling `avioding`**. Do not "fix" it: the templates are copied from the real released episodes and the student model must see the same text distribution.
  - `EpisodeSetup` frozen dataclass: `scene_id, seed, start: Pose, target_xy_z, target_scale, distractors: tuple[tuple[float, float, float], ...], instruction: str, start_edge: str, target_edge: str`
  - `sample_setup(scene: SceneFile, layout: Layout, rng: np.random.Generator, *, n_distractors_range=(3, 5)) -> EpisodeSetup` — pure, no simulator
  - `apply_setup(sim: Simulator, setup: EpisodeSetup) -> tuple[str, ...]` — spawns, returns the actual (uniquified) names in spawn order, target first
  - `clear_setup(sim: Simulator, names) -> None` — destroys, tolerating an already-missing name

**Before writing code, read `autofly_ue5/scenes/reachability.py:74` (`occupancy`) and confirm which polarity it returns (True = occupied, or True = free). Do not assume — the whole sampler inverts on it.** Also read `cell_centers` to learn the grid's index-to-metres mapping.

M3's collector will import this module unchanged, so it must not take a training-specific argument or touch any RL type.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_expert_episode.py
import math

import numpy as np
import pytest

from autofly_ue5.paths import ROOT
from autofly_ue5.scenes.model import load_scene_file
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import Pose


def scene_and_layout():
    import json

    from autofly_ue5.scenes.model import Bounds, Instance, Layout

    scene = load_scene_file(ROOT / "scenes" / "s01_white_pillars.json")
    raw = json.loads((ROOT / "runs" / "levels" / "s01.layout.json").read_text())["layout"]
    b = Bounds(**raw["bounds"])
    inst = tuple(Instance(**i) for i in raw["instances"])
    return scene, Layout(scene_id=raw["scene_id"], seed=raw["seed"], bounds=b, instances=inst)


def test_instruction_templates_are_verbatim_including_the_misspelling():
    from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES

    joined = " | ".join(INSTRUCTION_TEMPLATES)
    assert "avioding" in joined, "the released episodes misspell 'avoiding'; do not correct it"
    assert "go through and avoid the {obstacle} or other obstacles to reach the {target}" in INSTRUCTION_TEMPLATES


def test_start_and_target_are_on_opposite_edges():
    from autofly_ue5.expert.episode import OPPOSITE, sample_setup

    scene, layout = scene_and_layout()
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        assert OPPOSITE[s.start_edge] == s.target_edge, "a same-edge episode would be a metre long"


def test_start_is_in_the_start_band_and_altitude_band():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    lo, hi = scene.start_band
    a_lo, a_hi = scene.altitude_band
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        b = layout.bounds
        d = min(s.start.x - b.x_min, b.x_max - s.start.x, s.start.y - b.y_min, b.y_max - s.start.y)
        assert lo - 1e-6 <= d <= hi + 1e-6
        assert a_lo - 1e-6 <= -s.start.z <= a_hi + 1e-6, "z is NED; altitude is -z"


def test_the_crossing_is_long():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        d = math.hypot(s.target_xy_z[0] - s.start.x, s.target_xy_z[1] - s.start.y)
        assert d > 50.0, f"seed {seed} produced a {d:.1f} m episode; the scene is 70 m across"


def test_nothing_spawns_inside_an_obstacle():
    from autofly_ue5.expert.episode import sample_setup, spawn_clearance_m

    scene, layout = scene_and_layout()
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        points = [(s.start.x, s.start.y), (s.target_xy_z[0], s.target_xy_z[1])]
        points += [(d[0], d[1]) for d in s.distractors]
        for px, py in points:
            for inst in layout.instances:
                gap = math.hypot(px - inst.x, py - inst.y) - inst.radius_m
                assert gap >= spawn_clearance_m(), f"seed {seed}: point {px:.1f},{py:.1f} sits in {inst.tag}"


def test_distractor_count_and_spacing():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        assert 3 <= len(s.distractors) <= 5
        pts = [s.target_xy_z] + list(s.distractors)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) >= 4.0 - 1e-6


def test_sampling_is_deterministic_for_a_seed():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    a = sample_setup(scene, layout, np.random.default_rng(7))
    b = sample_setup(scene, layout, np.random.default_rng(7))
    assert a == b
    assert a != sample_setup(scene, layout, np.random.default_rng(8))


def test_apply_then_clear_round_trips_against_the_fake():
    from autofly_ue5.expert.episode import apply_setup, clear_setup, sample_setup

    scene, layout = scene_and_layout()
    setup = sample_setup(scene, layout, np.random.default_rng(3))
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(setup.start)
    names = apply_setup(sim, setup)
    assert len(names) == 1 + len(setup.distractors)
    clear_setup(sim, names)
    clear_setup(sim, names)  # must tolerate an already-destroyed name, so a failed episode can always clean up
```

- [ ] **Step 2: Run them and watch them fail**

Expected: `ModuleNotFoundError: No module named 'autofly_ue5.expert.episode'`.

- [ ] **Step 3: Implement**

Write `autofly_ue5/expert/episode.py` providing the interface above. Required behaviour, all pinned by the tests:

- `DRONE_RADIUS_M = 0.4`, `CLEARANCE_M = 1.0`, `spawn_clearance_m()` returns their sum — the same inflation the reachability check uses, so a sampled point is flyable by the same definition the scene was validated against.
- Rejection sampling: draw a candidate, reject it if the nearest obstacle surface is closer than `spawn_clearance_m()`, retry up to 200 times, and raise a clear `EpisodeSetupError` if it cannot place a point. Use the layout's `Instance` list directly (each carries `x`, `y`, `radius_m`) rather than rasterising an occupancy grid: it is exact, cheaper, and does not depend on `occupancy()`'s polarity.
- `sample_setup` draws, in this fixed order so a seed reproduces exactly: start edge → start position → start altitude → start yaw → target position on the opposite edge → distractor count → distractor positions → instruction template. The yaw is uniform over `[-pi, pi)` (spec §9.2: "a random yaw"), **not** pointed at the target — the expert must learn to turn, and AutoFly's own `a0` is a turn-in-place.
- The target reference point is the object's centre at `z = -1.0` (NED), i.e. 1 m above ground, with `target_scale = (1.0, 1.0, 2.0)` so a 2 m cylinder stands on the ground. Distractors use the same scale.
- Instruction: `rng.choice(INSTRUCTION_TEMPLATES).format(target=..., obstacle=scene.instruction_obstacle)`. Until M4 supplies the 60-target pool, the target name is the literal string `"target"` (decision D1) — leave a comment saying M4 replaces it, so the instruction reads "...to reach the target".
- `apply_setup` spawns the target first with material `"orange"` if the registry has it, else the first material that is not the obstacle palette's, then the distractors with the obstacle palette's material. Read `assets/registry.json` for the available material names rather than hard-coding — M1's registry has exactly two.
- `clear_setup` swallows `ObjectNotFoundError` per name so a partially-spawned episode can always be cleaned up.

- [ ] **Step 4: Run the tests** → all PASS. If `test_the_crossing_is_long` fails, the edge-opposition logic is wrong; if `test_nothing_spawns_inside_an_obstacle` fails, the clearance test is using the pillar centre rather than its surface.

- [ ] **Step 5: Commit**

```bash
git add autofly_ue5/expert/episode.py tests/test_expert_episode.py
git commit -m "$(cat <<'MSG'
Sample episode setups: opposite-edge crossings with a target and 3-5 distractors

Start and target go on opposite edges because band_mask measures distance to the nearest edge, so the
bands are rings and a naive draw can put both on the same side and produce a metre-long episode. Start
yaw is uniform, not aimed at the target: the expert has to learn to turn. Clearance reuses the same
inflation the scene's reachability check was validated with, so a sampled point is flyable by the same
definition. M3's collector imports this module unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: The Gymnasium environment

**Files:**
- Create: `autofly_ue5/expert/env.py`
- Test: `tests/test_expert_env.py`

**Interfaces:**
- Consumes: everything from Tasks 2-4, plus `autofly_ue5.sim.protocol.Simulator`.
- Produces: `AutoFlyEnv(gymnasium.Env)` with
  - `__init__(self, scene, layout, sim_factory: Callable[[], Simulator], *, map_path: str, instance: int, cfg: RewardConfig = RewardConfig(), seed_base: int = 0, max_episode_steps: int = 300)`
  - `observation_space = Dict({"depth": Box(0, 1, (1, 84, 84), float32), "vector": Box(-inf, inf, (8,), float32)})`
  - `action_space = Box(low=[0, -1, -1], high=[2, 1, 1], dtype=float32)`
  - `reset(seed=None, options=None) -> (obs, info)`, `step(action) -> (obs, reward, terminated, truncated, info)`, `close()`
  - `info` on a terminal step carries `{"outcome": str, "steps": int, "final_distance_m": float, "is_success": bool}`.
    **Measured correction (2026-09-16):** a plain `Monitor` does **not** copy `is_success` into its episode record —
    `info["episode"]` carries only `r`, `l` and `t`. Task 9 must therefore read `info["is_success"]` off the final
    step directly (which is what SB3's `EvalCallback._log_success_callback` does), or wrap the env as
    `Monitor(env, info_keywords=("is_success",))`. Writing Task 9 against the original wording would have made the
    gate report nothing rather than something wrong — loud rather than silent, but still wrong.

`sim_factory` is injected so the tests can pass `FakeSimulator` and training can pass the real backend; the env never imports `airsim_backend` itself.

**Lifecycle rules the env must obey (spec §7.1, non-negotiable):**
- `reset()` must call `sim.reset(start_pose)` **before** any `step()`; frame 0 of a session is corrupt.
- Previous-episode objects are destroyed before the new ones are spawned, or the scene accumulates actors across thousands of episodes (the final review of Plan 1 flagged this as M3's trap; the env owns it here).
- `command_velocity()` must be called before **every** `step()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_expert_env.py
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from autofly_ue5.sim.fake import FakeSimulator
from tests.test_expert_episode import scene_and_layout


def make_env(**kw):
    from autofly_ue5.expert.env import AutoFlyEnv

    scene, layout = scene_and_layout()
    return AutoFlyEnv(scene, layout, FakeSimulator, map_path="/Game/AutoFly/Maps/S01", instance=0, **kw)


def test_passes_the_gymnasium_api_checker():
    check_env(make_env(), skip_render_check=True)


def test_action_space_is_exactly_the_spec_action():
    env = make_env()
    assert np.allclose(env.action_space.low, [0.0, -1.0, -1.0])
    assert np.allclose(env.action_space.high, [2.0, 1.0, 1.0])


def test_reset_returns_an_observation_inside_the_space():
    env = make_env()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert info["outcome"] == "running" and info["steps"] == 0


def test_an_episode_terminates_and_reports_its_outcome():
    env = make_env()
    env.reset(seed=2)
    for _ in range(400):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        if terminated or truncated:
            break
    assert terminated or truncated, "an episode must end within the 300-step limit"
    assert info["outcome"] in {"success", "collision", "out_of_bounds", "timeout"}
    assert isinstance(info["is_success"], bool)


def test_flying_straight_at_a_target_succeeds():
    # The fake integrates the commanded velocity exactly, so a policy that points at the target and flies
    # must succeed. If this fails, the reward/geometry wiring disagrees with the action convention.
    import math

    from autofly_ue5.expert.obs import target_geometry

    env = make_env()
    env.reset(seed=5)
    outcome = None
    for _ in range(300):
        dist, bearing, _ = target_geometry(env._sim.observe().pose, env._setup.target_xy_z)
        yaw_rate = float(np.clip(bearing * 2.0, -1.0, 1.0))
        v = 2.0 if abs(bearing) < 0.3 else 0.3
        _, _, terminated, truncated, info = env.step(np.array([v, yaw_rate, 0.0], dtype=np.float32))
        if terminated or truncated:
            outcome = info["outcome"]
            break
    assert outcome == "success", f"a straight-line pilot ended in {outcome}"


def test_reset_clears_the_previous_episode_objects():
    env = make_env()
    env.reset(seed=1)
    first = set(env._sim._objects) if hasattr(env._sim, "_objects") else None
    env.reset(seed=2)
    second = set(env._sim._objects) if hasattr(env._sim, "_objects") else None
    if first is not None:
        assert len(second) == len(first), "objects accumulated across episodes"


def test_two_envs_with_different_seed_bases_diverge():
    a, b = make_env(seed_base=0), make_env(seed_base=1000)
    oa, _ = a.reset(seed=None)
    ob, _ = b.reset(seed=None)
    assert not np.array_equal(oa["vector"], ob["vector"]), "vectorised envs must not fly identical episodes"


def test_close_is_idempotent():
    env = make_env()
    env.reset(seed=1)
    env.close()
    env.close()
```

- [ ] **Step 2: Run them and watch them fail.**

- [ ] **Step 3: Implement**

`AutoFlyEnv` holds one `Simulator`, created lazily on the first `reset()` via `sim_factory()` then `launch(map_path, instance)`. Per-episode flow:

1. `clear_setup(sim, self._spawned)` if anything is spawned.
2. `setup = sample_setup(scene, layout, rng)` where `rng = np.random.default_rng(seed_base + episode_index)` — deterministic per env, and disjoint between envs because `seed_base` differs.
3. `obs = sim.reset(setup.start)` — **before** any step.
4. `self._spawned = apply_setup(sim, setup)`.
5. `self._prev_dist = target_geometry(obs.pose, setup.target_xy_z)[0]`.

Each `step(action)`:
1. Clip the action into the space, `sim.command_velocity(v_forward, yaw_rate, v_z)`, `sim.step(0.2)`, `obs = sim.observe()`.
2. `dist, bearing, _ = target_geometry(obs.pose, setup.target_xy_z)`; `altitude = -obs.pose.z`; `in_bounds` from the layout bounds.
3. `result = evaluate(prev_dist_m=self._prev_dist, dist_m=dist, ...)`, update `self._prev_dist`.
4. Return `encode(obs, setup.target_xy_z)`, `result.reward`, `result.terminated`, `result.truncated`, info.

Spawn the target **after** `sim.reset()` so the reset sweep cannot collide with it, and note that in a comment.

- [ ] **Step 4: Run the tests** → all PASS, including `check_env`.

- [ ] **Step 5: Commit**

```bash
git add autofly_ue5/expert/env.py tests/test_expert_env.py
git commit -m "$(cat <<'MSG'
Wrap the simulator in a Gymnasium environment for the SAC expert

One simulator per env, injected as a factory so tests run against FakeSimulator and training against the
real backend without the env importing either. reset() always resets the simulator before stepping it
(frame 0 of a session is corrupt, spec 7.1) and destroys the previous episode's objects first, so a long
run cannot accumulate actors. Per-env seed bases keep vectorised workers off identical episodes.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: SB3 feature extractor

**Files:**
- Create: `autofly_ue5/expert/features.py`
- Test: `tests/test_expert_features.py`

**Interfaces:**
- Produces: `DepthVectorExtractor(BaseFeaturesExtractor)` — strided CNN on `depth`, MLP on `vector`, concatenated (spec §8: "CNN (strided convolutions) on depth, MLP on the vector, shared by actor and twin critics"); `features_dim` default 256; `POLICY_KWARGS` dict ready to hand to SAC.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_expert_features.py
import numpy as np
import torch


def space():
    from gymnasium import spaces

    from autofly_ue5.expert.obs import DEPTH_SIZE, VECTOR_DIM

    return spaces.Dict({
        "depth": spaces.Box(0.0, 1.0, (1, DEPTH_SIZE, DEPTH_SIZE), np.float32),
        "vector": spaces.Box(-np.inf, np.inf, (VECTOR_DIM,), np.float32),
    })


def test_output_shape_and_dtype():
    from autofly_ue5.expert.features import DepthVectorExtractor

    ex = DepthVectorExtractor(space(), features_dim=256)
    out = ex({"depth": torch.zeros(4, 1, 84, 84), "vector": torch.zeros(4, 8)})
    assert out.shape == (4, 256) and out.dtype == torch.float32


def test_gradients_reach_both_branches():
    # A vector-only or depth-only extractor would silently train a blind or a target-less agent.
    from autofly_ue5.expert.features import DepthVectorExtractor

    ex = DepthVectorExtractor(space(), features_dim=256)
    obs = {"depth": torch.rand(2, 1, 84, 84, requires_grad=True),
           "vector": torch.rand(2, 8, requires_grad=True)}
    ex(obs).sum().backward()
    assert obs["depth"].grad is not None and obs["depth"].grad.abs().sum() > 0
    assert obs["vector"].grad is not None and obs["vector"].grad.abs().sum() > 0


def test_it_plugs_into_sac_and_takes_one_gradient_step():
    from stable_baselines3 import SAC

    from autofly_ue5.expert.features import POLICY_KWARGS
    from tests.test_expert_env import make_env

    model = SAC("MultiInputPolicy", make_env(), policy_kwargs=POLICY_KWARGS,
                buffer_size=500, learning_starts=10, batch_size=8, device="cpu", verbose=0)
    model.learn(total_timesteps=40)
    assert model.num_timesteps >= 40


def test_target_entropy_matches_the_spec():
    # spec 8: automatic entropy tuning with target entropy -3 (one per action dimension).
    from stable_baselines3 import SAC

    from autofly_ue5.expert.features import POLICY_KWARGS
    from tests.test_expert_env import make_env

    model = SAC("MultiInputPolicy", make_env(), policy_kwargs=POLICY_KWARGS,
                buffer_size=100, learning_starts=10, device="cpu")
    assert float(model.target_entropy) == -3.0
```

- [ ] **Step 2: Run them and watch them fail.**

- [ ] **Step 3: Implement**

```python
"""SAC feature extractor (spec §8): strided CNN on depth + MLP on the target vector, shared by actor and critics."""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class DepthVectorExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, vector_dim: int = 64):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space["depth"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),     # 84 -> 20
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),    # 20 -> 9
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),    # 9 -> 7
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.mlp = nn.Sequential(nn.Linear(observation_space["vector"].shape[0], vector_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(n_flat + vector_dim, features_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(torch.cat([self.cnn(observations["depth"]), self.mlp(observations["vector"])], dim=1))


POLICY_KWARGS = {
    "features_extractor_class": DepthVectorExtractor,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [256, 256],
}
```

Note the CNN geometry is the standard Nature-DQN stack, which is sized for 84×84 — that is why `DEPTH_SIZE` is 84.

- [ ] **Step 4: Run the tests** → all PASS. `test_it_plugs_into_sac` runs on CPU deliberately: it is a wiring test, not a speed test.

- [ ] **Step 5: Run the full suite and commit**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q
git add autofly_ue5/expert/features.py tests/test_expert_features.py
git commit -m "$(cat <<'MSG'
Add the SAC feature extractor: strided CNN on depth, MLP on the target vector

Nature-DQN convolution geometry, which is what fixes DEPTH_SIZE at 84. A gradient test asserts both
branches receive gradient: a silently depth-only or vector-only extractor would train a blind or a
target-less agent and still look like it was learning.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 7: Instance-scaling throughput measurement (LIVE — needs the GPU and the packaged simulator)

**Files:**
- Create: `scripts/measure_instances.py`
- Create: `docs/gates/m2_instances.json`
- Test: `tests/test_measure_instances.py` (offline: the arithmetic and the JSON shape, not the flight)

**Interfaces:**
- Consumes: `autofly_ue5.sim.airsim_backend.ProjectAirSimSimulator`, `autofly_ue5.expert.env.AutoFlyEnv`, `autofly_ue5.gpu`.
- Produces: `docs/gates/m2_instances.json` with `{per_n: {"1": {...}, "2": {...}, "4": {...}, "6": {...}}, chosen_n, projection}`, each entry carrying `env_steps_per_s_total`, `env_steps_per_s_per_instance`, `vram_mib`, `launch_s`, `episodes_completed`, `faults_ok`.

This is spec §8's throughput gate: *"measure environment steps per second per instance, the number of instances the RTX 4090 holds, and time to 95 % on s01; project the cost of all agents before M5."* M0 measured 7.43 steps/s for one instance and 5.67 each for two, but never beyond two, and never with an RL env in the loop.

**Hard rules for this task:**
- Run it through `scripts/run_job.sh` — it is a long job.
- Machine budget: RTX 4090, **24,564 MiB** total, ~566 MiB already in use; M1 measured **1,882 MiB** per packaged instance while capturing. 32 CPU cores, 61 GiB RAM.
- Stop climbing N the moment either (a) total VRAM would exceed **20,000 MiB**, or (b) total throughput *falls* versus the previous N. Record where it stopped and why. Do not push to out-of-memory — a GPU fault costs the whole session.
- Take VRAM samples *inside* the window when all N instances are capturing, exactly as M0's `two_instance_concurrency` check did; a sample taken before the last instance finishes launching is worthless.
- If any instance fails to launch, tear down every instance before returning, so a partial failure does not leave processes holding VRAM.

- [ ] **Step 1: Write the offline tests**

```python
# tests/test_measure_instances.py
import pytest


def test_projection_arithmetic():
    from scripts.measure_instances import project_cost

    # 30 env steps/s total, 400k steps needed per scene, 10 scenes.
    p = project_cost(steps_per_s_total=30.0, steps_needed=400_000, n_scenes=10)
    assert p["hours_per_scene"] == pytest.approx(400_000 / 30.0 / 3600.0, rel=1e-6)
    assert p["hours_all_scenes"] == pytest.approx(p["hours_per_scene"] * 10, rel=1e-6)
    assert p["days_all_scenes"] == pytest.approx(p["hours_all_scenes"] / 24.0, rel=1e-6)


def test_projection_refuses_nonsense():
    from scripts.measure_instances import project_cost

    with pytest.raises(ValueError):
        project_cost(steps_per_s_total=0.0, steps_needed=1, n_scenes=1)


def test_should_stop_climbing():
    from scripts.measure_instances import should_stop

    assert should_stop(vram_total_mib=20_500, prev_total=30.0, this_total=35.0)[0] is True
    assert should_stop(vram_total_mib=8_000, prev_total=30.0, this_total=28.0)[0] is True
    assert should_stop(vram_total_mib=8_000, prev_total=30.0, this_total=35.0)[0] is False
```

- [ ] **Step 2: Run them, watch them fail, then implement `project_cost`, `should_stop` and the measurement loop.**

The measurement loop, for each `n` in `(1, 2, 4, 6)`: build `n` `AutoFlyEnv`s on instances `0..n-1` in a `SubprocVecEnv`, `reset()`, then step with **random actions** for a fixed wall-clock window (180 s) after a 30 s warm-up, counting env steps across all workers. Random actions, not a policy: this measures the simulator, and a policy would add its own variable cost.

- [ ] **Step 3: Run it live**

```bash
scripts/run_job.sh start m2_instances -- env -u PYTHONPATH .venv/bin/python scripts/measure_instances.py --out docs/gates/m2_instances.json
scripts/run_job.sh wait m2_instances
```

- [ ] **Step 4: Read the result and record the chosen N**

Report the table (n, total steps/s, per-instance steps/s, VRAM) and the projection to the controller. `chosen_n` is the N with the highest total throughput that stayed inside the VRAM budget.

- [ ] **Step 5: Commit the script, the tests and the evidence.**

---

### Task 8: SAC training on s01 (LIVE — long job)

**Files:**
- Create: `autofly_ue5/expert/train.py`
- Create: `docs/gates/m2_train.json`
- Test: `tests/test_expert_train.py` (offline: config, seeding and resume logic against `FakeSimulator`)

**Interfaces:**
- Produces: `build_model(...)`, `make_vec_env(...)`, `main()` CLI with `--scene`, `--instances`, `--hours`, `--resume`, `--total-timesteps`, `--out`.
- Artefacts under `runs/expert/s01/`: `checkpoints/*.zip`, `best/`, `tensorboard/`, `monitor/*.csv`, `final.zip`.

**Configuration (SAC defaults unless the spec fixes them):**
- `MultiInputPolicy` with `POLICY_KWARGS` from Task 6.
- Automatic entropy tuning, `target_entropy = -3` (spec §8) — SB3's default for a 3-D action space already gives −3; assert it rather than setting it, so a silent SB3 change is caught.
- `buffer_size = 150_000`. **The arithmetic, so this is not a mystery later:** one observation is 84×84 float32 = 28,224 B of depth plus 32 B of vector; SB3's `DictReplayBuffer` stores `obs` and `next_obs`, so 150,000 transitions ≈ 150,000 × 28,256 × 2 ≈ **8.5 GiB** of the machine's 61 GiB. Do not raise this without redoing that sum. **Check whether `optimize_memory_usage=True` is supported for dict observations in SB3 2.9 before using it** — if it raises, keep it off and keep the buffer at 150k.
- `learning_starts = 5_000`, `batch_size = 256`, `train_freq = 1`, `gradient_steps = 1`, `gamma = 0.99`, `tau = 0.005`, `learning_rate = 3e-4`, `device = "cuda"`.
- `CheckpointCallback` every 10,000 steps, `EvalCallback` every 25,000 steps over 20 episodes into `best/`, `Monitor` wrapping every env.
- A `StopTrainingOnMaxTime`-style callback implementing D7's wall-clock budget, and `--resume` loading the newest checkpoint **and** its replay buffer, because a SAC resume that drops the buffer restarts exploration from scratch.

**Four hazards measured during Task 5–6's review. Each is invisible until GPU hours have already been spent, and each is cheap to prevent here.**

1. **Give every worker a distinct `seed_base`.** Two `AutoFlyEnv`s constructed with the same `seed_base` fly byte-identical episode streams — measured directly under a real `SubprocVecEnv`. N workers at the default would collect N copies of the same sequence: an N-fold loss of scene diversity that presents as "SAC plateaued", not as an error. Use `worker_seed_base(rank) = rank * 1_000_000` and keep the disjointness test.
2. **Stagger or serialise the workers' first `reset()`.** Each env launches its simulator lazily inside its first reset, so all N workers hit `check_gpu_for_launch` simultaneously, all read the same pre-launch VRAM figure, all pass, and can then collectively exhaust the GPU. `SubprocVecEnv.__init__` itself is safe — only the first reset launches.
3. **Sweep stale simulator instances before launching.** `AutoFlyEnv` has no `__del__`, so a hard crash of a previous run can leave Unreal children holding VRAM; `launch_process` then refuses with "already running" or "port already in use". Use Plan 1's `own_running_instances()` / `stop(instance)` pidfile machinery to reap them first.
4. **Wrap the rollout so a backend error retries instead of killing the run.** `AutoFlyEnv.step()` deliberately propagates `CameraPoseError`, `StepTimingError`, `CommandTimeoutError` and `StaleStateError`; under `SubprocVecEnv` an uncaught one kills the worker and aborts training. At ~300 steps × tens of thousands of episodes there are millions of chances. `reset()` can likewise raise `EpisodeSetupError` (measured 0 failures in 3000 s01 seeds, so rare but not impossible by construction). Checkpoint often and relaunch the failed instance rather than losing the run.

- [ ] **Step 1: Write the offline tests**

```python
# tests/test_expert_train.py
def test_target_entropy_is_minus_three():
    from autofly_ue5.expert.train import build_model
    from tests.test_expert_env import make_env

    m = build_model(make_env(), device="cpu", buffer_size=200, learning_starts=10)
    assert float(m.target_entropy) == -3.0


def test_seed_bases_are_disjoint_across_workers():
    from autofly_ue5.expert.train import worker_seed_base

    bases = [worker_seed_base(i) for i in range(8)]
    assert len(set(bases)) == 8
    # Each worker owns a range wide enough that two workers cannot collide within a run.
    assert min(b2 - b1 for b1, b2 in zip(sorted(bases), sorted(bases)[1:])) >= 1_000_000


def test_evaluation_seed_base_is_disjoint_from_every_training_worker():
    from autofly_ue5.expert.train import EVAL_SEED_BASE, worker_seed_base

    assert all(EVAL_SEED_BASE > worker_seed_base(i) + 1_000_000 for i in range(64))


def test_resume_picks_the_newest_checkpoint(tmp_path):
    from autofly_ue5.expert.train import newest_checkpoint

    (tmp_path / "rl_model_10000_steps.zip").write_text("a")
    (tmp_path / "rl_model_90000_steps.zip").write_text("b")
    (tmp_path / "rl_model_200000_steps.zip").write_text("c")
    assert newest_checkpoint(tmp_path).name == "rl_model_200000_steps.zip", "sort by step count, not by string"


def test_a_short_run_against_the_fake_learns_without_crashing(tmp_path):
    from autofly_ue5.expert.train import build_model
    from tests.test_expert_env import make_env

    m = build_model(make_env(), device="cpu", buffer_size=500, learning_starts=20, batch_size=8)
    m.learn(total_timesteps=60)
    assert m.num_timesteps >= 60
```

`test_resume_picks_the_newest_checkpoint` matters: sorting checkpoint filenames as strings puts `rl_model_90000_steps.zip` after `rl_model_200000_steps.zip` and would resume from an older model while reporting success.

- [ ] **Step 2: Run them, watch them fail, implement, watch them pass.**

- [ ] **Step 3: Launch the real run**

```bash
scripts/run_job.sh start m2_train -- env -u PYTHONPATH .venv/bin/python -m autofly_ue5.expert.train \
  --scene s01 --instances <chosen_n from Task 7> --hours <budget> --out docs/gates/m2_train.json
```

Monitor with `tensorboard --logdir runs/expert/s01/tensorboard`. The numbers to watch, in order of what they tell you:
1. `rollout/ep_rew_mean` rising — the agent is learning *something*.
2. `eval/success_rate` rising — it is learning the *right* thing.
3. `train/ent_coef` falling — exploration is annealing rather than stuck.
4. `rollout/ep_len_mean` falling toward ~170 — it is flying the crossing, not timing out.

**If `success_rate` is flat at ~0 after 100k steps**, stop and diagnose before spending more GPU time. In order of likelihood: the action convention is inverted somewhere (check `test_flying_straight_at_a_target_succeeds` still passes against the *real* backend, not just the fake); the altitude band is terminating episodes immediately (log the outcome histogram); the reward is dominated by the time penalty (raise `k_p`).

- [ ] **Step 4: Record the run**

`docs/gates/m2_train.json`: total timesteps, wall-clock, steps/s achieved, final and best `eval/success_rate`, the outcome histogram, the config used, the checkpoint SHA-256, and `faults_ok` from `autofly_ue5.gpu`.

- [ ] **Step 5: Commit the trainer, the tests and the run record.** The checkpoint `.zip` itself is **not** committed (it is tens of MB); it lives under `runs/` which is gitignored, and the manifest records its SHA-256.

---

### Task 9: Evaluation and the M2 gate (LIVE)

**Files:**
- Create: `autofly_ue5/expert/evaluate.py`, `scripts/m2_gate.py`
- Create: `docs/gates/m2_gate.json`
- Test: `tests/test_m2_gate.py`

**Interfaces:**
- Produces: `evaluate_policy_episodes(model, env, n_episodes, seed_base) -> EvalReport` with `success_rate`, `collision_rate`, `timeout_rate`, `out_of_bounds_rate`, `mean_steps`, `mean_final_distance_m`, `per_episode` list; and `scripts/m2_gate.py` writing `docs/gates/m2_gate.json`.

**The gate (all must hold):**
1. `success_rate >= 0.95` over **≥ 200** episodes drawn from `EVAL_SEED_BASE` (disjoint from every training worker, per D8).
2. Actions are **deterministic** (`deterministic=True`) — this is the expert that will fly M3's dataset, and a stochastic pilot would make the dataset unreproducible. (Spec §9.5 says the expert flies with *stochastic* actions during collection; the ≥95 % acceptance number is a property of the policy, so it is measured deterministically. Record both: run 200 deterministic and 200 stochastic episodes and report each.)
3. `collision_rate` recorded (no threshold — AutoFly reports 21.9 % for its own VLA; the expert should be far below).
4. `faults_ok` true — no GPU Xid faults, no reboot.
5. The Task 7 projection is carried into the gate file so M5's cost is on the record.
6. **Backend faults must be counted and reported separately from policy failures, and must never be scored as failed episodes.** Task 8 measured five recoverable hazards on this platform — `CameraPoseError`, `StepTimingError`, `StaleStateError`, `CommandTimeoutError`, and a raw `pynng.exceptions.Timeout` escaping the third-party client — at a rate that matters over hundreds of episodes. Reuse Task 8's `ResilientAutoFlyEnv` retry wrapper rather than writing another: an evaluation that silently counts a simulator hiccup as a policy failure would under-report success and could fail a genuinely good expert. Report `episodes_retried` and the per-hazard counts beside the success rate, and emit them as `0` rather than omitting them when nothing faults, so a clean run is distinguishable from broken counting.
7. **Every wait is bounded and the process must exit.** Task 8 confirmed that a crashed run hangs forever — the `projectairsim` client leaves a non-daemon thread alive, in the main process as well as in workers — and that a second failure during cleanup can prevent the gate file being written at all. The gate run must write its JSON even when the evaluation itself fails, and terminate rather than hang.

- [ ] **Step 1: Write the offline tests** (the report arithmetic, the seed disjointness, the threshold logic — using a scripted dummy policy against `FakeSimulator`, not a trained model).

```python
# tests/test_m2_gate.py
def test_rates_sum_to_one():
    from autofly_ue5.expert.evaluate import EvalReport

    r = EvalReport.from_outcomes(["success"] * 190 + ["collision"] * 5 + ["timeout"] * 4 + ["out_of_bounds"])
    assert r.success_rate == 0.95
    assert r.success_rate + r.collision_rate + r.timeout_rate + r.out_of_bounds_rate == 1.0
    assert r.n_episodes == 200


def test_gate_needs_both_the_rate_and_the_episode_count():
    from scripts.m2_gate import gate_passes

    assert gate_passes(success_rate=0.96, n_episodes=200, faults_ok=True) is True
    assert gate_passes(success_rate=0.94, n_episodes=200, faults_ok=True) is False
    assert gate_passes(success_rate=0.99, n_episodes=150, faults_ok=True) is False, "needs >= 200 episodes"
    assert gate_passes(success_rate=0.99, n_episodes=200, faults_ok=False) is False


def test_an_unsuccessful_run_is_reported_as_failing_not_rounded_up():
    from scripts.m2_gate import gate_passes

    assert gate_passes(success_rate=0.9499, n_episodes=200, faults_ok=True) is False
```

- [ ] **Step 2: Implement, run the offline tests.**

- [ ] **Step 3: Run the gate live**

```bash
scripts/run_job.sh start m2_gate -- env -u PYTHONPATH .venv/bin/python scripts/m2_gate.py \
  --model runs/expert/s01/best/best_model.zip --episodes 200 --out docs/gates/m2_gate.json
scripts/run_job.sh wait m2_gate
```

- [ ] **Step 4: Report honestly**

If the gate passes, M2 is complete. **If it does not, the gate file records `pass: false` and the real number — do not lower the threshold, do not re-run until a lucky seed passes, and do not hand-edit the evidence.** The controller then rules on which of D6's levers to pull and Task 8 re-runs. A 95 % claim that is really 88 % would poison every dataset and every student-model comparison built on top of it.

- [ ] **Step 5: Commit the evaluator, the gate, the tests and the evidence.**

---

## Milestone exit

M2 is complete when: the full suite passes; `docs/gates/m2_env_manifest.json`, `m2_instances.json`, `m2_train.json` and `m2_gate.json` all exist and describe the same run; `m2_gate.json` has `pass: true`; and the projection for the remaining nine experts is on the record so the user can decide whether M5 is affordable before it starts.

## Self-review notes

- **Spec coverage:** §8's observation (Task 2), action (Tasks 2/5), network (Task 6), reward (Task 3), acceptance ≥95 % over ≥200 episodes (Task 9), throughput gate (Task 7), stable-baselines3 + vectorised envs + one instance per env (Tasks 5/8). §9's episode setup steps 1–3 and 5 (Task 4/5); §9 step 4's `a0` is M3's, not M2's, and is deliberately out of scope. §8.1 is untouched — the AutoFly checkpoint pilot is optional and gated on a release that has not happened.
- **Interfaces:** `target_geometry` is defined once in Task 2 and consumed by Tasks 3 and 5; `EpisodeSetup` is defined in Task 4 and consumed by Task 5; `POLICY_KWARGS` is defined in Task 6 and consumed by Task 8; `EVAL_SEED_BASE`/`worker_seed_base` are defined in Task 8 and consumed by Task 9.
- **Known gap carried deliberately:** the target is a cylinder until M4 supplies the asset pool (D1). The expert is depth-only plus a privileged vector, so this cannot affect the expert's quality — only the look of M3's RGB frames, which M4 fixes.
