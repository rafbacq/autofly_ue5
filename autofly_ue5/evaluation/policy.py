"""What a policy sees and returns in the closed-loop evaluation (plan 6 item 6; spec §12 M6), and the reference adapters.

A policy is any object with two methods (`Policy`):

    reset(instruction: str) -> None           once per episode attempt, before its first act()
    act(obs: PolicyObservation) -> action     once per 0.2 s control step, until the episode ends

`act` returns AutoFly's action[3] in physical units: forward speed in [0, 2] m/s, yaw rate in [-1, 1] rad/s (positive
turns right, clockwise seen from above), vertical speed in [-1, 1] m/s (positive climbs). The harness clips it to that box
exactly as AutoFlyEnv.step() does (the collector recorded the expert's actions clipped the same way). Anything but three
finite numbers ends the run (`PolicyError`): replacing it silently would score a broken adapter as a policy.

`PolicyObservation` holds what one dataset record holds before its action:

- rgb: (256, 256, 3) uint8 RGB, the front camera frame, the record's observation.image_0 exactly. The harness builds it,
  and the state, with the collector's own code (collect.collector.record_inputs), so a policy is scored on the inputs it
  was trained on.
- instruction: the episode's language_instruction, naming the drawn target (expert/episode.py's two templates).
- state: float32[9] as dataset/state.py defines it. state[0] and state[1] are the distance and the bearing to the target
  from its true position. They are privileged: a policy that reads them can fly from them alone (plan 6 revision 1 item 4).
  So they are ZERO unless the harness runs with privileged_state=True, and `privileged_state` says which.
- step: the record's index within the episode; 0 is the a0-aligned start.
- extras: inputs only the reference adapters may use: "expert_obs", the SAC expert's own observation (depth and the
  target's true position). A student must never read it.

How a VLA implements it (the next phase; none is attached here):

    class OpenVLAPolicy:
        name = "openvla_autofly"                     # optional: names it in the record (or describe() -> dict)

        def __init__(self, checkpoint, statistics_path):
            self.model = ...                         # in its own process or venv if its dependencies clash with .venv
            self.stats = load_action_statistics(statistics_path, "autofly_ue5_proto")

        def reset(self, instruction):
            self.history = []                        # whatever it keeps per episode; a replayed attempt resets it too

        def act(self, obs):
            normalized = self.model.predict(image=obs.rgb, instruction=obs.instruction)  # in [-1, 1]^3, greedy
            return unnormalize_action(normalized, self.stats)

1. Inputs: obs.rgb and obs.instruction, preprocessed as in training (OpenVLA's processor resizes 256 to 224). Add
   obs.state only if the model was trained on observation.state, and then report it both ways, privileged_state off and
   on, as revision 1 item 4 asks of every baseline.
2. Output: de-normalise with the statistics the model was trained with, the train split's dataset_statistics.json of the
   merged dataset (revision 1 item 1, OpenVLA's schema). Per dimension a = 0.5 (n + 1)(q99 - q01) + q01 where its mask is
   true; all three of AutoFly's dimensions are (OpenVLA leaves only a gripper unmasked). That is `unnormalize_action`, and
   spec §8.1 decodes the AutoFly checkpoint the same way.
3. Decode greedily (do_sample=False), so a score can be reproduced.
4. Timing: the simulator runs in lock-step. Its clock advances only inside the harness's step, so act() may take as long
   as it needs and the dynamics do not change; the record keeps the time each action took. A model on the GPU counts as
   this process's own memory to the launch guard (gpu.check_gpu_for_launch), but the guard still wants 6 GB free for
   every simulator launch.
5. Run it with `autofly_ue5.evaluation.harness.run_condition(policy, <condition file>, instance=k, out_path=...)`, or add
   its loader to scripts/eval_harness.py's `build_policy`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from autofly_ue5.expert.obs import ObsConfig, obs_config_from_space

STATE_DIM = 9
# dataset/state.py's distance_to_target_m and bearing_to_target_rad: computed from the target's true position.
PRIVILEGED_STATE_INDICES = (0, 1)
ACTION_FIELDS = ("forward_m_s", "yaw_rate_rad_s", "vertical_m_s")


class PolicyError(RuntimeError):
    """A policy failed (it raised, or returned something that is not an action): the run stops, as no score is honest."""


class PrivilegedStateRequired(PolicyError):
    """A policy that steers by state[0] and state[1] was handed them zeroed."""


@dataclass(frozen=True, eq=False)
class PolicyObservation:
    rgb: np.ndarray
    instruction: str
    state: np.ndarray
    privileged_state: bool = False
    step: int = 0
    extras: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    """See the module docstring. Optional attributes the harness reads: `name` (str), `describe()` (a dict for the
    record), `requires_privileged_state` (bool: refused before launch without it) and `obs_config` (the expert
    observation the env must build for it, an `expert.obs.ObsConfig`)."""

    def reset(self, instruction: str) -> None: ...

    def act(self, obs: PolicyObservation) -> np.ndarray: ...


def zero_privileged(state: np.ndarray) -> np.ndarray:
    """A float32 copy of `state` with the privileged fields zeroed."""
    out = np.array(state, dtype=np.float32, copy=True)
    out[list(PRIVILEGED_STATE_INDICES)] = 0.0
    return out


def checked_action(action: Any) -> np.ndarray:
    """`action` as float32[3], or a PolicyError saying what is wrong with it."""
    try:
        values = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise PolicyError(f"a policy's action must be 3 numbers {ACTION_FIELDS}, got {action!r}") from err
    if values.size != 3:
        raise PolicyError(f"a policy's action must be 3 numbers {ACTION_FIELDS}, got shape {values.shape}: {action!r}")
    values = values.reshape(3)
    if not np.all(np.isfinite(values)):
        raise PolicyError(f"a policy's action must be finite, got {values.tolist()}")
    return values.astype(np.float32)


def describe_policy(policy: Any) -> dict:
    """What the record says of `policy`: its describe() if it has one, else its name."""
    describe = getattr(policy, "describe", None)
    if callable(describe):
        return dict(describe())
    return {"name": getattr(policy, "name", type(policy).__name__)}


class HoverPolicy:
    """Commands nothing: the floor every policy must beat (it times out wherever nothing pushes it)."""

    name = "hover"
    obs_config = None

    def reset(self, instruction: str) -> None:
        pass

    def act(self, obs: PolicyObservation) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def describe(self) -> dict:
        return {"name": self.name, "kind": "floor", "action": [0.0, 0.0, 0.0]}


class ScriptedPolicy:
    """A privileged straight-line baseline: it turns on the spot until the target's bearing (state[1]) is within
    `aligned_rad`, then flies at `cruise_m_s` steering onto it, holding `altitude_m` (state[8], not privileged), and ignores
    every obstacle. It shows what knowing where the target is buys without any avoidance; it needs the privileged
    state, so it refuses to act without it (`requires_privileged_state`, checked by the harness before anything launches)."""

    name = "scripted"
    requires_privileged_state = True
    obs_config = None

    def __init__(self, *, cruise_m_s: float = 2.0, altitude_m: float = 2.0, aligned_rad: float = 0.3,
                 turn_gain: float = 2.0, climb_gain: float = 1.0) -> None:
        self.cruise_m_s = cruise_m_s
        self.altitude_m = altitude_m
        self.aligned_rad = aligned_rad
        self.turn_gain = turn_gain
        self.climb_gain = climb_gain

    def reset(self, instruction: str) -> None:
        pass

    def act(self, obs: PolicyObservation) -> np.ndarray:
        if not obs.privileged_state:
            raise PrivilegedStateRequired(
                "the scripted baseline steers by state[0] and state[1], the target's true distance and bearing, which "
                "the harness zeroes unless it runs with privileged_state=True (scripts/eval_harness.py --privileged-state)")
        bearing, altitude = float(obs.state[1]), float(obs.state[8])
        forward = self.cruise_m_s if abs(bearing) <= self.aligned_rad else 0.0
        yaw_rate = float(np.clip(self.turn_gain * bearing, -1.0, 1.0))
        climb = float(np.clip(self.climb_gain * (self.altitude_m - altitude), -1.0, 1.0))
        return np.array([forward, yaw_rate, climb], dtype=np.float32)

    def describe(self) -> dict:
        return {"name": self.name, "kind": "privileged_straight_line", "requires_privileged_state": True,
                "cruise_m_s": self.cruise_m_s, "altitude_m": self.altitude_m, "aligned_rad": self.aligned_rad,
                "turn_gain": self.turn_gain, "climb_gain": self.climb_gain}


def checkpoint_obs_config(path: Path) -> ObsConfig:
    """The observation an SB3 checkpoint was trained on, read from its zip without building a model (the same reading
    as scripts/m2_gate.checkpoint_obs_config, which package code cannot import)."""
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _params, _variables = load_from_zip_file(path, device="cpu", load_data=True)
    return obs_config_from_space(data["observation_space"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExpertPolicy:
    """The SAC expert (an SB3 checkpoint) flying on its own observation, which the harness passes in
    extras["expert_obs"] built for `obs_config`. That observation holds the target's true position (expert/obs.py), so the
    expert is a privileged reference however the state is handed over, never a student. Stochastic is how it collected
    the dataset (spec §9 step 5), deterministic how it was gated."""

    name = "expert"

    def __init__(self, model: Any, *, deterministic: bool, obs_config: ObsConfig, checkpoint: Path | None = None,
                 sha256: str | None = None, trained_footprint_shape: str | None = None) -> None:
        self.model = model
        self.deterministic = bool(deterministic)
        self.obs_config = obs_config
        self.checkpoint = checkpoint
        self.sha256 = sha256
        self.trained_footprint_shape = trained_footprint_shape

    @classmethod
    def load(cls, checkpoint: Path, *, deterministic: bool, device: str = "auto",
             loader: Callable[[Path], Any] | None = None,
             obs_config_reader: Callable[[Path], ObsConfig] | None = None) -> "ExpertPolicy":
        """The checkpoint at `checkpoint`, its observation read from the zip; `loader` and `obs_config_reader` replace
        SB3's for tests."""
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"no expert checkpoint at {path}")
        obs_config = (obs_config_reader or checkpoint_obs_config)(path)
        if loader is None:
            from stable_baselines3 import SAC

            def loader(p: Path) -> Any:
                return SAC.load(p, device=device)

        try:  # what the harness's outline rule differs from, for the record; it never changes what is scored
            from autofly_ue5.expert.run_shape import trained_shape

            shape = trained_shape(path)[0]
        except ValueError:
            shape = None
        return cls(loader(path), deterministic=deterministic, obs_config=obs_config, checkpoint=path,
                   sha256=_file_sha256(path), trained_footprint_shape=shape)

    def reset(self, instruction: str) -> None:
        pass

    def act(self, obs: PolicyObservation) -> np.ndarray:
        own = obs.extras.get("expert_obs")
        if own is None:
            raise PolicyError("the expert flies on its own observation, which the harness passes in "
                              "extras['expert_obs']; this observation has none")
        action, _state = self.model.predict(own, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(3)

    def describe(self) -> dict:
        return {"name": self.name, "kind": "sac_expert", "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
                "sha256": self.sha256, "deterministic": self.deterministic, "obs_config": self.obs_config.to_json(),
                "privileged_observation": True, "trained_footprint_shape": self.trained_footprint_shape}


# --------------------------------------------------------------------------------------------------------
# A VLA's action, back in AutoFly's units.
# --------------------------------------------------------------------------------------------------------
def unnormalize_action(normalized: Any, stats: Mapping[str, Any]) -> np.ndarray:
    """OpenVLA's BOUNDS_Q99 inverse: a = 0.5 (n + 1)(q99 - q01) + q01 per dimension where `stats["mask"]` is true (all
    when it has none), n passed through elsewhere. `stats` is one dataset's "action" entry of dataset_statistics.json."""
    n = np.asarray(normalized, dtype=np.float64).reshape(-1)
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    mask = np.asarray(stats.get("mask", np.ones(q01.shape, dtype=bool)), dtype=bool)
    if not n.shape == q01.shape == q99.shape == mask.shape:
        raise ValueError(f"a normalised action of shape {n.shape} against statistics of shapes q01 {q01.shape}, "
                         f"q99 {q99.shape}, mask {mask.shape}")
    return np.where(mask, 0.5 * (n + 1.0) * (q99 - q01) + q01, n).astype(np.float32)


def load_action_statistics(path: Path, dataset_name: str | None = None) -> dict:
    """The "action" statistics of `dataset_name` in a dataset_statistics.json: OpenVLA's file keyed by dataset, or a
    dataset's own file with "action" at its top. The name may be left out when the file holds one dataset."""
    data = json.loads(Path(path).read_text())
    if "action" in data:
        stats = data["action"]
    else:
        keyed = {name: entry for name, entry in data.items() if isinstance(entry, dict) and "action" in entry}
        if dataset_name is not None:
            if dataset_name not in keyed:
                raise ValueError(f"{path}: no dataset {dataset_name!r} (it holds {sorted(keyed)})")
            stats = keyed[dataset_name]["action"]
        elif len(keyed) == 1:
            stats = next(iter(keyed.values()))["action"]
        else:
            raise ValueError(f"{path} holds the statistics of {sorted(keyed)}: name the dataset")
    for key in ("q01", "q99"):
        if len(stats.get(key) or ()) != 3:
            raise ValueError(f"{path}: the action statistics need 3 values of {key}, got {stats.get(key)!r}")
    return stats
