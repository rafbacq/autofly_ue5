"""Gymnasium environment wrapping a Simulator with the expert's obs/reward/episode stack (spec 7.1, 8, 9).

`AutoFlyEnv` is the only place that owns a `Simulator` instance and its lifecycle. Everything it needs to
turn that simulator into an RL environment already exists and is closed for editing: `episode.py` samples
and (de)spawns episodes, `obs.py` encodes observations and is the single source of target geometry, and
`reward.py` scores a step and decides termination. This module just sequences those calls in the order the
simulator's lifecycle requires.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from autofly_ue5.expert.episode import EpisodeSetup, apply_setup, clear_setup, sample_setup
from autofly_ue5.expert.obs import DEPTH_SIZE, VECTOR_DIM, encode, target_geometry
from autofly_ue5.expert.reward import Outcome, RewardConfig, evaluate, oob_kind
from autofly_ue5.scenes.model import Layout, SceneFile
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.types import CONTROL_DT_S, StartCollisionError


class AutoFlyEnv(gym.Env):
    """One simulator per env, injected as a factory so tests run `FakeSimulator` and training runs the
    real backend without this module importing either.

    Per-episode flow in `reset()`: destroy the previous episode's spawned objects, sample a new
    `EpisodeSetup`, reset the simulator to its start pose (before any `step()` -- a session's first
    rendered frame is corrupt, spec 7.1), *then* spawn the new target/distractors so the reset's settle
    sweep cannot collide with them, and destroy the ones spawned before it so a long run cannot
    accumulate actors in the scene.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        scene: SceneFile,
        layout: Layout,
        sim_factory: Callable[[], Simulator],
        *,
        map_path: str,
        instance: int,
        cfg: RewardConfig = RewardConfig(),
        seed_base: int = 0,
        max_episode_steps: int = 300,
    ) -> None:
        super().__init__()
        self._scene = scene
        self._layout = layout
        self._sim_factory = sim_factory
        self._map_path = map_path
        self._instance = instance
        # max_episode_steps is a separate knob from cfg.step_limit -- e.g. a shorter smoke-test episode
        # without hand-building a whole new RewardConfig just to change one field. Folding it into the
        # cfg actually used by evaluate() keeps classify()'s TIMEOUT check from disagreeing with it.
        self._cfg = dataclasses.replace(cfg, step_limit=max_episode_steps)
        self._seed_base = seed_base
        self._max_episode_steps = max_episode_steps

        self.observation_space = spaces.Dict({
            "depth": spaces.Box(0.0, 1.0, (1, DEPTH_SIZE, DEPTH_SIZE), dtype=np.float32),
            "vector": spaces.Box(-np.inf, np.inf, (VECTOR_DIM,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._sim: Simulator | None = None
        self._spawned: tuple[str, ...] = ()
        self._setup: EpisodeSetup | None = None
        self._prev_dist = 0.0
        self._step_index = 0
        # Advances by one on every seed=None reset() so a given env instance replays episodes 0, 1, 2,
        # ... in a fixed, reproducible order (offset per-env by seed_base, so vectorised workers stay
        # disjoint even when the caller resets every one of them with seed=None). An explicit seed does
        # NOT touch this counter -- see reset() for why.
        self._episode_index = 0

    def _ensure_launched(self) -> Simulator:
        if self._sim is None:
            # Kept only once launch() succeeds: a simulator whose launch raised is not connected, and reusing it
            # would turn every later reset() into "not connected" instead of a fresh launch.
            sim = self._sim_factory()
            sim.launch(self._map_path, self._instance)
            self._sim = sim
        return self._sim

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        # An explicit seed must pick out ONE deterministic, reproducible episode, distinct from every
        # other seed: an evaluation harness (Task 9's M2 gate) draws >= 200 episodes as
        # reset(seed=EVAL_SEED_BASE + i) for i in range(200) and needs 200 DIFFERENT episodes out of
        # that, not the same one 200 times. So an explicit seed feeds `sample_setup` directly --
        # `rng = np.random.default_rng(seed)` -- rather than being combined with the per-env counter:
        # combining them (e.g. seed_base + seed, or rewinding the counter to 0 on every seed) would
        # collapse many distinct seeds onto the same handful of episodes, or onto exactly one. When
        # seed is None (the normal per-episode call during training), we fall back to the counter-driven
        # stream so vectorised workers stay disjoint via seed_base without the caller having to stagger
        # seeds itself.
        super().reset(seed=seed)

        sim = self._ensure_launched()
        clear_setup(sim, self._spawned)  # previous episode's objects, destroyed before the next is sampled
        self._spawned = ()

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(self._seed_base + self._episode_index)
            self._episode_index += 1
        setup = sample_setup(self._scene, self._layout, rng)

        obs = sim.reset(setup.start)  # before any step() -- frame 0 of a session is corrupt (spec 7.1)
        # Spawn the target/distractors AFTER the reset, not before: reset()'s settle sweep could
        # otherwise collide with objects placed where the drone is about to be teleported to.
        self._spawned = apply_setup(sim, setup)
        self._setup = setup

        # `obs` above was rendered before apply_setup ran, so its depth frame is missing the
        # target/distractors just spawned. Re-render with a zero-velocity step so the observation this
        # method actually returns reflects the completed scene: M3's dataset collector inherits this
        # env's reset() contract and treats its first frame as a0, so a stale s_0 would propagate a wrong
        # "expert demonstration" into the dataset itself, not just cost SAC one bad transition in ~300.
        # This step must not count against the episode's step budget -- it is a render, not a policy
        # action -- so self._step_index is set to 0 afterward, not incremented.
        sim.command_velocity(0.0, 0.0, 0.0)
        sim.step(CONTROL_DT_S)
        obs = sim.observe()
        if obs.collided:
            # An episode that starts in contact is lost before its first action (C9): not a policy outcome.
            raise StartCollisionError(f"the episode's first observation already reports a collision at {obs.pose}")

        self._prev_dist, bearing, _ = target_geometry(obs.pose, setup.target_xy_z)
        self._step_index = 0

        info = self._info(Outcome.RUNNING, self._prev_dist, obs.pose, bearing, None)
        return encode(obs, setup.target_xy_z), info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if self._sim is None or self._setup is None:
            raise RuntimeError("AutoFlyEnv.step() called before reset()")

        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        v_forward, yaw_rate, v_z = (float(a) for a in action)

        self._sim.command_velocity(v_forward, yaw_rate, v_z)  # a fresh command is required before every step
        self._sim.step(CONTROL_DT_S)
        obs = self._sim.observe()

        dist, bearing, _ = target_geometry(obs.pose, self._setup.target_xy_z)
        altitude = -obs.pose.z  # NED: altitude above ground is -z
        bounds = self._layout.bounds
        in_bounds = bounds.x_min <= obs.pose.x <= bounds.x_max and bounds.y_min <= obs.pose.y <= bounds.y_max

        self._step_index += 1
        result = evaluate(
            prev_dist_m=self._prev_dist, dist_m=dist, bearing_rad=bearing, altitude_m=altitude,
            in_bounds=in_bounds, collided=obs.collided, step_index=self._step_index, cfg=self._cfg,
        )
        self._prev_dist = dist

        kind = oob_kind(in_bounds=in_bounds, altitude_m=altitude, cfg=self._cfg) if result.outcome is Outcome.OUT_OF_BOUNDS else None
        info = self._info(result.outcome, dist, obs.pose, bearing, kind)
        return encode(obs, self._setup.target_xy_z), result.reward, result.terminated, result.truncated, info

    def _info(self, outcome: Outcome, dist_m: float, pose, bearing_rad: float, oob: str | None) -> dict:
        # pose/bearing_deg/oob_kind: where an episode ended and which bound it left, for the run record (the
        # 2026-09-17 gate could not say either).
        return {
            "outcome": outcome.value,
            "steps": self._step_index,
            "final_distance_m": float(dist_m),
            "is_success": outcome is Outcome.SUCCESS,
            "pose": [float(pose.x), float(pose.y), float(pose.z), float(pose.yaw)],
            "bearing_deg": math.degrees(bearing_rad),
            "oob_kind": oob,
        }

    def close(self) -> None:
        # Detach first, then close: a relaunch may abandon a hung close() and install a new simulator, and the late
        # return of this call must not detach that replacement.
        sim, self._sim = self._sim, None
        if sim is not None:
            sim.close()
