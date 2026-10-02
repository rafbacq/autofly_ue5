"""Gymnasium environment wrapping a Simulator with the expert's obs/reward/episode stack (spec 7.1, 8, 9).

`AutoFlyEnv` is the only place that owns a `Simulator` instance and its lifecycle. Everything it needs to
turn that simulator into an RL environment lives elsewhere: `episode.py` samples and (de)spawns episodes,
`obs.py` encodes observations and is the single source of target geometry, and `reward.py` scores a step and
decides termination. This module just sequences those calls in the order the simulator's lifecycle requires.
(Earlier plans marked these modules "closed for editing"; the 2026-09-24 review reopened them -- see
docs/decisions/2026-09-25-code-review-findings.md.)
"""

from __future__ import annotations

import dataclasses
import math
import sys
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from autofly_ue5.expert.episode import EpisodeSetup, apply_setup, clear_setup, sample_setup
from autofly_ue5.expert.movers import MoverController, home_poses, park_poses
from autofly_ue5.expert.obs import DepthStacker, ObsConfig, encode_depth, encode_vector, obs_config_for_scene, target_geometry
from autofly_ue5.expert.reward import Outcome, RewardConfig, evaluate, oob_kind
from autofly_ue5.scenes.model import Layout, SceneFile
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.types import CONTROL_DT_S, CameraPoseError, KinematicsJumpError, Observation, Pose, StartCollisionError

# Step faults that a moving pillar the drone was about to touch can cause (spec §6.5, "defensive inference").
MOVER_INFERABLE_FAULTS = (CameraPoseError, KinematicsJumpError)


class AutoFlyEnv(gym.Env):
    """One simulator per env, injected as a factory so tests run `FakeSimulator` and training runs the
    real backend without this module importing either.

    Per-episode flow in `reset()`: destroy the previous episode's spawned objects, sample a new
    `EpisodeSetup`, reset the simulator to its start pose (before any `step()` -- a session's first
    rendered frame is corrupt, spec 7.1), *then* spawn the new target/distractors so the reset's settle
    sweep cannot collide with them, and destroy the ones spawned before it so a long run cannot
    accumulate actors in the scene.

    A dynamic scene (spec §6.5) also moves some of the level's pillars. Every pillar this env may have displaced is
    tracked in `_displaced` from before it moves until it is back home, and parked 50 m under its home before the
    reset sequence, so the drone and a pillar are never in one place: see `reset()`.
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
        obs_config: ObsConfig | None = None,
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

        # One float32 depth frame for a static scene (every M2 checkpoint's space); a stack for a dynamic one.
        self._obs_config = obs_config if obs_config is not None else obs_config_for_scene(scene)
        self.observation_space = self._obs_config.space()
        self._stacker = DepthStacker(self._obs_config)
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
        self._instances = {inst.tag: inst for inst in layout.instances}
        self._movers: MoverController | None = None
        # Pillars that may be away from home in the current simulator: marked before anything moves them, cleared only
        # once they are back home (or a fresh simulator has launched with every pillar home).
        self._displaced: set[str] = set()
        self._last_pose: Pose | None = None
        self._last_obs: dict[str, np.ndarray] | None = None
        self._last_raw: Observation | None = None
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
            self._displaced = set()  # a freshly launched level has every pillar at home
            self._movers = None
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
        self._movers = None
        if self._displaced:
            # Out of the way of the reset's up-across-down sequence: the drone may have ended on a vacated home spot,
            # and sending a pillar home there would put it inside the drone.
            sim.set_object_poses(park_poses(self._instances, self._displaced))

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

        # Movers to where their clocks start, and last episode's other movers home, in one batch now that the drone
        # stands at its start (>= start_keepout_m from every sweep). Marked displaced before anything moves, so a
        # failure partway through still leaves every one of them tracked for the next reset to park.
        movers = None
        new = {route.tag for route in setup.movers}
        if setup.movers or self._displaced:
            old_only = self._displaced - new
            self._displaced |= new
            batch = home_poses(self._instances, old_only)
            if setup.movers:
                d = self._scene.dynamic
                movers = MoverController(setup.movers, contact_m=d.contact_m, yield_margin_m=d.yield_margin_m,
                                         max_speed_m_s=d.speed_m_s[1], dt=CONTROL_DT_S)
                batch.update(movers.initial_poses())
            sim.set_object_poses(batch)

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
        if movers is not None and movers.nearest_gap((obs.pose.x, obs.pose.y)) <= movers.contact_m:
            raise StartCollisionError(f"the episode's first observation is within {movers.contact_m} m of a mover")
        self._displaced = new
        self._movers = movers
        if movers is not None:
            movers.record_frame(obs.pose)

        self._prev_dist, bearing, _ = target_geometry(obs.pose, setup.target_xy_z)
        self._step_index = 0
        self._last_pose = obs.pose
        self._last_raw = obs

        info = self._info(Outcome.RUNNING, self._prev_dist, obs.pose, bearing, None)
        if movers is not None:
            info["mover_routes"] = [route.to_json() for route in setup.movers]
        self._last_obs = {"depth": self._stacker.reset(encode_depth(obs.depth)),
                          "vector": encode_vector(obs.pose, obs.velocity_ned, obs.yaw_rate, setup.target_xy_z)}
        return self._last_obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if self._sim is None or self._setup is None:
            raise RuntimeError("AutoFlyEnv.step() called before reset()")

        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        v_forward, yaw_rate, v_z = (float(a) for a in action)

        before = self._last_pose
        movers = self._movers
        if movers is not None:
            moved = movers.advance((before.x, before.y))  # each mover yields to the drone where it stands now
            if moved:
                self._sim.set_object_poses(moved)

        self._sim.command_velocity(v_forward, yaw_rate, v_z)  # a fresh command is required before every step
        try:
            self._sim.step(CONTROL_DT_S)
        except MOVER_INFERABLE_FAULTS as err:
            # A camera left behind or a jump right next to a moving pillar is that pillar's doing: a fault would drop
            # it from training forever and make the gate replay the seed until it gives up (spec §6.5).
            if movers is not None and movers.nearest_gap((before.x, before.y)) <= movers.inference_radius_m:
                return self._inferred_mover_collision(err)
            raise
        obs = self._sim.observe()

        contact = movers.contact((before.x, before.y), (obs.pose.x, obs.pose.y)) if movers is not None else None
        collided = obs.collided or contact is not None
        source = "mover" if contact is not None else ("sim" if obs.collided else None)
        seen = movers.seen_recently(contact[0]) if contact is not None else None  # the frames the policy flew on
        if movers is not None:
            movers.record_frame(obs.pose)

        dist, bearing, _ = target_geometry(obs.pose, self._setup.target_xy_z)
        altitude = -obs.pose.z  # NED: altitude above ground is -z
        bounds = self._layout.bounds
        in_bounds = bounds.x_min <= obs.pose.x <= bounds.x_max and bounds.y_min <= obs.pose.y <= bounds.y_max

        self._step_index += 1
        result = evaluate(
            prev_dist_m=self._prev_dist, dist_m=dist, bearing_rad=bearing, altitude_m=altitude,
            in_bounds=in_bounds, collided=collided, step_index=self._step_index, cfg=self._cfg,
        )
        self._prev_dist = dist
        self._last_pose = obs.pose
        self._last_raw = obs

        kind = oob_kind(in_bounds=in_bounds, altitude_m=altitude, cfg=self._cfg) if result.outcome is Outcome.OUT_OF_BOUNDS else None
        info = self._info(result.outcome, dist, obs.pose, bearing, kind,
                          collision_source=source if result.outcome is Outcome.COLLISION else None, mover_in_view=seen)
        self._last_obs = {"depth": self._stacker.push(encode_depth(obs.depth)),
                          "vector": encode_vector(obs.pose, obs.velocity_ned, obs.yaw_rate, self._setup.target_xy_z)}
        return self._last_obs, result.reward, result.terminated, result.truncated, info

    @property
    def last_observation(self) -> Observation | None:
        """The simulator's full observation behind the last reset() or step() -- RGB included, which the expert never
        sees but M3's collector records (spec §9-§10)."""
        return self._last_raw

    def _inferred_mover_collision(self, err: Exception) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """End the episode as a collision on its last real observation, nothing having been observed this step."""
        pose = self._last_pose
        dist, bearing, _ = target_geometry(pose, self._setup.target_xy_z)
        self._step_index += 1
        result = evaluate(prev_dist_m=self._prev_dist, dist_m=dist, bearing_rad=bearing, altitude_m=-pose.z,
                          in_bounds=True, collided=True, step_index=self._step_index, cfg=self._cfg)
        info = self._info(result.outcome, dist, pose, bearing, None, collision_source="mover_inferred",
                          mover_in_view=None)
        info["inferred_from"] = f"{type(err).__name__}: {err}"
        print(f"MOVER instance {self._instance}: {info['inferred_from']} within "
              f"{self._movers.nearest_gap((pose.x, pose.y)):.2f} m of a mover: scored as a collision", file=sys.stderr)
        return self._last_obs, result.reward, result.terminated, result.truncated, info

    def _info(self, outcome: Outcome, dist_m: float, pose, bearing_rad: float, oob: str | None, *,
              collision_source: str | None = None, mover_in_view: bool | None = None) -> dict:
        # pose/bearing_deg/oob_kind: where an episode ended and which bound it left, for the run record (the
        # 2026-09-17 gate could not say either). collision_source ("sim", "mover", "mover_inferred"), n_movers,
        # movers (each mover's x, y this step) and mover_in_view: spec §6.5; empty for a static scene.
        return {
            "outcome": outcome.value,
            "steps": self._step_index,
            "final_distance_m": float(dist_m),
            "is_success": outcome is Outcome.SUCCESS,
            "pose": [float(pose.x), float(pose.y), float(pose.z), float(pose.yaw)],
            "bearing_deg": math.degrees(bearing_rad),
            "oob_kind": oob,
            "collision_source": collision_source,
            "n_movers": len(self._movers.routes) if self._movers is not None else 0,
            "movers": self._movers.info() if self._movers is not None else [],
            "mover_in_view": mover_in_view,
        }

    def close(self) -> None:
        # Detach first, then close: a relaunch may abandon a hung close() and install a new simulator, and the late
        # return of this call must not detach that replacement.
        sim, self._sim = self._sim, None
        if sim is not None:
            sim.close()
