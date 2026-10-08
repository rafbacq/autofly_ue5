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

from autofly_ue5.dataset.targets import TargetPool
from autofly_ue5.expert.altitude_margin import AltitudeMarginPenalty
from autofly_ue5.expert.episode import EpisodeSetup, a0_aligned, apply_setup, clear_setup, sample_setup
from autofly_ue5.expert.footprint_contact import (
    DEFAULT_SHAPE,
    FootprintContactRule,
    check_shape,
    footprint_obstacles,
    require_fields,
)
from autofly_ue5.expert.mover_clearance import MoverClearancePenalty, MoverClosingPenalty
from autofly_ue5.expert.movers import MoverController, home_poses, park_poses
from autofly_ue5.expert.obs import (
    MOVER_FEATURES,
    DepthStacker,
    ObsConfig,
    encode_depth,
    encode_vector,
    obs_config_for_scene,
    target_geometry,
)
from autofly_ue5.expert.reward import Outcome, RewardConfig, evaluate, oob_kind, reward_config_for_scene
from autofly_ue5.expert.static_contact import StaticContactBoundary
from autofly_ue5.frames import wrap_pi
from autofly_ue5.scenes.footprint_field import FootprintFields
from autofly_ue5.scenes.model import AssetRegistry, Layout, SceneFile, load_registry
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.types import CONTROL_DT_S, CameraPoseError, KinematicsJumpError, Observation, Pose, StartCollisionError

def action_space() -> spaces.Box:
    """Forward speed 0-2 m/s, yaw rate -1..1 rad/s, vertical speed -1..1 m/s (spec §8). A function, so a check that
    needs the policy's shapes (expert.warmstart's pre-check) can build it without an env."""
    return spaces.Box(low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
                      high=np.array([2.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)


# Step faults that a moving pillar the drone was about to touch can cause (spec §6.5, "defensive inference").
MOVER_INFERABLE_FAULTS = (CameraPoseError, KinematicsJumpError)

# Plan 6 step 2a. Live, a pool object spawned for the first time in a simulator session could be missing from the first
# frame after it spawned and showed within five steps (runs/m4/targets_settle_diag/settle.json: three of six models drew
# no pixel on step 1 and their whole outline by step 5, a fourth a fifth of it; every second spawn showed on step 1;
# docs/gates/m4_targets_probe.json counted five such first-frame misses). The collector records the reset's frame as the
# episode's first, so a reset that spawned an object new to the session hovers this many more steps before it returns.
PREWARM_HOLD_STEPS = 5


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

    A scene with mesh obstacles (trees, rocks: expert/footprint_contact.py) ends an episode near each obstacle,
    geometrically and on every step, because a body that reaches wood can wedge the simulator's actor. That is the
    task's rule, scored by the gate; `footprint_contact_margin_m` widens it for a training run only.
    `footprint_shape`: what the distance is measured to, the model's outline ("outline", 1.05 m: the task's rule since
    2026-10-08) or its footprint circle ("circle", 0.9 m: what s02_r1 trained under). `footprint_fields`: the outline fields
    (scenes/footprint_field.py; default: read now from assets/footprints/, and an asset without one is refused here).
    `registry`: the asset registry that decides which obstacles are meshes (default: read now). A trainer passes the
    registry and the fields it read before claiming its run root, so its identity, record and workers cannot disagree
    about them.

    `target_pool` (plan 5 C5, `autofly_ue5.dataset.targets`): the objects a collection's targets and distractors are,
    handed to the sampler for every episode; the instruction then names the target. A collection-time choice: the
    trainer and the gate pass none, so the expert keeps flying to the orange cylinder it learned on. A pool the scene's
    split cannot draw from (fewer than six usable entries, after the scene's own exclusions: the pool's
    target_exclusions.json, keyed by this scene's id) is refused here, before anything launches.

    `target_split` (evaluation only, plan 6 revision 1 item 5; needs a pool): None draws for the scene's own split;
    "train", "test_seen", "test_unseen" for that split; "unseen" an unseen target among seen distractors
    (`expert.episode.sample_setup` pins each form).

    The first-use hold (plan 6 step 2a, with a pool only): when an episode spawns an object, an entry's mesh and
    material, that this simulator session has not spawned before, `reset()` hovers PREWARM_HOLD_STEPS more steps after
    its re-render step, so the observation it returns (the collector's first record) shows every object. The objects
    count as warm once a hold completes; a launch, a relaunch included, starts cold. Not episode steps, like the
    re-render step; the reset info says how many (`prewarm_steps`). Without a pool nothing changes, call for call.
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
        mover_clearance: MoverClearancePenalty | None = None,
        mover_closing: MoverClosingPenalty | None = None,
        altitude_margin: AltitudeMarginPenalty | None = None,
        mover_contact_margin_m: float = 0.0,
        static_contact_m: float = 0.0,
        footprint_contact_margin_m: float = 0.0,
        registry: AssetRegistry | None = None,
        target_pool: TargetPool | None = None,
        footprint_shape: str = DEFAULT_SHAPE,
        footprint_fields: FootprintFields | None = None,
        target_split: str | None = None,
    ) -> None:
        super().__init__()
        self._footprint_shape = check_shape(footprint_shape)
        if target_split is not None and target_pool is None:
            raise ValueError(f"target_split {target_split!r} says which pool entries an episode draws: it needs a "
                             f"target pool")
        if target_pool is not None:
            # raises TargetPoolError for a split it cannot fill, after this scene's own exclusions (plan 6 step 2c)
            target_pool.draw_lists(scene.split if target_split is None else target_split, scene_id=scene.id)
        self._target_pool = target_pool
        self._target_split = target_split
        if not (math.isfinite(static_contact_m) and static_contact_m >= 0):
            raise ValueError(f"a static contact boundary must be a non-negative number of metres (0: physical contact "
                             f"only), got {static_contact_m}")
        # Training only (expert/static_contact.py): a step whose path passes this close to a static pillar's surface
        # ends the episode as a collision. 0 leaves physical contact, which the gate scores, as the only rule.
        self._static_contact_m = float(static_contact_m)
        self._static: StaticContactBoundary | None = None
        if not (math.isfinite(mover_contact_margin_m) and mover_contact_margin_m >= 0):
            raise ValueError(f"mover contact margin must be a non-negative number of metres, "
                             f"got {mover_contact_margin_m}")
        if mover_contact_margin_m and scene.dynamic is None:
            raise ValueError(f"scene {scene.id} has no moving pillars, so a mover contact margin cannot apply")
        # Training only: a mover contact ends the episode this much outside the task's rule (MoverController). The
        # expert observes, and the gate scores, the task's rule.
        self._mover_contact_margin_m = float(mover_contact_margin_m)
        if (mover_clearance is not None or mover_closing is not None) and scene.dynamic is None:
            raise ValueError(f"scene {scene.id} has no moving pillars, so a mover penalty cannot apply")
        if mover_clearance is not None and mover_closing is not None:
            raise ValueError("a run pays one kind of mover penalty: the clearance penalty or the closing penalty")
        # Training only (expert/mover_clearance.py): a run that asks for one pays it on every step, the gate never.
        self._mover_clearance = mover_clearance
        self._mover_closing = mover_closing
        self._closing_depths: list[float] | None = None  # each mover's depth inside the margin after the last step
        self._altitude_margin = altitude_margin  # expert/altitude_margin.py: any scene, training only
        if not (math.isfinite(footprint_contact_margin_m) and footprint_contact_margin_m >= 0):
            raise ValueError(f"a footprint contact margin must be a non-negative number of metres, "
                             f"got {footprint_contact_margin_m}")
        # The task's contact rule for mesh obstacles (expert/footprint_contact.py), with a training run's margin on top.
        # Which instances it covers is the registry's say (a measured model, never an engine primitive), decided once.
        self._footprint_obstacles = footprint_obstacles(layout.instances, registry if registry is not None else load_registry())
        self._footprint_tags = frozenset(inst.tag for inst in self._footprint_obstacles)
        if footprint_contact_margin_m and not self._footprint_obstacles:
            raise ValueError(f"scene {scene.id} has no mesh obstacle, so a footprint contact margin cannot apply")
        if self._footprint_obstacles and scene.dynamic is not None:
            raise ValueError(f"scene {scene.id} has moving pillars and mesh obstacles: moving a mesh would leave its "
                             f"<tag>_cards actor (the leaves, which have no collision) behind, and how the mover rule and "
                             f"the footprint rule share an obstacle is not designed")
        if static_contact_m and all(inst.tag in self._footprint_tags for inst in layout.instances):
            # The boundary covers pillars and boxes only: here it would end nothing, yet a run's identity would name it.
            raise ValueError(f"scene {scene.id} has no pillar or box (every obstacle is a mesh, under the footprint rule), "
                             f"so a static contact boundary cannot apply")
        self._footprint_contact_margin_m = float(footprint_contact_margin_m)
        self._footprint_fields: FootprintFields | None = None
        if self._footprint_obstacles and self._footprint_shape == "outline":
            # Read (or checked) once: an asset without a field, two horizontal scales or a field too short for the
            # rule's distance is refused here, before anything launches, never measured by its circle instead.
            self._footprint_fields = require_fields({inst.asset for inst in self._footprint_obstacles}, footprint_fields)
            FootprintContactRule(self._footprint_obstacles, training_margin_m=self._footprint_contact_margin_m,
                                 shape="outline", fields=self._footprint_fields)
        self._footprint: FootprintContactRule | None = None
        self._scene = scene
        self._layout = layout
        self._sim_factory = sim_factory
        self._map_path = map_path
        self._instance = instance
        # max_episode_steps is a separate knob from cfg.step_limit -- e.g. a shorter smoke-test episode
        # without hand-building a whole new RewardConfig just to change one field. Folding it into the
        # cfg actually used by evaluate() keeps classify()'s TIMEOUT check from disagreeing with it.
        self._cfg = dataclasses.replace(reward_config_for_scene(scene, cfg), step_limit=max_episode_steps)
        if altitude_margin is not None:
            altitude_margin.check_band(self._cfg.altitude_band_m)
        self._seed_base = seed_base
        self._max_episode_steps = max_episode_steps

        # One float32 depth frame for a static scene (every M2 checkpoint's space); a stack for a dynamic one.
        self._obs_config = obs_config if obs_config is not None else obs_config_for_scene(scene)
        self.observation_space = self._obs_config.space()
        self._stacker = DepthStacker(self._obs_config)
        self.action_space = action_space()

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
        # The pool objects (asset, material) this simulator session has spawned and held for: the first-use hold.
        self._warm: set[tuple[str | None, str | None]] = set()
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
            self._warm = set()  # and has loaded no pool object yet
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
        self._static = None
        self._footprint = None
        if self._displaced:
            # Out of the way of the reset's up-across-down sequence: the drone may have ended on a vacated home spot,
            # and sending a pillar home there would put it inside the drone.
            sim.set_object_poses(park_poses(self._instances, self._displaced))

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(self._seed_base + self._episode_index)
            self._episode_index += 1
        # Without a pool, M2's own call, argument for argument (the seed tests wrap sample_setup by that signature).
        pool_kwargs = {"pool": self._target_pool} if self._target_pool is not None else {}
        if self._target_split is not None:
            pool_kwargs["target_split"] = self._target_split
        setup = sample_setup(self._scene, self._layout, rng, **pool_kwargs)
        a0 = (options or {}).get("a0")
        if a0 == "sector8":  # dataset collection only: AutoFly's coarse guidance, applied before recording (M3)
            setup = a0_aligned(setup)
        elif a0 is not None:
            raise ValueError(f"unknown a0 option {a0!r} (supported: 'sector8')")

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
                                         max_speed_m_s=d.speed_m_s[1], dt=CONTROL_DT_S,
                                         termination_margin_m=self._mover_contact_margin_m)
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
        prewarm_steps = self._first_use_hold(sim, setup) if self._target_pool is not None else None
        obs = sim.observe()
        if obs.collided:
            # An episode that starts in contact is lost before its first action (C9): not a policy outcome.
            raise StartCollisionError(f"the episode's first observation already reports a collision at {obs.pose}")
        # termination_m: where a contact ends this env's episodes (the task's rule, or a training run's wider one). A
        # start inside it would lose its first step to an unavoidable collision, so it is retried like a start in
        # contact.
        if movers is not None and movers.nearest_gap((obs.pose.x, obs.pose.y)) <= movers.termination_m:
            raise StartCollisionError(f"the episode's first observation is within {movers.termination_m} m of a mover")
        # A training run's static boundary covers the solids (pillars, boxes); the mesh obstacles are the footprint
        # rule's, whatever the run asked for.
        static = (StaticContactBoundary.for_episode([i for i in self._layout.instances if i.tag not in self._footprint_tags],
                                                    new, self._static_contact_m)
                  if self._static_contact_m else None)
        if static is not None and static.nearest_gap((obs.pose.x, obs.pose.y)) <= static.boundary_m:
            raise StartCollisionError(f"the episode's first observation is within {static.boundary_m} m of a static "
                                      f"pillar")
        footprint = (FootprintContactRule.for_episode(self._footprint_obstacles, new,
                                                      training_margin_m=self._footprint_contact_margin_m,
                                                      shape=self._footprint_shape, fields=self._footprint_fields)
                     if self._footprint_obstacles else None)
        if footprint is not None and footprint.nearest_gap((obs.pose.x, obs.pose.y)) <= footprint.termination_m:
            raise StartCollisionError(f"the episode's first observation is within {footprint.termination_m} m of a mesh "
                                      f"obstacle's footprint {footprint.shape}")
        self._displaced = new
        self._movers = movers
        self._static = static
        self._footprint = footprint
        if movers is not None:
            movers.record_frame(obs.pose)
        self._closing_depths = None
        if self._mover_closing is not None and movers is not None:
            self._closing_depths = self._mover_closing.depths(movers.gaps((obs.pose.x, obs.pose.y)),
                                                              contact_m=movers.contact_m)

        self._prev_dist, bearing, _ = target_geometry(obs.pose, setup.target_xy_z)
        self._step_index = 0
        self._last_pose = obs.pose
        self._last_raw = obs

        info = self._info(Outcome.RUNNING, self._prev_dist, obs.pose, bearing, None)
        if movers is not None:
            info["mover_routes"] = [route.to_json() for route in setup.movers]
        if prewarm_steps is not None:  # with a pool only: every pool-free reset's info stays as it was
            info["prewarm_steps"] = prewarm_steps
        self._last_obs = {"depth": self._stacker.reset(encode_depth(obs.depth)),
                          "vector": encode_vector(obs.pose, obs.velocity_ned, obs.yaw_rate, setup.target_xy_z),
                          **self._mover_obs(obs.pose)}
        return self._last_obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if self._sim is None or self._setup is None:
            raise RuntimeError("AutoFlyEnv.step() called before reset()")
        if self._footprint_obstacles and self._footprint is None:
            # reset() stores the episode before its start checks, so after one of them refused the start a step would
            # fly without the footprint rule, and a body could reach wood unscored.
            raise RuntimeError("AutoFlyEnv.step(): reset() did not complete (a start check refused the episode); "
                               "reset() again before stepping")

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
        # The task's footprint rule for mesh obstacles, then a training run's static boundary, unless something else
        # already ended the step: a physical contact is scored as the contact it is.
        footprint = (self._footprint.contact((before.x, before.y), (obs.pose.x, obs.pose.y))
                     if self._footprint is not None and contact is None and not obs.collided else None)
        static = (self._static.contact((before.x, before.y), (obs.pose.x, obs.pose.y))
                  if self._static is not None and contact is None and not obs.collided and footprint is None else None)
        collided = obs.collided or contact is not None or static is not None or footprint is not None
        source = ("mover" if contact is not None else
                  "sim" if obs.collided else
                  self._footprint.source(footprint[1]) if footprint is not None else
                  "static_margin" if static is not None else None)
        seen = movers.seen_recently(contact[0]) if contact is not None else None  # the frames the policy flew on
        mover_contact = None
        if contact is not None:
            # How the contact happened (the 2026-10-03 s01d gate could not say): the surface gap, where the mover
            # stood relative to the heading the step began with, and whether it moved this step or yielded.
            index, gap = contact
            mx, my = movers.positions[index]
            mover_contact = {
                "gap_m": round(float(gap), 4),
                "bearing_deg": round(math.degrees(wrap_pi(math.atan2(my - before.y, mx - before.x) - before.yaw)), 2),
                "mover_moving": bool(movers.moving[index]),
                "route_kind": movers.routes[index].kind,
                "drone_forward_m_s": round(v_forward, 3),
            }
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
                          collision_source=source if result.outcome is Outcome.COLLISION else None, mover_in_view=seen,
                          mover_contact=mover_contact if result.outcome is Outcome.COLLISION else None)
        if footprint is not None:
            # Where the rule ended the episode (the key exists only on such a step: every pillar scene keeps its infos
            # exactly), and a log line per ending, so a run can be read for which trees end its flights and from where.
            # gap_m is the shape's (the outline's or the circle's); circle_gap_m the circle's either way, for diagnostics.
            tag, gap = footprint
            tree = self._instances[tag]
            shape = self._footprint.shape
            contact_info = info["footprint_contact"] = {
                "tag": tag,
                "asset": tree.asset,
                "gap_m": round(gap, 4),
                "rule": source,
                "bearing_deg": round(math.degrees(wrap_pi(math.atan2(tree.y - before.y, tree.x - before.x) - before.yaw)), 2),
                "drone_forward_m_s": round(v_forward, 3),
                "shape": shape,
                "circle_gap_m": round(self._footprint.circle_gap(tag, (before.x, before.y), (obs.pose.x, obs.pose.y)), 4),
            }
            print(f"FOOTPRINT instance {self._instance}: {tag} ({tree.asset}) at {contact_info['gap_m']:.2f} m from its "
                  f"{shape} ({source}), bearing {contact_info['bearing_deg']:+.0f} deg, forward "
                  f"{contact_info['drone_forward_m_s']:.2f} m/s", file=sys.stderr)
        elif source == "sim" and self._footprint is not None:
            # A physical contact in a mesh scene, which the rule exists to prevent. The nearest obstacle tells apart a
            # spawned target or distractor (placed 1.4 m or more from every circle: a larger gap) and wood the rule did
            # not stop (a gap inside the rule's line: under the circle, a centre inside the circle).
            a_xy, b_xy = (before.x, before.y), (obs.pose.x, obs.pose.y)
            nearest = self._footprint.nearest(a_xy, b_xy)
            if nearest is not None:
                tag, gap = nearest
                shape = self._footprint.shape
                # None: beyond every outline field's reach (3 m at unit scale), e.g. a spawned target far from trees
                info["footprint_nearest"] = {"tag": tag, "asset": self._instances[tag].asset,
                                             "gap_m": round(gap, 4) if math.isfinite(gap) else None, "shape": shape,
                                             "circle_gap_m": round(self._footprint.circle_gap(tag, a_xy, b_xy), 4)}
                # Training records keep counts only (the s02 smoke of 2026-10-08 logged "sim: 1" of 207 episodes and
                # could not say what the body hit), so the log keeps each one.
                where = f"{gap:.2f} m away" if math.isfinite(gap) else "beyond its field's reach"
                print(f"SIM-CONTACT instance {self._instance}: physical contact with the nearest {shape} {tag} "
                      f"({self._instances[tag].asset}) {where}, forward {v_forward:.2f} m/s", file=sys.stderr)
        if static is not None:
            # Where the boundary was crossed, as for a mover contact; the key exists only on such a step, so every run
            # without a boundary keeps its infos exactly.
            tag, gap = static
            pillar = self._instances[tag]
            contact_info = info["static_contact"] = {
                "tag": tag,
                "gap_m": round(gap, 4),
                "bearing_deg": round(math.degrees(wrap_pi(math.atan2(pillar.y - before.y, pillar.x - before.x)
                                                          - before.yaw)), 2),
                "drone_forward_m_s": round(v_forward, 3),
            }
            # Training records keep only counts, so the log keeps each ending: in view of the forward camera or not.
            print(f"STATIC-MARGIN instance {self._instance}: {tag} at {contact_info['gap_m']:.2f} m, bearing "
                  f"{contact_info['bearing_deg']:+.0f} deg, forward {contact_info['drone_forward_m_s']:.2f} m/s",
                  file=sys.stderr)
        reward = self._with_penalties(result.reward, obs.pose, info)
        self._last_obs = {"depth": self._stacker.push(encode_depth(obs.depth)),
                          "vector": encode_vector(obs.pose, obs.velocity_ned, obs.yaw_rate, self._setup.target_xy_z),
                          **self._mover_obs(obs.pose)}
        return self._last_obs, reward, result.terminated, result.truncated, info

    def _first_use_hold(self, sim: Simulator, setup: EpisodeSetup) -> int:
        """PREWARM_HOLD_STEPS more hover steps when this episode spawned a pool object (an entry's mesh and material: a
        primitive in a colour new to the session loads that material too) the session has not held for yet, else none.
        The objects count as warm only once the hold completed: one a fault cut short is held again next time."""
        objects = {(entry.asset, entry.material) for entry in (setup.target, *setup.distractor_entries)}
        if objects <= self._warm:
            return 0
        for _ in range(PREWARM_HOLD_STEPS):
            sim.command_velocity(0.0, 0.0, 0.0)
            sim.step(CONTROL_DT_S)
        self._warm |= objects
        return PREWARM_HOLD_STEPS

    def _with_penalties(self, reward: float, pose: Pose, info: dict) -> float:
        """`reward` less the run's training penalties for a step ending at `pose` (expert/mover_clearance.py,
        expert/altitude_margin.py), each named in `info`. Without any, both are left exactly as they were, so every
        earlier run's rewards and infos stand."""
        if self._movers is not None and (self._mover_clearance is not None or self._mover_closing is not None):
            gaps = self._movers.gaps((pose.x, pose.y))
            if self._mover_clearance is not None:
                penalty = self._mover_clearance(gaps, contact_m=self._movers.contact_m)
                info["mover_clearance_penalty"] = penalty
                reward -= penalty
            if self._mover_closing is not None:
                depths = self._mover_closing.depths(gaps, contact_m=self._movers.contact_m)
                penalty = self._mover_closing(self._closing_depths, depths) if self._closing_depths is not None else 0.0
                self._closing_depths = depths
                info["mover_closing_penalty"] = penalty
                reward -= penalty
        if self._altitude_margin is not None:
            penalty = self._altitude_margin(-pose.z, band_m=self._cfg.altitude_band_m)  # NED: altitude is -z
            info["altitude_margin_penalty"] = penalty
            reward -= penalty
        return reward

    def _mover_obs(self, pose: Pose) -> dict[str, np.ndarray]:
        """The `movers` key when the observation has one (a dynamic scene's default since 2026-10-03), else nothing."""
        slots = self._obs_config.mover_slots
        if not slots:
            return {}
        if self._movers is None:
            return {"movers": np.zeros(slots * MOVER_FEATURES, dtype=np.float32)}
        return {"movers": self._movers.observation(pose, slots)}

    @property
    def contact_rules(self) -> dict | None:
        """The contact rules this env applies beyond the simulator's own collision event (for run records): the
        footprint rule and this env's margin on a mesh scene; None where physical contact is the only rule."""
        if not self._footprint_obstacles:
            return None
        rule = FootprintContactRule(self._footprint_obstacles, training_margin_m=self._footprint_contact_margin_m,
                                    shape=self._footprint_shape, fields=self._footprint_fields)
        return {"footprint": rule.to_json()}

    @property
    def target_pool(self) -> TargetPool | None:
        """The target pool every episode draws from (plan 5 C5), or None: the orange cylinder."""
        return self._target_pool

    @property
    def target_split(self) -> str | None:
        """The split an evaluation draws its targets for instead of the scene's (plan 6 revision 1 item 5), or None."""
        return self._target_split

    @property
    def setup(self) -> EpisodeSetup | None:
        """The current episode's setup (M3's collector records it as provenance)."""
        return self._setup

    @property
    def layout(self) -> Layout:
        """The layout this env flies (the collector's first-frame check ray-casts its obstacles)."""
        return self._layout

    @property
    def spawned(self) -> tuple[str, ...]:
        """The current episode's spawned object names, target first (provenance, spec §10.2)."""
        return self._spawned

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
        reward = self._with_penalties(result.reward, pose, info)
        return self._last_obs, reward, result.terminated, result.truncated, info

    def _info(self, outcome: Outcome, dist_m: float, pose, bearing_rad: float, oob: str | None, *,
              collision_source: str | None = None, mover_in_view: bool | None = None,
              mover_contact: dict | None = None) -> dict:
        # pose/bearing_deg/oob_kind: where an episode ended and which bound it left, for the run record (the
        # 2026-09-17 gate could not say either). collision_source ("sim", "mover", "mover_inferred", "footprint", and in
        # training "static_margin" and "footprint_margin"), n_movers,
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
            "mover_contact": mover_contact,
        }

    def close(self) -> None:
        # Detach first, then close: a relaunch may abandon a hung close() and install a new simulator, and the late
        # return of this call must not detach that replacement.
        sim, self._sim = self._sim, None
        if sim is not None:
            sim.close()
