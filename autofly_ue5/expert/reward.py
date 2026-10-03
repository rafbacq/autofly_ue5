"""Expert reward and episode termination (spec §8). Pure functions: no simulator, no randomness."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum

# Recorded with every training session and gate run. A replay buffer or checkpoint trained under one version must not
# be resumed or compared under another (train.prepare_run_root refuses the resume).
# 2: no progress credit inside the success radius (C8). 3 (2026-10-03): a scene may set its own coefficients
# (reward_config_for_scene); s01d sets r_bounds = r_collision. Static scenes' rewards are unchanged.
REWARD_VERSION = "3-scene-reward-overrides"


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
    altitude_band_m: tuple[float, float] = (1.0, 3.0)   # hard operating bounds; leaving them ends the episode


def reward_config_for_scene(scene, base: RewardConfig = RewardConfig()) -> RewardConfig:
    """`base` with the coefficients the scene file sets for itself (its "reward" block, validated against the
    schema). s01d sets r_bounds = r_collision: at 5 against 10, s01d_r1's late policy learned to dive out of the
    altitude band whenever a collision looked likely (final.zip: 58 of its 62 out-of-bounds gate episodes, 2026-10-03)."""
    return dataclasses.replace(base, **dict(scene.reward))


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
    if not in_bounds or not (low <= altitude_m <= high):
        return Outcome.OUT_OF_BOUNDS
    if dist_m <= cfg.success_radius_m and abs(bearing_rad) <= math.radians(cfg.success_yaw_deg):
        return Outcome.SUCCESS
    if step_index >= cfg.step_limit:
        return Outcome.TIMEOUT
    return Outcome.RUNNING


def oob_kind(*, in_bounds: bool, altitude_m: float, cfg: RewardConfig) -> str | None:
    """Which bound an OUT_OF_BOUNDS step left, for the run record: "lateral" (the scene's x/y bounds, checked first),
    "altitude_low" or "altitude_high"; None while inside every bound."""
    low, high = cfg.altitude_band_m
    if not in_bounds:
        return "lateral"
    if altitude_m < low:
        return "altitude_low"
    if altitude_m > high:
        return "altitude_high"
    return None


def step_reward(prev_dist_m: float, dist_m: float, bearing_rad: float,
                outcome: Outcome, cfg: RewardConfig) -> float:
    # Progress stops counting at the success radius (2026-09-24 review, C8). Paid all the way in, closing from 5 m to
    # 2 m misaligned and turning at the end earned more than succeeding aligned at 5 m, and the trained policy learned
    # exactly that (final.zip's successes ended a median 2.35 m out; 7 of its 16 real gate failures left the bounds
    # within 5 m of a target that sits 0-3 m from the edge). Inside the radius only the alignment bonus and the time
    # penalty remain, so turning onto the target at once is the best move. Still potential-based, so it cannot be
    # farmed by oscillating across the radius.
    r_s = cfg.success_radius_m
    r = cfg.k_p * (max(prev_dist_m, r_s) - max(dist_m, r_s)) - cfg.k_t
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
