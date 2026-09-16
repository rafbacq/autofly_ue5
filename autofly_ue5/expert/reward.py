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
    altitude_band_m: tuple[float, float] = (1.0, 3.0)   # hard operating bounds; leaving them ends the episode


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
