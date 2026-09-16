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
