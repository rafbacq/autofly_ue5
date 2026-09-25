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


# ------------------------------------------------------------------------------------------------------
# C8 (2026-09-24 review): no progress credit inside the success radius. With it, closing in misaligned and
# turning at the end out-earned an aligned success at 5 m -- final.zip's successes ended a median 2.35 m from the
# target (best_model's 4.89 m), and 7 of its 16 real failures left the bounds within 5 m of the target.
# ------------------------------------------------------------------------------------------------------
def test_the_reward_version_names_the_clamp():
    from autofly_ue5.expert.reward import REWARD_VERSION

    assert REWARD_VERSION == "2-no-progress-inside-success-radius"


def test_moving_inside_the_success_radius_earns_no_progress():
    from autofly_ue5.expert.reward import Outcome, step_reward

    c = cfg()
    inside = step_reward(4.0, 3.0, math.pi / 2, Outcome.RUNNING, c)  # misaligned, 1 m closer, well inside 5 m
    assert inside == pytest.approx(-c.k_t + c.k_h * math.cos(math.pi / 2))


def test_crossing_into_the_radius_credits_only_the_part_outside_it():
    from autofly_ue5.expert.reward import Outcome, step_reward

    c = cfg()
    assert step_reward(6.0, 4.0, 0.0, Outcome.RUNNING, c) == pytest.approx(1.0 - c.k_t + c.k_h)
    assert step_reward(6.0, 4.0, 0.0, Outcome.SUCCESS, c) == pytest.approx(11.09)  # still > r_success
    assert step_reward(4.0, 6.0, 0.0, Outcome.RUNNING, c) == pytest.approx(-1.0 - c.k_t + c.k_h)  # leaving costs it back


def test_an_aligned_success_at_5_m_is_worth_at_least_a_misaligned_dive_to_2_m():
    # Discounted (gamma 0.99) returns from 5.4 m out at 0.4 m/step: succeed now, or keep a 20-degree offset,
    # close to 2 m at 0.4*cos(20 deg) per step, then spend two steps turning onto the target.
    from autofly_ue5.expert.reward import Outcome, step_reward

    c, gamma = cfg(), 0.99
    aligned = step_reward(5.4, 5.0, 0.0, Outcome.SUCCESS, c)
    offset, dist, rewards = math.radians(20), 5.4, []
    while dist > 2.0:
        nxt = max(dist - 0.4 * math.cos(offset), 2.0)
        rewards.append(step_reward(dist, nxt, offset, Outcome.RUNNING, c))
        dist = nxt
    rewards.append(step_reward(dist, dist, offset / 2, Outcome.RUNNING, c))
    rewards.append(step_reward(dist, dist, 0.0, Outcome.SUCCESS, c))
    dive = sum(r * gamma**i for i, r in enumerate(rewards))
    assert aligned >= dive


@pytest.mark.parametrize("in_bounds, altitude, expected", [
    (True, 2.0, None), (False, 2.0, "lateral"), (True, 0.5, "altitude_low"), (True, 3.5, "altitude_high"),
    (False, 0.5, "lateral"),
])
def test_oob_kind_names_which_bound_was_left(in_bounds, altitude, expected):
    from autofly_ue5.expert.reward import oob_kind

    assert oob_kind(in_bounds=in_bounds, altitude_m=altitude, cfg=cfg()) == expected
