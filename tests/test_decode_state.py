"""scripts/decode_state.py: recovers time order from shuffled rows, tests the state[9] hypotheses by the spec's rule,
and finds where the turn-in-place actions are. On a synthetic episode with known fields, and on the two real episodes
when this host has them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.decode_state import REAL_EPISODES, decode, load_episode, shared_order


def _synthetic(n: int, rng: np.random.Generator, order: np.ndarray) -> dict:
    """A curving, climbing flight toward a target 7 m short of its end, rows written in `order`. state[2] is altitude
    minus 1.5 m, state[5] the true yaw rate; state[1] and state[4] are noise."""
    t = np.arange(n) * 0.2
    yaw = 0.4 * np.sin(t / 3.0)
    x = np.cumsum(0.38 * np.cos(yaw))
    y = np.cumsum(0.38 * np.sin(yaw))
    z = 2.0 + 0.4 * np.sin(t / 4.0)
    tx, ty = x[-1] + 7.0, y[-1]
    d = np.hypot(tx - x, ty - y)
    states = np.stack([d, rng.normal(size=n), z - 1.5, np.full(n, 1.9), rng.normal(size=n),
                       np.gradient(np.unwrap(yaw), 0.2), x, y, z], 1)
    actions = np.stack([np.full(n, 1.95), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)], 1)
    actions[n - 5] = [0.05, 0.95, 0.0]  # a turn in place near the end
    return {"states": states[order], "actions": actions[order], "instruction": "x", "source": "synthetic", "split": "s"}


def test_a_synthetic_episode_is_reordered_and_decoded_by_the_rule():
    rng = np.random.default_rng(0)
    perm_long = rng.permutation(60)
    perm_short = np.array([t - (t > 7) for t in perm_long if t != 7 and t - (t > 7) < 50])
    report = decode({"train": _synthetic(60, rng, perm_long), "val": _synthetic(50, rng, perm_short)})
    assert report["adopted"]["state[2]"] and "altitude" in report["adopted"]["state[2]"]
    assert report["adopted"]["state[1]"] == [] and report["adopted"]["state[4]"] == []
    assert "yaw_rate" in report["adopted"]["state[5]"]  # (lateral acceleration ties with it at constant speed)
    assert report["order"]["shared_permutation"] == {"dropped_step_of_longer": 7, "file_rows_on_the_same_time_step": 50,
                                                     "of": 50}
    train = report["episodes"]["train"]
    assert train["turn_in_place_time_steps"] == [55]
    assert train["median_step_m"]["time_order"] == pytest.approx(0.38, abs=0.01)
    assert train["state2_vs_altitude_fit"]["intercept"] == pytest.approx(-1.5, abs=1e-6)
    assert math.dist(train["target_xy"], train["end_state"][6:8]) == pytest.approx(7.0, abs=1e-3)


def test_shared_order_needs_no_drop_for_identical_orders():
    order = np.array([3, 0, 2, 1])
    assert shared_order(order, order) == {"dropped_step_of_longer": None, "file_rows_on_the_same_time_step": 4, "of": 4}


@pytest.mark.skipif(not (REAL_EPISODES / "train.jsonl").is_file(), reason="the real AutoFly episodes are not on this host")
def test_the_real_episodes_decode_as_recorded_in_plan_4():
    report = decode({split: load_episode(REAL_EPISODES / f"{split}.jsonl") for split in ("train", "val")})
    assert set(report["adopted"]["state[2]"]) == {"altitude", "height_above_start"}
    assert all(not report["adopted"][f"state[{k}]"] for k in (1, 4, 5))
    assert report["order"]["shared_permutation"]["file_rows_on_the_same_time_step"] == 70
    train, val = report["episodes"]["train"], report["episodes"]["val"]
    assert train["file_row_0"]["time_step"] == 75 and val["file_row_0"]["time_step"] == 74
    assert train["first_actions_in_time_order"][0][0] > 1.9 and val["first_actions_in_time_order"][0][0] > 1.9
