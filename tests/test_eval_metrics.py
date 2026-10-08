"""The evaluation harness's metrics (spec §3.1, §12 M6; plan 6 item 6): success, collision, out-of-bounds and timeout
rates with Wilson intervals, and the paper's path efficiency PER = L_opt / max(L, L_opt) over successful trials,
reported as an interval because L_opt is not defined by the paper: the straight line to the success disc bounds it from
below, the shortest path on the scene's inflated grid from above (MEMORY.md, "Scene learnings")."""

from __future__ import annotations

import math

import pytest

from autofly_ue5.scenes.model import Bounds, Instance, Layout

BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)


def _pillar(tag: str, x: float, y: float, radius: float = 0.5) -> Instance:
    return Instance(tag=tag, asset="cylinder", x=x, y=y, z_center=-5.0, yaw=0.0, scale=(2 * radius, 2 * radius, 10.0),
                    material="white", radius_m=radius, height_m=10.0)


def _wall(x: float = 0.0, y_from: float = -20.0, y_to: float = 20.0, step: float = 1.0) -> tuple[Instance, ...]:
    """Pillars touching along x = `x`: no 1.4 m-inflated gap between y_from and y_to."""
    n = int(round((y_to - y_from) / step)) + 1
    return tuple(_pillar(f"wall_{i}", x, y_from + i * step) for i in range(n))


# --------------------------------------------------------------------------------------------------------
# Wilson intervals.
# --------------------------------------------------------------------------------------------------------
def test_the_wilson_interval_reproduces_the_projects_own_reported_interval():
    # CLAUDE.md, M2d: the accepted expert flew 370/400 = 0.925 (CI 0.895-0.947).
    from autofly_ue5.evaluation.metrics import wilson_interval

    lo, hi = wilson_interval(370, 400)
    assert round(lo, 3) == 0.895 and round(hi, 3) == 0.947


def test_the_wilson_interval_stays_inside_zero_and_one_at_the_extremes():
    from autofly_ue5.evaluation.metrics import wilson_interval

    z2 = 1.959963984540054 ** 2
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and hi == pytest.approx(z2 / (20 + z2))
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and lo == pytest.approx(20 / (20 + z2))
    assert wilson_interval(0, 0) is None, "no episodes, no interval"
    for bad in ((-1, 5), (6, 5), (1, -1)):
        with pytest.raises(ValueError):
            wilson_interval(*bad)


# --------------------------------------------------------------------------------------------------------
# Path length and L_opt.
# --------------------------------------------------------------------------------------------------------
def test_the_flown_path_length_is_the_sum_of_horizontal_step_lengths():
    from autofly_ue5.evaluation.metrics import horizontal_path_length_m

    assert horizontal_path_length_m([(0.0, 0.0), (3.0, 4.0), (3.0, 4.0), (6.0, 8.0)]) == pytest.approx(10.0)
    assert horizontal_path_length_m([(1.0, 2.0)]) == 0.0
    assert horizontal_path_length_m([]) == 0.0


def test_l_opt_is_bounded_by_the_straight_line_to_the_success_disc_and_the_grid_path():
    from autofly_ue5.evaluation.metrics import L_OPT_INFLATE_M, L_OPT_RESOLUTION_M, l_opt_bounds

    open_field = Layout(scene_id="t", seed=0, bounds=BOUNDS, instances=())
    bounds = l_opt_bounds(open_field, (-30.0, 0.0), (30.0, 0.0), success_radius_m=5.0)
    assert bounds["straight_m"] == pytest.approx(55.0)
    assert 55.0 <= bounds["grid_m"] <= 55.0 + 2 * L_OPT_RESOLUTION_M, "an open field's grid path is the straight line"
    assert (bounds["inflate_m"], bounds["resolution_m"]) == (L_OPT_INFLATE_M, L_OPT_RESOLUTION_M) == (1.4, 0.5)

    walled = Layout(scene_id="t", seed=0, bounds=BOUNDS, instances=_wall())
    around = l_opt_bounds(walled, (-30.0, 0.0), (30.0, 0.0), success_radius_m=5.0)
    assert around["straight_m"] == pytest.approx(55.0), "the lower bound ignores obstacles"
    assert around["grid_m"] > 60.0, "the grid path goes round the wall"


def test_l_opt_can_leave_out_obstacles_that_were_not_where_the_layout_puts_them():
    # A dynamic scene's movers leave their homes: the grid is drawn without them (and says how many it left out).
    from autofly_ue5.evaluation.metrics import l_opt_bounds

    walled = Layout(scene_id="t", seed=0, bounds=BOUNDS, instances=_wall())
    tags = {inst.tag for inst in walled.instances}
    bounds = l_opt_bounds(walled, (-30.0, 0.0), (30.0, 0.0), success_radius_m=5.0, exclude_tags=tags)
    assert bounds["grid_m"] <= 56.0 and bounds["excluded_obstacles"] == len(tags)


def test_an_unreachable_success_disc_has_no_grid_optimum():
    from autofly_ue5.evaluation.metrics import l_opt_bounds

    ring = tuple(_pillar(f"ring_{i}", 30.0 + 7.0 * math.cos(a), 7.0 * math.sin(a))
                 for i, a in enumerate(i * 2 * math.pi / 60 for i in range(60)))
    boxed_in = Layout(scene_id="t", seed=0, bounds=BOUNDS, instances=ring)
    bounds = l_opt_bounds(boxed_in, (-30.0, 0.0), (30.0, 0.0), success_radius_m=1.0)
    assert bounds["grid_m"] is None and bounds["straight_m"] == pytest.approx(59.0)


def test_per_is_l_opt_over_the_longer_of_the_flight_and_l_opt_at_both_ends_of_the_interval():
    from autofly_ue5.evaluation.metrics import per_bounds

    assert per_bounds(60.0, {"straight_m": 55.0, "grid_m": 58.0}) == {"lower": pytest.approx(55 / 60),
                                                                     "upper": pytest.approx(58 / 60)}
    assert per_bounds(50.0, {"straight_m": 55.0, "grid_m": 58.0}) == {"lower": 1.0, "upper": 1.0}, \
        "a flight shorter than L_opt is perfectly efficient, never above 1"
    assert per_bounds(60.0, {"straight_m": 55.0, "grid_m": None}) == {"lower": pytest.approx(55 / 60), "upper": None}
    assert per_bounds(0.0, {"straight_m": 0.0, "grid_m": 0.0}) == {"lower": None, "upper": None}


# --------------------------------------------------------------------------------------------------------
# Summaries.
# --------------------------------------------------------------------------------------------------------
def _episode(outcome: str, *, lower: float | None = None, upper: float | None = None, steps: int = 100,
             length: float = 60.0, source: str | None = None, act_s: float = 0.01) -> dict:
    per = {"lower": lower, "upper": upper} if outcome == "success" or lower is not None else {"lower": None, "upper": None}
    return {"outcome": outcome, "steps": steps, "path_length_m": length, "per": per,
            "collision_source": source if outcome == "collision" else None,
            "policy_act_s": {"mean": act_s, "max": act_s}}


def test_a_summary_holds_each_rate_with_its_wilson_interval_and_the_rates_sum_to_one():
    from autofly_ue5.evaluation.metrics import summarise, wilson_interval

    episodes = ([_episode("success", lower=0.9, upper=0.95)] * 7 + [_episode("collision", source="sim")] * 2
                + [_episode("out_of_bounds")])
    summary = summarise(episodes)
    assert summary["episodes"] == 10
    assert summary["outcomes"] == {"success": 7, "collision": 2, "out_of_bounds": 1, "timeout": 0}
    assert summary["success_rate"] == {"value": 0.7, "wilson95": list(wilson_interval(7, 10))}
    assert summary["collision_rate"]["value"] == 0.2 and summary["out_of_bounds_rate"]["value"] == 0.1
    assert summary["timeout_rate"] == {"value": 0.0, "wilson95": list(wilson_interval(0, 10))}
    assert sum(summary[k]["value"] for k in ("success_rate", "collision_rate", "out_of_bounds_rate", "timeout_rate")) \
        == pytest.approx(1.0)
    assert summary["collision_sources"] == {"sim": 2}


def test_per_is_averaged_over_successes_only_and_over_the_same_episodes_at_both_ends():
    from autofly_ue5.evaluation.metrics import summarise

    episodes = [_episode("success", lower=0.8, upper=0.9), _episode("success", lower=0.6, upper=0.7),
                _episode("success", lower=0.5, upper=None),  # no grid optimum: left out of both ends
                _episode("collision", lower=0.2, upper=0.3, source="footprint")]  # never a success's PER
    per = summarise(episodes)["per"]
    assert per["successes"] == 3 and per["episodes"] == 2 and per["without_grid_optimum"] == 1
    assert per["lower"] == pytest.approx(0.7) and per["upper"] == pytest.approx(0.8)


def test_an_empty_summary_reports_zero_episodes_and_no_rates():
    from autofly_ue5.evaluation.metrics import summarise

    summary = summarise([])
    assert summary["episodes"] == 0 and summary["success_rate"] == {"value": None, "wilson95": None}
    assert summary["per"]["lower"] is None and summary["mean_steps"] is None


def test_an_outcome_outside_the_four_is_counted_and_still_in_the_denominator():
    # AutoFlyEnv's own step limit ends every episode; an outer guard firing instead must not vanish from the rates.
    from autofly_ue5.evaluation.metrics import summarise

    summary = summarise([_episode("success", lower=1.0, upper=1.0), _episode("exceeded_max_steps")])
    assert summary["outcomes"]["exceeded_max_steps"] == 1 and summary["success_rate"]["value"] == 0.5
