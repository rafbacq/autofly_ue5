import pytest


def test_projection_arithmetic():
    from scripts.measure_instances import project_cost

    # 30 env steps/s total, 400k steps needed per scene, 10 scenes.
    p = project_cost(steps_per_s_total=30.0, steps_needed=400_000, n_scenes=10)
    assert p["hours_per_scene"] == pytest.approx(400_000 / 30.0 / 3600.0, rel=1e-6)
    assert p["hours_all_scenes"] == pytest.approx(p["hours_per_scene"] * 10, rel=1e-6)
    assert p["days_all_scenes"] == pytest.approx(p["hours_all_scenes"] / 24.0, rel=1e-6)


def test_projection_refuses_nonsense():
    from scripts.measure_instances import project_cost

    with pytest.raises(ValueError):
        project_cost(steps_per_s_total=0.0, steps_needed=1, n_scenes=1)


def test_projection_refuses_negative_inputs():
    from scripts.measure_instances import project_cost

    with pytest.raises(ValueError):
        project_cost(steps_per_s_total=30.0, steps_needed=-1, n_scenes=1)
    with pytest.raises(ValueError):
        project_cost(steps_per_s_total=30.0, steps_needed=1, n_scenes=0)


def test_should_stop_climbing():
    from scripts.measure_instances import should_stop

    assert should_stop(vram_total_mib=20_500, prev_total=30.0, this_total=35.0)[0] is True
    assert should_stop(vram_total_mib=8_000, prev_total=30.0, this_total=28.0)[0] is True
    assert should_stop(vram_total_mib=8_000, prev_total=30.0, this_total=35.0)[0] is False


def test_should_stop_reason_names_the_trigger():
    from scripts.measure_instances import should_stop

    over_budget, reason = should_stop(vram_total_mib=20_500, prev_total=30.0, this_total=35.0)
    assert over_budget is True
    assert "vram" in reason.lower() or "budget" in reason.lower()

    fell, reason = should_stop(vram_total_mib=8_000, prev_total=30.0, this_total=28.0)
    assert fell is True
    assert "throughput" in reason.lower() or "fell" in reason.lower()


def test_should_stop_with_no_previous_measurement_only_checks_vram():
    from scripts.measure_instances import should_stop

    assert should_stop(vram_total_mib=8_000, prev_total=None, this_total=5.0)[0] is False
    assert should_stop(vram_total_mib=20_500, prev_total=None, this_total=5.0)[0] is True


def test_should_stop_boundary_is_not_over_at_exactly_the_budget():
    from scripts.measure_instances import should_stop

    assert should_stop(vram_total_mib=20_000, prev_total=None, this_total=5.0)[0] is False


def test_choose_best_n_picks_highest_throughput_within_budget():
    from scripts.measure_instances import choose_best_n

    per_n = {
        "1": {"env_steps_per_s_total": 7.0, "vram_mib": 2_000},
        "2": {"env_steps_per_s_total": 11.0, "vram_mib": 4_000},
        "4": {"env_steps_per_s_total": 9.0, "vram_mib": 20_500},  # over budget: disqualified
    }
    assert choose_best_n(per_n) == 2


def test_choose_best_n_with_empty_input():
    from scripts.measure_instances import choose_best_n

    assert choose_best_n({}) is None
