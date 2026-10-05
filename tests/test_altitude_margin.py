"""The training-only altitude margin penalty (2026-10-05): a per-step cost inside a margin below the altitude band's
ceiling and above its floor, zero in the middle of the band. Nothing in the task's reward cares where the drone is
inside the band until it leaves it, and every s01d expert lost episodes to slow climbs and dives out of it. Off unless a
run asks for it, then recorded in the run identity."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from autofly_ue5.expert.altitude_margin import AltitudeMarginPenalty
from autofly_ue5.expert.mover_clearance import MoverClosingPenalty
from autofly_ue5.sim.fake import FakeSimulator
from tests.test_dynamic_env import make_dynamic_env

BAND = (1.0, 3.0)
UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)


@pytest.mark.parametrize("altitude, expected", [
    (2.0, 0.0), (1.5, 0.0), (2.5, 0.0),     # the middle of the band, and the margin's inner edges
    (1.25, 0.1), (2.75, 0.1),               # half way into the margin
    (1.0, 0.2), (3.0, 0.2),                 # on the band's edges: the full k
    (0.8, 0.2), (3.3, 0.2),                 # outside it (the step that ends the episode): capped
])
def test_the_penalty_ramps_from_zero_at_the_margin_to_k_at_the_band_s_edge(altitude, expected):
    assert AltitudeMarginPenalty(k=0.2, margin_m=0.5)(altitude, band_m=BAND) == pytest.approx(expected)


@pytest.mark.parametrize("k, margin", [(0.0, 0.5), (-0.1, 0.5), (0.2, 0.0), (math.nan, 0.5), (0.2, math.inf),
                                       (0.2, 1.01)])
def test_a_penalty_that_could_not_mean_anything_is_refused(k, margin):
    # A margin over half the 2 m band would leave no altitude free of it.
    with pytest.raises(ValueError):
        AltitudeMarginPenalty(k=k, margin_m=margin).check_band(BAND)


def test_it_serialises_for_the_run_identity():
    assert AltitudeMarginPenalty(k=0.2, margin_m=0.5).to_json() == {"k": 0.2, "margin_m": 0.5}


def _climb(env, seed=1_000_070):
    env.reset(seed=seed)
    steps = []
    for _ in range(300):
        _obs, reward, terminated, truncated, info = env.step(UP)
        steps.append((reward, info))
        if terminated or truncated:
            return steps
    raise AssertionError("never left the band")


def test_climbing_out_pays_more_the_closer_it_gets_and_changes_nothing_else():
    plain, shaped = make_dynamic_env(), make_dynamic_env(altitude_margin=AltitudeMarginPenalty(k=0.2, margin_m=0.5))
    a, b = _climb(plain), _climb(shaped)
    assert len(a) == len(b)
    paid = []
    for (r_plain, info_plain), (r_shaped, info_shaped) in zip(a, b):
        p = info_shaped["altitude_margin_penalty"]
        assert r_shaped == pytest.approx(r_plain - p)
        assert {k: v for k, v in info_shaped.items() if k != "altitude_margin_penalty"} == info_plain
        paid.append(p)
    assert b[-1][1]["oob_kind"] == "altitude_high"
    assert paid == sorted(paid), "the higher it climbs, the more each step costs"
    assert paid[-1] == pytest.approx(0.2) and paid[0] < 0.2


def test_a_static_scene_pays_it_too():
    from autofly_ue5.expert.env import AutoFlyEnv
    from tests.test_expert_episode import scene_and_layout

    scene, layout = scene_and_layout()
    env = AutoFlyEnv(scene, layout, FakeSimulator, map_path="/Game/AutoFly/Maps/S01", instance=0,
                     altitude_margin=AltitudeMarginPenalty(k=0.2, margin_m=0.5))
    steps = _climb(env)
    assert steps[-1][1]["altitude_margin_penalty"] == pytest.approx(0.2)


def test_it_combines_with_the_mover_closing_penalty():
    both = make_dynamic_env(mover_closing=MoverClosingPenalty(k=3.0, margin_m=1.0),
                            altitude_margin=AltitudeMarginPenalty(k=0.2, margin_m=0.5))
    plain = make_dynamic_env()
    for (r_plain, _ip), (r_both, info) in zip(_climb(plain), _climb(both)):
        assert r_both == pytest.approx(r_plain - info["mover_closing_penalty"] - info["altitude_margin_penalty"])


def test_without_it_the_info_is_what_it_was():
    for _reward, info in _climb(make_dynamic_env()):
        assert "altitude_margin_penalty" not in info


# --------------------------------------------------------------------------------------------------------
# The run identity, the trainer and the log.
# --------------------------------------------------------------------------------------------------------
def test_the_run_identity_names_it_only_when_a_run_uses_it():
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.train import run_identity
    from autofly_ue5.scenes.resolve import resolve_scene

    resolved, obs = resolve_scene("s01d"), ObsConfig(3, "float16", mover_slots=4)
    plain = run_identity(resolved, obs)
    assert "altitude_margin_penalty" not in plain
    shaped = run_identity(resolved, obs, altitude_margin=AltitudeMarginPenalty(k=0.2, margin_m=0.5))
    assert shaped == {**plain, "altitude_margin_penalty": {"k": 0.2, "margin_m": 0.5}}


def _train(tmp_path, monkeypatch, *flags, scene="s01d"):
    from autofly_ue5.expert import train
    from tests.test_dynamic_plumbing import _dynamic_fake_factory, _no_launch

    _no_launch(monkeypatch)
    if scene == "s01d":
        monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: _dynamic_fake_factory())
    else:
        monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: FakeSimulator)
    seen = []
    real = train.make_vec_env

    def spy(*args, **kwargs):
        seen.append((kwargs.get("altitude_margin"), kwargs.get("mover_closing")))
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "make_vec_env", spy)
    out = tmp_path / "train.json"
    code = train.main(["--scene", scene, *flags, "--run-root", str(tmp_path / "run"), "--out", str(out),
                       "--device", "cpu", "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "10",
                       "--learning-starts", "100", "--buffer-size", "100", "--eval-freq", "1000"])
    return code, out, seen


def test_a_training_run_takes_it_with_the_closing_penalty_and_records_both(tmp_path, monkeypatch):
    code, out, seen = _train(tmp_path, monkeypatch, "--altitude-margin-penalty", "0.2", "0.5",
                             "--mover-closing-penalty", "3", "1.0")
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["identity"]["altitude_margin_penalty"] == {"k": 0.2, "margin_m": 0.5}
    assert record["identity"]["mover_closing_penalty"] == {"k": 3.0, "margin_m": 1.0}
    assert record["config"]["altitude_margin_penalty"] == {"k": 0.2, "margin_m": 0.5}
    assert seen == [(AltitudeMarginPenalty(k=0.2, margin_m=0.5), MoverClosingPenalty(k=3.0, margin_m=1.0)),
                    (None, None)], "training pays them; evaluation does not"


def test_a_static_scene_may_train_with_it(tmp_path, monkeypatch):
    code, out, _seen = _train(tmp_path, monkeypatch, "--altitude-margin-penalty", "0.2", "0.5", scene="s01")
    record = json.loads(out.read_text())
    assert code == 0 and record["identity"]["altitude_margin_penalty"] == {"k": 0.2, "margin_m": 0.5}


@pytest.mark.parametrize("values, why", [(["0", "0.5"], "k"), (["0.2", "1.5"], "margin")])
def test_a_penalty_that_cannot_apply_is_refused_before_the_run_root_is_claimed(tmp_path, monkeypatch, capsys,
                                                                               values, why):
    code, _out, seen = _train(tmp_path, monkeypatch, "--altitude-margin-penalty", *values)
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


def test_the_outcome_log_carries_each_penalty_an_episode_paid():
    from types import SimpleNamespace

    from autofly_ue5.expert.train import OutcomeHistogramCallback
    from tests.test_mover_clearance import _Logger

    cb = OutcomeHistogramCallback()
    cb.model = SimpleNamespace(logger=_Logger())
    steps = (([{"outcome": "running", "mover_closing_penalty": 1.0, "altitude_margin_penalty": 0.1}], [False]),
             ([{"outcome": "success", "mover_closing_penalty": 0.0, "altitude_margin_penalty": 0.2}], [True]))
    for infos, dones in steps:
        cb.locals = {"infos": infos, "dones": dones}
        cb._on_step()
    assert cb.model.logger.records["outcomes/mover_closing_penalty"] == pytest.approx(1.0)
    assert cb.model.logger.records["outcomes/altitude_margin_penalty"] == pytest.approx(0.3)
