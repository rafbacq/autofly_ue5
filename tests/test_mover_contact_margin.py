"""Safety-margin training against moving pillars (2026-10-05): a training run may end an episode as a mover collision
`margin` outside the task's contact rule (1.0 m from a mover's surface). Every s01d expert's mover collisions were
near misses clustered just inside the rule (medians 0.93-0.98 m), so the experts learned to fly to the boundary they
trained on. Training on a wider one moves where they fly; the gate keeps the task's rule. What the expert observes is
unchanged, so nothing differs between training and evaluation but where an episode ends."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from autofly_ue5.sim.fake import FakeSimulator
from tests.test_dynamic_env import make_dynamic_env


def _controller(margin=0.0):
    from autofly_ue5.expert.movers import MoverController
    from autofly_ue5.scenes.motion import MoverRoute

    from tests.test_dynamic_env import dynamic_scene

    scene, layout = dynamic_scene()
    import autofly_ue5.expert.episode as episode

    setup = episode.sample_setup(scene, layout, np.random.default_rng(1_000_070))
    assert setup.movers and isinstance(setup.movers[0], MoverRoute)
    d = scene.dynamic
    return MoverController(setup.movers, contact_m=d.contact_m, yield_margin_m=d.yield_margin_m,
                           max_speed_m_s=d.speed_m_s[1], dt=0.2, termination_margin_m=margin)


def test_the_margin_moves_where_a_contact_fires_and_nothing_else_the_controller_does():
    from autofly_ue5.sim.types import Pose

    plain, wide = _controller(), _controller(0.3)
    (x, y), r = plain.positions[0], plain.routes[0].footprint.radius_m
    for gap, fires_plain, fires_wide in ((1.5, False, False), (1.2, False, True), (0.9, True, True)):
        point = (x - r - gap, y)
        assert (plain.contact(point, point) is not None) == fires_plain, gap
        assert (wide.contact(point, point) is not None) == fires_wide, gap
    pose = Pose(x - r - 1.2, y, -2.0, 0.0)
    assert np.array_equal(plain.observation(pose, 4), wide.observation(pose, 4)), "the expert sees the task's rule"
    assert plain.yield_distance_m == wide.yield_distance_m, "movers yield where they always did"
    assert wide.inference_radius_m == pytest.approx(plain.inference_radius_m + 0.3)


def _fly_recording(env, seed):
    steps = []
    obs, info = env.reset(seed=seed)
    observations = [obs]
    for _ in range(300):
        pose = env._last_pose
        xy = min(info["movers"], key=lambda m: math.dist(m, (pose.x, pose.y)))
        bearing = math.atan2(xy[1] - pose.y, xy[0] - pose.x) - pose.yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        action = np.array([2.0 if abs(bearing) < 0.2 else 0.0, float(np.clip(2.0 * bearing, -1, 1)), 0.0], np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append((reward, info))
        observations.append(obs)
        if terminated or truncated:
            return steps, observations
    raise AssertionError("never ended")


def test_flying_at_a_mover_ends_sooner_and_farther_out_with_the_margin_and_sees_the_same_on_the_way():
    plain, wide = make_dynamic_env(), make_dynamic_env(mover_contact_margin_m=0.3)
    margin_hits = 0
    for seed in range(1_000_020, 1_000_030):
        (a, obs_a), (b, obs_b) = _fly_recording(plain, seed), _fly_recording(wide, seed)
        end_a, end_b = a[-1][1], b[-1][1]
        if end_b["collision_source"] != "mover":
            continue
        assert len(b) <= len(a), "the wider rule can only end an episode sooner"
        assert end_b["mover_contact"]["gap_m"] <= 1.3 + 1e-9
        margin_hits += end_b["mover_contact"]["gap_m"] > 1.0
        for o_a, o_b in zip(obs_a[:len(b)], obs_b[:len(b)]):  # the same flight up to the wider rule's end
            assert all(np.array_equal(o_a[k], o_b[k]) for k in o_a)
        if end_a["collision_source"] == "mover":
            assert end_a["mover_contact"]["gap_m"] <= 1.0 + 1e-9, "the task's own rule is untouched"
    assert margin_hits > 0, "some flights must end in the margin, between 1.0 and 1.3 m"


def test_without_a_margin_nothing_changes():
    a, _ = _fly_recording(make_dynamic_env(), 1_000_021)
    b, _ = _fly_recording(make_dynamic_env(mover_contact_margin_m=0.0), 1_000_021)
    assert [(r, i) for r, i in a] == [(r, i) for r, i in b]


@pytest.mark.parametrize("margin", [-0.1, math.nan, math.inf])
def test_a_margin_that_could_not_mean_anything_is_refused(margin):
    with pytest.raises(ValueError, match="margin"):
        make_dynamic_env(mover_contact_margin_m=margin)


def test_a_static_scene_refuses_a_margin():
    from autofly_ue5.expert.env import AutoFlyEnv
    from tests.test_expert_episode import scene_and_layout

    scene, layout = scene_and_layout()
    with pytest.raises(ValueError, match="moving"):
        AutoFlyEnv(scene, layout, FakeSimulator, map_path="/Game/AutoFly/Maps/S01", instance=0,
                   mover_contact_margin_m=0.3)


def test_the_vec_env_hands_it_to_every_worker(tmp_path):
    import functools

    from autofly_ue5.expert.vec import make_vec_env
    from tests.test_dynamic_env import dynamic_scene, scene_objects

    scene, layout = dynamic_scene()
    vec = make_vec_env(scene, layout, 1, map_path="/Game/AutoFly/Maps/S01", monitor_dir=tmp_path / "m",
                       sim_factory=functools.partial(FakeSimulator, scene_objects=scene_objects(layout)),
                       sim_root=tmp_path / "sim", mover_contact_margin_m=0.3)
    try:
        assert vec.envs[0].unwrapped._mover_contact_margin_m == 0.3
    finally:
        vec.close()


# --------------------------------------------------------------------------------------------------------
# The run identity and the trainer.
# --------------------------------------------------------------------------------------------------------
def test_the_run_identity_names_it_only_when_a_run_uses_it():
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.train import run_identity
    from autofly_ue5.scenes.resolve import resolve_scene

    resolved, obs = resolve_scene("s01d"), ObsConfig(3, "float16", mover_slots=4)
    plain = run_identity(resolved, obs)
    assert "mover_contact_margin_m" not in plain and run_identity(resolved, obs, mover_contact_margin_m=0.0) == plain
    assert run_identity(resolved, obs, mover_contact_margin_m=0.3) == {**plain, "mover_contact_margin_m": 0.3}


def _train(tmp_path, monkeypatch, *flags, scene="s01d"):
    from autofly_ue5.expert import train
    from tests.test_dynamic_plumbing import _dynamic_fake_factory, _no_launch

    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: _dynamic_fake_factory())
    seen = []
    real = train.make_vec_env

    def spy(*args, **kwargs):
        seen.append(kwargs.get("mover_contact_margin_m"))
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "make_vec_env", spy)
    out = tmp_path / "train.json"
    code = train.main(["--scene", scene, *flags, "--run-root", str(tmp_path / "run"), "--out", str(out),
                       "--device", "cpu", "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "10",
                       "--learning-starts", "100", "--buffer-size", "100", "--eval-freq", "1000"])
    return code, out, seen


def test_a_training_run_takes_it_from_the_command_line_and_records_it(tmp_path, monkeypatch):
    code, out, seen = _train(tmp_path, monkeypatch, "--mover-contact-margin", "0.3")
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["identity"]["mover_contact_margin_m"] == 0.3
    assert record["config"]["mover_contact_margin_m"] == 0.3
    assert seen == [0.3, 0.0], "training ends episodes at the wider rule; evaluation keeps the task's"


@pytest.mark.parametrize("scene, value, why", [("s01", "0.3", "moving"), ("s01d", "-0.2", "margin")])
def test_a_margin_that_cannot_apply_is_refused_before_the_run_root_is_claimed(tmp_path, monkeypatch, capsys, scene,
                                                                              value, why):
    code, _out, seen = _train(tmp_path, monkeypatch, "--mover-contact-margin", value, scene=scene)
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


def test_run_5_s_exact_combination_trains_and_records_all_three(tmp_path, monkeypatch):
    # docs/runbook-m2d.md 6d: run 4's per-step clearance penalty plus both margins, in one command.
    from autofly_ue5.expert.altitude_margin import AltitudeMarginPenalty
    from autofly_ue5.expert.mover_clearance import MoverClearancePenalty

    code, out, seen = _train(tmp_path, monkeypatch, "--mover-clearance-penalty", "0.5", "1.0",
                             "--mover-contact-margin", "0.3", "--altitude-margin-penalty", "0.1", "0.5")
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    identity = record["identity"]
    assert identity["mover_clearance_penalty"] == MoverClearancePenalty(0.5, 1.0).to_json()
    assert identity["altitude_margin_penalty"] == AltitudeMarginPenalty(0.1, 0.5).to_json()
    assert identity["mover_contact_margin_m"] == 0.3
    assert seen == [0.3, 0.0]


def test_a_start_inside_the_training_boundary_is_retried_not_charged(monkeypatch):
    # Review of 7fcb8a8 (2026-10-05): the start check used the task's 1.0 m, so in training a start 1.0-1.3 m from a
    # mover would lose its first step to an unavoidable -10. s01d keeps starts 6 m from every mover, so it cannot happen
    # there; but the start check belongs with where a contact ends the episode.
    from autofly_ue5.expert.movers import MoverController
    from autofly_ue5.sim.types import StartCollisionError

    monkeypatch.setattr(MoverController, "nearest_gap", lambda self, xy: 1.15)
    with pytest.raises(StartCollisionError):
        make_dynamic_env(mover_contact_margin_m=0.3).reset(seed=1_000_070)
    make_dynamic_env().reset(seed=1_000_070)  # outside the task's 1.0 m: a valid start
