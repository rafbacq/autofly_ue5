"""Safety-margin training against static pillars (2026-10-06, run 6): a training run may end an episode as a collision
when the drone's path passes within a boundary of a static pillar's surface. Run 5's ten static gate collisions ended
0.52-0.63 m from a pillar's surface, and its successful flights passed pillars at 0.5-0.9 m: the expert flew to the
boundary it trained on, which against a static pillar was physical contact. The mover margin (test_mover_contact_margin)
moved that boundary for movers and cut run 5's gate mover collisions 35 -> 7. The expert observes, and the gate scores,
physical contact only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from autofly_ue5.sim.fake import FakeSimulator
from tests.test_dynamic_env import dynamic_scene, make_dynamic_env, scene_objects


# --------------------------------------------------------------------------------------------------------
# The boundary itself (pure).
# --------------------------------------------------------------------------------------------------------
def _boundary(boundary_m=1.1, pillars=(("p0", 0.0, 0.0, 0.5), ("p1", 10.0, 0.0, 0.4))):
    from autofly_ue5.expert.static_contact import StaticContactBoundary

    return StaticContactBoundary(pillars, boundary_m)


@pytest.mark.parametrize("gap, fires", [(1.3, False), (1.1001, False), (1.1, True), (0.8, True), (0.2, True)])
def test_a_point_fires_within_the_boundary_of_a_pillars_surface(gap, fires):
    point = (-0.5 - gap, 0.0)
    hit = _boundary().contact(point, point)
    assert (hit is not None) == fires
    if fires:
        assert hit[0] == "p0" and hit[1] == pytest.approx(gap)


def test_a_step_that_sweeps_past_a_pillar_fires_even_with_both_ends_outside():
    # 0.4 m a step at full speed: the swept segment is what passed by, as for movers (motion.first_contact).
    a, b = (-3.0, 1.4), (3.0, 1.4)  # 0.9 m from p0's surface at the closest point, 3 m away at both ends
    hit = _boundary().contact(a, b)
    assert hit is not None and hit[0] == "p0" and hit[1] == pytest.approx(0.9)
    assert _boundary().contact((-3.0, 1.7), (3.0, 1.7)) is None  # 1.2 m at the closest point


def test_the_nearest_pillar_is_the_one_named():
    hit = _boundary(pillars=(("far", 0.0, 0.0, 0.5), ("near", 2.6, 0.0, 0.5))).contact((1.3, 0.0), (1.3, 0.0))
    assert hit is not None and hit[1] == pytest.approx(0.8)  # both are 0.8 m away; either may be named
    hit = _boundary(pillars=(("far", 0.0, 0.0, 0.5), ("near", 2.4, 0.0, 0.5))).contact((1.3, 0.0), (1.3, 0.0))
    assert hit == ("near", pytest.approx(0.6))


def test_nearest_gap_measures_from_the_surface():
    assert _boundary().nearest_gap((-2.0, 0.0)) == pytest.approx(1.5)
    assert _boundary().nearest_gap((10.0, 3.0)) == pytest.approx(2.6)


def test_an_episodes_movers_are_not_static():
    from autofly_ue5.expert.static_contact import StaticContactBoundary
    from tests.test_expert_episode import scene_and_layout

    _scene, layout = scene_and_layout()
    movers = {inst.tag for inst in layout.instances[:10]}
    boundary = StaticContactBoundary.for_episode(layout.instances, movers, 1.1)
    assert set(boundary.tags) == {inst.tag for inst in layout.instances} - movers
    home = layout.instances[0]  # a mover's home is vacated once it moves; every other pillar is >= 3.4 m away
    assert boundary.contact((home.x, home.y), (home.x, home.y)) is None
    everything = StaticContactBoundary.for_episode(layout.instances, set(), 1.1)
    assert everything.contact((home.x, home.y), (home.x, home.y))[0] == home.tag


@pytest.mark.parametrize("value", [0.0, -0.1, math.nan, math.inf])
def test_a_boundary_that_could_not_mean_anything_is_refused(value):
    with pytest.raises(ValueError, match="boundary"):
        _boundary(boundary_m=value)


# --------------------------------------------------------------------------------------------------------
# The environment.
# --------------------------------------------------------------------------------------------------------
def _nearest_static_pillar(env, pose):
    movers = {route.tag for route in env.setup.movers}
    return min((inst for inst in env._layout.instances if inst.tag not in movers),
               key=lambda inst: math.hypot(inst.x - pose.x, inst.y - pose.y))


def _fly_at_a_static_pillar(env, seed):
    """Turn to the nearest static pillar and fly straight at it; returns each step's (reward, info) and observation."""
    obs, _info = env.reset(seed=seed)
    target = _nearest_static_pillar(env, env._last_pose)
    steps, observations = [], [obs]
    for _ in range(300):
        pose = env._last_pose
        bearing = math.atan2(target.y - pose.y, target.x - pose.x) - pose.yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        action = np.array([2.0 if abs(bearing) < 0.2 else 0.0, float(np.clip(2.0 * bearing, -1, 1)), 0.0], np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append((reward, info))
        observations.append(obs)
        if terminated or truncated:
            return steps, observations
    raise AssertionError("never ended")


SEEDS = range(1_000_020, 1_000_030)


def test_flying_at_a_static_pillar_ends_at_the_boundary_and_sees_the_same_on_the_way():
    plain, bounded = make_dynamic_env(), make_dynamic_env(static_contact_m=1.1)
    boundary_hits = 0
    for seed in SEEDS:
        (a, obs_a), (b, obs_b) = _fly_at_a_static_pillar(plain, seed), _fly_at_a_static_pillar(bounded, seed)
        end_a, end_b = a[-1][1], b[-1][1]
        assert len(b) <= len(a), "the boundary can only end an episode sooner"
        for (r_a, i_a), (r_b, i_b) in zip(a[:len(b) - 1], b[:len(b) - 1]):
            assert (r_a, i_a) == (r_b, i_b), "every step before the boundary's is untouched"
        for o_a, o_b in zip(obs_a[:len(b)], obs_b[:len(b)]):  # what the expert saw on the way: the same flight
            assert all(np.array_equal(o_a[k], o_b[k]) for k in o_a)
        if end_b["collision_source"] != "static_margin":
            assert a == b, "a flight the boundary never ended is the plain flight, step for step"
            continue
        boundary_hits += 1
        contact = end_b["static_contact"]
        assert end_b["outcome"] == "collision" and 0.6 < contact["gap_m"] <= 1.1 + 1e-9
        assert contact["tag"] == _nearest_static_pillar(bounded, bounded._last_pose).tag
        assert -180.0 <= contact["bearing_deg"] <= 180.0
        assert b[-1][0] <= -10.0 + 0.5, "it is scored as the collision it stands for (r_collision 10)"
        assert end_a["outcome"] == "collision" and end_a["collision_source"] == "sim", \
            "without the boundary the same flight hits the pillar"
        assert "static_contact" not in end_a
    assert boundary_hits >= 5, "flying straight at a pillar must end at the boundary"


def test_each_step_checks_the_path_it_flew_not_only_where_it_ended(monkeypatch):
    # Review of 3a0c3b7: the env must pass the swept segment (a full-speed step covers 0.4 m), not the end point alone.
    from autofly_ue5.expert.static_contact import StaticContactBoundary

    real = StaticContactBoundary.contact
    for seed in SEEDS:
        calls = []

        def spy(self, a_xy, b_xy):
            calls.append((tuple(a_xy), tuple(b_xy)))
            return real(self, a_xy, b_xy)

        monkeypatch.setattr(StaticContactBoundary, "contact", spy)
        steps, _ = _fly_at_a_static_pillar(make_dynamic_env(static_contact_m=1.1), seed)
        if steps[-1][1]["collision_source"] != "static_margin":
            continue  # a mover or the bounds ended it: the boundary was not consulted on its last step
        ends = [tuple(info["pose"][:2]) for _reward, info in steps]
        assert len(calls) == len(steps)
        for (_a, b), end in zip(calls, ends):
            assert b == pytest.approx(end), "each call ends where its step ended"
        for (a, _b), previous_end in zip(calls[1:], ends[:-1]):
            assert a == pytest.approx(previous_end), "and starts where the step before it ended"
        assert sum(math.dist(a, b) > 0.3 for a, b in calls) >= 3, "full-speed steps are checked as segments"
        return
    raise AssertionError("no flight ended at the boundary")


def test_a_mover_contact_takes_precedence_and_carries_no_static_detail(monkeypatch):
    # The boundary is consulted only when nothing else ended the step: a stray static_contact on a mover step would
    # misreport what ended it.
    from autofly_ue5.expert.movers import MoverController
    from autofly_ue5.expert.static_contact import StaticContactBoundary
    from tests.test_mover_contact_margin import _fly_recording

    hit = {"now": False}
    real_mover = MoverController.contact

    def mover_spy(self, before_xy, after_xy):
        result = real_mover(self, before_xy, after_xy)
        hit["now"] = result is not None
        return result

    monkeypatch.setattr(MoverController, "contact", mover_spy)
    monkeypatch.setattr(StaticContactBoundary, "contact",
                        lambda self, a, b: (self.tags[0], 0.5) if hit["now"] else None)  # fires only with a mover
    ended_on_mover = 0
    for seed in SEEDS:
        steps, _ = _fly_recording(make_dynamic_env(static_contact_m=1.1), seed)
        end = steps[-1][1]
        if end["collision_source"] == "mover":
            ended_on_mover += 1
            assert "static_contact" not in end
    assert ended_on_mover > 0


def test_a_physical_contact_takes_precedence_and_carries_no_static_detail(monkeypatch):
    from autofly_ue5.expert.static_contact import StaticContactBoundary

    from tests.test_dynamic_env import ROTOR  # where the fake's pillars collide

    real = StaticContactBoundary.contact

    def physical_only(self, a, b):  # the boundary fires only where the simulator also reports contact
        result = real(self, a, b)
        return result if result is not None and result[1] <= ROTOR else None

    monkeypatch.setattr(StaticContactBoundary, "contact", physical_only)
    sim_ends = 0
    for seed in SEEDS:
        steps, _ = _fly_at_a_static_pillar(make_dynamic_env(static_contact_m=1.1), seed)
        end = steps[-1][1]
        if end["collision_source"] == "sim":
            sim_ends += 1
            assert "static_contact" not in end
    assert sim_ends > 0


def test_each_boundary_ending_is_logged_with_where_it_happened(capsys):
    # Training records keep counts only; the log keeps each ending's pillar, gap and bearing (in view of the forward
    # camera or not), so a run can be read for where its static contacts happen.
    env = make_dynamic_env(static_contact_m=1.1)
    for seed in SEEDS:
        steps, _ = _fly_at_a_static_pillar(env, seed)
        if steps[-1][1]["collision_source"] == "static_margin":
            contact = steps[-1][1]["static_contact"]
            line = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("STATIC-MARGIN")]
            assert line == [f"STATIC-MARGIN instance 0: {contact['tag']} at {contact['gap_m']:.2f} m, bearing "
                            f"{contact['bearing_deg']:+.0f} deg, forward {contact['drone_forward_m_s']:.2f} m/s"]
            return
    raise AssertionError("no flight ended at the boundary")


def test_without_a_boundary_nothing_changes():
    a, _ = _fly_at_a_static_pillar(make_dynamic_env(), 1_000_021)
    b, _ = _fly_at_a_static_pillar(make_dynamic_env(static_contact_m=0.0), 1_000_021)
    assert a == b
    assert all("static_contact" not in info for _reward, info in b)


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf])
def test_the_env_refuses_a_boundary_that_could_not_mean_anything(value):
    with pytest.raises(ValueError, match="boundary"):
        make_dynamic_env(static_contact_m=value)


def test_each_episode_counts_only_the_pillars_that_stay_home():
    env = make_dynamic_env(static_contact_m=1.1)
    for seed in SEEDS:
        env.reset(seed=seed)
        movers = {route.tag for route in env.setup.movers}
        assert movers and set(env._static.tags) == {inst.tag for inst in env._layout.instances} - movers


def test_a_static_scene_takes_it_too():
    # Every pillar of a static scene stays home. (Static s01 itself is unchanged without it: test_static_s01_golden.)
    from autofly_ue5.expert.env import AutoFlyEnv
    from tests.test_expert_episode import scene_and_layout

    scene, layout = scene_and_layout()
    env = AutoFlyEnv(scene, layout, FakeSimulator, map_path="/Game/AutoFly/Maps/S01", instance=0, static_contact_m=1.1)
    env.reset(seed=3)
    assert set(env._static.tags) == {inst.tag for inst in layout.instances}


def test_a_start_inside_the_boundary_is_retried_not_charged(monkeypatch):
    # As for the mover margin (f8dbce8): a start inside the boundary would lose its first step to an unavoidable -10.
    # Starts keep 1.4 m from every pillar (episode.spawn_clearance_m), so a boundary under that never meets one.
    from autofly_ue5.expert.static_contact import StaticContactBoundary
    from autofly_ue5.sim.types import StartCollisionError

    monkeypatch.setattr(StaticContactBoundary, "nearest_gap", lambda self, xy: 1.05)
    with pytest.raises(StartCollisionError, match="static pillar"):
        make_dynamic_env(static_contact_m=1.1).reset(seed=1_000_070)
    make_dynamic_env().reset(seed=1_000_070)  # without the boundary, 1.05 m from a pillar is a valid start


def test_the_vec_env_hands_it_to_every_worker(tmp_path):
    import functools

    from autofly_ue5.expert.vec import make_vec_env

    scene, layout = dynamic_scene()
    vec = make_vec_env(scene, layout, 1, map_path="/Game/AutoFly/Maps/S01", monitor_dir=tmp_path / "m",
                       sim_factory=functools.partial(FakeSimulator, scene_objects=scene_objects(layout)),
                       sim_root=tmp_path / "sim", static_contact_m=1.1)
    try:
        assert vec.envs[0].unwrapped._static_contact_m == 1.1
    finally:
        vec.close()


def test_the_outcome_callback_logs_boundary_collisions_on_their_own_curve():
    from types import SimpleNamespace

    from stable_baselines3.common.logger import Logger

    from autofly_ue5.expert.train import OutcomeHistogramCallback

    logger = Logger(folder=None, output_formats=[])
    cb = OutcomeHistogramCallback()
    cb.model = SimpleNamespace(logger=logger)
    for info in ({"outcome": "success"}, {"outcome": "collision", "collision_source": "static_margin"}):
        cb.update_locals({"infos": [info], "dones": [True]})
        cb._on_step()
    values = logger.name_to_value
    assert values["outcomes/collision_static_margin"] == 0.5 and values["outcomes/collision"] == 0.5
    assert values["outcomes/collision_sim"] == 0.0 and cb.collision_sources == {"static_margin": 1}


# --------------------------------------------------------------------------------------------------------
# The run identity and the trainer.
# --------------------------------------------------------------------------------------------------------
def test_the_run_identity_names_it_only_when_a_run_uses_it():
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.train import run_identity
    from autofly_ue5.scenes.resolve import resolve_scene

    resolved, obs = resolve_scene("s01d"), ObsConfig(3, "float16", mover_slots=4)
    plain = run_identity(resolved, obs)
    assert "static_contact_m" not in plain and run_identity(resolved, obs, static_contact_m=0.0) == plain
    assert run_identity(resolved, obs, static_contact_m=1.1) == {**plain, "static_contact_m": 1.1}


def _train(tmp_path, monkeypatch, *flags, scene="s01d"):
    from autofly_ue5.expert import train
    from tests.test_dynamic_plumbing import _dynamic_fake_factory, _no_launch

    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: _dynamic_fake_factory())
    seen = []
    real = train.make_vec_env

    def spy(*args, **kwargs):
        seen.append((kwargs.get("static_contact_m"), kwargs.get("mover_contact_margin_m")))
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "make_vec_env", spy)
    out = tmp_path / "train.json"
    code = train.main(["--scene", scene, *flags, "--run-root", str(tmp_path / "run"), "--out", str(out),
                       "--device", "cpu", "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "10",
                       "--learning-starts", "100", "--buffer-size", "100", "--eval-freq", "1000"])
    return code, out, seen


def test_a_training_run_takes_it_from_the_command_line_and_records_it(tmp_path, monkeypatch):
    code, out, seen = _train(tmp_path, monkeypatch, "--static-contact-m", "1.1")
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["identity"]["static_contact_m"] == 1.1
    assert record["config"]["static_contact_m"] == 1.1
    assert seen == [(1.1, 0.0), (0.0, 0.0)], "training ends episodes at the boundary; evaluation keeps contact"


@pytest.mark.parametrize("value, why", [("-0.2", "non-negative"), ("nan", "non-negative"), ("1.2", "1.1 m"),
                                        ("1.4", "1.1 m"), ("2.0", "1.1 m")])
def test_a_boundary_that_cannot_apply_is_refused_before_the_run_root_is_claimed(tmp_path, monkeypatch, capsys,
                                                                                value, why):
    code, _out, seen = _train(tmp_path, monkeypatch, "--static-contact-m", value)
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


def test_run_6_s_exact_combination_trains_and_records_every_term(tmp_path, monkeypatch):
    # docs/runbook-m2d.md 6e: run 5's command with a 0.5 m mover margin, the static boundary and a V-shaped altitude cost.
    from autofly_ue5.expert.altitude_margin import AltitudeMarginPenalty
    from autofly_ue5.expert.mover_clearance import MoverClearancePenalty

    code, out, seen = _train(tmp_path, monkeypatch, "--mover-clearance-penalty", "0.5", "1.0",
                             "--mover-contact-margin", "0.5", "--static-contact-m", "1.1",
                             "--altitude-margin-penalty", "0.2", "1.0")
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    identity = record["identity"]
    assert identity["mover_clearance_penalty"] == MoverClearancePenalty(0.5, 1.0).to_json()
    assert identity["altitude_margin_penalty"] == AltitudeMarginPenalty(0.2, 1.0).to_json()
    assert identity["mover_contact_margin_m"] == 0.5 and identity["static_contact_m"] == 1.1
    assert seen == [(1.1, 0.5), (0.0, 0.0)]
