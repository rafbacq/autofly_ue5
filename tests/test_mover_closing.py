"""The training-only mover closing penalty (2026-10-05): a charge for every centimetre the drone closes on a moving
pillar inside a margin outside its contact boundary, and nothing for time spent there. A pass costs k times its depth,
whatever its speed, and braking or hovering beside a mover is free. Off unless a run asks for it, then recorded in the
run identity."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from autofly_ue5.expert.mover_clearance import MoverClearancePenalty, MoverClosingPenalty
from autofly_ue5.sim.types import Pose
from tests.test_dynamic_env import ZERO, make_dynamic_env

CONTACT = 1.0


@pytest.mark.parametrize("gaps, expected", [
    ([], []),
    ([5.0, 2.0], [0.0, 0.0]),   # outside the margin, and exactly on it
    ([1.5], [0.5]),             # half way in
    ([1.0, 0.4], [1.0, 1.0]),   # on the contact boundary, and inside it (capped)
])
def test_each_mover_s_depth_inside_the_margin(gaps, expected):
    assert MoverClosingPenalty(k=3.0, margin_m=1.0).depths(gaps, contact_m=CONTACT) == pytest.approx(expected)


def test_only_closing_in_costs_and_a_pass_costs_k_times_its_depth():
    p = MoverClosingPenalty(k=3.0, margin_m=1.0)
    assert p([0.0, 0.2], [0.5, 0.1]) == pytest.approx(1.5)   # closing on the first, backing off the second
    assert p([0.5], [0.5]) == 0.0                            # standing still beside it
    assert p([0.8], [0.3]) == 0.0                            # backing off
    # A pass: in to 0.3 m of clearance (depth 0.7) and out again, at any number of steps.
    for clearances in ([2.0, 0.6, 0.3, 0.6, 2.0], [2.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.5, 0.7, 2.0]):
        depths = [p.depths([1.0 + c], contact_m=CONTACT) for c in clearances]
        assert sum(p(a, b) for a, b in zip(depths, depths[1:])) == pytest.approx(3.0 * 0.7)


@pytest.mark.parametrize("k, margin", [(0.0, 1.0), (-1.0, 1.0), (3.0, 0.0), (math.nan, 1.0), (3.0, math.inf)])
def test_a_closing_penalty_that_could_not_mean_anything_is_refused(k, margin):
    with pytest.raises(ValueError):
        MoverClosingPenalty(k=k, margin_m=margin)


def test_it_serialises_for_the_run_identity():
    assert MoverClosingPenalty(k=3.0, margin_m=1.0).to_json() == {"k": 3.0, "margin_m": 1.0}


def _fly_recording(env, seed):
    steps = []
    _obs, info = env.reset(seed=seed)
    for _ in range(300):
        pose = env._last_pose
        xy = min(info["movers"], key=lambda m: math.dist(m, (pose.x, pose.y)))
        bearing = math.atan2(xy[1] - pose.y, xy[0] - pose.x) - pose.yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        action = np.array([2.0 if abs(bearing) < 0.2 else 0.0, float(np.clip(2.0 * bearing, -1, 1)), 0.0], np.float32)
        _obs, reward, terminated, truncated, info = env.step(action)
        steps.append((reward, info))
        if terminated or truncated:
            return steps
    raise AssertionError("never ended")


def test_the_penalty_is_subtracted_from_each_step_s_reward_and_changes_nothing_else():
    plain, closing = make_dynamic_env(), make_dynamic_env(mover_closing=MoverClosingPenalty(k=3.0, margin_m=1.0))
    charged = hits = 0
    for seed in range(1_000_020, 1_000_026):
        a, b = _fly_recording(plain, seed), _fly_recording(closing, seed)
        assert len(a) == len(b), "the penalty must not change the flight"
        for (r_plain, info_plain), (r_closing, info_closing) in zip(a, b):
            paid = info_closing["mover_closing_penalty"]
            assert paid >= 0.0 and r_closing == pytest.approx(r_plain - paid)
            assert {k: v for k, v in info_closing.items() if k != "mover_closing_penalty"} == info_plain
            charged += paid > 0
        if b[-1][1]["collision_source"] == "mover":
            # It flew in from outside the margin to the contact boundary: about k in all (more if two movers were near).
            assert sum(info["mover_closing_penalty"] for _r, info in b) >= 3.0 * 0.9
            hits += 1
    assert charged > 0 and hits > 0


def _hover_beside_a_mover(env, steps=5):
    """Put the drone 1.5 m from a mover's surface (inside the margin, inside the yield distance), then hover."""
    from autofly_ue5.expert.obs import target_geometry

    env.reset(seed=1_000_070)
    route = env._setup.movers[0]
    x, y = route.position(env._movers.taus[0])
    env._sim._pose = env._last_pose = Pose(x - (route.footprint.radius_m + 1.5), y, env._last_pose.z, 0.0)
    env._prev_dist = target_geometry(env._last_pose, env._setup.target_xy_z)[0]
    return [env.step(ZERO) for _ in range(steps)]


def test_hovering_beside_a_stopped_mover_is_free_after_the_approach():
    results = _hover_beside_a_mover(make_dynamic_env(mover_closing=MoverClosingPenalty(k=3.0, margin_m=1.0)))
    paid = [info["mover_closing_penalty"] for *_rest, info in results]
    assert paid[0] == pytest.approx(3.0 * 0.5, abs=0.05), "the teleport in is a closing of depth 0.5"
    assert paid[1:] == pytest.approx([0.0] * (len(paid) - 1), abs=1e-9), "waiting beside it costs nothing"


def test_the_per_step_penalty_by_contrast_charges_every_step_of_the_wait():
    results = _hover_beside_a_mover(make_dynamic_env(mover_clearance=MoverClearancePenalty(k=0.5, margin_m=1.0)))
    assert all(info["mover_clearance_penalty"] == pytest.approx(0.25, abs=0.05) for *_rest, info in results)


def test_a_static_scene_refuses_it():
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.sim.fake import FakeSimulator
    from tests.test_expert_episode import scene_and_layout

    scene, layout = scene_and_layout()
    with pytest.raises(ValueError, match="moving"):
        AutoFlyEnv(scene, layout, FakeSimulator, map_path="/Game/AutoFly/Maps/S01", instance=0,
                   mover_closing=MoverClosingPenalty(k=3.0, margin_m=1.0))


def test_one_run_pays_one_kind_of_mover_penalty():
    with pytest.raises(ValueError, match="one"):
        make_dynamic_env(mover_clearance=MoverClearancePenalty(k=0.5, margin_m=1.0),
                         mover_closing=MoverClosingPenalty(k=3.0, margin_m=1.0))


def test_the_vec_env_hands_it_to_every_worker(tmp_path):
    import functools

    from autofly_ue5.expert.vec import make_vec_env
    from autofly_ue5.sim.fake import FakeSimulator
    from tests.test_dynamic_env import dynamic_scene, scene_objects

    scene, layout = dynamic_scene()
    penalty = MoverClosingPenalty(k=3.0, margin_m=1.0)
    vec = make_vec_env(scene, layout, 1, map_path="/Game/AutoFly/Maps/S01", monitor_dir=tmp_path / "m",
                       sim_factory=functools.partial(FakeSimulator, scene_objects=scene_objects(layout)),
                       sim_root=tmp_path / "sim", mover_closing=penalty)
    try:
        assert vec.envs[0].unwrapped._mover_closing == penalty
    finally:
        vec.close()


# --------------------------------------------------------------------------------------------------------
# The run identity, the trainer and the log.
# --------------------------------------------------------------------------------------------------------
def test_the_run_identity_names_it_only_when_a_run_uses_it():
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.train import run_identity
    from autofly_ue5.scenes.resolve import resolve_scene

    resolved, obs = resolve_scene("s01d"), ObsConfig(3, "float16", mover_slots=4)
    plain = run_identity(resolved, obs)
    assert "mover_closing_penalty" not in plain
    closing = run_identity(resolved, obs, mover_closing=MoverClosingPenalty(k=3.0, margin_m=1.0))
    assert closing == {**plain, "mover_closing_penalty": {"k": 3.0, "margin_m": 1.0}}


def _train(tmp_path, monkeypatch, *flags, scene="s01d"):
    from autofly_ue5.expert import train
    from tests.test_dynamic_plumbing import _dynamic_fake_factory, _no_launch

    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: _dynamic_fake_factory())
    seen = []
    real = train.make_vec_env

    def spy(*args, **kwargs):
        seen.append(kwargs.get("mover_closing"))
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "make_vec_env", spy)
    out = tmp_path / "train.json"
    code = train.main(["--scene", scene, *flags, "--run-root", str(tmp_path / "run"), "--out", str(out),
                       "--device", "cpu", "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "10",
                       "--learning-starts", "100", "--buffer-size", "100", "--eval-freq", "1000"])
    return code, out, seen


def test_a_training_run_takes_it_from_the_command_line_and_records_it(tmp_path, monkeypatch):
    code, out, seen = _train(tmp_path, monkeypatch, "--mover-closing-penalty", "3", "1.0")
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    expected = {"k": 3.0, "margin_m": 1.0}
    assert record["identity"]["mover_closing_penalty"] == expected
    assert record["config"]["mover_closing_penalty"] == expected
    assert "mover_clearance_penalty" not in record["identity"]
    assert seen == [MoverClosingPenalty(k=3.0, margin_m=1.0), None], "training pays it; evaluation does not"


@pytest.mark.parametrize("scene, flags, why", [
    ("s01", ["--mover-closing-penalty", "3", "1.0"], "moving"),
    ("s01d", ["--mover-closing-penalty", "0", "1.0"], "k"),
    ("s01d", ["--mover-closing-penalty", "3", "1.0", "--mover-clearance-penalty", "0.5", "1.0"], "one"),
])
def test_a_closing_penalty_that_cannot_apply_is_refused_before_the_run_root_is_claimed(tmp_path, monkeypatch, capsys,
                                                                                       scene, flags, why):
    code, _out, seen = _train(tmp_path, monkeypatch, *flags, scene=scene)
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


def test_the_outcome_log_carries_the_closing_penalty_each_episode_paid():
    from types import SimpleNamespace

    from autofly_ue5.expert.train import OutcomeHistogramCallback
    from tests.test_mover_clearance import _Logger

    cb = OutcomeHistogramCallback()
    cb.model = SimpleNamespace(logger=_Logger())
    collision = {"outcome": "collision", "collision_source": "mover", "mover_closing_penalty": 2.0}
    for infos, dones in (([{"outcome": "running", "mover_closing_penalty": 1.0}], [False]), ([collision], [True])):
        cb.locals = {"infos": infos, "dones": dones}
        cb._on_step()
    assert cb.model.logger.records["outcomes/mover_closing_penalty"] == pytest.approx(3.0)
    assert "outcomes/mover_clearance_penalty" not in cb.model.logger.records
