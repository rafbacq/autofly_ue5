"""The training-only mover-clearance penalty (2026-10-04): a graded cost inside a margin outside a moving pillar's
contact boundary, so the expert learns to keep the extra clearance a mover needs. Off unless a run asks for it, and
then recorded in the run identity; static scenes and every earlier run are untouched."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from autofly_ue5.expert.mover_clearance import MoverClearancePenalty
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import Pose
from tests.test_dynamic_env import ZERO, FaultsOnStep, make_dynamic_env

CONTACT = 1.0


@pytest.mark.parametrize("gaps, expected", [
    ([], 0.0),
    ([5.0], 0.0),            # clearance 4 m: far outside the margin
    ([2.0], 0.0),            # clearance 1 m: exactly the margin
    ([1.5], 0.25),           # clearance 0.5 m: half way in
    ([1.0], 0.5),            # on the contact boundary: the full k
    ([0.5], 0.5),            # inside it (the step that ends as a contact): capped at k
    ([1.5, 1.0, 9.0], 0.75), # every mover inside the margin counts
])
def test_the_penalty_ramps_from_zero_at_the_margin_to_k_at_the_contact_boundary(gaps, expected):
    assert MoverClearancePenalty(k=0.5, margin_m=1.0)(gaps, contact_m=CONTACT) == pytest.approx(expected)


@pytest.mark.parametrize("k, margin", [(0.0, 1.0), (-0.5, 1.0), (0.5, 0.0), (0.5, -1.0), (math.nan, 1.0),
                                       (0.5, math.inf)])
def test_a_penalty_that_could_not_mean_anything_is_refused(k, margin):
    with pytest.raises(ValueError):
        MoverClearancePenalty(k=k, margin_m=margin)


def test_it_serialises_for_the_run_identity():
    assert MoverClearancePenalty(k=0.5, margin_m=1.0).to_json() == {"k": 0.5, "margin_m": 1.0}


def _fly_recording(env, seed):
    """_fly_at_nearest_mover's flight, keeping every step's reward and info."""
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
    penalty = MoverClearancePenalty(k=0.5, margin_m=1.0)
    plain, shaped = make_dynamic_env(), make_dynamic_env(mover_clearance=penalty)
    charged = 0
    for seed in range(1_000_020, 1_000_026):
        a, b = _fly_recording(plain, seed), _fly_recording(shaped, seed)
        assert len(a) == len(b), "the penalty must not change the flight"
        for (r_plain, info_plain), (r_shaped, info_shaped) in zip(a, b):
            p = info_shaped["mover_clearance_penalty"]
            assert r_shaped == pytest.approx(r_plain - p)
            assert {k: v for k, v in info_shaped.items() if k != "mover_clearance_penalty"} == info_plain
            assert 0.0 <= p <= 0.5 * len(info_shaped["movers"])
            charged += p > 0
        assert b[-1][1]["outcome"] == a[-1][1]["outcome"]
    assert charged > 0, "flying at movers must come inside the margin"


def test_far_from_every_mover_it_costs_nothing():
    shaped = make_dynamic_env(mover_clearance=MoverClearancePenalty(k=0.5, margin_m=1.0))
    shaped.reset(seed=1_000_071)  # the start is >= 6 m from every mover's sweep
    _obs, _r, _te, _tr, info = shaped.step(ZERO)
    assert info["mover_clearance_penalty"] == 0.0


def test_without_it_a_dynamic_scene_s_info_and_reward_are_what_they_were():
    plain = make_dynamic_env()
    for reward, info in _fly_recording(plain, 1_000_021):
        assert "mover_clearance_penalty" not in info


def test_an_inferred_mover_collision_pays_it_too():
    penalty = MoverClearancePenalty(k=0.5, margin_m=1.0)
    rewards = {}
    for name, kw in (("plain", {}), ("shaped", {"mover_clearance": penalty})):
        env = make_dynamic_env(sim_cls=FaultsOnStep, **kw)
        env.reset(seed=1_000_070)
        route = env._setup.movers[0]
        x, y = route.position(env._movers.taus[0])
        from autofly_ue5.expert.obs import target_geometry

        env._sim._pose = env._last_pose = Pose(x - (route.footprint.radius_m + 1.2), y, env._last_pose.z, 0.0)
        env._prev_dist = target_geometry(env._last_pose, env._setup.target_xy_z)[0]
        FaultsOnStep.fail_at = env._sim.steps_taken + 1
        try:
            _obs, rewards[name], terminated, _tr, info = env.step(ZERO)
        finally:
            FaultsOnStep.fail_at = None
        assert terminated and info["collision_source"] == "mover_inferred"
    # 1.2 m from the surface is clearance 0.2 m: 0.5 * (1 - 0.2) = 0.4, give or take the mover's own step
    assert rewards["plain"] - rewards["shaped"] == pytest.approx(0.4, abs=0.15)


def test_a_static_scene_refuses_it():
    from autofly_ue5.expert.env import AutoFlyEnv
    from tests.test_expert_episode import scene_and_layout

    scene, layout = scene_and_layout()
    with pytest.raises(ValueError, match="moving"):
        AutoFlyEnv(scene, layout, FakeSimulator, map_path="/Game/AutoFly/Maps/S01", instance=0,
                   mover_clearance=MoverClearancePenalty(k=0.5, margin_m=1.0))


def test_the_vec_env_hands_it_to_every_worker(tmp_path):
    import functools

    from autofly_ue5.expert.vec import make_vec_env
    from tests.test_dynamic_env import dynamic_scene, scene_objects

    scene, layout = dynamic_scene()
    penalty = MoverClearancePenalty(k=0.5, margin_m=1.0)
    vec = make_vec_env(scene, layout, 1, map_path="/Game/AutoFly/Maps/S01", monitor_dir=tmp_path / "m",
                       sim_factory=functools.partial(FakeSimulator, scene_objects=scene_objects(layout)),
                       sim_root=tmp_path / "sim", mover_clearance=penalty)
    try:
        assert vec.envs[0].unwrapped._mover_clearance == penalty
    finally:
        vec.close()


# --------------------------------------------------------------------------------------------------------
# A run that uses it says so, and cannot be resumed without it (or with another one).
# --------------------------------------------------------------------------------------------------------
def test_the_run_identity_names_it_only_when_a_run_uses_it():
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.train import run_identity
    from autofly_ue5.scenes.resolve import resolve_scene

    resolved, obs = resolve_scene("s01d"), ObsConfig(3, "float16", mover_slots=4)
    plain = run_identity(resolved, obs)
    assert "mover_clearance_penalty" not in plain, "run 3's identity must serialise exactly as before (its resume)"
    shaped = run_identity(resolved, obs, mover_clearance=MoverClearancePenalty(k=0.5, margin_m=1.0))
    assert shaped == {**plain, "mover_clearance_penalty": {"k": 0.5, "margin_m": 1.0}}


def test_a_resume_refuses_a_run_trained_with_another_penalty(tmp_path):
    from autofly_ue5.expert.train import prepare_run_root
    from tests.test_dynamic_plumbing import _checkpoint, _identity

    with_it = _identity(mover_clearance_penalty={"k": 0.5, "margin_m": 1.0})
    assert prepare_run_root(tmp_path, resume=False, reward_version="v3", seed=0, identity=with_it) == 0
    _checkpoint(tmp_path)
    for other in (_identity(), _identity(mover_clearance_penalty={"k": 1.0, "margin_m": 1.0})):
        with pytest.raises(RuntimeError, match="mover_clearance_penalty"):
            prepare_run_root(tmp_path, resume=True, reward_version="v3", seed=0, identity=other)
    assert prepare_run_root(tmp_path, resume=True, reward_version="v3", seed=0, identity=with_it) == 1


def _train(tmp_path, monkeypatch, *flags, scene="s01d"):
    from autofly_ue5.expert import train
    from tests.test_dynamic_plumbing import _dynamic_fake_factory, _no_launch

    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: _dynamic_fake_factory())
    seen = []
    real = train.make_vec_env

    def spy(*args, **kwargs):
        seen.append(kwargs.get("mover_clearance"))
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "make_vec_env", spy)
    out = tmp_path / "train.json"
    code = train.main(["--scene", scene, *flags, "--run-root", str(tmp_path / "run"), "--out", str(out),
                       "--device", "cpu", "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "10",
                       "--learning-starts", "100", "--buffer-size", "100", "--eval-freq", "1000"])
    return code, out, seen


def test_a_training_run_takes_it_from_the_command_line_and_records_it(tmp_path, monkeypatch):
    code, out, seen = _train(tmp_path, monkeypatch, "--mover-clearance-penalty", "0.5", "1.0")
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    expected = {"k": 0.5, "margin_m": 1.0}
    assert record["identity"]["mover_clearance_penalty"] == expected
    assert record["config"]["mover_clearance_penalty"] == expected
    assert json.loads((tmp_path / "run" / "sessions.json").read_text())["sessions"][0]["identity"] == record["identity"]
    # The training workers pay it; the evaluation env scores the task's own reward.
    assert seen == [MoverClearancePenalty(k=0.5, margin_m=1.0), None]


def test_without_the_flag_a_run_and_its_record_are_as_before(tmp_path, monkeypatch):
    code, out, seen = _train(tmp_path, monkeypatch)
    record = json.loads(out.read_text())
    assert code == 0 and "mover_clearance_penalty" not in record["identity"]
    assert record["config"]["mover_clearance_penalty"] is None
    assert seen == [None, None]


@pytest.mark.parametrize("scene, values, why", [("s01", ["0.5", "1.0"], "moving"), ("s01d", ["0", "1.0"], "k"),
                                                ("s01d", ["0.5", "-1"], "margin")])
def test_a_penalty_that_cannot_apply_is_refused_before_the_run_root_is_claimed(tmp_path, monkeypatch, capsys, scene,
                                                                               values, why):
    code, _out, seen = _train(tmp_path, monkeypatch, "--mover-clearance-penalty", *values, scene=scene)
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


# --------------------------------------------------------------------------------------------------------
# The training log shows what the penalty costs per episode (TensorBoard: outcomes/mover_clearance_penalty).
# --------------------------------------------------------------------------------------------------------
class _Logger:
    def __init__(self):
        self.records = {}

    def record(self, key, value, exclude=None):
        self.records[key] = value


def _outcomes():
    from types import SimpleNamespace

    from autofly_ue5.expert.train import OutcomeHistogramCallback

    cb = OutcomeHistogramCallback()
    cb.model = SimpleNamespace(logger=_Logger())  # what SB3's init_callback attaches

    def step(infos, dones):
        cb.locals = {"infos": infos, "dones": dones}
        cb._on_step()

    return cb, step


def test_the_outcome_log_carries_the_penalty_each_episode_paid():
    cb, step = _outcomes()
    step([{"outcome": "running", "mover_clearance_penalty": 0.25},
          {"outcome": "running", "mover_clearance_penalty": 0.0}], [False, False])
    step([{"outcome": "collision", "collision_source": "mover", "mover_clearance_penalty": 0.5},
          {"outcome": "running", "mover_clearance_penalty": 0.1}], [True, False])
    assert cb.model.logger.records["outcomes/mover_clearance_penalty"] == pytest.approx(0.75)
    step([{"outcome": "running", "mover_clearance_penalty": 0.0},
          {"outcome": "success", "mover_clearance_penalty": 0.0}], [False, True])
    assert cb.model.logger.records["outcomes/mover_clearance_penalty"] == pytest.approx((0.75 + 0.1) / 2)


def test_a_faulted_episode_s_penalty_is_dropped_with_the_episode():
    cb, step = _outcomes()
    step([{"outcome": "running", "mover_clearance_penalty": 0.4}], [False])
    step([{"outcome": "running", "sim_fault": "CameraPoseError"}], [True])
    step([{"outcome": "success", "mover_clearance_penalty": 0.0}], [True])
    assert cb.model.logger.records["outcomes/mover_clearance_penalty"] == 0.0


def test_a_run_without_the_penalty_logs_nothing_about_it():
    cb, step = _outcomes()
    step([{"outcome": "success"}], [True])
    assert "outcomes/success" in cb.model.logger.records
    assert "outcomes/mover_clearance_penalty" not in cb.model.logger.records
