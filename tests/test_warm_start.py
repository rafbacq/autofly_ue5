"""Warm-starting a new SAC from an earlier checkpoint whose observation lacked the mover input (2026-10-03).

s01d_r1's best_model (0.775 on the gate) already flies the field; s01d_r2 adds the mover input. The new network is
the old one plus a mover branch whose way into the head starts at zero, so at step 0 it acts and values exactly like
its source, and the warm-up fills the new replay buffer with the policy's own behaviour, not uniform random actions.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from autofly_ue5.expert.obs import ObsConfig

OLD = ObsConfig(3, "float16")
NEW = ObsConfig(3, "float16", mover_slots=4)


def _model(config, algorithm=None, learning_starts=50):
    from tests.test_dynamic_env import make_dynamic_env

    from autofly_ue5.expert.train import build_model

    kwargs = {} if algorithm is None else {"algorithm": algorithm}
    return build_model(DummyVecEnv([lambda: make_dynamic_env(obs_config=config)]), device="cpu", buffer_size=200,
                       learning_starts=learning_starts, verbose=0, **kwargs)


def _saved_source(tmp_path):
    old = _model(OLD)
    with torch.no_grad():
        old.log_ent_coef.fill_(math.log(0.0071))
    path = tmp_path / "source.zip"
    old.save(path)
    return old, path


def _obs(rng, config):
    sample = config.space().sample()
    sample["depth"] = rng.random(sample["depth"].shape).astype(sample["depth"].dtype)
    sample["vector"] = rng.normal(size=sample["vector"].shape).astype(np.float32)
    if "movers" in sample:
        sample["movers"] = rng.uniform(-1, 1, size=sample["movers"].shape).astype(np.float32)
    return sample


def test_a_warm_started_model_acts_and_values_exactly_like_its_source(tmp_path):
    from autofly_ue5.expert.warmstart import warm_start

    old, path = _saved_source(tmp_path)
    new = _model(NEW)
    report = warm_start(new, path)
    assert set(report["widened"]) == {f"{net}.features_extractor.head.0.weight" for net in ("actor", "critic", "critic_target")}
    assert len(report["fresh"]) == 6 and all(".mover_mlp." in name for name in report["fresh"])
    assert report["copied"] == len(new.policy.state_dict()) - 9
    assert report["log_ent_coef"] == pytest.approx(math.log(0.0071))
    assert float(new.log_ent_coef) == pytest.approx(math.log(0.0071))
    rng = np.random.default_rng(0)
    for _ in range(5):
        new_obs = _obs(rng, NEW)
        old_obs = {k: v for k, v in new_obs.items() if k != "movers"}
        a_new, _ = new.predict(new_obs, deterministic=True)
        a_old, _ = old.predict(old_obs, deterministic=True)
        np.testing.assert_allclose(a_new, a_old, atol=1e-6)
        act = torch.as_tensor(rng.uniform(-1, 1, size=(1, 3)), dtype=torch.float32)
        t_new, _ = new.policy.obs_to_tensor(new_obs)
        t_old, _ = old.policy.obs_to_tensor(old_obs)
        with torch.no_grad():
            for net in ("critic", "critic_target"):
                q_new = torch.cat(getattr(new.policy, net)(t_new, act))
                q_old = torch.cat(getattr(old.policy, net)(t_old, act))
                torch.testing.assert_close(q_new, q_old, atol=1e-5, rtol=0)


def test_a_source_of_another_architecture_is_refused(tmp_path):
    from autofly_ue5.expert.warmstart import warm_start

    one_frame = _model(ObsConfig(1, "float32"))
    path = tmp_path / "one_frame.zip"
    one_frame.save(path)
    with pytest.raises(ValueError, match="cnn.0.weight"):
        warm_start(_model(NEW), path)


def test_the_warmup_acts_with_the_policy_instead_of_at_random():
    from stable_baselines3 import SAC

    from autofly_ue5.expert.warmstart import PolicyWarmupSAC

    for algorithm, expected in ((SAC, 0), (PolicyWarmupSAC, 6)):
        model = _model(NEW, algorithm=algorithm, learning_starts=100)
        calls = []
        original = model.predict
        model.predict = lambda *a, **kw: (calls.append(1), original(*a, **kw))[1]
        model.learn(total_timesteps=6)
        assert len(calls) == expected, algorithm.__name__


# ---------------------------------------------------------------------------------------------------------------
# train.py --warm-start
# ---------------------------------------------------------------------------------------------------------------
def _train(tmp_path, monkeypatch, *extra):
    from tests.test_dynamic_plumbing import _dynamic_fake_factory, _no_launch

    from autofly_ue5.expert import train

    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: _dynamic_fake_factory())
    out = tmp_path / "train.json"
    code = train.main(["--scene", "s01d", "--run-root", str(tmp_path / "run"), "--out", str(out), "--device", "cpu",
                       "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "30", "--learning-starts", "20",
                       "--buffer-size", "200", "--batch-size", "8", "--eval-freq", "1000", *extra])
    return code, out


def test_training_can_warm_start_and_records_where_from(tmp_path, monkeypatch):
    from autofly_ue5.expert.train import sha256_of

    _old, path = _saved_source(tmp_path)
    code, out = _train(tmp_path, monkeypatch, "--warm-start", str(path))
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    ws = record["warm_start"]
    assert ws["path"] == str(path) and ws["sha256"] == sha256_of(path) and len(ws["widened"]) == 3
    sessions = json.loads((tmp_path / "run" / "sessions.json").read_text())["sessions"]
    assert sessions[0]["warm_start"]["sha256"] == ws["sha256"]


def test_a_warm_start_is_refused_with_resume_or_without_a_readable_source(tmp_path, monkeypatch, capsys):
    code, _out = _train(tmp_path, monkeypatch, "--warm-start", str(tmp_path / "missing.zip"))
    assert code == 2 and "missing.zip" in capsys.readouterr().err and not (tmp_path / "run").exists()
    _old, path = _saved_source(tmp_path)
    code, _out = _train(tmp_path, monkeypatch, "--warm-start", str(path), "--resume")
    assert code == 2 and "--warm-start" in capsys.readouterr().err
