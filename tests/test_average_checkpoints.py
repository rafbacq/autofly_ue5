"""scripts/average_checkpoints.py: the mean of several checkpoints' policy weights, saved as an ordinary checkpoint.

Run 6 (2026-10-06): neighbouring checkpoints differed sharply in deterministic flight (stage 1: step_500000 left the
bounds in 1 of 40 episodes, step_520000 in 15). The mean of its 250k-550k checkpoints flew 100 fresh validation
episodes with no exit at all, at 0.94 against the selected checkpoint's 0.83 on the same episodes."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from tests.test_warm_start import NEW, OLD, _model, _obs


def _saved(tmp_path, name, config=NEW, scale=None):
    model = _model(config)
    if scale is not None:  # distinct weights, so a mean is not any one of them
        with torch.no_grad():
            for p in model.policy.parameters():
                p.mul_(scale)
    path = tmp_path / f"{name}.zip"
    model.save(path)
    return path


def _state(path):
    from stable_baselines3 import SAC

    return SAC.load(path, device="cpu").policy.state_dict()


def test_the_average_is_the_mean_of_every_policy_tensor_and_flies_as_a_checkpoint(tmp_path):
    from stable_baselines3 import SAC

    from scripts.average_checkpoints import average

    sources = [_saved(tmp_path, f"c{i}", scale=s) for i, s in enumerate((0.5, 1.0, 2.0))]
    out = tmp_path / "avg" / "avg.zip"
    average(sources, out)
    states = [_state(p) for p in sources]
    averaged = _state(out)
    assert set(averaged) == set(states[0]) and any(k.startswith("actor.") for k in averaged)
    for key, value in averaged.items():
        assert torch.allclose(value, sum(s[key] for s in states) / 3, atol=1e-6), key
    model = SAC.load(out, device="cpu")
    action, _ = model.predict(_obs(np.random.default_rng(0), NEW), deterministic=True)
    assert action.shape == (3,) and np.all(np.isfinite(action))


def test_the_average_of_one_checkpoint_flies_exactly_like_it(tmp_path):
    from stable_baselines3 import SAC

    from scripts.average_checkpoints import average

    source = _saved(tmp_path, "only")
    out = tmp_path / "avg.zip"
    average([source], out)
    a, b = SAC.load(source, device="cpu"), SAC.load(out, device="cpu")
    rng = np.random.default_rng(1)
    for _ in range(5):
        obs = _obs(rng, NEW)
        assert np.array_equal(a.predict(obs, deterministic=True)[0], b.predict(obs, deterministic=True)[0])


def test_its_record_names_every_source_and_the_result_by_hash(tmp_path):
    from scripts.average_checkpoints import average

    sources = [_saved(tmp_path, f"c{i}") for i in range(2)]
    out = tmp_path / "avg.zip"
    record = average(sources, out)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert record == json.loads(out.with_suffix(".json").read_text())
    assert record["sha256"] == sha(out) and record["n_sources"] == 2
    assert record["sources"] == [{"path": str(p), "sha256": sha(p)} for p in sources]


def test_it_refuses_to_overwrite_an_output_or_to_average_nothing(tmp_path):
    from scripts.average_checkpoints import average

    source = _saved(tmp_path, "c0")
    taken = tmp_path / "taken.zip"
    taken.write_bytes(b"evidence")
    with pytest.raises(FileExistsError):
        average([source], taken)
    assert taken.read_bytes() == b"evidence"
    with pytest.raises(ValueError, match="no checkpoints"):
        average([], tmp_path / "none.zip")


def test_it_refuses_checkpoints_of_different_networks(tmp_path):
    from scripts.average_checkpoints import average

    with pytest.raises(ValueError, match="another network"):
        average([_saved(tmp_path, "movers", NEW), _saved(tmp_path, "no_movers", OLD)], tmp_path / "avg.zip")
    assert not (tmp_path / "avg.zip").exists()


def test_the_command_line_takes_the_output_then_the_sources(tmp_path, capsys):
    from scripts.average_checkpoints import main

    sources = [_saved(tmp_path, f"c{i}") for i in range(2)]
    assert main([str(tmp_path / "avg.zip"), *map(str, sources)]) == 0
    assert "mean of 2 checkpoints" in capsys.readouterr().out
    assert (tmp_path / "avg.zip").is_file()
