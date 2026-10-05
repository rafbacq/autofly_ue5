"""A clean stop on request (2026-10-04): `<run root>/STOP` ends a training session the way its wall-clock budget does,
so its record and final.zip are written and its simulators torn down. Killing the job instead loses the record (run 2
was stopped that way on 2026-10-03 and left none)."""

from __future__ import annotations

import functools
import json

from autofly_ue5.sim.fake import FakeSimulator
from tests.test_dynamic_env import dynamic_scene, scene_objects


def test_the_callback_stops_only_once_the_request_exists(tmp_path):
    from autofly_ue5.expert.train import StopOnRequest

    cb = StopOnRequest(tmp_path / "STOP", every=1, verbose=0)
    assert cb._on_step() is True and cb.requested is False
    (tmp_path / "STOP").touch()
    assert cb._on_step() is False and cb.requested is True


def test_it_looks_only_every_few_steps(tmp_path):
    from autofly_ue5.expert.train import StopOnRequest

    (tmp_path / "STOP").touch()
    cb = StopOnRequest(tmp_path / "STOP", every=3, verbose=0)
    results = []
    for _ in range(3):
        cb.n_calls += 1  # what BaseCallback.on_step() does before calling _on_step()
        results.append(cb._on_step())
    assert results == [True, True, False]


class _RequestsAStop(FakeSimulator):
    """Writes the run's STOP request on its `at`-th step, as a person would from a shell."""
    path = None
    at = 0

    def step(self, dt=0.2):
        if self.steps_taken + 1 == _RequestsAStop.at:
            _RequestsAStop.path.touch()
        return super().step(dt)


def _main(tmp_path, monkeypatch, factory):
    from autofly_ue5.expert import train
    from tests.test_dynamic_plumbing import _no_launch

    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: factory)
    out = tmp_path / "train.json"
    code = train.main(["--scene", "s01d", "--run-root", str(tmp_path / "run"), "--out", str(out), "--device", "cpu",
                       "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "3000", "--learning-starts", "20",
                       "--buffer-size", "300", "--batch-size", "8", "--eval-freq", "100000",
                       "--checkpoint-freq", "100000"])
    return code, out


def test_a_requested_stop_ends_the_session_with_its_record_and_final_model(tmp_path, monkeypatch):
    _scene, layout = dynamic_scene()
    _RequestsAStop.path, _RequestsAStop.at = tmp_path / "run" / "STOP", 60
    code, out = _main(tmp_path, monkeypatch, functools.partial(_RequestsAStop, scene_objects=scene_objects(layout)))
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["stopped_on_request"] is True
    assert 0 < record["num_timesteps"] < 3000, "it stopped on the request, not at the step cap"
    assert (tmp_path / "run" / "final.zip").is_file()
    assert record["checkpoint"]["sha256"]


def test_a_session_that_runs_out_its_budget_says_it_was_not_stopped(tmp_path, monkeypatch):
    from autofly_ue5.expert import train
    from tests.test_dynamic_plumbing import _dynamic_fake_factory, _no_launch

    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: _dynamic_fake_factory())
    out = tmp_path / "train.json"
    assert train.main(["--scene", "s01d", "--run-root", str(tmp_path / "run"), "--out", str(out), "--device", "cpu",
                       "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "10", "--learning-starts", "100",
                       "--buffer-size", "100", "--eval-freq", "1000"]) == 0
    assert json.loads(out.read_text())["stopped_on_request"] is False


def test_a_pending_request_refuses_to_start_or_resume_before_claiming_the_run_root(tmp_path, monkeypatch, capsys):
    from tests.test_dynamic_plumbing import _dynamic_fake_factory

    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "STOP").touch()
    code, out = _main(tmp_path, monkeypatch, _dynamic_fake_factory())
    assert code == 2 and "STOP" in capsys.readouterr().err
    assert not (tmp_path / "run" / "sessions.json").exists() and not out.exists()
