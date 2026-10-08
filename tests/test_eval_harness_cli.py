"""scripts/eval_harness.py: `write-episodes` writes the frozen condition files once and never over them; `run` flies a
policy (hover, the privileged scripted baseline, an expert checkpoint) on a condition and writes its record, after
refusing, before anything launches, a busy slot, an existing record, a policy it cannot build or fly honestly. Every test
here flies FakeSimulator through the script's sim factory hook, with Project AirSim's launch stubbed to fail."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_eval_episodes import _gates, fake_resolver
from tests.test_eval_harness import LaunchCounter, _condition, fake_sims


@pytest.fixture
def cli(monkeypatch):
    import scripts.eval_harness as cli
    from tests.test_dynamic_plumbing import _no_launch

    _no_launch(monkeypatch)  # whatever a test gets wrong, no real simulator can start
    monkeypatch.setattr(cli, "SIM_FACTORY", fake_sims)
    return cli


def _run_args(tmp_path, condition, *extra, out="out/record.json", instance=5):
    return ["run", "--condition", str(condition), "--instance", str(instance), "--sim-root", str(tmp_path / "sim"),
            "--out", str(tmp_path / out), *extra]


def test_write_episodes_writes_every_condition_once_and_never_over_one(tmp_path, cli, monkeypatch):
    from autofly_ue5.evaluation.episodes import load_condition

    monkeypatch.setattr(cli, "resolve_scene", lambda scene_id, **kwargs: fake_resolver()(scene_id))
    out = tmp_path / "eval_episodes"
    common = ["write-episodes", "--out-dir", str(out), "--episodes-per-scene", "2", "--gates-dir",
              str(_gates(tmp_path, m4_s05r_gate=True))]
    assert cli.main(common) == 0
    assert sorted(p.name for p in out.iterdir()) == ["seen.json", "test_seen.json", "unseen_targets.json"]
    assert load_condition(out / "test_seen.json").scenes["s06r"]["status_at_write"] == "pending"
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert cli.main(common) == 2 and cli.main([*common, "--condition", "seen"]) == 2
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before, "a frozen file is never written over"
    (out / "seen.json").unlink()
    assert cli.main([*common, "--condition", "seen"]) == 0 and len(load_condition(out / "seen.json").episodes) == 18


def test_run_flies_a_policy_end_to_end_and_writes_its_record_and_summary(tmp_path, cli, capsys):
    condition = _condition(tmp_path, n=2)
    assert cli.main(_run_args(tmp_path, condition, "--policy", "hover", "--max-steps", "3")) == 0
    record = json.loads((tmp_path / "out" / "record.json").read_text())
    assert record["status"] == "ok" and record["policy"]["name"] == "hover" and len(record["episodes"]) == 2
    assert all(e["outcome"] == "timeout" and e["steps"] == 3 for e in record["episodes"]), "hover is the floor"
    assert record["privileged_state"] is False and record["provenance"]["argv"][0] == "run"
    assert "git_head" in record["provenance"] and (tmp_path / "out" / "record.md").is_file()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "ok" and printed["overall"]["success_rate"]["value"] == 0.0


def test_run_flies_the_scripted_baseline_with_the_privileged_state(tmp_path, cli):
    condition = _condition(tmp_path, n=1)
    assert cli.main(_run_args(tmp_path, condition, "--policy", "scripted", "--privileged-state")) == 0
    record = json.loads((tmp_path / "out" / "record.json").read_text())
    assert record["privileged_state"] is True and record["episodes"][0]["outcome"] == "success"


def test_run_flies_an_expert_checkpoint_on_the_observation_it_was_trained_on(tmp_path, cli):
    # A real (untrained) SAC checkpoint saved on s01's observation: the script reads its observation from the zip,
    # loads it with SB3 and flies it, as it will the accepted experts.
    from stable_baselines3.common.vec_env import DummyVecEnv

    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.train import build_model
    from autofly_ue5.scenes.resolve import resolve_scene
    from autofly_ue5.sim.fake import FakeSimulator

    resolved = resolve_scene("s01")
    vec = DummyVecEnv([lambda: AutoFlyEnv(resolved.scene, resolved.layout, FakeSimulator, map_path="/x", instance=0)])
    checkpoint = tmp_path / "expert.zip"
    build_model(vec, device="cpu", buffer_size=100, learning_starts=1000, verbose=0).save(checkpoint)
    condition = _condition(tmp_path, n=1)
    assert cli.main(_run_args(tmp_path, condition, "--policy", f"expert:{checkpoint}", "--deterministic", "--device",
                              "cpu", "--max-steps", "3")) == 0
    record = json.loads((tmp_path / "out" / "record.json").read_text())
    policy = record["policy"]
    assert policy["name"] == "expert" and policy["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert policy["deterministic"] is True and policy["obs_config"] == {"depth_frames": 1, "depth_dtype": "float32"}
    assert record["status"] == "ok" and record["episodes"][0]["steps"] >= 1


@pytest.mark.parametrize("args, says", [
    (["--policy", "scripted"], "privileged"),
    (["--policy", "hover", "--deterministic"], "deterministic"),
    (["--policy", "expert:/nowhere/model.zip"], "no expert checkpoint"),
    (["--policy", "teleport"], "unknown policy"),
])
def test_run_refuses_a_policy_it_cannot_build_or_fly_honestly_before_anything_launches(tmp_path, cli, monkeypatch,
                                                                                       capsys, args, says):
    launches = LaunchCounter()
    monkeypatch.setattr(cli, "SIM_FACTORY", launches)
    assert cli.main(_run_args(tmp_path, _condition(tmp_path, n=1), *args)) == 2
    assert says in capsys.readouterr().err and launches.launches == 0
    assert not (tmp_path / "out").exists()


def test_run_refuses_an_existing_record_a_record_that_is_not_json_and_a_missing_condition(tmp_path, cli, capsys):
    condition = _condition(tmp_path, n=1)
    existing = tmp_path / "out" / "record.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"another": "run"}\n')
    assert cli.main(_run_args(tmp_path, condition, "--policy", "hover")) == 2
    assert existing.read_text() == '{"another": "run"}\n' and "never written over" in capsys.readouterr().err
    assert cli.main(_run_args(tmp_path, condition, "--policy", "hover", out="out/record.md")) == 2
    assert cli.main(_run_args(tmp_path, tmp_path / "missing.json", "--policy", "hover", out="out/other.json")) == 2
    assert not (tmp_path / "out" / "other.json").exists()


def test_run_refuses_a_slot_another_live_run_holds_before_touching_it(tmp_path, cli, capsys):
    from autofly_ue5.sim.process import instance_dir, stop
    from tests.test_process import _init_owner, _sleeper

    condition = _condition(tmp_path, n=1)
    _sleeper(7, tmp_path / "sim", owner=_init_owner())
    log = instance_dir(7, tmp_path / "sim") / "client.log"
    log.write_text("the other run's log\n")
    try:
        assert cli.main(_run_args(tmp_path, condition, "--policy", "hover", instance=7)) == 2
    finally:
        stop(7, grace_s=2.0, run_root=tmp_path / "sim")
    assert "slot 7 holds simulator pid" in capsys.readouterr().err
    assert log.read_text() == "the other run's log\n" and not (tmp_path / "out").exists()


def test_run_refuses_the_unseen_condition_when_the_env_cannot_draw_unseen_targets(tmp_path, cli, monkeypatch, capsys):
    import autofly_ue5.evaluation.harness as harness

    monkeypatch.setattr(harness, "env_accepts_target_split", lambda: False)
    condition = _condition(tmp_path, name="unseen_targets", n=1)
    assert cli.main(_run_args(tmp_path, condition, "--policy", "hover")) == 2
    assert "target_split" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_a_condition_is_a_file_or_the_name_of_a_committed_one():
    from autofly_ue5.evaluation.episodes import EPISODES_DIR
    from scripts.eval_harness import condition_path

    assert condition_path("test_seen") == EPISODES_DIR / "test_seen.json"
    assert condition_path("runs/eval/mine.json") == Path("runs/eval/mine.json")
