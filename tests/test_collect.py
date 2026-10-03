"""M3's collector, raw store and validator (spec §9-§11), against FakeSimulator and a scripted pilot: success-only
episodes in time order with AutoFly's four fields, rejects with their reason, provenance per episode, faults replayed,
and a validator that passes a good store and catches a bad one."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import CameraPoseError
from tests.test_expert_episode import scene_and_layout
from tests.test_expert_train import _FlakyFakeSimulator
from tests.test_m2_gate import _StraightAtTargetModel

SEED_BASE = 400_000_000
PROVENANCE = {"expert": {"path": "best_model.zip", "sha256": "0" * 64}, "platform": {"projectairsim": "test"},
              "collector_commit": "test"}


class _Crashes(_StraightAtTargetModel):
    """Flies straight but never climbs out of a dive: an episode that ends badly, for the rejects."""

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action, _ = super().predict(observation, deterministic=deterministic)
        action[2] = -1.0  # descend until the altitude band ends the episode
        return action, None


def _env(tmp_path, sim_factory=FakeSimulator):
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.resilient import ResilientAutoFlyEnv

    scene, layout = scene_and_layout()
    base = AutoFlyEnv(scene, layout, sim_factory, map_path="/Game/AutoFly/Maps/S01", instance=0)
    return scene, ResilientAutoFlyEnv(base, instance=0, sim_root=tmp_path / "sim")


def _collect(tmp_path, model=None, n_keep=3, sim_factory=FakeSimulator, max_attempts=20):
    from autofly_ue5.collect.collector import collect
    from autofly_ue5.dataset.raw import RawDatasetWriter

    scene, env = _env(tmp_path, sim_factory)
    writer = RawDatasetWriter(tmp_path / "data", "pilot", scene=scene, provenance=PROVENANCE)
    summary = collect(model or _StraightAtTargetModel(), env, scene=scene, writer=writer, seed_base=SEED_BASE,
                      n_keep=n_keep, target_name="orange cylinder", max_attempts=max_attempts,
                      layout_sha256="test-layout")
    return summary, tmp_path / "data" / "pilot"


def test_collected_episodes_hold_autofly_s_fields_in_time_order(tmp_path):
    from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES

    summary, root = _collect(tmp_path)
    assert summary["kept"] == 3 and summary["status"] == "ok"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["counts"]["episodes"] == 3 and len(manifest["episodes"]) == 3
    for entry in manifest["episodes"]:
        ep = root / entry["path"]
        steps = np.load(ep / "steps.npz")
        n = entry["steps"]
        assert steps["state"].shape == (n, 9) and steps["state"].dtype == np.float32
        assert steps["action"].shape == (n, 3) and steps["action"].dtype == np.float32
        assert np.all(np.diff(steps["sim_time_ns"]) == 200_000_000), "exactly one 0.2 s step per record"
        assert np.all(steps["state"][1:, 0] <= steps["state"][:-1, 0] + 0.5), "flying at the target"
        assert steps["state"][-1, 0] <= 5.0 + 0.4 * 2, "the last record is within reach of the 5 m success radius"
        frames = sorted((ep / "frames").glob("*.png"))
        assert len(frames) == n
        img = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        assert img.shape == (256, 256, 3)
        text = entry["instruction"]
        assert "orange cylinder" in text and "white pillars" in text
        assert any(text == t.format(target="orange cylinder", obstacle="white pillars") for t in INSTRUCTION_TEMPLATES)
        prov = json.loads((root / "provenance" / f"{entry['id']}.json").read_text())
        assert prov["seed"] == entry["seed"] and prov["termination"] == "success" and prov["a0"]["mode"] == "sector8"
        assert abs(math.remainder(prov["start_pose"][3], math.pi / 4)) < 1e-9
        assert len(prov["sim_time_ns"]) == n and prov["expert"]["sha256"] == "0" * 64
        assert prov["scene"]["sha256"] and prov["layout_sha256"] and len(prov["distractors"]) >= 3


def test_the_first_record_is_the_aligned_start_with_the_expert_s_own_first_action(tmp_path):
    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    steps = np.load(root / entry["path"] / "steps.npz")
    assert steps["state"][0, 3] == pytest.approx(0.0, abs=1e-6), "recording starts from the hover"
    assert abs(steps["state"][0, 1]) <= math.radians(22.5) + 1e-6, "facing the target's sector (a0)"
    assert (steps["state"][0, 6], steps["state"][0, 7]) == (0.0, 0.0)


def test_failed_episodes_go_to_the_rejects_with_their_reason(tmp_path):
    summary, root = _collect(tmp_path, model=_Crashes(), n_keep=1, max_attempts=3)
    assert summary["kept"] == 0 and summary["rejected"] == 3 and summary["status"] == "incomplete"
    rejects = sorted((tmp_path / "data" / "rejects" / "pilot").glob("*.json"))
    assert len(rejects) == 3
    record = json.loads(rejects[0].read_text())
    assert record["reason"] == "out_of_bounds" and record["provenance"]["seed"] == SEED_BASE
    assert json.loads((root / "manifest.json").read_text())["counts"]["episodes"] == 0


def test_a_backend_fault_replays_the_seed_and_never_reaches_the_store(tmp_path):
    import functools

    factory = functools.partial(_FlakyFakeSimulator, fail_on_step_calls=(3,), error=CameraPoseError)
    summary, root = _collect(tmp_path, n_keep=1, sim_factory=factory)
    assert summary["kept"] == 1 and summary["fault_retries"] == 1
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    assert entry["seed"] == SEED_BASE, "the faulted seed was flown again, not skipped"


def test_the_validator_passes_a_good_store_and_names_what_is_wrong_with_a_bad_one(tmp_path):
    from autofly_ue5.validate.dataset import validate_dataset

    _summary, root = _collect(tmp_path, n_keep=3)
    report = validate_dataset(root)
    assert report["pass"], report["failures"]
    assert report["counts"]["episodes"] == 3 and report["counts"]["records"] > 100
    assert {"speed_m_s", "altitude_m", "episode_steps"} <= set(report["distributions"])
    # Break one episode: a NaN state and a missing frame.
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    ep = root / entry["path"]
    data = dict(np.load(ep / "steps.npz"))
    data["state"][2, 4] = np.nan
    np.savez(ep / "steps.npz", **data)
    sorted((ep / "frames").glob("*.png"))[-1].unlink()
    bad = validate_dataset(root)
    assert not bad["pass"]
    assert any("NaN" in f for f in bad["failures"]) and any("frames" in f for f in bad["failures"])


def test_an_existing_dataset_is_never_overwritten(tmp_path):
    from autofly_ue5.dataset.raw import RawDatasetWriter

    scene, _env_ = _env(tmp_path)
    (tmp_path / "data" / "pilot").mkdir(parents=True)
    (tmp_path / "data" / "pilot" / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        RawDatasetWriter(tmp_path / "data", "pilot", scene=scene, provenance=PROVENANCE)


def test_the_pilot_cli_collects_validates_exports_and_records_the_gate(tmp_path):
    from autofly_ue5.expert.obs import ObsConfig
    from scripts.collect_dataset import run

    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a real model; the fake loader ignores it")
    record = run(scene="s01", model_path=checkpoint, scene_config="scene_autofly_s01_fast.jsonc", name="pilot",
                 n_episodes=2, instance=5, out_path=tmp_path / "m3_gate.json", data_root=tmp_path / "data",
                 sim_factory=FakeSimulator, load_model=lambda path: _StraightAtTargetModel(),
                 obs_config_reader=lambda path: ObsConfig(), sim_root=tmp_path / "sim", platform={"test": True})
    assert record["status"] == "ok" and record["error"] is None, record["error"]
    assert record["collection"]["kept"] == 2 and record["validation"]["pass"]
    assert record["pass"] == record["faults_ok"], "with a clean audit, a kept and validated pilot passes"
    assert record["seed_base"] == SEED_BASE and record["expert"]["deterministic"] is False
    assert record["scene_config"]["file"] == "scene_autofly_s01_fast.jsonc" and record["a0"] == "sector8"
    assert record["rlds"]["splits"] == {"train": 2}
    assert json.loads((tmp_path / "m3_gate.json").read_text())["collection"]["kept"] == 2
    prov = json.loads(next((tmp_path / "data" / "pilot" / "provenance").glob("*.json")).read_text())
    assert prov["platform"] == {"test": True} and prov["expert"]["sha256"] == record["expert"]["sha256"]
    assert not (tmp_path / "sim" / "inst5" / "pid.json").exists(), "its simulator slot is left empty"


def test_the_pilot_cli_refuses_an_existing_dataset_or_record_before_launching_anything(tmp_path):
    from autofly_ue5 import evidence
    from scripts.collect_dataset import main

    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"x")
    common = ["--model", str(checkpoint), "--scene-config", "scene_autofly_s01_fast.jsonc", "--name", "pilot",
              "--data-root", str(tmp_path / "data")]
    (tmp_path / "data" / "pilot").mkdir(parents=True)
    (tmp_path / "data" / "pilot" / "manifest.json").write_text("{}")
    assert main([*common, "--out", str(tmp_path / "m3.json")]) == 2, "an existing dataset"
    assert main([*common, "--name", "other", "--out", str(tmp_path / "m3.json")]) == 2, "a checkpoint that is no zip"
    assert main([*common, "--name", "other", "--out", str(tmp_path / "m3.json"), "--scene-config", "nope.jsonc"]) == 2
    (evidence.GATES_DIR / "m3_gate.json").write_text("{}")  # the default record (a scratch dir, see conftest)
    assert main([*common, "--name", "other"]) == 2, "an existing gate record"
    assert not (tmp_path / "m3.json").exists() and not (tmp_path / "data" / "other").exists()


def test_the_start_check_allows_float32_poses_but_not_a_recording_that_starts_a_step_late(tmp_path):
    # Live (2026-10-03 smoke): record 0 sat 1 um from the start (float32 poses), record 1 already 5-11 mm away.
    from autofly_ue5.validate.dataset import validate_dataset

    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    path = root / entry["path"] / "steps.npz"
    data = dict(np.load(path))
    data["state"][0, 6:8] = [-1.0e-7, 1.03e-6]  # the live smoke's own first record
    np.savez(path, **data)
    assert validate_dataset(root)["pass"]
    data["state"][0, 6:8] = [0.0052, 0.0]  # where the live drone was one step later
    np.savez(path, **data)
    assert any("not at the start" in f for f in validate_dataset(root)["failures"])


def test_the_pilot_cli_refuses_a_slot_another_live_run_holds(tmp_path, capsys):
    from scripts.collect_dataset import main
    from tests.test_process import _init_owner, _sleeper

    from autofly_ue5.sim.process import instance_dir, stop

    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"x")
    _sleeper(7, tmp_path / "sim", owner=_init_owner())
    log = instance_dir(7, tmp_path / "sim") / "client.log"
    log.write_text("the other run's log\n")
    try:
        assert main(["--model", str(checkpoint), "--scene-config", "scene_autofly_s01_fast.jsonc", "--name", "pilot",
                     "--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "m3.json"), "--instance", "7",
                     "--sim-root", str(tmp_path / "sim")]) == 2
    finally:
        stop(7, grace_s=2.0, run_root=tmp_path / "sim")
    assert "slot 7 holds simulator pid" in capsys.readouterr().err
    assert log.read_text() == "the other run's log\n" and not (tmp_path / "data").exists()


class _DiesAfter(_StraightAtTargetModel):
    """A pilot that raises something unrecoverable after `calls` predictions: a crash partway through a pilot."""

    def __init__(self, calls: int) -> None:
        super().__init__()
        self.left = calls

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.left -= 1
        if self.left < 0:
            raise RuntimeError("the pilot crashed")
        return super().predict(observation, deterministic=deterministic)


class _RefusedLaunch(FakeSimulator):
    """A launch refused before anything starts (the GPU guard on a shared host)."""

    def launch(self, map_path, instance):
        raise RuntimeError("launch refused: a foreign GPU job holds 4.4 GB")


def _pilot(tmp_path, *, model=None, sim_factory=FakeSimulator, n_episodes=2, name="pilot", out=None, **kwargs):
    """The pilot as main() runs it, into the default record docs/gates/m3_gate.json (a scratch dir here, see conftest)."""
    from autofly_ue5 import evidence
    from autofly_ue5.expert.obs import ObsConfig
    from scripts.collect_dataset import run

    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a real model; the fake loader ignores it")
    return run(scene="s01", model_path=checkpoint, scene_config="scene_autofly_s01_fast.jsonc", name=name,
               n_episodes=n_episodes, instance=5, out_path=out or evidence.GATES_DIR / "m3_gate.json",
               data_root=tmp_path / "data",
               sim_factory=sim_factory, load_model=lambda path: model or _StraightAtTargetModel(),
               obs_config_reader=lambda path: ObsConfig(), sim_root=tmp_path / "sim", platform={"test": True}, **kwargs)


def test_a_pilot_that_crashes_partway_is_recorded_as_a_failed_gate_with_its_counts(tmp_path):
    from autofly_ue5 import evidence
    from autofly_ue5.validate.dataset import validate_dataset

    record = _pilot(tmp_path, model=_DiesAfter(400), n_episodes=5)
    assert record["status"] == "failed" and "the pilot crashed" in record["error"] and record["pass"] is False
    stored = json.loads((tmp_path / "data" / "pilot" / "manifest.json").read_text())["counts"]["episodes"]
    assert stored >= 1 and record["collection"]["kept"] == stored, "the counts survive the crash"
    assert record["collection"]["attempted"] == stored + record["collection"]["rejected"]
    assert json.loads((evidence.GATES_DIR / "m3_gate.json").read_text())["status"] == "failed", "evidence, not 'never started'"
    assert record["validation"]["pass"] and validate_dataset(tmp_path / "data" / "pilot")["pass"]
    assert record["rlds"]["splits"] == {"train": stored}, "what was kept is still exported"


def test_a_pilot_whose_every_episode_failed_is_recorded_as_a_failed_gate(tmp_path):
    from autofly_ue5 import evidence

    record = _pilot(tmp_path, model=_Crashes(), n_episodes=1, max_attempts=2)
    assert record["status"] == "incomplete" and record["pass"] is False
    assert record["collection"]["rejected"] == 2 and record["collection"]["kept"] == 0
    assert (evidence.GATES_DIR / "m3_gate.json").is_file(), "two flown episodes are evidence, not a run that never started"
    assert not (evidence.NOT_STARTED_DIR / "m3_gate.json").exists()


def test_a_refused_launch_claims_nothing_so_the_same_name_can_be_retried(tmp_path):
    from autofly_ue5 import evidence

    record = _pilot(tmp_path, sim_factory=_RefusedLaunch)
    assert record["status"] == "failed" and "launch refused" in record["error"]
    assert not (tmp_path / "data" / "pilot").exists() and not (tmp_path / "data" / "rejects" / "pilot").exists()
    assert not (evidence.GATES_DIR / "m3_gate.json").exists() and (evidence.NOT_STARTED_DIR / "m3_gate.json").is_file()
    assert _pilot(tmp_path)["collection"]["kept"] == 2, "the retry under the same name works"


def test_the_gate_needs_a_usable_rlds_export(tmp_path, monkeypatch):
    import scripts.collect_dataset as cli

    def broken_export(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "export_rlds", broken_export)
    record = _pilot(tmp_path)
    assert record["validation"]["pass"] and record["rlds"] == {"error": "OSError: disk full"}
    assert record["pass"] is False


def test_the_gate_needs_the_tfds_read_back_when_one_was_asked_for(tmp_path):
    fake_python = tmp_path / "fake_tfds_python"
    fake_python.write_text('#!/bin/sh\necho \'{"pass": false, "mismatches": ["frames differ"]}\'\nexit 1\n')
    fake_python.chmod(0o755)
    record = _pilot(tmp_path, check_python=fake_python)
    assert record["rlds"]["tfds_check"]["pass"] is False and record["pass"] is False


def test_a_validator_or_audit_that_raises_still_leaves_an_honest_record(tmp_path, monkeypatch):
    import scripts.collect_dataset as cli

    def boom(*args, **kwargs):
        raise IndexError("validator bug")

    monkeypatch.setattr(cli, "validate_dataset", boom)
    monkeypatch.setattr(cli, "audit_engine_faults", boom)
    record = _pilot(tmp_path)
    assert record["validation"]["pass"] is False and "validator bug" in record["validation"]["failures"][0]
    assert record["faults_ok"] is False and "validator bug" in record["engine_faults"]["error"]
    assert record["pass"] is False and record["written_to"].endswith("m3_gate.json")


def test_the_record_never_overwrites_one_that_appeared_while_the_pilot_ran(tmp_path):
    out = tmp_path / "m3_gate.json"
    out.write_text('{"another": "pilot"}\n')
    record = _pilot(tmp_path, out=out)
    assert json.loads(out.read_text()) == {"another": "pilot"}
    conflict = Path(record["written_to"])
    assert conflict != out and conflict.parent == out.parent and json.loads(conflict.read_text())["collection"]["kept"] == 2


def test_a_dataset_name_with_leftover_rejects_is_refused(tmp_path):
    from autofly_ue5.dataset.raw import RawDatasetWriter

    scene, _env_ = _env(tmp_path)
    (tmp_path / "data" / "rejects" / "pilot").mkdir(parents=True)
    (tmp_path / "data" / "rejects" / "pilot" / "s01_400000000.json").write_text("{}")
    with pytest.raises(FileExistsError):
        RawDatasetWriter(tmp_path / "data", "pilot", scene=scene, provenance=PROVENANCE)


def _rewrite_json(path, edit):
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def test_the_validator_reports_a_malformed_episode_instead_of_crashing_on_it(tmp_path):
    from autofly_ue5.validate.dataset import validate_dataset

    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    path = root / entry["path"] / "steps.npz"
    data = dict(np.load(path))
    data["state"] = data["state"][:, :8]
    np.savez(path, **data)
    report = validate_dataset(root)
    assert not report["pass"] and any("state is float32" in f for f in report["failures"])


def test_the_validator_checks_each_instruction_names_its_own_target_and_the_scene_s_obstacles(tmp_path):
    from autofly_ue5.validate.dataset import validate_dataset

    _summary, root = _collect(tmp_path, n_keep=1)
    manifest = root / "manifest.json"
    original = json.loads(manifest.read_text())["episodes"][0]["instruction"]
    _rewrite_json(manifest, lambda m: m["episodes"][0].update(instruction=original.replace("orange cylinder", "target")))
    assert any("instruction" in f for f in validate_dataset(root)["failures"]), "Plan 2's placeholder must not pass"
    _rewrite_json(manifest, lambda m: m["episodes"][0].update(instruction=original.replace("white pillars", "trees")))
    assert any("instruction" in f for f in validate_dataset(root)["failures"]), "another scene's obstacles"
    _rewrite_json(manifest, lambda m: m["episodes"][0].update(instruction=original))
    assert validate_dataset(root)["pass"]


def test_the_validator_compares_provenance_with_the_record_by_value(tmp_path):
    from autofly_ue5.validate.dataset import validate_dataset

    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    prov = root / "provenance" / f"{entry['id']}.json"
    _rewrite_json(prov, lambda p: p.update(seed=p["seed"] + 1))
    assert any("seed" in f for f in validate_dataset(root)["failures"])
    _rewrite_json(prov, lambda p: p.update(seed=p["seed"] - 1, sim_time_ns=[t + 1 for t in p["sim_time_ns"]]))
    assert any("simulator times" in f for f in validate_dataset(root)["failures"])


def test_provenance_names_the_target_and_distractors_and_the_card_marks_the_placeholder_name(tmp_path):
    # Spec §10.2: target and distractor names; U3: "orange cylinder" is marked a placeholder until M4's pool.
    record = _pilot(tmp_path, n_episodes=1)
    root = tmp_path / "data" / "pilot"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["card"]["target_names"] == {"orange cylinder": "placeholder_until_M4"}
    prov = json.loads((root / "provenance" / f"{manifest['episodes'][0]['id']}.json").read_text())
    target = prov["target"]
    assert target["name"] == "orange cylinder" and target["name_status"] == "placeholder_until_M4"
    assert target["asset"] == "cylinder" and target["material"] == "orange" and target["spawned_as"].startswith("target")
    assert len(target["xyz"]) == 3 and len(target["scale"]) == 3
    assert len(prov["distractors"]) >= 3
    for i, d in enumerate(prov["distractors"]):
        assert d["asset"] == "cylinder" and d["material"] == "mesh_default" and d["spawned_as"].startswith(f"distractor_{i}")
        assert len(d["xyz"]) == 3
    assert record["pass"] == record["faults_ok"]


def test_the_rlds_shards_sit_where_spec_10_1_puts_them(tmp_path):
    # data/<dataset_name>/1.0.0/, beside the same dataset's provenance/ and manifest.json; the read-back report beside it.
    fake_python = tmp_path / "fake_tfds_python"
    fake_python.write_text('#!/bin/sh\necho \'{"pass": true}\'\n')
    fake_python.chmod(0o755)
    record = _pilot(tmp_path, n_episodes=1, check_python=fake_python)
    root = tmp_path / "data" / "pilot"
    assert Path(record["rlds"]["path"]) == root / "1.0.0"
    assert len(list((root / "1.0.0").glob("pilot-train.tfrecord-*"))) == 1
    assert json.loads((root / "1.0.0" / "dataset_info.json").read_text())["name"] == "pilot"
    assert json.loads((root / "tfds_check.json").read_text()) == {"pass": True} and record["pass"] == record["faults_ok"]


def test_a_dataset_name_must_be_a_valid_tfds_name(tmp_path):
    from autofly_ue5.dataset.raw import RawDatasetWriter

    scene, _env_ = _env(tmp_path)
    for bad in ("S01-Pilot", "1pilot", "pilot run", ""):
        with pytest.raises(ValueError):
            RawDatasetWriter(tmp_path / "data", bad, scene=scene, provenance=PROVENANCE)
