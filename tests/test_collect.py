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


def _env(tmp_path, sim_factory=FakeSimulator, target_pool=None):
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.resilient import ResilientAutoFlyEnv

    scene, layout = scene_and_layout()
    base = AutoFlyEnv(scene, layout, sim_factory, map_path="/Game/AutoFly/Maps/S01", instance=0, target_pool=target_pool)
    return scene, ResilientAutoFlyEnv(base, instance=0, sim_root=tmp_path / "sim")


def _collect(tmp_path, model=None, n_keep=3, sim_factory=FakeSimulator, max_attempts=20, target_pool=None,
             target_name="orange cylinder"):
    from autofly_ue5.collect.collector import collect
    from autofly_ue5.dataset.raw import RawDatasetWriter

    scene, env = _env(tmp_path, sim_factory, target_pool)
    writer = RawDatasetWriter(tmp_path / "data", "pilot", scene=scene, provenance=PROVENANCE)
    summary = collect(model or _StraightAtTargetModel(), env, scene=scene, writer=writer, seed_base=SEED_BASE,
                      n_keep=n_keep, target_name=target_name, max_attempts=max_attempts,
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


def test_each_record_s_frame_and_state_come_from_the_one_function_the_evaluation_harness_shares(tmp_path, monkeypatch):
    # The harness (autofly_ue5/evaluation) hands a policy what a record holds before its action; both build it with
    # collector.record_inputs, so a policy is never scored on inputs that differ from the ones it trained on.
    import autofly_ue5.collect.collector as collector
    from autofly_ue5.dataset.state import autofly_state

    real = collector.record_inputs
    calls = []

    def spy(raw, setup):
        rgb, state = real(raw, setup)
        assert rgb is not raw.rgb and np.array_equal(rgb, raw.rgb), "a copy of the frame, not the simulator's buffer"
        assert np.array_equal(state, autofly_state(raw, setup.start, setup.target_xy_z)) and state.dtype == np.float32
        calls.append(raw.sim_time_ns)
        return rgb, state

    monkeypatch.setattr(collector, "record_inputs", spy)
    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    assert calls == np.load(root / entry["path"] / "steps.npz")["sim_time_ns"].tolist(), "one call per record, in order"


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
    assert not any(k.startswith("footprint") for k in record["provenance"]), "a pillar scene's provenance is as before"
    assert json.loads((root / "manifest.json").read_text())["counts"]["episodes"] == 0


def test_an_episode_the_footprint_rule_ended_is_rejected_with_the_tree_and_where_it_was_reached(tmp_path):
    # The rejects are where a tree scene's footprint endings land: each says which tree, how close, from where.
    from autofly_ue5.collect.collector import collect
    from autofly_ue5.dataset.raw import RawDatasetWriter
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.footprint_contact import OUTLINE_CONTACT_M
    from autofly_ue5.expert.resilient import ResilientAutoFlyEnv
    from tests.test_footprint_contact import AtTheNearestTree, recording_tree_factory, tree_scene_and_layout

    scene, layout = tree_scene_and_layout()
    sims = []
    base = AutoFlyEnv(scene, layout, recording_tree_factory(layout, sims), map_path="/Game/AutoFly/Maps/S99", instance=0)
    env = ResilientAutoFlyEnv(base, instance=0, sim_root=tmp_path / "sim")
    writer = RawDatasetWriter(tmp_path / "data", "trees", scene=scene, provenance=PROVENANCE)
    summary = collect(AtTheNearestTree(layout, sims), env, scene=scene, writer=writer, seed_base=SEED_BASE, n_keep=1,
                      target_name="orange cylinder", max_attempts=3, layout_sha256="test-layout")
    assert summary["kept"] == 0 and summary["rejected"] == 3
    rejects = [json.loads(p.read_text()) for p in sorted((tmp_path / "data" / "rejects" / "trees").glob("*.json"))]
    ended = [r for r in rejects if r["provenance"]["collision_source"] == "footprint"]
    assert ended, [r["provenance"]["collision_source"] for r in rejects]
    assets = {inst.tag: inst.asset for inst in layout.instances}
    for record in ended:
        contact = record["provenance"]["footprint_contact"]
        assert record["reason"] == "collision" and contact["rule"] == "footprint"
        assert assets[contact["tag"]] == contact["asset"] and contact["gap_m"] <= OUTLINE_CONTACT_M + 1e-9
        assert contact["shape"] == "outline", "the env's own default: the task's rule"
        assert {"bearing_deg", "drone_forward_m_s"} <= set(contact)
    assert all("footprint_contact" not in r["provenance"] for r in rejects if r["provenance"]["collision_source"] != "footprint")


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


def test_a_store_collected_for_the_merged_export_skips_its_own_and_still_passes(tmp_path, monkeypatch):
    # Plan 6 revision 1 item 1: one merged dataset is exported from every store, so a store's own RLDS copy would only
    # double the disk and the time. Skipped, the record says why; the validator still runs and still decides.
    import scripts.collect_dataset as cli

    monkeypatch.setattr(cli, "export_rlds", lambda *a, **k: pytest.fail("exported a store the merged export covers"))
    record = _pilot(tmp_path, export=False)
    assert record["rlds"] == {"skipped": "merged export (plan 6 revision 1)"}
    assert record["validation"]["pass"] and record["pass"] is True, record
    assert not (tmp_path / "data" / "pilot" / "1.0.0").exists()
    failing = _pilot(tmp_path, export=False, name="bad", n_episodes=3, max_attempts=2)  # 2 flights cannot keep 3
    assert failing["rlds"] == {"skipped": "merged export (plan 6 revision 1)"} and failing["pass"] is False


def test_the_pilot_cli_takes_no_export_and_refuses_a_read_back_of_nothing(tmp_path, monkeypatch, capsys):
    from tests.test_run_shape import run_root

    model = run_root(tmp_path, "s01_r2") / "final.zip"
    code, kwargs = _pilot_main_s01(tmp_path, monkeypatch, model, "--no-export")
    assert code == 0 and kwargs["export"] is False
    code, kwargs = _pilot_main_s01(tmp_path, monkeypatch, model)
    assert code == 0 and kwargs["export"] is True, "the default exports, as M3's pilot did"
    code, kwargs = _pilot_main_s01(tmp_path, monkeypatch, model, "--no-export", "--check-python", "/bin/true")
    assert code == 2 and kwargs is None and "--check-python" in capsys.readouterr().err


def _pilot_main_s01(tmp_path, monkeypatch, model, *argv):
    """The pilot CLI on s01 with run() replaced: (exit code, run()'s keyword arguments or None)."""
    import scripts.collect_dataset as cli

    monkeypatch.setattr(cli, "checkpoint_obs_config", lambda path: None)
    called = []
    monkeypatch.setattr(cli, "run", lambda **kwargs: called.append(kwargs) or {
        "status": "ok", "pass": True, "error": None, "collection": None, "validation": None})
    code = cli.main(["--model", str(model), "--scene-config", "scene_autofly_s01_fast.jsonc", "--name", "pilot",
                     "--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "m3.json"), "--sim-root",
                     str(tmp_path / "sim"), *argv])
    return code, (called[0] if called else None)


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


def test_the_validator_holds_the_provenance_s_target_name_to_the_record_s(tmp_path):
    # The rebalancing detector queries the manifest's name for each episode (scripts/detect_targets.py); with a target
    # pool the names differ per episode, so a provenance that names another target would make the detector look for the
    # wrong object.
    from autofly_ue5.validate.dataset import validate_dataset

    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    prov = root / "provenance" / f"{entry['id']}.json"
    _rewrite_json(prov, lambda p: p["target"].update(name="blue hatchback"))
    assert any("target name" in f for f in validate_dataset(root)["failures"])
    _rewrite_json(prov, lambda p: p["target"].update(name=entry["target_name"]))
    assert validate_dataset(root)["pass"]


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


# ------------------------------------------------------------------------------------------------------------------
# The target pool (plan 5 C5): a collection draws each episode's target and distractors from it, and the record says so.
# ------------------------------------------------------------------------------------------------------------------
class _SpawnRecordingSim(FakeSimulator):
    spawns: list = []

    def spawn(self, name, asset, pose, scale, material=None):
        _SpawnRecordingSim.spawns.append((name, asset, pose, tuple(scale), material))
        return super().spawn(name, asset, pose, scale, material)


def test_a_pool_collection_names_each_episode_s_own_target_and_records_its_entries(tmp_path):
    from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES
    from autofly_ue5.validate.dataset import validate_dataset
    from tests.test_dataset_targets import primitive_pool

    pool = primitive_pool(tmp_path)
    entries = {e.name: e for e in pool.entries}
    _SpawnRecordingSim.spawns = []
    summary, root = _collect(tmp_path, n_keep=4, target_pool=pool, target_name=None, sim_factory=_SpawnRecordingSim)
    assert summary["kept"] == 4 and validate_dataset(root)["pass"]
    manifest = json.loads((root / "manifest.json").read_text())
    names = [entry["target_name"] for entry in manifest["episodes"]]
    assert len(set(names)) > 1 and set(names) <= set(pool_names := [e.name for e in pool.for_split("train")]), names
    assert manifest["counts"]["by_target"] == {n: names.count(n) for n in set(names)}
    for entry in manifest["episodes"]:
        assert entry["instruction"] in {t.format(target=entry["target_name"], obstacle="white pillars")
                                        for t in INSTRUCTION_TEMPLATES}
        prov = json.loads((root / "provenance" / f"{entry['id']}.json").read_text())
        target = prov["target"]
        drawn = entries[target["name"]]
        assert target["name"] == entry["target_name"] and target["name_status"] == "pool"
        assert {k: target[k] for k in drawn.to_json()} == drawn.to_json()
        assert target["material_applied"] == (drawn.material or "mesh_default")
        assert target["spawned_as"].startswith("target") and len(target["xyz"]) == 3
        assert 3 <= len(prov["distractors"]) <= 5
        distractor_names = [d["name"] for d in prov["distractors"]]
        assert len(set(distractor_names)) == len(distractor_names) and target["name"] not in distractor_names
        for i, d in enumerate(prov["distractors"]):
            assert d["name"] in pool_names and {k: d[k] for k in entries[d["name"]].to_json()} == entries[d["name"]].to_json()
            assert d["spawned_as"].startswith(f"distractor_{i}") and len(d["xyz"]) == 3
    assets = {asset for _name, asset, _pose, _scale, _material in _SpawnRecordingSim.spawns}
    assert assets <= {"Cylinder", "Cube", "boulder_01"}, "short Unreal names, never a registry key or a package path"


def test_a_collection_names_its_target_one_way_only(tmp_path):
    # With a pool each episode's entry names it; without one --target-name does (M3). Both, or neither, is refused.
    from autofly_ue5.collect.collector import collect
    from autofly_ue5.dataset.raw import RawDatasetWriter
    from tests.test_dataset_targets import primitive_pool

    for target_pool, target_name in ((primitive_pool(tmp_path), "orange cylinder"), (None, None)):
        scene, env = _env(tmp_path, target_pool=target_pool)
        writer = RawDatasetWriter(tmp_path / "data", "pilot", scene=scene, provenance=PROVENANCE)
        with pytest.raises(ValueError, match="target"):
            collect(_StraightAtTargetModel(), env, scene=scene, writer=writer, seed_base=SEED_BASE, n_keep=1,
                    target_name=target_name, max_attempts=1)
        assert not (tmp_path / "data" / "pilot").exists()


def test_the_pilot_cli_with_a_pool_lists_its_names_in_the_card_and_records_the_pool(tmp_path):
    from tests.test_dataset_targets import SEEN_USABLE, primitive_pool, write_pool

    path = write_pool(tmp_path / "pool" / "targets.json")
    pool = primitive_pool(tmp_path / "pool")
    record = _pilot(tmp_path, n_episodes=2, targets=path)
    assert record["status"] == "ok" and record["validation"]["pass"], record["error"]
    # the pool as the scene draws from it: its names after the scene's exclusions (none here), and which table
    assert record["target_name"] is None and record["target_pool"] == pool.to_json("train", scene_id="s01")
    assert record["target_pool"]["scene_id"] == "s01" and record["target_pool"]["excluded"] == {}
    root = tmp_path / "data" / "pilot"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["card"]["target_names"] == {name: "pool" for name in SEEN_USABLE}
    for entry in manifest["episodes"]:
        prov = json.loads((root / "provenance" / f"{entry['id']}.json").read_text())
        assert prov["target_pool"] == record["target_pool"] and prov["target"]["name"] == entry["target_name"]


def test_each_pool_episode_records_its_first_use_hold_and_which_objects_were_stand_ins(tmp_path):
    # Plan 6 steps 2a/2b: whether the reset hovered for an object new to the session before the first record, and for
    # the target and every distractor whether it was a low-poly stand-in; the validator takes both.
    from autofly_ue5.expert.env import PREWARM_HOLD_STEPS
    from tests.test_dataset_targets import primitive_pool_entries, write_pool

    entries = [{**e, "stand_in": True} if e["name"] == "grey boulder" else e for e in primitive_pool_entries()]
    record = _pilot(tmp_path, n_episodes=3, targets=write_pool(tmp_path / "pool" / "targets.json", entries))
    assert record["status"] == "ok" and record["validation"]["pass"], (record["error"], record["validation"])
    root = tmp_path / "data" / "pilot"
    episodes = [json.loads((root / "provenance" / f"{e['id']}.json").read_text())
                for e in json.loads((root / "manifest.json").read_text())["episodes"]]
    episodes += [json.loads(p.read_text())["provenance"] for p in (tmp_path / "data" / "rejects" / "pilot").glob("*.json")]
    holds = [prov["prewarm_steps"] for prov in episodes]
    assert holds[0] == PREWARM_HOLD_STEPS and set(holds) <= {0, PREWARM_HOLD_STEPS}, holds
    for prov in episodes:
        for obj in (prov["target"], *prov["distractors"]):
            assert obj["stand_in"] is (obj["name"] == "grey boulder"), obj


def test_a_pool_free_episode_records_no_first_use_hold(tmp_path):
    record = _pilot(tmp_path, n_episodes=1)
    prov = json.loads(next((tmp_path / "data" / "pilot" / "provenance").glob("*.json")).read_text())
    assert record["status"] == "ok" and "prewarm_steps" not in prov and "stand_in" not in prov["target"]
    assert "first_frame_visibility" not in prov and "first_frame_visibility" not in record["collection"]


# Plan 6 revision 1, the "Targets" done check: every pool episode, kept or rejected, records whether its first recorded
# frame shows its target (validate/visibility.py). On s01's ground without its pillars, so nothing but a distractor can
# stand between the drone and its target.
def _open_field_collect(tmp_path, sim_factory, *, n_keep=3, max_attempts=8):
    import dataclasses

    from autofly_ue5.collect.collector import collect
    from autofly_ue5.dataset.raw import RawDatasetWriter
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.resilient import ResilientAutoFlyEnv
    from tests.test_dataset_targets import primitive_pool

    scene, layout = scene_and_layout()
    base = AutoFlyEnv(scene, dataclasses.replace(layout, instances=()), sim_factory, map_path="/Game/AutoFly/Maps/S01",
                      instance=0, target_pool=primitive_pool(tmp_path / "pool"))
    env = ResilientAutoFlyEnv(base, instance=0, sim_root=tmp_path / "sim")
    writer = RawDatasetWriter(tmp_path / "data", "field", scene=scene, provenance=PROVENANCE)
    summary = collect(_StraightAtTargetModel(), env, scene=scene, writer=writer, seed_base=SEED_BASE, n_keep=n_keep,
                      max_attempts=max_attempts, layout_sha256="test")
    root = tmp_path / "data"
    provenance = [json.loads(p.read_text()) for p in sorted((root / "field" / "provenance").glob("*.json"))]
    provenance += [json.loads(p.read_text())["provenance"] for p in sorted((root / "rejects" / "field").glob("*.json"))]
    return summary, provenance


def test_each_pool_episode_records_whether_its_first_frame_shows_its_target(tmp_path):
    from autofly_ue5.validate.visibility import STATUSES
    from tests.test_visibility import RenderingFake

    summary, provenance = _open_field_collect(tmp_path, RenderingFake)
    records = [prov["first_frame_visibility"] for prov in provenance]
    assert len(records) == summary["attempted"] >= 3
    for prov, record in zip(provenance, records):
        assert record["prewarm_steps"] == prov["prewarm_steps"] and record["status"] in STATUSES
        assert {"status", "range_m", "pixel", "depth_window_min_m", "prewarm_steps"} <= set(record)
        if record["status"] == "occluded":
            assert record["occluder"]["by"].startswith("distractor_"), record
    statuses = [record["status"] for record in records]
    assert "visible" in statuses and set(statuses) <= {"visible", "occluded"}, statuses
    assert summary["first_frame_visibility"] == {status: statuses.count(status) for status in STATUSES}


def test_a_first_frame_without_its_target_is_recorded_missing_and_said_so(tmp_path, capsys):
    summary, provenance = _open_field_collect(tmp_path, FakeSimulator)  # its depth frames show nothing at all
    statuses = [prov["first_frame_visibility"]["status"] for prov in provenance]
    assert "missing" in statuses and set(statuses) <= {"missing", "occluded"}, statuses
    assert summary["first_frame_visibility"]["missing"] == statuses.count("missing")
    assert capsys.readouterr().err.count("FIRST-FRAME MISSING") == statuses.count("missing")


def test_the_pilot_cli_lists_only_what_its_scene_draws_after_the_pool_s_exclusions(tmp_path):
    from tests.test_dataset_targets import WIDE_SEEN, wide_pool

    pool = wide_pool(tmp_path / "pool")  # s01 excludes its cylinders
    record = _pilot(tmp_path, n_episodes=2, targets=Path(pool.path))
    assert record["status"] == "ok" and record["validation"]["pass"], record["error"]
    names = [n for n in WIDE_SEEN if n not in ("orange cylinder", "grid pole")]
    assert record["target_pool"]["names"] == names and set(record["target_pool"]["excluded"]) == {"cylinder"}
    manifest = json.loads((tmp_path / "data" / "pilot" / "manifest.json").read_text())
    assert manifest["card"]["target_names"] == {name: "pool" for name in names}


def test_the_pilot_cli_refuses_a_table_that_leaves_its_scene_too_few_targets_before_launching(tmp_path, monkeypatch,
                                                                                            capsys):
    import scripts.collect_dataset as cli
    from tests.test_dataset_targets import wide_pool

    monkeypatch.setattr(cli, "checkpoint_obs_config", lambda path: None)  # the pool must be what refuses
    monkeypatch.setattr(cli, "run", lambda **kwargs: pytest.fail("main() went on to collect"))
    pool = wide_pool(tmp_path / "pool", {"version": 1, "scenes": {"s01": {"cylinder": "pillars", "sphere": "stones"}}})
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"x")
    assert cli.main(["--model", str(checkpoint), "--scene-config", "scene_autofly_s01_fast.jsonc", "--name", "pilot",
                     "--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "m3.json"), "--targets",
                     pool.path, "--sim-root", str(tmp_path / "sim")]) == 2
    assert "5 usable entries for split 'train' in scene 's01' after its exclusions" in capsys.readouterr().err
    assert not (tmp_path / "m3.json").exists() and not (tmp_path / "data").exists()


def test_the_pilot_cli_refuses_a_bad_pool_or_a_target_name_beside_one_before_launching_anything(tmp_path, monkeypatch,
                                                                                               capsys):
    import scripts.collect_dataset as cli
    from tests.test_dataset_targets import primitive_pool_entries, write_pool

    # The checkpoint is not a model: its reader is stubbed so each refusal below is the pool's, not the checkpoint's.
    monkeypatch.setattr(cli, "checkpoint_obs_config", lambda path: None)
    monkeypatch.setattr(cli, "run", lambda **kwargs: pytest.fail("main() went on to collect"))
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"x")
    common = ["--model", str(checkpoint), "--scene-config", "scene_autofly_s01_fast.jsonc", "--name", "pilot",
              "--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "m3.json"), "--sim-root",
              str(tmp_path / "sim")]
    small = write_pool(tmp_path / "small.json", [e for e in primitive_pool_entries() if e["name"] != "white box"])
    good = write_pool(tmp_path / "good.json")
    assert cli.main([*common, "--targets", str(small)]) == 2, "five seen entries cannot fill a train episode"
    assert "5 usable entries for split 'train'" in capsys.readouterr().err
    assert cli.main([*common, "--targets", str(tmp_path / "missing.json")]) == 2
    assert "missing.json" in capsys.readouterr().err
    assert cli.main([*common, "--targets", str(good), "--target-name", "orange cylinder"]) == 2
    assert "--target-name is the pool-free fallback" in capsys.readouterr().err
    assert not (tmp_path / "m3.json").exists() and not (tmp_path / "data").exists()


@pytest.mark.parametrize("shape", ["outline", "circle"])
def test_a_tree_scene_pilot_records_the_footprint_rule_it_collected_under(tmp_path, monkeypatch, shape):
    # A mesh scene's episodes end at the footprint rule, so which shape the expert flew under is part of what each
    # episode is: the record and every episode's provenance carry the rule (a pillar scene's stay M3's, key for key).
    import functools

    import scripts.collect_dataset as cli
    from autofly_ue5.expert.obs import ObsConfig
    from scripts import m2_gate
    from tests.test_footprint_contact import _tree_resolved, outline_fields_for, outline_rules, tree_scene_and_layout, wood

    _scene, layout = tree_scene_and_layout()
    expected = (outline_rules(outline_fields_for(layout), 11) if shape == "outline"
                else {"footprint": {"contact_m": 0.9, "obstacles": 11, "shape": "circle"}})
    monkeypatch.setattr(cli, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    monkeypatch.setattr(m2_gate, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a real model; the fake loader ignores it")
    record = cli.run(scene="s99", model_path=checkpoint, scene_config="scene_autofly_s02_fast.jsonc", name="trees",
                     n_episodes=1, max_attempts=4, instance=5, out_path=tmp_path / "m3.json", data_root=tmp_path / "data",
                     sim_factory=functools.partial(FakeSimulator, obstacles=wood(layout)),
                     load_model=lambda path: _StraightAtTargetModel(), obs_config_reader=lambda path: ObsConfig(),
                     sim_root=tmp_path / "sim", platform={"test": True}, footprint_shape=shape)
    assert record["error"] is None, record["error"]
    assert record["contact_rules"] == expected
    episodes = [json.loads(p.read_text())["provenance"] for p in (tmp_path / "data" / "rejects" / "trees").glob("*.json")]
    episodes += [json.loads(p.read_text()) for p in (tmp_path / "data" / "trees" / "provenance").glob("*.json")]
    assert episodes and all(prov["contact_rules"] == expected for prov in episodes)
    for prov in episodes:
        if "footprint_contact" in prov:
            assert prov["footprint_contact"]["shape"] == shape


def test_a_pillar_pilot_names_no_footprint_rule(tmp_path):
    record = _pilot(tmp_path, n_episodes=1)
    assert record["status"] == "ok" and "contact_rules" not in record
    prov = json.loads(next((tmp_path / "data" / "pilot" / "provenance").glob("*.json")).read_text())
    assert "contact_rules" not in prov


def test_the_pilot_cli_takes_the_shape_and_refuses_a_mesh_asset_without_a_field_before_launching(tmp_path, monkeypatch,
                                                                                                capsys):
    import dataclasses

    import autofly_ue5.expert.footprint_contact as footprint_module
    import scripts.collect_dataset as cli
    from tests.test_footprint_contact import LEAN, _tree_resolved, lean_tree, tree_scene_and_layout

    parsed = cli.build_arg_parser().parse_args(["--model", "m", "--scene-config", "c", "--name", "n"])
    assert parsed.footprint_shape is None and parsed.against_trained_shape is False, "no default: the run decides"
    registry, _fields, inst = lean_tree(tmp_path / "lean")
    scene, layout = tree_scene_and_layout()
    with_lean = dataclasses.replace(layout, instances=(*layout.instances, inst))
    monkeypatch.setattr(footprint_module, "load_registry", lambda: registry)
    monkeypatch.setattr(cli, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path, scene, with_lean))
    monkeypatch.setattr(cli, "checkpoint_obs_config", lambda path: None)
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"x")
    argv = ["--scene", "s99", "--model", str(checkpoint), "--scene-config", "scene_autofly_s02_fast.jsonc", "--name",
            "trees", "--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "m3.json"), "--sim-root",
            str(tmp_path / "sim"), "--footprint-shape", "outline"]
    assert cli.main(argv) == 2
    assert f"no footprint field for {LEAN}" in capsys.readouterr().err
    assert not (tmp_path / "data").exists() and not (tmp_path / "m3.json").exists()


def _pilot_main(tmp_path, monkeypatch, model, *argv):
    """The pilot CLI on the test tree scene with run() replaced: (exit code, run()'s keyword arguments or None)."""
    import scripts.collect_dataset as cli
    from tests.test_footprint_contact import _tree_resolved

    monkeypatch.setattr(cli, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    monkeypatch.setattr(cli, "checkpoint_obs_config", lambda path: None)
    called = []
    monkeypatch.setattr(cli, "run", lambda **kwargs: called.append(kwargs) or {
        "status": "ok", "pass": True, "error": None, "collection": None, "validation": None})
    code = cli.main(["--scene", "s99", "--model", str(model), "--scene-config", "scene_autofly_s02_fast.jsonc", "--name",
                     "trees", "--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "m3.json"), "--sim-root",
                     str(tmp_path / "sim"), *argv])
    return code, (called[0] if called else None)


def test_the_pilot_collects_under_the_shape_its_experts_run_trained_under(tmp_path, monkeypatch, capsys):
    # The expert's run names its footprint shape (expert/run_shape.py): a collection by s02_r1's circle-trained expert
    # rejects by the circle unless told, on purpose, otherwise.
    from tests.test_run_shape import LEGACY, run_root

    s02_r1 = run_root(tmp_path, "s02_r1", LEGACY)
    model = s02_r1 / "averaged" / "avg_1k_end.zip"
    code, kwargs = _pilot_main(tmp_path, monkeypatch, model)
    assert code == 0 and kwargs["footprint_shape"] == "circle" and kwargs["against_trained_shape"] is False
    code, kwargs = _pilot_main(tmp_path, monkeypatch, model, "--footprint-shape", "outline")
    assert code == 2 and kwargs is None and "--footprint-shape outline contradicts" in capsys.readouterr().err
    code, kwargs = _pilot_main(tmp_path, monkeypatch, model, "--footprint-shape", "outline", "--against-trained-shape")
    assert code == 0 and kwargs["footprint_shape"] == "outline" and kwargs["against_trained_shape"] is True
    loose = tmp_path / "loose.zip"
    loose.write_bytes(b"x")
    code, kwargs = _pilot_main(tmp_path, monkeypatch, loose)
    assert code == 2 and kwargs is None and "pass --footprint-shape outline or" in capsys.readouterr().err
    assert not (tmp_path / "data").exists() and not (tmp_path / "m3.json").exists()


def test_the_pilots_library_call_records_the_shape_its_expert_trained_under_when_it_scores_another(tmp_path, monkeypatch):
    import functools

    import scripts.collect_dataset as cli
    from autofly_ue5.expert.obs import ObsConfig
    from scripts import m2_gate
    from tests.test_footprint_contact import _tree_resolved, outline_fields_for, outline_rules, tree_scene_and_layout, wood
    from tests.test_run_shape import LEGACY, run_root

    _scene, layout = tree_scene_and_layout()
    monkeypatch.setattr(cli, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    monkeypatch.setattr(m2_gate, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    s02_r1 = run_root(tmp_path, "s02_r1", LEGACY)
    common = dict(scene="s99", scene_config="scene_autofly_s02_fast.jsonc", n_episodes=1, max_attempts=2, instance=5,
                  sim_factory=functools.partial(FakeSimulator, obstacles=wood(layout)),
                  load_model=lambda path: _StraightAtTargetModel(), obs_config_reader=lambda path: ObsConfig(),
                  sim_root=tmp_path / "sim", platform={"test": True})
    with pytest.raises(ValueError, match="contradicts"):
        cli.run(model_path=s02_r1 / "final.zip", name="a", out_path=tmp_path / "a.json", data_root=tmp_path / "data",
                footprint_shape="outline", **common)
    record = cli.run(model_path=s02_r1 / "final.zip", name="b", out_path=tmp_path / "b.json", data_root=tmp_path / "data",
                     footprint_shape="outline", against_trained_shape=True, **common)
    expected = outline_rules(outline_fields_for(layout), 11)
    expected["footprint"]["trained_shape"] = "circle"
    assert record["error"] is None and record["contact_rules"] == expected
    record = cli.run(model_path=s02_r1 / "final.zip", name="c", out_path=tmp_path / "c.json", data_root=tmp_path / "data",
                     **common)
    assert record["contact_rules"] == {"footprint": {"contact_m": 0.9, "obstacles": 11, "shape": "circle"}}
