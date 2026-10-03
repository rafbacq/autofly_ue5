"""M3's collector, raw store and validator (spec §9-§11), against FakeSimulator and a scripted pilot: success-only
episodes in time order with AutoFly's four fields, rejects with their reason, provenance per episode, faults replayed,
and a validator that passes a good store and catches a bad one."""

from __future__ import annotations

import json
import math

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
