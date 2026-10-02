"""Training, the gate, the renderer and the throughput script on a dynamic scene (s01d), and the evidence rules that keep
a smoke run from touching a committed record (CLAUDE.md). Offline: FakeSimulator only, every simulator record under
tmp_path, and any path to a real launch stubbed to fail."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from autofly_ue5.expert.obs import ObsConfig
from autofly_ue5.paths import ROOT
from autofly_ue5.sim.fake import FakeSimulator
from tests.test_dynamic_env import dynamic_scene, make_dynamic_env, scene_objects
from tests.test_m2_gate import _StraightAtTargetModel

STATIC_OBS = ObsConfig(1, "float32")
STACKED_OBS = ObsConfig(3, "float16")


def _no_launch(monkeypatch):
    from autofly_ue5.sim import airsim_backend

    def no_launch(self, map_path, instance):
        raise AssertionError("tried to launch a real simulator")

    monkeypatch.setattr(airsim_backend.ProjectAirSimSimulator, "launch", no_launch)


def _dynamic_fake_factory():
    _scene, layout = dynamic_scene()
    return functools.partial(FakeSimulator, scene_objects=scene_objects(layout))


# --------------------------------------------------------------------------------------------------------
# Evidence paths.
# --------------------------------------------------------------------------------------------------------
def test_each_scene_has_its_own_default_evidence_and_others_must_name_one():
    from autofly_ue5.evidence import GATES_DIR, default_evidence_path

    assert default_evidence_path("s01", "train") == GATES_DIR / "m2_train.json"
    assert default_evidence_path("s01d", "gate") == GATES_DIR / "m2d_gate.json"
    assert default_evidence_path("s01d", "instances") == GATES_DIR / "m2d_instances.json"
    with pytest.raises(ValueError, match="--out"):
        default_evidence_path("s02", "train")


def test_a_committed_record_is_never_written_over(tmp_path):
    from autofly_ue5.evidence import GATES_DIR, refuse_existing_evidence

    with pytest.raises(FileExistsError, match="m2_gate.json"):
        refuse_existing_evidence(GATES_DIR / "m2_gate.json")
    refuse_existing_evidence(GATES_DIR / "m2d_no_such_record.json")  # a new record is fine
    (tmp_path / "out.json").write_text("{}")
    refuse_existing_evidence(tmp_path / "out.json")  # outside docs/gates: a scratch file may be replaced


def _scratch_gates(tmp_path, monkeypatch) -> Path:
    """A stand-in docs/gates/ holding a copy of each committed m2 record, so these tests exercise the refusal without
    ever pointing a run at the real evidence -- even before the refusal exists (a RED run once wrote over it)."""
    from autofly_ue5 import evidence

    gates = tmp_path / "gates"
    gates.mkdir()
    for name in ("m2_train.json", "m2_gate.json", "m2_instances.json"):
        (gates / name).write_text('{"recorded": true}')
    monkeypatch.setattr(evidence, "GATES_DIR", gates)
    return gates


@pytest.mark.parametrize("explicit_out", [False, True])
def test_training_refuses_to_overwrite_a_committed_record_before_claiming_anything(tmp_path, monkeypatch, explicit_out):
    from autofly_ue5.expert import train

    gates = _scratch_gates(tmp_path, monkeypatch)
    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "make_vec_env", lambda *a, **k: pytest.fail("training started"))
    argv = ["--scene", "s01"] + (["--out", str(gates / "m2_train.json")] if explicit_out else [])
    code = train.main([*argv, "--run-root", str(tmp_path / "run"), "--device", "cpu", "--sim-root", str(tmp_path / "sim")])
    assert code == 2 and not (tmp_path / "run").exists()
    assert (gates / "m2_train.json").read_text() == '{"recorded": true}'


def test_the_gate_and_the_throughput_script_refuse_too(tmp_path, monkeypatch):
    from scripts import m2_gate, measure_instances

    gates = _scratch_gates(tmp_path, monkeypatch)
    _no_launch(monkeypatch)
    monkeypatch.setattr(m2_gate, "run", lambda **k: pytest.fail("the gate ran"))
    monkeypatch.setattr(measure_instances, "run", lambda *a, **k: pytest.fail("the measurement ran"))
    assert m2_gate.main(["--scene", "s01", "--run-root", str(tmp_path)]) == 2
    assert m2_gate.main(["--scene", "s01", "--out", str(gates / "m2_gate.json")]) == 2
    assert measure_instances.main(["--scene", "s01"]) == 2
    assert all(p.read_text() == '{"recorded": true}' for p in gates.iterdir())


def test_the_real_evidence_directory_is_docs_gates():
    from autofly_ue5.evidence import GATES_DIR

    assert GATES_DIR == ROOT / "docs" / "gates"


# --------------------------------------------------------------------------------------------------------
# Run identity, and the host preflight.
# --------------------------------------------------------------------------------------------------------
def _identity(**changes) -> dict:
    identity = {"scene": "s01d", "scene_sha256": "a" * 64, "base_scene": "s01", "base_layout_sha256": "b" * 64,
                "obs_config": STACKED_OBS.to_json(), "dynamic": {"count": [8, 12]}}
    identity.update(changes)
    return identity


def _checkpoint(run_root: Path) -> None:
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints" / "rl_model_10000_steps.zip").write_text("model")
    (run_root / "checkpoints" / "rl_model_replay_buffer_10000_steps.pkl").write_text("buffer")


def test_a_resume_must_fly_the_same_scene_with_the_same_observation(tmp_path):
    from autofly_ue5.expert.train import prepare_run_root

    assert prepare_run_root(tmp_path, resume=False, reward_version="v2", seed=0, identity=_identity()) == 0
    _checkpoint(tmp_path)
    assert prepare_run_root(tmp_path, resume=True, reward_version="v2", seed=0, identity=_identity()) == 1
    recorded = json.loads((tmp_path / "sessions.json").read_text())["sessions"]
    assert recorded[0]["identity"] == _identity()
    for change in ({"obs_config": STATIC_OBS.to_json()}, {"scene": "s01"}, {"dynamic": {"count": [4, 6]}},
                   {"base_layout_sha256": "c" * 64}):
        with pytest.raises(RuntimeError, match=next(iter(change))):
            prepare_run_root(tmp_path, resume=True, reward_version="v2", seed=0, identity=_identity(**change))
    assert len(json.loads((tmp_path / "sessions.json").read_text())["sessions"]) == 2, "a refusal records nothing"


def test_a_run_from_before_identities_resumes_only_as_static_one_frame(tmp_path):
    from autofly_ue5.expert.train import prepare_run_root

    # runs/expert/s01_r2/sessions.json, as M2 run 2 wrote it.
    (tmp_path / "sessions.json").write_text(json.dumps({"sessions": [
        {"index": 0, "started": "2026-09-25 19:14:45", "resume": False, "reward_version": "v2", "seed": 0}]}))
    _checkpoint(tmp_path)
    static = _identity(scene="s01", obs_config=STATIC_OBS.to_json(), dynamic=None)
    with pytest.raises(RuntimeError, match="obs_config"):
        prepare_run_root(tmp_path, resume=True, reward_version="v2", seed=0, identity=_identity())
    assert prepare_run_root(tmp_path, resume=True, reward_version="v2", seed=0, identity=static) == 1


def test_replay_buffer_arithmetic():
    from autofly_ue5.expert.train import replay_buffer_bytes

    # 150k transitions x (obs + next_obs): one float32 frame 8.47 GB, three float16 frames 12.70 GB.
    assert replay_buffer_bytes(150_000, STATIC_OBS) == pytest.approx(8.47e9, rel=0.005)
    assert replay_buffer_bytes(150_000, STACKED_OBS) == pytest.approx(12.70e9, rel=0.005)
    assert replay_buffer_bytes(150_000, ObsConfig(3, "float32")) == pytest.approx(25.4e9, rel=0.005)


def test_the_preflight_refuses_a_buffer_the_host_cannot_hold(tmp_path, monkeypatch):
    from autofly_ue5.expert import train

    monkeypatch.setattr(train, "available_ram_bytes", lambda: 20 * 2**30)
    report = train.host_preflight(buffer_bytes=2 * 2**30, run_root=tmp_path)
    assert report["ok"] and report["available_ram_bytes"] == 20 * 2**30 and report["free_disk_bytes"] > 0
    with pytest.raises(RuntimeError, match="RAM"):
        train.host_preflight(buffer_bytes=12 * 2**30, run_root=tmp_path)


# --------------------------------------------------------------------------------------------------------
# Training s01d end to end on the fake.
# --------------------------------------------------------------------------------------------------------
def test_training_on_s01d_stacks_depth_records_its_identity_and_counts_mover_collisions(tmp_path, monkeypatch):
    from autofly_ue5.expert import train

    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(): _dynamic_fake_factory())
    out = tmp_path / "train.json"
    code = train.main(["--scene", "s01d", "--run-root", str(tmp_path / "run"), "--out", str(out), "--device", "cpu",
                       "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "60", "--learning-starts", "20",
                       "--buffer-size", "300", "--batch-size", "8", "--eval-freq", "40", "--eval-episodes", "1",
                       "--checkpoint-freq", "50"])
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["scene"] == "s01d" and record["map"] == "/Game/AutoFly/Maps/S01"
    assert record["obs_config"] == STACKED_OBS.to_json()
    identity = record["identity"]
    assert identity["scene"] == "s01d" and identity["base_scene"] == "s01" and identity["dynamic"]["count"] == [8, 12]
    assert record["host"]["ok"] and record["host"]["replay_buffer_bytes"] > 0
    assert "collision_sources" in record
    sessions = json.loads((tmp_path / "run" / "sessions.json").read_text())["sessions"]
    assert sessions[0]["identity"] == identity
    assert (tmp_path / "run" / "final.zip").is_file()
    assert (tmp_path / "sim" / "inst0" / "client.log").is_file(), "every simulator record under --sim-root"


def test_the_depth_stack_can_be_set_on_the_command_line(tmp_path, monkeypatch):
    from autofly_ue5.expert import train

    _no_launch(monkeypatch)
    monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(): _dynamic_fake_factory())
    out = tmp_path / "train.json"
    train.main(["--scene", "s01d", "--depth-frames", "1", "--run-root", str(tmp_path / "run"), "--out", str(out),
                "--device", "cpu", "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "10",
                "--learning-starts", "100", "--buffer-size", "100", "--eval-freq", "1000"])
    assert json.loads(out.read_text())["obs_config"] == STATIC_OBS.to_json()


def test_vec_envs_take_the_observation_config(tmp_path):
    from autofly_ue5.expert.vec import make_vec_env

    scene, layout = dynamic_scene()
    vec = make_vec_env(scene, layout, 1, map_path="/x", monitor_dir=tmp_path / "mon", sim_factory=_dynamic_fake_factory(),
                       sim_root=tmp_path / "sim", obs_config=ObsConfig(2, "float16"))
    assert vec.observation_space["depth"].shape == (2, 84, 84)
    vec.close()


# --------------------------------------------------------------------------------------------------------
# The gate.
# --------------------------------------------------------------------------------------------------------
def test_a_checkpoints_observation_config_is_read_from_its_zip(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from autofly_ue5.expert.train import build_model
    from scripts.m2_gate import checkpoint_obs_config

    model = build_model(DummyVecEnv([make_dynamic_env]), device="cpu", buffer_size=50, learning_starts=100, verbose=0)
    model.save(tmp_path / "m.zip")
    assert checkpoint_obs_config(tmp_path / "m.zip") == STACKED_OBS


def _gate(tmp_path, scene, model_paths, *, reader, sim_factory, n=2):
    from autofly_ue5.expert.seeds import EVAL_SEED_BASE
    from scripts.m2_gate import run

    return run(scene=scene, model_paths=model_paths, conditions=["deterministic"], n_episodes=n, seed_base=EVAL_SEED_BASE,
               instance=6, out_path=tmp_path / "gate.json", sim_factory=sim_factory,
               load_model=lambda path: _StraightAtTargetModel(), sim_root=tmp_path / "sim", obs_config_reader=reader,
               instances_path=tmp_path / "none.json")


def _fake_checkpoints(tmp_path, *names) -> dict:
    paths = {}
    for name in names:
        paths[name] = tmp_path / f"{name}.zip"
        paths[name].write_bytes(b"scripted")
    return paths


def test_the_gate_refuses_checkpoints_that_see_different_observations_before_launching(tmp_path):
    def never(*_a, **_k):
        raise AssertionError("the gate launched a simulator for checkpoints it cannot compare")

    configs = {"best_model": STACKED_OBS, "final": STATIC_OBS}
    gate = _gate(tmp_path, "s01d", _fake_checkpoints(tmp_path, "best_model", "final"),
                 reader=lambda p: configs[p.stem], sim_factory=never)
    assert gate["status"] == "failed" and "disagree" in gate["error"] and gate["pass"] is False


def test_the_s01d_gate_flies_the_checkpoints_observation_and_records_mover_outcomes(tmp_path):
    gate = _gate(tmp_path, "s01d", _fake_checkpoints(tmp_path, "best_model"), reader=lambda p: STACKED_OBS,
                 sim_factory=_dynamic_fake_factory(), n=3)
    assert gate["status"] == "ok" and gate["scene"] == "s01d" and gate["obs_config"] == STACKED_OBS.to_json()
    det = gate["checkpoints"]["best_model"]["deterministic"]
    assert det["n_episodes"] == 3 and "collision_sources" in det
    for episode in det["per_episode"]:
        assert episode["n_movers"] >= 2 and "collision_source" in episode and "mover_in_view" in episode


def test_the_gate_takes_its_throughput_projection_from_its_own_scene(tmp_path):
    from scripts.m2_gate import default_instances_path

    assert default_instances_path("s01") == ROOT / "docs" / "gates" / "m2_instances.json"
    assert default_instances_path("s01d") == ROOT / "docs" / "gates" / "m2d_instances.json"
    assert default_instances_path("s02") is None


# --------------------------------------------------------------------------------------------------------
# The renderer.
# --------------------------------------------------------------------------------------------------------
def test_the_renderer_refuses_a_gate_record_of_another_scene(tmp_path):
    import scripts.render_episodes as rv

    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"scene": "s01", "checkpoints": {}}))
    checkpoint = tmp_path / "m.zip"
    checkpoint.write_bytes(b"scripted")
    with pytest.raises(ValueError, match="s01"):
        rv.run(scene="s01d", checkpoint_name="best_model", checkpoint_path=checkpoint, episodes=[0],
               condition="deterministic", seed_base=100_000_000, instance=7, out_dir=tmp_path / "viz", gate_path=gate,
               sim_factory=_dynamic_fake_factory(), load_model=lambda p: _StraightAtTargetModel(),
               sim_root=tmp_path / "sim", obs_config_reader=lambda p: STACKED_OBS)


def test_the_renderer_draws_s01d_movers_frame_by_frame(tmp_path):
    import scripts.render_episodes as rv

    checkpoint = tmp_path / "m.zip"
    checkpoint.write_bytes(b"scripted")
    summary = rv.run(scene="s01d", checkpoint_name="best_model", checkpoint_path=checkpoint, episodes=[0, 1],
                     condition="deterministic", seed_base=100_000_000, instance=7, out_dir=tmp_path / "viz",
                     gate_path=None, sim_factory=_dynamic_fake_factory(), load_model=lambda p: _StraightAtTargetModel(),
                     sim_root=tmp_path / "sim", obs_config_reader=lambda p: STACKED_OBS, fps=5.0, scale=1, hold_s=0.0)
    assert summary["status"] == "ok", summary["error"]
    for episode in summary["episodes"]:
        routes = episode["mover_routes"]
        assert len(routes) >= 2 and {"tag", "kind", "speed_m_s"} <= set(routes[0])
        assert len(episode["mover_positions"]) == len(episode["trajectory"])
        assert all(len(frame) == len(routes) for frame in episode["mover_positions"])
        assert "collision_source" in episode
        assert (tmp_path / "viz" / episode["video"]).is_file() and (tmp_path / "viz" / episode["map"]).is_file()
    first, last = summary["episodes"][0]["mover_positions"][0], summary["episodes"][0]["mover_positions"][-1]
    assert first != last, "the movers moved during the episode"


def test_a_static_render_still_works_without_movers(tmp_path):
    import scripts.render_episodes as rv

    checkpoint = tmp_path / "m.zip"
    checkpoint.write_bytes(b"scripted")
    summary = rv.run(scene="s01", checkpoint_name="best_model", checkpoint_path=checkpoint, episodes=[0],
                     condition="deterministic", seed_base=100_000_000, instance=7, out_dir=tmp_path / "viz",
                     gate_path=None, sim_factory=FakeSimulator, load_model=lambda p: _StraightAtTargetModel(),
                     sim_root=tmp_path / "sim", obs_config_reader=lambda p: STATIC_OBS, fps=5.0, scale=1, hold_s=0.0)
    assert summary["status"] == "ok" and summary["episodes"][0]["mover_routes"] == []


# --------------------------------------------------------------------------------------------------------
# Throughput.
# --------------------------------------------------------------------------------------------------------
def test_throughput_is_measured_on_the_scene_asked_for(tmp_path, monkeypatch):
    import scripts.measure_instances as mi

    seen = {}

    def fake_measure_n(n, scene, layout, *args, map_path, sim_factory, **kwargs):
        seen.update(scene=scene.id, map=map_path, factory=sim_factory)
        return {"env_steps_per_s_total": 1.0, "env_steps_per_s_per_instance": 1.0, "vram_mib": 100}

    monkeypatch.setattr(mi, "measure_n", fake_measure_n)
    monkeypatch.setattr(mi, "sweep_orphaned_instances", lambda *a, **k: [])
    monkeypatch.setattr(mi, "gpu_memory_mib", lambda: (300, 24564))
    monkeypatch.setattr(mi, "audit_engine_faults", lambda **k: {"ok": True})
    gate = mi.run(tmp_path / "instances.json", candidate_ns=(1,), scene="s01d", warmup_s=0, timed_s=0)
    assert seen["scene"] == "s01d" and seen["map"] == "/Game/AutoFly/Maps/S01"
    assert seen["factory"].keywords["movable_objects"][:2] == ("obs_0000", "obs_0001")
    assert gate["scene"] == "s01d" and gate["scene_config"]["file"] == "scene_autofly_s01.jsonc"


# --------------------------------------------------------------------------------------------------------
# The dynamic scene's report (scripts/build_scenes.py).
# --------------------------------------------------------------------------------------------------------
def test_building_a_dynamic_scene_reports_its_movers_on_training_seeds(tmp_path):
    from autofly_ue5.expert.seeds import EVAL_SEED_BASE, worker_seed_base
    from scripts.build_scenes import dynamic_report

    report = dynamic_report("s01d", n_seeds=12)
    assert report["scene"] == "s01d" and report["base_scene"] == "s01" and report["n_seeds"] == 12
    assert report["seeds"] == [worker_seed_base(0), worker_seed_base(0) + 11]
    assert report["seeds"][1] < EVAL_SEED_BASE, "statistics never come from the gate's episodes"
    placed = report["movers_placed"]
    assert 2 <= placed["min"] <= placed["mean"] <= placed["max"] <= 12 and sum(placed["histogram"].values()) == 12
    assert set(report["guard"]) <= {"ok", "repaired", "static_unreachable"}
    assert report["path_ratio"]["max"] <= 1.2 + 1e-9
    assert report["reset_ms"]["median"] > 0 and set(report["route_kinds"]) <= {"pingpong", "orbit"}


def test_build_scenes_writes_the_dynamic_report_and_no_level(tmp_path):
    from scripts.build_scenes import main

    assert main(["scenes/s01d_moving_pillars.json", "--out-dir", str(tmp_path), "--report-seeds", "5"]) == 0
    assert json.loads((tmp_path / "s01d.dynamic_report.json").read_text())["n_seeds"] == 5
    assert not (tmp_path / "s01d.level.json").exists() and not (tmp_path / "s01d.layout.json").exists()
