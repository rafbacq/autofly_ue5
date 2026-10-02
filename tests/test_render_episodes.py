"""Offline tests for scripts/render_episodes.py: the recording simulator, frame bookkeeping, fault replay, the gate
comparison and the files it writes -- all against FakeSimulator and a scripted pilot, never a live simulator."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.render_episodes as rv
from autofly_ue5.expert.episode import sample_setup
from autofly_ue5.expert.obs import ObsConfig
from autofly_ue5.expert.seeds import EVAL_SEED_BASE
from autofly_ue5.expert.train import scene_and_layout, sha256_of
from autofly_ue5.scenes.model import Bounds
from autofly_ue5.sim.airsim_backend import CameraPoseError
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import Pose
from tests.test_expert_train import _FlakyFakeSimulator
from tests.test_m2_gate import _StraightAtTargetModel


def _checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "model.zip"
    path.write_bytes(b"not a real model; the scripted loader ignores it")
    return path


def _render(root: Path, episodes, *, gate_path=None, sim_factory=FakeSimulator):
    return rv.run(scene="s01", checkpoint_name="best_model", checkpoint_path=_checkpoint(root), episodes=episodes,
                  condition="deterministic", seed_base=EVAL_SEED_BASE, instance=3, out_dir=root / "viz",
                  gate_path=gate_path, sim_factory=sim_factory, load_model=lambda path: _StraightAtTargetModel(),
                  sim_root=root / "sim", fps=5.0, scale=1, hold_s=0.0, obs_config_reader=lambda path: ObsConfig())


def _scripted_gate_episodes(root: Path, n: int) -> list[dict]:
    """What the scripted pilot really scores on the first n gate seeds, measured the way the gate measures it."""
    from autofly_ue5.expert.evaluate import evaluate_policy_episodes
    from scripts.m2_gate import build_eval_env

    env = build_eval_env("s01", instance=4, sim_factory=FakeSimulator, sim_root=root / "sim_ref")
    try:
        return evaluate_policy_episodes(_StraightAtTargetModel(), env, n_episodes=n, seed_base=EVAL_SEED_BASE).per_episode
    finally:
        env.close()


def _write_gate(path: Path, checkpoint: Path, per_episode: list[dict], *, sha256: str | None = None) -> Path:
    gate = {"checkpoints": {"best_model": {"path": str(checkpoint), "sha256": sha256 or sha256_of(checkpoint),
                                           "deterministic": {"per_episode": per_episode}}}}
    path.write_text(json.dumps(gate))
    return path


def _video_frames(path: Path) -> tuple[int, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n, size


def test_world_to_px_puts_north_up_and_east_right():
    bounds = Bounds(x_min=-35.0, x_max=35.0, y_min=-35.0, y_max=35.0)
    assert rv.world_to_px(35.0, -35.0, bounds, 512, 8) == pytest.approx((8.0, 8.0)), "north-west corner is top-left"
    assert rv.world_to_px(-35.0, 35.0, bounds, 512, 8) == pytest.approx((504.0, 504.0)), "south-east is bottom-right"
    assert rv.world_to_px(0.0, 0.0, bounds, 512, 8) == pytest.approx((256.0, 256.0))
    col, row = rv.world_to_px(10.0, 0.0, bounds, 512, 8)
    assert col == pytest.approx(256.0) and row < 256.0, "+x (north) is up"


def test_depth_colouring_reads_no_hit_as_far():
    img = rv.depth_to_bgr(np.array([[0.5, np.inf], [30.0, 100.0]], dtype=np.float32))
    assert img.shape == (2, 2, 3) and img.dtype == np.uint8
    assert (img[0, 1] == img[1, 0]).all() and (img[1, 0] == img[1, 1]).all(), "sky and beyond the clip are far"
    assert not (img[0, 0] == img[1, 0]).all(), "a near obstacle must look different from far"


def test_recording_simulator_keeps_the_latest_observation_and_this_episodes_spawns():
    recorder = rv.FrameRecorder()
    sim = rv.RecordingSimulator(FakeSimulator(), recorder)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    start = Pose(1.0, 2.0, -2.0, 0.5)

    sim.reset(start)
    assert recorder.last.pose == start
    sim.spawn("target", "Cylinder", Pose(10.0, 0.0, -1.0, 0.0), (1.0, 1.0, 2.0), None)
    sim.spawn("distractor_0", "Cylinder", Pose(0.0, 10.0, -1.0, 0.0), (1.0, 1.0, 2.0), None)
    assert recorder.spawned == [("target", (10.0, 0.0, -1.0)), ("distractor_0", (0.0, 10.0, -1.0))]

    sim.command_velocity(1.0, 0.0, 0.0)
    sim.step()
    obs = sim.observe()
    assert recorder.last is obs and obs.pose.x > start.x
    assert sim.steps_taken > 0, "everything else is forwarded to the wrapped simulator"

    sim.reset(start)
    assert recorder.spawned == [], "a new episode starts with nothing spawned"


def test_recording_simulator_forwards_and_records_every_scene_object_move():
    # Without its own set_object_poses, __getattr__ would forward the call and the recorder would never know where
    # the movers were (s01d, spec §6.5).
    recorder = rv.FrameRecorder()
    fake = FakeSimulator(scene_objects={"obs_0003": (Pose(4.0, 0.0, -5.0, 0.0), 0.5)})
    sim = rv.RecordingSimulator(fake, recorder)
    sim.launch("/Game/AutoFly/Maps/S01", 0)

    sim.set_object_poses({"obs_0003": Pose(5.0, 1.0, -5.0, 0.0)})
    assert fake.scene_object_poses()["obs_0003"] == Pose(5.0, 1.0, -5.0, 0.0)
    assert recorder.object_poses == {"obs_0003": (5.0, 1.0, -5.0)}
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    assert recorder.object_poses == {"obs_0003": (5.0, 1.0, -5.0)}, "a reset does not move the level's objects"


def test_capture_frame_refuses_an_observation_that_is_not_the_steps_own():
    recorder = rv.FrameRecorder()
    fake = FakeSimulator()
    fake.launch("/Game/AutoFly/Maps/S01", 0)
    recorder.last = fake.reset(Pose(0.0, 0.0, -2.0, 0.0))
    info = {"pose": [5.0, 0.0, -2.0, 0.0], "final_distance_m": 1.0, "bearing_deg": 0.0}
    with pytest.raises(RuntimeError, match="wrong frame"):
        rv.capture_frame(recorder, info, step=1, action=None, reward=0.0)


def test_renders_gate_episodes_and_compares_them_with_the_gate(tmp_path):
    truth = _scripted_gate_episodes(tmp_path, 2)
    wrong = dict(truth[1], outcome="collision", steps=3)
    gate = _write_gate(tmp_path / "gate.json", _checkpoint(tmp_path), [truth[0], wrong])

    summary = _render(tmp_path, [0, 1], gate_path=gate)

    assert summary["status"] == "ok", summary["error"]
    assert summary["checkpoint"]["matches_gate_sha256"] is True
    first, second = summary["episodes"]
    assert (first["seed"], first["outcome"], first["steps"]) == (EVAL_SEED_BASE, truth[0]["outcome"], truth[0]["steps"])
    assert first["matches_gate"] is True and first["final_pose_offset_m"] == pytest.approx(0.0, abs=1e-6)
    assert second["matches_gate"] is False, "a replay that differs from the gate is recorded, not hidden"
    assert second["gate"]["outcome"] == "collision" and second["gate"]["steps"] == 3
    assert (summary["n_compared"], summary["n_matching_gate"]) == (2, 1)

    scene, layout = scene_and_layout("s01")
    out = tmp_path / "viz"
    for ep in summary["episodes"]:
        n_frames, size = _video_frames(out / ep["video"])
        assert n_frames == ep["steps"] + 1, "one frame for the reset plus one per step"
        assert size == rv.frame_size(1)
        assert ep["video_codec"] in ("h264", "mp4v")
        assert cv2.imread(str(out / ep["map"])) is not None
        assert len(ep["trajectory"]) == ep["steps"] + 1 and len(ep["commands"]) == ep["steps"] + 1
        assert ep["commands"][0] is None and len(ep["commands"][1]) == 3
        setup = sample_setup(scene, layout, np.random.default_rng(ep["seed"]))
        assert ep["target_xy_z"] == pytest.approx(list(setup.target_xy_z)), "the map's target is the one spawned"
        assert len(ep["distractors"]) == len(setup.distractors)
        assert ep["trajectory"][0][:2] == pytest.approx([setup.start.x, setup.start.y], abs=1e-6)
    assert cv2.imread(str(out / "overview.png")) is not None
    assert json.loads((out / "summary.json").read_text())["n_matching_gate"] == 1, "the record on disk is complete"
    assert (tmp_path / "sim" / "inst3" / "client.log").is_file(), "client logs go under the given sim_root"


def test_a_checkpoint_other_than_the_gated_one_is_flagged(tmp_path):
    truth = _scripted_gate_episodes(tmp_path, 1)
    gate = _write_gate(tmp_path / "gate.json", _checkpoint(tmp_path), truth, sha256="0" * 64)
    summary = _render(tmp_path, [0], gate_path=gate)
    assert summary["status"] == "ok", summary["error"]
    assert summary["checkpoint"]["matches_gate_sha256"] is False


def test_without_a_gate_record_nothing_is_compared(tmp_path):
    summary = _render(tmp_path, [0], gate_path=tmp_path / "no_gate.json")
    assert summary["status"] == "ok", summary["error"]
    assert summary["checkpoint"]["matches_gate_sha256"] is None
    assert summary["episodes"][0]["matches_gate"] is None and summary["n_compared"] == 0


def test_a_backend_fault_mid_episode_is_replayed_and_leaves_no_frames_behind(tmp_path):
    # Step call #1 is AutoFlyEnv.reset()'s own render step and #2 the first policy step, so #3 faults mid-episode
    # after one frame has been captured: that attempt's frames must be dropped and the seed flown again.
    clean = _render(tmp_path / "clean", [0])
    flaky = _render(tmp_path / "flaky", [0],
                    sim_factory=lambda: _FlakyFakeSimulator(fail_on_step_calls=(3,), error=CameraPoseError))

    assert flaky["status"] == "ok", flaky["error"]
    ep, ref = flaky["episodes"][0], clean["episodes"][0]
    assert ep["retries"] == 1 and ref["retries"] == 0
    assert (ep["outcome"], ep["steps"]) == (ref["outcome"], ref["steps"])
    np.testing.assert_allclose(np.array(ep["trajectory"]), np.array(ref["trajectory"]))
    assert _video_frames(tmp_path / "flaky" / "viz" / ep["video"])[0] == ep["steps"] + 1
    assert flaky["backend_faults"]["fault_counts"]["CameraPoseError"] == 1


def test_refuses_to_overwrite_an_earlier_render(tmp_path):
    (tmp_path / "viz").mkdir()
    (tmp_path / "viz" / "summary.json").write_text("{}")

    def _must_not_launch():
        raise AssertionError("a refused run must not build a simulator")

    with pytest.raises(FileExistsError):
        _render(tmp_path, [0], sim_factory=_must_not_launch)
    assert (tmp_path / "viz" / "summary.json").read_text() == "{}"


def test_rejects_repeated_or_negative_episode_indices(tmp_path):
    with pytest.raises(ValueError):
        _render(tmp_path, [1, 1])
    with pytest.raises(ValueError):
        _render(tmp_path, [-1])


def test_cli_resolves_the_checkpoint_under_the_run_root():
    args = rv.build_arg_parser().parse_args(
        ["--run-root", "runs/expert/x", "--checkpoint", "final", "--episodes", "1", "2", "--out-dir", "o"])
    assert rv.checkpoint_path(args) == Path("runs/expert/x/final.zip")
    assert args.episodes == [1, 2] and args.condition == "deterministic"
    with pytest.raises(SystemExit):
        rv.build_arg_parser().parse_args(["--out-dir", "o"])
