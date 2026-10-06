"""Dataset rebalancing (spec §10.3; AutoFly App. A.2.4): scripts/detect_targets.py scores every frame for its episode's
target, autofly_ue5/dataset/rebalance.py turns the scores into phases and resampling weights, and
scripts/rebalance_dataset.py writes data/<name>/rebalance.json. All offline: a fake detector stands in for Grounding
DINO, and the raw store comes from the collector against FakeSimulator (tests/test_collect.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.test_collect import _collect


class _MeanPixelDetector:
    """Scores a frame by its mean pixel value, so a test can recompute every score from the PNG itself and check that
    the right frames were scored in the right order. Rejects what the real adapter would mis-handle: a prompt that does
    not follow Grounding DINO's convention, or an image that is not the store's 256 x 256 RGB."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.frames_scored = 0
        self._fail_on_call = fail_on_call

    def identity(self) -> dict:
        return {"model": "fake/mean-pixel", "revision": "0"}

    def run_info(self) -> dict:
        return {"device": "cpu"}

    def detect(self, images, prompt: str):
        self.calls += 1
        if self._fail_on_call is not None and self.calls == self._fail_on_call:
            raise RuntimeError("the detector fell over")
        assert prompt == prompt.lower().strip() and prompt.endswith("."), f"prompt {prompt!r} breaks the convention"
        out = []
        for image in images:
            assert isinstance(image, Image.Image) and image.mode == "RGB" and image.size == (256, 256)
            out.append((float(np.asarray(image).mean() / 255.0), [0.5, 0.5, 0.1, 0.2]))
        self.frames_scored += len(images)
        return out


def _expected_scores(root: Path, entry: dict) -> list[float]:
    frames = sorted((root / entry["path"] / "frames").glob("*.png"))
    return [float(np.asarray(Image.open(f).convert("RGB")).mean() / 255.0) for f in frames]


# ------------------------------------------------------------------------------------------------------------------
# scripts/detect_targets.py
# ------------------------------------------------------------------------------------------------------------------


def test_the_prompt_follows_grounding_dino_s_convention():
    from scripts.detect_targets import prompt_for

    assert prompt_for("Orange Cylinder") == "orange cylinder."
    assert prompt_for("  blue hatchback ") == "blue hatchback."
    with pytest.raises(ValueError):
        prompt_for("")
    with pytest.raises(ValueError):
        prompt_for("orange cylinder. white pillar")  # a period separates phrases: that would be two queries


def test_every_frame_of_every_episode_is_scored_in_order_for_its_own_target(tmp_path):
    from scripts.detect_targets import FORMAT, run

    _summary, root = _collect(tmp_path, n_keep=3)
    out = tmp_path / "detections.json"
    detector = _MeanPixelDetector()
    doc = run(root, out, detector, batch=4)
    manifest = json.loads((root / "manifest.json").read_text())
    saved = json.loads(out.read_text())
    assert saved == doc
    assert doc["format"] == FORMAT and doc["dataset"] == "pilot" and doc["dataset_created"] == manifest["created"]
    assert doc["complete"] is True and doc["episodes_total"] == 3 and doc["detector"]["model"] == "fake/mean-pixel"
    assert detector.frames_scored == manifest["counts"]["records"]
    (run_record,) = doc["runs"]
    assert run_record["device"] == "cpu" and run_record["frames_scored"] == manifest["counts"]["records"]
    assert run_record["episodes_scored"] == 3 and run_record["finished"] >= run_record["started"]
    for entry in manifest["episodes"]:
        ep = doc["episodes"][entry["id"]]
        assert ep["query"] == "orange cylinder" and ep["prompt"] == "orange cylinder."
        assert len(ep["scores"]) == len(ep["boxes"]) == entry["steps"]
        assert ep["scores"] == pytest.approx(_expected_scores(root, entry), abs=1e-6)
        assert all(0.0 <= s <= 1.0 for s in ep["scores"])


def test_a_limited_run_is_resumed_without_rescoring_and_another_detector_or_store_is_refused(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=3)
    out = tmp_path / "detections.json"
    first = _MeanPixelDetector()
    doc = run(root, out, first, episodes=2)
    assert doc["complete"] is False and len(doc["episodes"]) == 2
    manifest = json.loads((root / "manifest.json").read_text())
    assert first.frames_scored == sum(e["steps"] for e in manifest["episodes"][:2])

    second = _MeanPixelDetector()
    doc = run(root, out, second)
    assert doc["complete"] is True and len(doc["episodes"]) == 3
    assert second.frames_scored == manifest["episodes"][2]["steps"], "the two scored episodes were kept, not redone"

    third = _MeanPixelDetector()
    assert run(root, out, third)["complete"] is True and third.frames_scored == 0

    class _Other(_MeanPixelDetector):
        def identity(self):
            return {**super().identity(), "model": "fake/other"}

    with pytest.raises(FileExistsError, match="detector"):
        run(root, out, _Other())

    other_store = _collect(tmp_path / "again", n_keep=1)[1]
    with pytest.raises(FileExistsError, match="store"):
        run(other_store, out, _MeanPixelDetector())


def test_a_crash_mid_run_leaves_a_valid_file_holding_the_finished_episodes(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=3)
    out = tmp_path / "detections.json"
    manifest = json.loads((root / "manifest.json").read_text())
    calls_for_first = -(-manifest["episodes"][0]["steps"] // 8)  # ceil: batches of 8 for the first episode
    with pytest.raises(RuntimeError, match="fell over"):
        run(root, out, _MeanPixelDetector(fail_on_call=calls_for_first + 1), batch=8)
    saved = json.loads(out.read_text())
    assert list(saved["episodes"]) == [manifest["episodes"][0]["id"]] and saved["complete"] is False


def test_a_store_whose_frames_do_not_match_its_manifest_is_refused(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    sorted((root / entry["path"] / "frames").glob("*.png"))[-1].unlink()
    with pytest.raises(ValueError, match=entry["id"]):
        run(root, tmp_path / "detections.json", _MeanPixelDetector())


# ------------------------------------------------------------------------------------------------------------------
# autofly_ue5/dataset/rebalance.py
# ------------------------------------------------------------------------------------------------------------------


def test_the_paper_s_own_numbers_come_out_of_the_formulas():
    from autofly_ue5.dataset.rebalance import kl_from_uniform, resampling_weights, target_distribution

    p0 = (0.73, 0.27)  # App. A.2.4's measured phase distribution
    assert target_distribution(p0, alpha=0.0) == (0.5, 0.5)
    assert resampling_weights(p0) == pytest.approx((0.685, 1.852), abs=0.001)  # the paper's 0.68 / 1.85
    # Eq. 10 on (0.73, 0.27) gives 0.110 nats; the paper prints "approximately 0.36 nats", which no base or convention
    # of that equation reproduces for these two numbers. The code follows the equation.
    assert kl_from_uniform(p0) == pytest.approx(0.1099, abs=0.0005)
    assert target_distribution(p0, alpha=1.0) == p0 and resampling_weights(p0, alpha=1.0) == (1.0, 1.0)
    assert target_distribution(p0, alpha=0.5) == pytest.approx((0.615, 0.385))
    with pytest.raises(ValueError):
        target_distribution(p0, alpha=1.5)


def test_a_record_is_target_seeking_from_the_first_confident_detection_on():
    from autofly_ue5.dataset.rebalance import first_detection, phase_records

    assert first_detection([0.1, 0.7, 0.71, 0.2], 0.7) == 2, "strictly above the threshold"
    assert first_detection([0.1, 0.2], 0.7) is None
    assert first_detection([0.9, 0.1], 0.7) == 0
    assert phase_records(5, None) == (5, 0) and phase_records(5, 0) == (0, 5) and phase_records(5, 2) == (2, 3)
    for bad in ([0.1, float("nan")], [1.2], [-0.1], []):
        with pytest.raises(ValueError):
            first_detection(bad, 0.7)


def test_phase_distribution_kl_and_the_degenerate_case():
    import math

    from autofly_ue5.dataset.rebalance import kl_from_uniform, phase_distribution, resampling_weights

    assert phase_distribution((73, 27)) == (0.73, 0.27)
    assert phase_distribution((10, 0)) == (1.0, 0.0)
    assert kl_from_uniform((1.0, 0.0)) == pytest.approx(math.log(2)), "0 log 0 is 0"
    assert kl_from_uniform((0.5, 0.5)) == 0.0
    assert resampling_weights((1.0, 0.0)) == (0.5, None), "a phase nobody flew has no weight"
    with pytest.raises(ValueError):
        phase_distribution((0, 0))


def test_the_weights_even_out_the_records_each_phase_contributes():
    from autofly_ue5.dataset.rebalance import phase_distribution, resampling_weights

    counts = (17_000, 3_000)
    weights = resampling_weights(phase_distribution(counts))
    assert weights[0] * counts[0] == pytest.approx(weights[1] * counts[1]) == pytest.approx(sum(counts) / 2)


def test_stratified_resampling_follows_the_paper_s_sizes_and_replacement_rule():
    from autofly_ue5.dataset.rebalance import resample_sizes, stratified_resample

    assert resample_sizes((100, 40), (0.685, 1.852)) == (69, 74), "round half up: 68.5 -> 69, 74.08 -> 74"
    assert resample_sizes((100, 0), (0.5, None)) == (50, 0)
    with pytest.raises(ValueError):
        resample_sizes((100, 3), (0.5, None))  # sub-trajectories in a phase that has no weight: inconsistent inputs
    groups = (list(range(100)), list(range(40)))
    drawn = stratified_resample(groups, (0.685, 1.852), np.random.default_rng(0))
    assert len(drawn[0]) == 69 and len(set(drawn[0])) == 69 and set(drawn[0]) <= set(groups[0]), "w < 1: without replacement"
    assert len(drawn[1]) == 74 and set(drawn[1]) <= set(groups[1]) and len(set(drawn[1])) < 74, "w > 1: with replacement"
    assert stratified_resample(groups, (0.685, 1.852), np.random.default_rng(0)) == drawn, "a seed reproduces the draw"
    assert stratified_resample(groups, (1.0, 1.0), np.random.default_rng(0)) == (groups[0], groups[1]), "w = 1 keeps all"


def test_a_box_is_on_target_when_it_covers_the_target_s_image_column():
    import math

    from autofly_ue5.dataset.rebalance import box_on_target

    # 90 deg HFOV pinhole: the target's bearing b lands at image x = 0.5 + 0.5 tan(b), positive to the right. The two
    # boxes are Grounding DINO's own on pilot frames 83 (the target at 36 m) and 0 (something at the image edge).
    assert box_on_target([0.39, 0.517, 0.021, 0.033], bearing_rad=math.radians(-11.5)) is True
    assert box_on_target([0.985, 0.441, 0.03, 0.325], bearing_rad=math.radians(-15.4)) is False
    assert box_on_target([0.5, 0.5, 0.05, 0.1], bearing_rad=math.radians(50)) is None, "outside the camera's view"


def _manifest(steps: tuple[int, ...]) -> dict:
    return {"name": "pilot", "created": "2026-10-06 12:00:00",
            "episodes": [{"id": f"e{i}", "steps": n, "target_name": "orange cylinder", "scene": "s01"} for i, n in enumerate(steps)],
            "counts": {"episodes": len(steps), "records": sum(steps)}}


def _detections(manifest: dict, scores: dict[str, list[float]], **overrides) -> dict:
    from scripts.detect_targets import FORMAT

    doc = {"format": FORMAT, "dataset": manifest["name"], "dataset_created": manifest["created"], "score": "test",
           "detector": {"model": "fake", "revision": "0"}, "episodes_total": len(manifest["episodes"]),
           "complete": len(scores) == len(manifest["episodes"]), "runs": [],
           "episodes": {eid: {"query": "orange cylinder", "prompt": "orange cylinder.", "scores": s,
                              "boxes": [[0.5, 0.5, 0.1, 0.2]] * len(s)} for eid, s in scores.items()}}
    doc.update(overrides)
    return doc


def test_build_rebalance_labels_every_episode_and_weights_the_whole_store():
    from autofly_ue5.dataset.rebalance import FORMAT, build_rebalance

    manifest = _manifest((10, 10))
    detections = _detections(manifest, {"e0": [0.1] * 7 + [0.8, 0.6, 0.9], "e1": [0.1] * 10})
    result = build_rebalance(manifest, detections, threshold=0.7, alpha=0.0)
    assert result["format"] == FORMAT and result["dataset"] == "pilot" and result["threshold"] == 0.7
    assert result["phases"] == ["obstacle_avoidance", "target_seeking"]
    assert result["coverage"] == {"episodes": 2, "of": 2, "partial": False}
    assert result["counts"]["records"] == [17, 3] and result["counts"]["sub_trajectories"] == [2, 1]
    assert result["counts"]["records_above_threshold"] == 2, "the per-record reading, for comparison: 0.8 and 0.9"
    assert result["counts"]["episodes_never_detected"] == 1 and result["counts"]["episodes_detected_at_first_record"] == 0
    assert result["p0"] == pytest.approx([0.85, 0.15]) and result["p_target"] == [0.5, 0.5]
    assert result["weights"] == pytest.approx([0.5 / 0.85, 0.5 / 0.15]) and result["resample_sizes"] == [1, 3]
    assert result["degenerate"] is False
    assert result["detection_persistence"] == pytest.approx(2 / 3), "records from the first detection on that are above the threshold"
    e0, e1 = result["episodes"]
    assert e0 == {"id": "e0", "records": 10, "first_detection": 7, "phase_records": [7, 3]}
    assert e1 == {"id": "e1", "records": 10, "first_detection": None, "phase_records": [10, 0]}


def test_build_rebalance_adds_the_privileged_checks_when_given_the_states():
    import math

    from autofly_ue5.dataset.rebalance import build_rebalance

    manifest = _manifest((4, 4))
    detections = _detections(manifest, {"e0": [0.1, 0.1, 0.8, 0.9], "e1": [0.9, 0.9, 0.9, 0.9]})
    detections["episodes"]["e0"]["boxes"][2] = [0.39, 0.5, 0.02, 0.03]  # on a target at bearing -11.5 deg
    detections["episodes"]["e1"]["boxes"][0] = [0.98, 0.5, 0.02, 0.03]  # not on a target dead ahead
    states = {eid: np.zeros((4, 9), dtype=np.float32) for eid in ("e0", "e1")}
    states["e0"][:, 0] = [40.0, 38.0, 36.0, 34.0]
    states["e0"][:, 1] = math.radians(-11.5)
    states["e1"][:, 0] = [60.0, 58.0, 56.0, 54.0]
    result = build_rebalance(manifest, detections, threshold=0.7, alpha=0.0, states=states)
    e0, e1 = result["episodes"]
    assert e0["first_detection_distance_m"] == pytest.approx(36.0) and e0["first_detection_on_target"] is True
    assert e1["first_detection_distance_m"] == pytest.approx(60.0) and e1["first_detection_on_target"] is False
    assert result["first_detection"] == {"distance_m_median": pytest.approx(48.0), "on_target": 1, "off_target": 1,
                                         "target_out_of_view": 0}


def test_build_rebalance_refuses_detections_that_do_not_match_the_store():
    from autofly_ue5.dataset.rebalance import build_rebalance

    manifest = _manifest((3, 3))
    good = {"e0": [0.1, 0.8, 0.9], "e1": [0.1, 0.1, 0.1]}
    with pytest.raises(ValueError, match="e1"):
        build_rebalance(manifest, _detections(manifest, {"e0": good["e0"]}), threshold=0.7, alpha=0.0)
    partial = build_rebalance(manifest, _detections(manifest, {"e0": good["e0"]}), threshold=0.7, alpha=0.0, allow_partial=True)
    assert partial["coverage"] == {"episodes": 1, "of": 2, "partial": True} and partial["counts"]["records"] == [1, 2]
    with pytest.raises(ValueError, match="records"):
        build_rebalance(manifest, _detections(manifest, {**good, "e1": [0.1, 0.1]}), threshold=0.7, alpha=0.0)
    with pytest.raises(ValueError, match="store"):
        build_rebalance(manifest, _detections(manifest, good, dataset_created="another time"), threshold=0.7, alpha=0.0)
    with pytest.raises(ValueError, match="store"):
        build_rebalance(manifest, _detections(manifest, good, dataset="other"), threshold=0.7, alpha=0.0)
    with pytest.raises(ValueError, match="format"):
        build_rebalance(manifest, _detections(manifest, good, format="something_else/9"), threshold=0.7, alpha=0.0)
    wrong_query = _detections(manifest, good)
    wrong_query["episodes"]["e0"]["query"] = "blue cone"
    with pytest.raises(ValueError, match="query"):
        build_rebalance(manifest, wrong_query, threshold=0.7, alpha=0.0)
    stale = _detections(manifest, {"e0": good["e0"], "e1": good["e1"], "e9": [0.1]})
    with pytest.raises(ValueError, match="e9"):
        build_rebalance(manifest, stale, threshold=0.7, alpha=0.0)


def test_a_degenerate_store_is_reported_not_crashed_and_the_writer_refuses_to_overwrite(tmp_path):
    from autofly_ue5.dataset.rebalance import build_rebalance, write_rebalance

    manifest = _manifest((3, 3))
    result = build_rebalance(manifest, _detections(manifest, {"e0": [0.1] * 3, "e1": [0.2] * 3}), threshold=0.7, alpha=0.0)
    assert result["degenerate"] is True and result["weights"] == [0.5, None] and result["p0"] == [1.0, 0.0]
    assert result["resample_sizes"] == [1, 0] and result["detection_persistence"] is None
    out = tmp_path / "rebalance.json"
    write_rebalance(out, result)
    assert json.loads(out.read_text()) == result
    with pytest.raises(FileExistsError):
        write_rebalance(out, result)


# ------------------------------------------------------------------------------------------------------------------
# scripts/rebalance_dataset.py
# ------------------------------------------------------------------------------------------------------------------


def _rising_detections(root: Path) -> Path:
    """A detections file for a fake store whose scores cross 0.7 at 60 % of every episode."""
    from scripts.detect_targets import FORMAT

    manifest = json.loads((root / "manifest.json").read_text())
    episodes = {}
    for entry in manifest["episodes"]:
        n = entry["steps"]
        scores = [round(i / (n - 1), 6) if n > 1 else 1.0 for i in range(n)]
        episodes[entry["id"]] = {"query": entry["target_name"], "prompt": entry["target_name"] + ".", "scores": scores,
                                 "boxes": [[0.5, 0.5, 0.1, 0.2]] * n}
    doc = {"format": FORMAT, "dataset": manifest["name"], "dataset_created": manifest["created"], "score": "test",
           "detector": {"model": "fake", "revision": "0"}, "episodes_total": len(manifest["episodes"]), "complete": True,
           "runs": [], "episodes": episodes}
    path = root / "detections.json"
    path.write_text(json.dumps(doc))
    return path


def test_the_cli_writes_rebalance_json_into_the_store_and_refuses_a_second_time(tmp_path, capsys):
    from autofly_ue5.dataset.rebalance import FORMAT
    from scripts.rebalance_dataset import main

    _summary, root = _collect(tmp_path, n_keep=3)
    _rising_detections(root)
    assert main(["--raw", str(root)]) == 0
    result = json.loads((root / "rebalance.json").read_text())
    assert result["format"] == FORMAT and result["coverage"]["partial"] is False and result["threshold"] == 0.7
    manifest = json.loads((root / "manifest.json").read_text())
    for entry, ep in zip(manifest["episodes"], result["episodes"]):
        n = entry["steps"]
        assert ep["id"] == entry["id"] and ep["first_detection"] == next(i for i in range(n) if i / (n - 1) > 0.7)
        assert ep["first_detection_distance_m"] is not None and ep["first_detection_on_target"] in (True, False, None)
    assert 0.6 < result["p0"][0] < 0.8 and result["detections"]["sha256"]
    assert "weights" in capsys.readouterr().out
    assert main(["--raw", str(root)]) != 0, "rebalance.json exists: refused, never replaced"


def test_the_cli_takes_partial_detections_only_with_the_flag_and_only_outside_the_store(tmp_path):
    from scripts.rebalance_dataset import main

    _summary, root = _collect(tmp_path, n_keep=2)
    path = _rising_detections(root)
    doc = json.loads(path.read_text())
    first_id = json.loads((root / "manifest.json").read_text())["episodes"][0]["id"]
    doc["episodes"] = {first_id: doc["episodes"][first_id]}
    doc["complete"] = False
    path.write_text(json.dumps(doc))
    assert main(["--raw", str(root)]) != 0, "one episode of two: refused without --allow-partial"
    assert not (root / "rebalance.json").exists()
    assert main(["--raw", str(root), "--allow-partial"]) != 0, "a partial result must not pose as the store's own"
    assert not (root / "rebalance.json").exists()
    out = tmp_path / "probe" / "rebalance.json"
    assert main(["--raw", str(root), "--allow-partial", "--out", str(out), "--threshold", "0.5"]) == 0
    result = json.loads(out.read_text())
    assert result["coverage"] == {"episodes": 1, "of": 2, "partial": True} and result["threshold"] == 0.5
