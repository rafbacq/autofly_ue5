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
    from autofly_ue5.dataset.detections import parse
    from scripts.detect_targets import FORMAT, run

    _summary, root = _collect(tmp_path, n_keep=3)
    out = tmp_path / "detections.jsonl"
    detector = _MeanPixelDetector()
    doc = run(root, out, detector, batch=4)
    manifest = json.loads((root / "manifest.json").read_text())
    saved = parse(out.read_bytes()).doc
    assert saved == doc and doc["sparse"] is None
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


def test_a_pool_store_is_scored_for_the_target_each_episode_s_provenance_names(tmp_path):
    # Plan 5 C5: with a target pool every episode has its own target, and the detector must look for that one. The
    # collector writes the drawn entry's name into the manifest entry the detector reads and into the provenance; the
    # validator holds the two equal (tests/test_collect.py).
    from autofly_ue5.dataset.rebalance import build_rebalance
    from scripts.detect_targets import run
    from tests.test_dataset_targets import primitive_pool

    _summary, root = _collect(tmp_path, n_keep=4, target_pool=primitive_pool(tmp_path), target_name=None)
    doc = run(root, tmp_path / "detections.jsonl", _MeanPixelDetector(), batch=4)
    manifest = json.loads((root / "manifest.json").read_text())
    queries = []
    for entry in manifest["episodes"]:
        name = json.loads((root / "provenance" / f"{entry['id']}.json").read_text())["target"]["name"]
        assert doc["episodes"][entry["id"]]["query"] == name and doc["episodes"][entry["id"]]["prompt"] == name + "."
        queries.append(name)
    assert len(set(queries)) > 1, "the store's episodes look for different targets"
    assert build_rebalance(manifest, doc)["coverage"]["episodes"] == 4


def test_a_limited_run_is_resumed_without_rescoring_and_another_detector_or_store_is_refused(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=3)
    out = tmp_path / "detections.jsonl"
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
    out = tmp_path / "detections.jsonl"
    manifest = json.loads((root / "manifest.json").read_text())
    calls_for_first = -(-manifest["episodes"][0]["steps"] // 8)  # ceil: batches of 8 for the first episode
    with pytest.raises(RuntimeError, match="fell over"):
        run(root, out, _MeanPixelDetector(fail_on_call=calls_for_first + 1), batch=8)
    from autofly_ue5.dataset.detections import parse
    saved = parse(out.read_bytes()).doc
    assert list(saved["episodes"]) == [manifest["episodes"][0]["id"]] and saved["complete"] is False


def _rewrite_episode_line(out: Path, eid: str, change) -> None:
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    for line in lines:
        if line.get("kind") == "episode" and line["id"] == eid:
            change(line)
    out.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_a_stored_episode_whose_length_disagrees_with_the_manifest_is_refused_on_resume(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=2)
    out = tmp_path / "detections.jsonl"
    doc = run(root, out, _MeanPixelDetector(), episodes=1)
    (eid,) = doc["episodes"]
    _rewrite_episode_line(out, eid, lambda line: line["scores"].pop())
    with pytest.raises(ValueError, match=eid):
        run(root, out, _MeanPixelDetector())


def test_a_second_scorer_of_the_same_file_is_refused_while_the_first_is_alive(tmp_path):
    import os

    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=1)
    out = tmp_path / "detections.jsonl"
    lock = out.with_suffix(out.suffix + ".lock")
    lock.write_text(f"{os.getpid()}\n")  # a live scorer: this very process
    with pytest.raises(FileExistsError, match="lock"):
        run(root, out, _MeanPixelDetector())
    lock.write_text("999999999\n")  # a scorer that died with the host: its lock is stale and taken over
    doc = run(root, out, _MeanPixelDetector())
    assert doc["complete"] is True and not lock.exists(), "the lock is released when the run ends"


def test_bad_arguments_are_refused_before_anything_is_written(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=1)
    out = tmp_path / "detections.jsonl"
    for kwargs in ({"batch": 0}, {"batch": -1}, {"episodes": -1}):
        with pytest.raises(ValueError):
            run(root, out, _MeanPixelDetector(), **kwargs)
    assert not out.exists()

    class _Short(_MeanPixelDetector):
        def detect(self, images, prompt):
            return super().detect(images, prompt)[:-1]

    with pytest.raises(ValueError, match="results"):
        run(root, out, _Short())


def test_a_frame_that_is_not_the_store_s_size_is_refused(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    frame = sorted((root / entry["path"] / "frames").glob("*.png"))[3]
    Image.new("RGB", (128, 256)).save(frame)
    with pytest.raises(ValueError, match=frame.name):
        run(root, tmp_path / "detections.jsonl", _MeanPixelDetector())


def test_the_header_follows_the_manifest_on_resume(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=2)
    out = tmp_path / "detections.jsonl"
    run(root, out, _MeanPixelDetector(), episodes=1)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["episodes"] = manifest["episodes"][:1]  # as if the first run had seen a store still being collected
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert run(root, out, _MeanPixelDetector())["episodes_total"] == 1


def test_the_score_is_the_best_query_s_confidence_for_the_phrase_and_its_box():
    torch = pytest.importorskip("torch")
    from scripts.detect_targets import reduce_outputs

    logits = torch.full((2, 3, 5), -torch.inf)  # (batch, queries, text positions); padded positions are -inf
    logits[0, 0, :3] = torch.tensor([-2.0, 0.5, -1.0])
    logits[0, 1, :3] = torch.tensor([-1.0, -1.0, 2.0])   # the best query of image 0 (sigmoid(2.0) = 0.8808)
    logits[0, 2, :3] = torch.tensor([0.0, 0.0, 0.0])
    logits[1, 0, :3] = torch.tensor([-3.0, -3.0, -3.0])  # image 1: nothing confident anywhere
    logits[1, 1, :3] = torch.tensor([-4.0, -2.5, -4.0])
    logits[1, 2, :3] = torch.tensor([-5.0, -5.0, -5.0])
    boxes = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 100.0
    out = reduce_outputs(logits, boxes)
    assert out[0][0] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item()) and out[0][1] == pytest.approx(boxes[0, 1].tolist())
    assert out[1][0] == pytest.approx(torch.sigmoid(torch.tensor(-2.5)).item()) and out[1][1] == pytest.approx(boxes[1, 1].tolist())


def test_a_store_whose_frames_do_not_match_its_manifest_is_refused(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    sorted((root / entry["path"] / "frames").glob("*.png"))[-1].unlink()
    with pytest.raises(ValueError, match=entry["id"]):
        run(root, tmp_path / "detections.jsonl", _MeanPixelDetector())


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
    assert box_on_target([0.5, 0.5, 0.05, 0.1], bearing_rad=math.radians(50)) is False, "the target is not in the image at all"


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
                                         "off_target_out_of_view": 0}
    states["e1"][0, 1] = math.radians(60)  # e1's first "detection" came with the target behind the camera's edge
    result = build_rebalance(manifest, detections, threshold=0.7, alpha=0.0, states=states)
    assert result["episodes"][1]["first_detection_on_target"] is False
    assert result["first_detection"]["off_target"] == 1 and result["first_detection"]["off_target_out_of_view"] == 1


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
    short_boxes = _detections(manifest, good)
    short_boxes["episodes"]["e0"]["boxes"] = short_boxes["episodes"]["e0"]["boxes"][:2]
    with pytest.raises(ValueError, match="boxes"):
        build_rebalance(manifest, short_boxes, threshold=0.7, alpha=0.0)
    with pytest.raises(ValueError, match="e0.*rows"):
        build_rebalance(manifest, _detections(manifest, good), threshold=0.7, alpha=0.0,
                        states={"e0": np.zeros((2, 9), np.float32), "e1": np.zeros((3, 9), np.float32)})
    nan = _detections(manifest, {**good, "e1": [0.1, float("nan"), 0.1]})
    with pytest.raises(ValueError, match="e1"):
        build_rebalance(manifest, nan, threshold=0.7, alpha=0.0)
    for bad in (1.0, 70.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="threshold"):
            build_rebalance(manifest, _detections(manifest, good), threshold=bad, alpha=0.0)


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
    assert json.loads(out.read_text()) == result, "a refused write leaves the file as it was"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rebalance.json"], "no temporary file is left behind"


def test_the_writer_puts_the_whole_document_in_place_in_one_step(tmp_path, monkeypatch):
    """The order in which the write reached the disk: data synced in a private file, then linked to its name, then the
    directory synced -- so a crash leaves either no file or the whole one, never a truncated rebalance.json."""
    import os

    from autofly_ue5.dataset.rebalance import write_rebalance

    events = []
    real_fsync, real_link = os.fsync, os.link
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append(("fsync", os.path.realpath(os.readlink(f"/proc/self/fd/{fd}")))),
                                                  real_fsync(fd)))
    monkeypatch.setattr(os, "link", lambda src, dst: (events.append(("link", os.path.realpath(src), os.path.realpath(dst))),
                                                       real_link(src, dst)))
    out = tmp_path / "rebalance.json"
    write_rebalance(out, {"a": 1})
    final = os.path.realpath(out)
    (link_at,) = [i for i, e in enumerate(events) if e[0] == "link" and e[2] == final]
    tmp_name = events[link_at][1]
    assert tmp_name.startswith(final + ".tmp.") and ("fsync", tmp_name) in events[:link_at], "the data is synced before the name appears"
    assert ("fsync", os.path.dirname(final)) in events[link_at + 1:], "the directory entry is synced after"
    assert json.loads(out.read_text()) == {"a": 1} and sorted(p.name for p in tmp_path.iterdir()) == ["rebalance.json"]


def test_the_two_stages_agree_on_the_detections_format():
    from autofly_ue5.dataset.rebalance import DETECTIONS_FORMAT
    from scripts.detect_targets import FORMAT

    assert FORMAT == DETECTIONS_FORMAT


# ------------------------------------------------------------------------------------------------------------------
# scripts/rebalance_dataset.py
# ------------------------------------------------------------------------------------------------------------------


def test_a_degenerate_result_is_refused_at_the_store_s_own_path_but_can_be_inspected_elsewhere(tmp_path):
    from scripts.detect_targets import run
    from scripts.rebalance_dataset import main

    _summary, root = _collect(tmp_path, n_keep=2)
    run(root, root / "detections.jsonl", _MeanPixelDetector())  # noise frames score about 0.5: never above 0.7
    assert main(["--raw", str(root)]) != 0 and not (root / "rebalance.json").exists()
    out = tmp_path / "look" / "rebalance.json"
    assert main(["--raw", str(root), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["degenerate"] is True and result["counts"]["episodes_never_detected"] == 2
    assert main(["--raw", str(root), "--threshold", "70"]) != 0, "a percentage is not a confidence"


def _rising_detections(root: Path) -> Path:
    """A detections file for a fake store whose scores cross 0.7 at 60 % of every episode, in the pilot's whole-document
    form (data/s01_pilot/detections.json), which the rebalancing stage still reads."""
    from autofly_ue5.dataset.detections import FORMAT_V1 as FORMAT

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
    from autofly_ue5.dataset.rebalance import FORMAT, box_on_target
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
        state = np.load(root / entry["path"] / "steps.npz")["state"]
        assert ep["first_detection_distance_m"] == pytest.approx(float(state[ep["first_detection"], 0]))
        box = json.loads((root / "detections.json").read_text())["episodes"][entry["id"]]["boxes"][ep["first_detection"]]
        assert ep["first_detection_on_target"] is box_on_target(box, float(state[ep["first_detection"], 1]))
    assert 0.6 < result["p0"][0] < 0.8
    import hashlib
    assert result["detections"]["sha256"] == hashlib.sha256((root / "detections.json").read_bytes()).hexdigest()
    assert "weights" in capsys.readouterr().out
    assert main(["--raw", str(root)]) != 0, "rebalance.json exists: refused, never replaced"


def test_detections_written_by_stage_one_feed_stage_two_end_to_end(tmp_path):
    from scripts.detect_targets import run
    from scripts.rebalance_dataset import main

    _summary, root = _collect(tmp_path, n_keep=2)
    run(root, root / "detections.jsonl", _MeanPixelDetector())
    out = tmp_path / "e2e" / "rebalance.json"  # noise frames average about 0.5: at 0.49 every episode is detected at
    assert main(["--raw", str(root), "--threshold", "0.49", "--out", str(out)]) == 0  # once, a degenerate split
    result = json.loads(out.read_text())
    assert result["coverage"]["partial"] is False and result["counts"]["episodes_detected_at_first_record"] == 2
    assert result["detector"]["model"] == "fake/mean-pixel" and result["p0"] == [0.0, 1.0] and result["degenerate"] is True
    assert result["detection_persistence"] == 1.0 and result["first_detection"]["on_target"] + result["first_detection"]["off_target"] == 2


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
    assert main(["--raw", str(root), "--allow-partial", "--out", str(root / "rebalance.json")]) != 0
    assert main(["--raw", str(root), "--allow-partial", "--out", str(root / "probe" / "rebalance.json")]) != 0
    assert not (root / "rebalance.json").exists() and not (root / "probe").exists()
    out = tmp_path / "probe" / "rebalance.json"
    assert main(["--raw", str(root), "--allow-partial", "--out", str(out), "--threshold", "0.5"]) == 0
    result = json.loads(out.read_text())
    assert result["coverage"] == {"episodes": 1, "of": 2, "partial": True} and result["threshold"] == 0.5


# ------------------------------------------------------------------------------------------------------------------
# scripts/detect_targets.py: the appendable cache and the sparse search (plan 6 step 5)
# ------------------------------------------------------------------------------------------------------------------


def _pixels(image) -> bytes:
    import hashlib

    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).digest()


class _ScriptedDetector(_MeanPixelDetector):
    """Scores each frame from a table keyed by its pixels, so a test chooses every record's score and can replay the
    sparse search offline on the same scores."""

    def __init__(self, table: dict[bytes, float]) -> None:
        super().__init__()
        self._table = table

    def detect(self, images, prompt):
        self.calls += 1
        assert prompt == prompt.lower().strip() and prompt.endswith("."), f"prompt {prompt!r} breaks the convention"
        out = [(self._table[_pixels(image)], [0.5, 0.5, 0.1, 0.2]) for image in images]
        self.frames_scored += len(images)
        return out


def _flicker_profile(n: int) -> list[float]:
    """The pilot's shape: a ramp far out, one record above 0.7 a third of the way in, a hover just below 0.7, then the
    sustained detection."""
    f = n // 3
    scores = []
    for i in range(n):
        if i < f:
            v = 0.3 + 0.3 * i / max(f, 1)
        elif i == f:
            v = 0.71
        elif i < 2 * n // 3:
            v = 0.65
        else:
            v = 0.85
        scores.append(round(v, 6))
    return scores


def _scripted(root: Path) -> tuple[_ScriptedDetector, dict[str, list[float]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    table, scores = {}, {}
    for entry in manifest["episodes"]:
        frames = sorted((root / entry["path"] / "frames").glob("*.png"))
        scores[entry["id"]] = _flicker_profile(len(frames))
        for frame, score in zip(frames, scores[entry["id"]]):
            table[_pixels(Image.open(frame))] = score
    assert len(table) == sum(len(s) for s in scores.values()), "every frame is distinct, so the table is unambiguous"
    return _ScriptedDetector(table), scores


def test_the_cache_grows_by_one_synced_line_per_episode_and_is_never_rewritten(tmp_path, monkeypatch):
    import os

    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=3)
    out = tmp_path / "detections.jsonl"
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(os.path.realpath(os.readlink(f"/proc/self/fd/{fd}"))),
                                                  real_fsync(fd)))
    snapshots = []
    run(root, out, _MeanPixelDetector(), log=lambda *a, **k: snapshots.append((out.read_bytes(), synced.count(os.path.realpath(out)))))
    final = out.read_bytes()
    assert len(snapshots) == 3
    for (before, _), (after, _) in zip(snapshots, snapshots[1:] + [(final, None)]):
        assert after.startswith(before) and len(after) > len(before), "appended, never rewritten"
    assert [s for _, s in snapshots] == [2, 3, 4], "the run line and each episode's line are synced before the next"
    kinds = [json.loads(line)["kind"] for line in final.decode().splitlines()]
    assert kinds == ["header", "run", "episode", "episode", "episode", "run_end"]


def test_a_resumed_run_cuts_off_a_torn_last_line_and_continues_after_the_last_complete_one(tmp_path):
    from autofly_ue5.dataset.detections import parse
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=3)
    manifest = json.loads((root / "manifest.json").read_text())
    out = tmp_path / "detections.jsonl"
    run(root, out, _MeanPixelDetector(), episodes=2)
    complete = out.read_bytes()
    torn = json.dumps({"kind": "episode", "id": manifest["episodes"][2]["id"], "query": "orange cylinder"}).encode()[:25]
    out.write_bytes(complete + torn)  # the host froze mid-append
    assert parse(out.read_bytes()).torn is True
    second = _MeanPixelDetector()
    messages = []
    doc = run(root, out, second, log=lambda *a, **k: messages.append(" ".join(map(str, a))))
    assert second.frames_scored == manifest["episodes"][2]["steps"], "only the episode whose line was torn is scored"
    final = out.read_bytes()
    assert final.startswith(complete) and final[len(complete):].startswith(b'{"kind":"run"'), "the fragment is cut off"
    parsed = parse(final)
    assert parsed.torn is False and doc["complete"] is True
    assert list(parsed.doc["episodes"]) == [e["id"] for e in manifest["episodes"]]
    assert any("torn" in m for m in messages), "the cut is reported"

    lines = final.split(b"\n")
    lines[2] = b'{"kind":"episode","id":'  # a damaged line that is not the last one is corruption, not a torn append
    out.write_bytes(b"\n".join(lines))
    with pytest.raises(ValueError, match="line 3"):
        run(root, out, _MeanPixelDetector())


def test_the_sparse_mode_scores_only_what_the_search_visits_and_stride_one_is_the_dense_first_detection(tmp_path):
    from autofly_ue5.dataset.detections import parse
    from autofly_ue5.dataset.sparse import replay
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=3)
    dense_detector, scores = _scripted(root)
    dense = run(root, tmp_path / "dense.jsonl", dense_detector)
    exact_detector, _ = _scripted(root)
    exact = run(root, tmp_path / "s1.jsonl", exact_detector, sparse={"stride": 1, "threshold": 0.7, "alert": 0.7}, batch=1)
    sparse_detector, _ = _scripted(root)
    out = tmp_path / "s4.jsonl"
    sparse = run(root, out, sparse_detector, sparse={"stride": 4, "threshold": 0.7, "alert": 0.5}, batch=3)
    assert parse(out.read_bytes()).doc == sparse
    for eid, s in scores.items():
        d = next(i for i, v in enumerate(dense["episodes"][eid]["scores"]) if v > 0.7)
        assert d == len(s) // 3
        assert exact["episodes"][eid]["first_detection"] == d, "stride 1 is the dense mode's first detection"
        assert exact["episodes"][eid]["visited"] == list(range(d + 1))
        first, visited = replay(s, stride=4, threshold=0.7, alert=0.5)
        ep = sparse["episodes"][eid]
        assert ep["first_detection"] == first == d
        assert set(visited) <= set(ep["visited"]) and ep["visited"] == sorted(set(ep["visited"]))
        assert ep["scores"] == [s[i] for i in ep["visited"]] and len(ep["boxes"]) == len(ep["visited"])
        assert ep["records"] == len(s) and ep["frames_scored"] == len(ep["visited"])
    assert (sparse["sparse"]["stride"], sparse["sparse"]["threshold"], sparse["sparse"]["alert"]) == (4, 0.7, 0.5)
    assert dense["sparse"] is None and exact["sparse"]["stride"] == 1
    scored = sum(len(e["visited"]) for e in sparse["episodes"].values())
    assert sparse_detector.frames_scored == scored < dense_detector.frames_scored


def test_a_file_is_continued_only_by_the_search_that_started_it(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=2)
    out = tmp_path / "detections.jsonl"
    search = {"stride": 8, "threshold": 0.7, "alert": 0.5}
    run(root, out, _MeanPixelDetector(), episodes=1)
    with pytest.raises(FileExistsError, match="search"):
        run(root, out, _MeanPixelDetector(), sparse=search)
    sparse_out = tmp_path / "sparse.jsonl"
    run(root, sparse_out, _MeanPixelDetector(), episodes=1, sparse=search)
    for other in ({**search, "stride": 4}, {**search, "threshold": 0.75}, {**search, "alert": 0.55}, None):
        with pytest.raises(FileExistsError, match="search"):
            run(root, sparse_out, _MeanPixelDetector(), sparse=other)
    assert run(root, sparse_out, _MeanPixelDetector(), sparse=search)["complete"] is True


def test_the_pilot_s_whole_document_file_is_never_continued_and_the_default_cache_is_detections_jsonl(tmp_path, monkeypatch):
    import scripts.detect_targets as detect_targets

    _summary, root = _collect(tmp_path, n_keep=1)
    _rising_detections(root)  # the pilot's form at <raw>/detections.json
    with pytest.raises(FileExistsError, match="autofly_ue5_detections/1"):
        detect_targets.run(root, root / "detections.json", _MeanPixelDetector())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(detect_targets, "GroundingDino", lambda *a, **k: _MeanPixelDetector())
    assert detect_targets.main(["--raw", str(root)]) != 0, "the store is scored already: no silent second scoring"
    assert not (root / "detections.jsonl").exists()
    (root / "detections.json").unlink()
    assert detect_targets.main(["--raw", str(root)]) == 0 and (root / "detections.jsonl").is_file()


def test_the_cli_runs_the_sparse_search_with_the_study_s_alert_by_default(tmp_path, monkeypatch):
    import scripts.detect_targets as detect_targets
    from autofly_ue5.dataset.detections import parse
    from autofly_ue5.dataset.sparse import describe

    _summary, root = _collect(tmp_path, n_keep=2)
    detector, _scores = _scripted(root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(detect_targets, "GroundingDino", lambda *a, **k: detector)
    assert detect_targets.main(["--raw", str(root), "--sparse-stride", "8", "--batch", "2"]) == 0
    doc = parse((root / "detections.jsonl").read_bytes()).doc
    assert doc["sparse"] == describe(stride=8, threshold=0.7, alert=detect_targets.DEFAULT_SPARSE_ALERT)
    assert doc["complete"] is True
    for bad in (["--sparse-stride", "0"], ["--sparse-stride", "8", "--sparse-alert", "0.8"], ["--sparse-alert", "0.5"],
                ["--sparse-threshold", "0.7"]):
        with pytest.raises(SystemExit):
            detect_targets.main(["--raw", str(root), "--out", str(tmp_path / "x.jsonl"), *bad])
    assert not (tmp_path / "x.jsonl").exists()


def test_the_prompt_is_the_target_phrase_the_instruction_names_never_a_registry_key(tmp_path):
    from scripts.detect_targets import run

    _summary, root = _collect(tmp_path, n_keep=2)
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(entry["target_name"] in entry["instruction"] for entry in manifest["episodes"])
    manifest["episodes"][1]["target_name"] = "kenney_car_sedan"  # a registry key, not the phrase the instruction names
    (root / "manifest.json").write_text(json.dumps(manifest))
    detector = _MeanPixelDetector()
    with pytest.raises(ValueError, match="instruction"):
        run(root, tmp_path / "detections.jsonl", detector)
    assert detector.frames_scored == 0, "refused before anything is scored"


# ------------------------------------------------------------------------------------------------------------------
# autofly_ue5/dataset/rebalance.py: sparse detections, and several stores at once
# ------------------------------------------------------------------------------------------------------------------


def _sparse_detections(manifest: dict, scores: dict[str, list[float]], *, stride=8, threshold=0.7, alert=0.5,
                       **overrides) -> dict:
    from autofly_ue5.dataset.detections import FORMAT
    from autofly_ue5.dataset.sparse import describe, replay

    episodes = {}
    for eid, s in scores.items():
        first, visited = replay(s, stride=stride, threshold=threshold, alert=alert)
        order = sorted(visited)
        episodes[eid] = {"query": "orange cylinder", "prompt": "orange cylinder.", "records": len(s), "visited": order,
                         "scores": [s[i] for i in order], "boxes": [[0.5, 0.5, 0.1, 0.2]] * len(order),
                         "first_detection": first}
    doc = {"format": FORMAT, "dataset": manifest["name"], "dataset_created": manifest["created"], "score": "test",
           "detector": {"model": "fake", "revision": "0"}, "sparse": describe(stride, threshold, alert), "runs": [],
           "episodes": episodes}
    doc.update(overrides)
    return doc


def _flicker(n: int = 40) -> list[float]:
    scores = [0.3] * n
    scores[5], scores[8] = 0.71, 0.62
    scores[30:] = [0.85] * (n - 30)
    return scores


def test_build_rebalance_takes_a_sparse_search_s_first_detections_after_replaying_them():
    from autofly_ue5.dataset.rebalance import build_rebalance

    manifest = _manifest((40, 40))
    scores = {"e0": _flicker(), "e1": [0.3] * 40}
    sparse = _sparse_detections(manifest, scores)
    result = build_rebalance(manifest, sparse, threshold=0.7)
    dense = build_rebalance(manifest, _detections(manifest, scores), threshold=0.7)
    assert [e["first_detection"] for e in result["episodes"]] == [5, None] == [e["first_detection"] for e in dense["episodes"]]
    assert result["counts"]["records"] == dense["counts"]["records"] == [45, 35]
    assert result["p0"] == dense["p0"] and result["weights"] == dense["weights"]
    assert result["sparse"] == sparse["sparse"] and dense["sparse"] is None
    assert result["counts"]["records_above_threshold"] is None and result["detection_persistence"] is None, \
        "a sparse run never saw most records"
    assert dense["counts"]["records_above_threshold"] == 11


def test_build_rebalance_refuses_a_sparse_record_it_cannot_reproduce():
    import copy

    from autofly_ue5.dataset.rebalance import build_rebalance

    manifest = _manifest((40,))
    good = _sparse_detections(manifest, {"e0": _flicker()})
    assert good["episodes"]["e0"]["visited"] == [0, 1, 2, 3, 4, 5, 8]
    assert build_rebalance(manifest, good, threshold=0.7)["episodes"][0]["first_detection"] == 5
    with pytest.raises(ValueError, match="threshold"):
        build_rebalance(manifest, good, threshold=0.75)  # the search stopped at 0.7's first detection: 0.75's is unknown

    def broken(change):
        doc = copy.deepcopy(good)
        change(doc["episodes"]["e0"])
        return doc

    cases = {
        "first detection": lambda ep: ep.update(first_detection=30),
        "record 3": lambda ep: [ep[k].pop(3) for k in ("visited", "scores", "boxes")],  # a window record it needed
        "40 records": lambda ep: ep.update(records=39),
        "visited": lambda ep: ep.update(visited=ep["visited"][::-1]),
        "boxes": lambda ep: ep["boxes"].pop(),
        "scores": lambda ep: ep["scores"].pop(),
    }
    for match, change in cases.items():
        with pytest.raises(ValueError, match=match):
            build_rebalance(manifest, broken(change), threshold=0.7)
    with pytest.raises(ValueError, match="alert"):
        build_rebalance(manifest, {**good, "sparse": {**good["sparse"], "alert": 0.9}}, threshold=0.7)


def test_the_privileged_checks_read_a_sparse_run_s_box_at_its_first_detection():
    import math

    from autofly_ue5.dataset.rebalance import build_rebalance

    manifest = _manifest((40,))
    doc = _sparse_detections(manifest, {"e0": _flicker()})
    ep = doc["episodes"]["e0"]
    ep["boxes"][ep["visited"].index(5)] = [0.39, 0.5, 0.02, 0.03]  # on a target at bearing -11.5 deg
    states = {"e0": np.zeros((40, 9), np.float32)}
    states["e0"][:, 0] = np.linspace(60.0, 5.0, 40)
    states["e0"][:, 1] = math.radians(-11.5)
    (e0,) = build_rebalance(manifest, doc, threshold=0.7, states=states)["episodes"]
    assert e0["first_detection_on_target"] is True
    assert e0["first_detection_distance_m"] == pytest.approx(float(states["e0"][5, 0]))


def _two_stores():
    a = {**_manifest((10, 10)), "name": "s07_a"}
    b = {**_manifest((20, 12)), "name": "s09_a"}
    b["episodes"][0]["target_name"], b["episodes"][1]["target_name"] = "red cube", "blue cone"
    det_a = _detections(a, {"e0": [0.1] * 7 + [0.8, 0.6, 0.9], "e1": [0.1] * 10})
    det_b = _detections(b, {"e0": [0.1] * 5 + [0.9] * 15, "e1": [0.2] * 12})
    det_b["episodes"]["e0"]["query"], det_b["episodes"]["e1"]["query"] = "red cube", "blue cone"
    return a, det_a, b, det_b


def test_the_union_of_several_stores_sums_their_records_and_weights_them_together():
    from autofly_ue5.dataset.rebalance import (UNION_FORMAT, build_rebalance, build_union_rebalance, phase_distribution,
                                               resampling_weights)

    a, det_a, b, det_b = _two_stores()
    union = build_union_rebalance([(a, det_a, None), (b, det_b, None)], threshold=0.7)
    assert union["format"] == UNION_FORMAT and union["datasets"] == ["s07_a", "s09_a"] and union["threshold"] == 0.7
    records = [7 + 10 + 5 + 12, 3 + 0 + 15 + 0]
    assert union["counts"]["records"] == records and union["counts"]["sub_trajectories"] == [4, 2]
    assert union["counts"]["episodes_never_detected"] == 2 and union["counts"]["records_above_threshold"] == 2 + 15
    p0 = phase_distribution(records)
    assert union["p0"] == list(p0) and union["weights"] == list(resampling_weights(p0))
    assert union["detection_persistence"] == pytest.approx((2 + 15) / (3 + 15))
    assert union["coverage"] == {"episodes": 4, "of": 4, "partial": False,
                                 "by_store": {"s07_a": {"episodes": 2, "of": 2, "partial": False},
                                              "s09_a": {"episodes": 2, "of": 2, "partial": False}}}
    for manifest, detections in ((a, det_a), (b, det_b)):
        single = build_rebalance(manifest, detections, threshold=0.7)
        store = union["stores"][manifest["name"]]
        for key in ("counts", "p0", "weights", "detection_persistence", "sparse", "degenerate"):
            assert store[key] == single[key], key
    assert [(e["store"], e["id"], e["target"], e["first_detection"]) for e in union["episodes"]] == [
        ("s07_a", "e0", "orange cylinder", 7), ("s07_a", "e1", "orange cylinder", None),
        ("s09_a", "e0", "red cube", 5), ("s09_a", "e1", "blue cone", None)]
    targets = union["by_target"]
    assert (targets["orange cylinder"]["episodes"], targets["orange cylinder"]["never_detected"]) == (2, 1)
    assert (targets["red cube"]["episodes"], targets["red cube"]["never_detected"]) == (1, 0)
    assert targets["red cube"]["first_detection_fraction_median"] == pytest.approx(5 / 20)
    assert (targets["blue cone"]["episodes"], targets["blue cone"]["never_detected"]) == (1, 1)
    assert union["targets_never_detected"] == ["blue cone"], "a target the detector never found in any episode"
    assert union["stores"]["s09_a"]["by_target"]["blue cone"]["never_detected"] == 1


def test_the_union_reports_distances_per_target_from_the_states():
    from autofly_ue5.dataset.rebalance import build_union_rebalance

    a, det_a, b, det_b = _two_stores()
    states_a = {"e0": np.zeros((10, 9), np.float32), "e1": np.zeros((10, 9), np.float32)}
    states_b = {"e0": np.zeros((20, 9), np.float32), "e1": np.zeros((12, 9), np.float32)}
    states_a["e0"][:, 0] = 30.0
    states_b["e0"][:, 0] = 12.0
    union = build_union_rebalance([(a, det_a, states_a), (b, det_b, states_b)], threshold=0.7)
    assert union["by_target"]["orange cylinder"]["distance_m_median"] == pytest.approx(30.0)
    assert union["by_target"]["red cube"]["distance_m_median"] == pytest.approx(12.0)
    assert union["by_target"]["blue cone"]["distance_m_median"] is None
    assert union["first_detection"]["distance_m_median"] == pytest.approx(21.0)
    assert union["stores"]["s09_a"]["first_detection"]["distance_m_median"] == pytest.approx(12.0)
    assert union["episodes"][0]["first_detection_distance_m"] == pytest.approx(30.0)


def test_the_union_refuses_to_mix_searches_or_detectors_unless_told():
    from autofly_ue5.dataset.rebalance import build_union_rebalance

    a = {**_manifest((40,)), "name": "a"}
    b = {**_manifest((40,)), "name": "b"}
    dense_a = _detections(a, {"e0": _flicker()})
    sparse_b = _sparse_detections(b, {"e0": _flicker()})
    with pytest.raises(ValueError, match="search"):
        build_union_rebalance([(a, dense_a, None), (b, sparse_b, None)], threshold=0.7)
    mixed = build_union_rebalance([(a, dense_a, None), (b, sparse_b, None)], threshold=0.7, allow_mixed_search=True)
    assert mixed["stores"]["a"]["sparse"] is None and mixed["stores"]["b"]["sparse"]["stride"] == 8
    assert mixed["counts"]["records_above_threshold"] is None and mixed["detection_persistence"] is None
    assert mixed["stores"]["a"]["detection_persistence"] is not None, "the dense store keeps its own"
    with pytest.raises(ValueError, match="search"):
        build_union_rebalance([(a, _sparse_detections(a, {"e0": _flicker()}), None),
                               (b, _sparse_detections(b, {"e0": _flicker()}, stride=4), None)], threshold=0.7)
    with pytest.raises(ValueError, match="detector"):
        build_union_rebalance([(a, dense_a, None), (b, _detections(b, {"e0": _flicker()}, detector={"model": "other"}), None)],
                              threshold=0.7)
    with pytest.raises(ValueError, match="twice"):
        build_union_rebalance([(a, dense_a, None), (a, dense_a, None)], threshold=0.7)
    with pytest.raises(ValueError):
        build_union_rebalance([], threshold=0.7)


# ------------------------------------------------------------------------------------------------------------------
# scripts/rebalance_dataset.py: either cache form, sparse detections, several stores
# ------------------------------------------------------------------------------------------------------------------


def _rename_store(root: Path, name: str) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["name"] = name
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_the_store_s_detections_are_found_in_either_form_but_never_guessed_between(tmp_path):
    from scripts.detect_targets import run
    from scripts.rebalance_dataset import main

    _summary, root = _collect(tmp_path, n_keep=2)
    _rising_detections(root)  # detections.json, the pilot's form
    run(root, root / "detections.jsonl", _MeanPixelDetector())  # and the appendable one
    assert main(["--raw", str(root), "--out", str(tmp_path / "x.json")]) != 0, "two caches: which one is meant?"
    assert main(["--raw", str(root), "--detections", str(root / "detections.json"), "--out", str(tmp_path / "x.json")]) == 0
    (root / "detections.json").unlink()
    assert main(["--raw", str(root), "--threshold", "0.49", "--out", str(tmp_path / "y.json")]) == 0
    assert json.loads((tmp_path / "y.json").read_text())["detections"]["path"].endswith("detections.jsonl")


def test_a_sparse_scoring_feeds_the_weights_end_to_end(tmp_path):
    from scripts.detect_targets import run
    from scripts.rebalance_dataset import main

    _summary, root = _collect(tmp_path, n_keep=3)
    detector, scores = _scripted(root)
    run(root, root / "detections.jsonl", detector, sparse={"stride": 4, "threshold": 0.7, "alert": 0.5})
    assert main(["--raw", str(root), "--threshold", "0.75"]) != 0, "the search answers for its own threshold only"
    assert not (root / "rebalance.json").exists()
    assert main(["--raw", str(root)]) == 0
    result = json.loads((root / "rebalance.json").read_text())
    assert [ep["first_detection"] for ep in result["episodes"]] == [len(scores[ep["id"]]) // 3 for ep in result["episodes"]]
    assert result["sparse"]["stride"] == 4 and result["detection_persistence"] is None
    assert result["by_target"]["orange cylinder"]["episodes"] == 3
    assert result["by_target"]["orange cylinder"]["distance_m_median"] is not None


def test_the_cli_writes_one_document_for_several_stores(tmp_path):
    import hashlib

    from scripts.rebalance_dataset import main

    _s, a = _collect(tmp_path / "a", n_keep=2)
    _s, b = _collect(tmp_path / "b", n_keep=1)
    _rename_store(b, "pilot_b")
    _rising_detections(a)
    _rising_detections(b)
    out = tmp_path / "union" / "rebalance.json"
    assert main(["--raw", str(a), "--raw", str(b)]) != 0, "several stores need an --out: the union is no store's own"
    assert main(["--raw", str(a), "--raw", str(b), "--detections", str(a / "detections.json"), "--out", str(out)]) != 0
    assert main(["--raw", str(a), "--raw", str(b), "--allow-partial", "--out", str(b / "union.json")]) != 0
    assert not out.exists() and not (b / "union.json").exists()
    assert main(["--raw", str(a), "--raw", str(b), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["datasets"] == ["pilot", "pilot_b"] and result["coverage"]["episodes"] == 3
    assert result["stores"]["pilot"]["detections"]["sha256"] == hashlib.sha256((a / "detections.json").read_bytes()).hexdigest()
    assert {e["store"] for e in result["episodes"]} == {"pilot", "pilot_b"}
    assert result["by_target"]["orange cylinder"]["episodes"] == 3
    assert not (a / "rebalance.json").exists() and not (b / "rebalance.json").exists()
    assert main(["--raw", str(a), "--raw", str(b), "--out", str(out)]) != 0, "an existing document is never replaced"
