"""scripts/rebalance_figures.py: what a rebalance document (scripts/rebalance_dataset.py: one store's, or the union of
several) says, as figures and the numbers they draw: the phase shares overall and per scene, each target's first
confident detection (the distance, with the episodes never detected marked) and the resampling weights against the
paper's. Synthetic documents are made by autofly_ue5/dataset/rebalance.py itself; the M3 pilot's document is the
committed copy in the status report (results/2026-10-07-project-status/records/s01_pilot_rebalance.json, the same bytes
as data/s01_pilot/rebalance.json): dense, one store, from before by_target existed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autofly_ue5.paths import ROOT

PNG = b"\x89PNG\r\n\x1a\n"
PILOT_DOCUMENT = ROOT / "results" / "2026-10-07-project-status" / "records" / "s01_pilot_rebalance.json"
OUTPUTS = ["rebalance_phases.png", "rebalance_summary.json", "rebalance_summary.md", "rebalance_targets.png"]


def _distances(n: int) -> np.ndarray:
    return np.linspace(60.0, 5.0, n)


def _store(name: str, scene: str, episodes, *, sparse: bool = False):
    """(manifest, detections, states) of a store whose episodes are (seed, target, scores); the target stands dead ahead,
    so every box at the image's centre is on it."""
    from autofly_ue5.dataset.detections import FORMAT
    from autofly_ue5.dataset.sparse import describe, replay

    manifest = {"name": name, "created": "2026-10-08 12:00:00",
                "episodes": [{"id": f"{scene}_{seed}", "scene": scene, "steps": len(scores), "target_name": target,
                              "path": f"episodes/{scene}_{seed}"} for seed, target, scores in episodes]}
    scored = {}
    for seed, target, scores in episodes:
        entry = {"query": target, "prompt": f"{target}."}
        if sparse:
            first, visited = replay(scores, stride=4, threshold=0.7, alert=0.5)
            order = sorted(visited)
            entry.update(records=len(scores), visited=order, scores=[scores[i] for i in order],
                         boxes=[[0.5, 0.5, 0.1, 0.2]] * len(order), first_detection=first)
        else:
            entry.update(scores=scores, boxes=[[0.5, 0.5, 0.1, 0.2]] * len(scores))
        scored[f"{scene}_{seed}"] = entry
    detections = {"format": FORMAT, "dataset": name, "dataset_created": manifest["created"], "score": "test",
                  "detector": {"model": "fake", "revision": "0"},
                  "sparse": describe(4, 0.7, 0.5) if sparse else None, "runs": [], "episodes": scored}
    states = {}
    for seed, _target, scores in episodes:
        state = np.zeros((len(scores), 9), dtype=np.float32)
        state[:, 0] = _distances(len(scores))
        states[f"{scene}_{seed}"] = state
    return manifest, detections, states


def _two_stores(sparse: bool = False):
    return [_store("s07_c1", "s07", [(1, "red cube", [0.1] * 7 + [0.8] * 3), (2, "blue cone", [0.1] * 10)], sparse=sparse),
            _store("s09_c1", "s09", [(1, "red cube", [0.1] * 5 + [0.9] * 15), (2, "orange cylinder", [0.2] * 12),
                                     (3, "orange cylinder", [0.1] * 2 + [0.95] * 8)], sparse=sparse)]


def _union_document(tmp_path: Path, sparse: bool = False) -> Path:
    from autofly_ue5.dataset.rebalance import build_union_rebalance, write_rebalance

    path = tmp_path / ("union_sparse.json" if sparse else "union.json")
    write_rebalance(path, build_union_rebalance(_two_stores(sparse), threshold=0.7))
    return path


def _build(tmp_path, document, **kw):
    from scripts.rebalance_figures import build

    return build(document, tmp_path / "figures", **kw)


def test_a_union_document_s_phase_shares_overall_and_per_scene(tmp_path):
    document = _union_document(tmp_path)
    doc = json.loads(document.read_text())
    summary = _build(tmp_path, document)
    overall = summary["overall"]
    assert overall["records"] == [36, 26] == doc["counts"]["records"]
    assert overall["p0"] == pytest.approx([36 / 62, 26 / 62]) and overall["weights"] == pytest.approx(doc["weights"])
    assert overall["episodes"] == 5 and overall["episodes_never_detected"] == 2 and overall["threshold"] == 0.7
    scenes = summary["scenes"]
    assert list(scenes) == ["s07", "s09"]
    assert scenes["s07"]["records"] == [17, 3] and scenes["s07"]["shares"] == pytest.approx([0.85, 0.15])
    assert scenes["s09"]["records"] == [19, 23]
    assert scenes["s09"]["shares"] == pytest.approx([19 / 42, 23 / 42], abs=1e-4), "shares to 4 decimals"
    assert scenes["s09"]["stores"] == ["s09_c1"] and scenes["s09"]["never_detected"] == 1
    assert summary["scene_source"] == "episode ids", "the collector names an episode <scene>_<seed>"
    assert summary["paper"] == {"p0": [0.73, 0.27], "weights": [0.68, 1.85], "threshold": 0.7}
    assert summary["stores"]["s07_c1"]["records"] == [17, 3]
    out = tmp_path / "figures"
    assert sorted(p.name for p in out.iterdir()) == OUTPUTS
    for name in ("rebalance_phases.png", "rebalance_targets.png"):
        assert (out / name).read_bytes()[:8] == PNG
    assert json.loads((out / "rebalance_summary.json").read_text()) == json.loads(json.dumps(summary))
    md = (out / "rebalance_summary.md").read_text()
    assert "| all | 5 | 62 | 0.581 | 0.419 |" in md and "| s07 stacked boxes | 2 | 20 | 0.850 | 0.150 |" in md


def test_each_target_s_first_detection_distance_and_its_episodes_never_detected(tmp_path):
    summary = _build(tmp_path, _union_document(tmp_path))
    targets = summary["targets"]
    assert list(targets) == ["orange cylinder", "red cube", "blue cone"], "farthest median first, never detected last"
    red = targets["red cube"]
    expected = [float(_distances(10)[7]), float(_distances(20)[5])]
    assert red["episodes"] == 2 and red["never_detected"] == 0
    assert red["distances_m"] == pytest.approx(sorted(expected), abs=1e-3)
    assert red["distance_m_median"] == pytest.approx(np.median(expected), abs=1e-3), "distances to the millimetre"
    orange = targets["orange cylinder"]
    assert orange["episodes"] == 2 and orange["never_detected"] == 1
    assert orange["distance_m_median"] == pytest.approx(_distances(10)[2], abs=1e-3)
    assert targets["blue cone"] == {**targets["blue cone"], "episodes": 1, "never_detected": 1, "distances_m": [],
                                    "distance_m_median": None}
    assert summary["targets_never_detected"] == ["blue cone"]
    assert summary["first_detection"]["on_target"] == 3 and summary["first_detection"]["off_target"] == 0
    md = (tmp_path / "figures" / "rebalance_summary.md").read_text()
    assert "| blue cone | 1 | **1** |" in md


def test_a_sparse_union_is_drawn_without_the_counts_only_dense_scores_give(tmp_path):
    summary = _build(tmp_path, _union_document(tmp_path, sparse=True))
    assert summary["overall"]["records_above_threshold"] is None and summary["overall"]["detection_persistence"] is None
    assert summary["overall"]["records"] == [36, 26], "a sparse search finds the same first detections here"
    assert summary["search"] == {"s07_c1": "sparse", "s09_c1": "sparse"}
    assert (tmp_path / "figures" / "rebalance_targets.png").is_file()


def test_the_m3_pilot_s_dense_single_store_document_is_read_as_written():
    doc = json.loads(PILOT_DOCUMENT.read_text())
    assert doc["format"] == "autofly_ue5_rebalance/1" and "by_target" not in doc, "the pilot predates by_target"


def test_the_pilot_document_s_numbers_come_through(tmp_path):
    from scripts.rebalance_figures import UNRECORDED_TARGET

    doc = json.loads(PILOT_DOCUMENT.read_text())
    summary = _build(tmp_path, PILOT_DOCUMENT)
    assert summary["kind"] == "single store" and summary["datasets"] == ["s01_pilot"]
    assert summary["overall"]["p0"] == doc["p0"] and summary["overall"]["weights"] == doc["weights"]
    assert summary["overall"]["records"] == [11572, 6166] and summary["overall"]["episodes"] == 100
    assert list(summary["scenes"]) == ["s01"] and summary["scenes"]["s01"]["shares"] == pytest.approx(doc["p0"], abs=1e-4)
    assert list(summary["targets"]) == [UNRECORDED_TARGET], "the single-store document does not name the targets"
    pilot = summary["targets"][UNRECORDED_TARGET]
    assert pilot["episodes"] == 100 and pilot["never_detected"] == 0
    assert pilot["distance_m_median"] == pytest.approx(doc["first_detection"]["distance_m_median"], abs=1e-3)
    assert summary["first_detection"]["on_target"] == 99 and summary["overall"]["detection_persistence"] == pytest.approx(
        doc["detection_persistence"])


def test_a_store_s_manifest_names_the_targets_and_scenes_a_single_store_document_leaves_out(tmp_path):
    from autofly_ue5.dataset.rebalance import build_rebalance

    manifest, detections, states = _two_stores()[1]
    doc = build_rebalance(manifest, detections, threshold=0.7, states=states)
    for key in ("by_target", "targets_never_detected", "sparse"):  # as the pilot's, written before these existed
        doc.pop(key)
    for episode in doc["episodes"]:
        episode["id"] = episode["id"].replace("s09_", "ep")  # ids that do not say their scene
    for entry in manifest["episodes"]:
        entry["id"] = entry["id"].replace("s09_", "ep")
    path = tmp_path / "old.json"
    path.write_text(json.dumps(doc))
    raw = tmp_path / "data" / "s09_c1"
    raw.mkdir(parents=True)
    (raw / "manifest.json").write_text(json.dumps(manifest))
    bare = _build(tmp_path, path)
    assert list(bare["scenes"]) == ["unknown"] and bare["scene_source"] is None
    named = _build(tmp_path / "named", path, raws=[raw])
    assert list(named["scenes"]) == ["s09"] and named["scene_source"] == "manifests"
    assert sorted(named["targets"]) == ["orange cylinder", "red cube"]
    assert named["targets"]["orange cylinder"]["never_detected"] == 1


def test_a_document_of_another_kind_is_refused_and_outputs_over_committed_evidence_too(tmp_path):
    from autofly_ue5 import evidence
    from scripts.rebalance_figures import RebalanceFigureError, main

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"format": "autofly_ue5_detections/2"}))
    with pytest.raises(RebalanceFigureError, match="rebalance document"):
        _build(tmp_path, other)
    assert main(["--rebalance", str(other), "--out-dir", str(tmp_path / "x")]) == 2
    document = _union_document(tmp_path)
    assert main(["--rebalance", str(document), "--out-dir", str(tmp_path / "ok")]) == 0
    gates_out = evidence.GATES_DIR / "rebalance"  # the suite's scratch stand-in for docs/gates/ (tests/conftest.py)
    gates_out.mkdir(parents=True)
    (gates_out / "rebalance_summary.json").write_text("{}")
    assert main(["--rebalance", str(document), "--out-dir", str(gates_out)]) == 2
    assert (gates_out / "rebalance_summary.json").read_text() == "{}" and not (gates_out / "rebalance_phases.png").exists()
