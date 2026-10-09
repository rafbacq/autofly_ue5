"""The sparse search for an episode's first confident detection (plan 6 step 5, Revision 1 item 6): the strategy
(autofly_ue5/dataset/sparse.py), the cache format shared by the detector and the rebalancing stage
(autofly_ue5/dataset/detections.py), and the study that validates the search against the s01 pilot's dense scores
(scripts/sparse_search_study.py). All offline."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _dense_first(scores, threshold):
    above = [i for i, s in enumerate(scores) if s > threshold]
    return above[0] if above else None


def _random_episode(rng, n):
    """A score profile shaped like the pilot's: low far out, rising with noise, a flicker zone around the threshold."""
    base = np.clip(np.linspace(0.25, 0.9, n) + rng.normal(0.0, 0.06, n), 0.0, 1.0)
    return [round(float(v), 6) for v in base]


# ------------------------------------------------------------------------------------------------------------------
# autofly_ue5/dataset/sparse.py
# ------------------------------------------------------------------------------------------------------------------


def test_the_coarse_probes_are_every_stride_th_record_and_the_last():
    from autofly_ue5.dataset.sparse import coarse_probes

    assert coarse_probes(10, 4) == [0, 4, 8, 9]
    assert coarse_probes(9, 4) == [0, 4, 8], "the last record is a probe already"
    assert coarse_probes(1, 8) == [0]
    assert coarse_probes(5, 1) == [0, 1, 2, 3, 4]
    assert coarse_probes(10, 4, offset=2) == [0, 2, 6, 9], "the study's shifted grids keep the first and last record"
    for bad in ((0, 4, 0), (10, 0, 0), (10, 4, 4), (10, 4, -1), (10, True, 0)):
        with pytest.raises(ValueError):
            coarse_probes(bad[0], bad[1], offset=bad[2])


def test_at_stride_one_the_search_is_the_dense_first_detection_and_stops_there():
    from autofly_ue5.dataset.sparse import replay

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 120))
        scores = _random_episode(rng, n)
        first, visited = replay(scores, stride=1, threshold=0.7, alert=0.7)
        assert first == _dense_first(scores, 0.7)
        assert list(visited) == list(range(len(visited))), "frames in order, none skipped"
        assert len(visited) == (n if first is None else first + 1), "nothing is scored after the first detection"


def test_with_every_window_alerting_any_stride_finds_the_dense_first_detection():
    from autofly_ue5.dataset.sparse import replay

    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 150))
        scores = [max(s, 1e-6) for s in _random_episode(rng, n)]  # every probe scores above an alert level of 0
        for stride in (2, 3, 5, 8, 16):
            first, _visited = replay(scores, stride=stride, threshold=0.7, alert=0.0)
            assert first == _dense_first(scores, 0.7), (stride, scores)


def test_the_search_never_reports_a_frame_below_the_threshold_nor_one_before_the_dense_first():
    from autofly_ue5.dataset.sparse import replay

    rng = np.random.default_rng(2)
    for _ in range(300):
        n = int(rng.integers(1, 200))
        scores = _random_episode(rng, n)
        dense = _dense_first(scores, 0.7)
        for stride in (2, 4, 8, 13):
            for alert in (0.7, 0.6, 0.5):
                first, visited = replay(scores, stride=stride, threshold=0.7, alert=alert)
                assert all(visited[i] == scores[i] for i in visited)
                if first is not None:
                    assert scores[first] > 0.7 and dense is not None and first >= dense
                    assert all(visited[i] <= 0.7 for i in visited if i < first), "the first above the threshold in order"


def test_the_alert_windows_catch_a_flicker_that_the_plain_coarse_scan_misses():
    from autofly_ue5.dataset.sparse import replay

    # The pilot's typical first detection: one record just above 0.7 among records hovering below it, the sustained
    # detection later. Probes at 0, 8, 16, ...
    scores = [0.3] * 40
    scores[5] = 0.71   # the flicker, between probes 0 and 8
    scores[8] = 0.62   # the probe after it hovers below the threshold but above the alert level
    for i in range(30, 40):
        scores[i] = 0.85
    assert replay(scores, stride=8, threshold=0.7, alert=0.7)[0] == 30, "the plain scan: a window only before a hit"
    first, visited = replay(scores, stride=8, threshold=0.7, alert=0.5)
    assert first == 5 and sorted(visited) == [0, 1, 2, 3, 4, 5, 8], "the window before the alerting probe 8 is scanned"

    scores[8] = 0.45   # both probes at or below the alert level: the search's one bet, lost here
    assert replay(scores, stride=8, threshold=0.7, alert=0.5)[0] == 30


def test_a_window_after_an_alerting_probe_is_scanned_too():
    from autofly_ue5.dataset.sparse import replay

    scores = [0.3] * 30
    scores[8] = 0.6    # alerts the windows on both sides
    scores[11] = 0.72  # the flicker after it, before the low probe 16
    first, visited = replay(scores, stride=8, threshold=0.7, alert=0.5)
    assert first == 11
    assert sorted(visited) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16]


def test_batching_changes_which_extra_frames_are_scored_never_the_result():
    from autofly_ue5.dataset.sparse import search

    rng = np.random.default_rng(3)
    for _ in range(100):
        n = int(rng.integers(1, 200))
        scores = _random_episode(rng, n)
        reference = None
        for batch in (1, 2, 4, 7):
            calls = []

            def score(indices, calls=calls):
                assert 1 <= len(indices) <= batch and indices == sorted(set(indices))
                calls.append(list(indices))
                return [scores[i] for i in indices]

            first, visited = search(n, stride=8, threshold=0.7, alert=0.5, score=score, batch=batch)
            flat = [i for c in calls for i in c]
            assert len(flat) == len(set(flat)) == len(visited), "no frame is scored twice"
            if reference is None:
                reference = (first, set(visited))
            else:
                assert first == reference[0] and reference[1] <= set(visited)
                assert len(visited) - len(reference[1]) <= 2 * (batch - 1), "a batch overshoots by at most batch - 1, twice"


def test_bad_scores_and_parameters_are_refused():
    from autofly_ue5.dataset.sparse import describe, search

    with pytest.raises(ValueError, match="returned"):
        search(10, stride=4, threshold=0.7, alert=0.5, score=lambda idx: [0.1] * (len(idx) + 1))
    for bad in (float("nan"), 1.2, -0.1, True, "0.5"):
        with pytest.raises(ValueError):
            search(10, stride=4, threshold=0.7, alert=0.5, score=lambda idx, bad=bad: [bad] * len(idx))
    for kwargs in ({"stride": 0}, {"threshold": 1.0}, {"threshold": 70.0}, {"alert": 0.8}, {"alert": -0.1},
                   {"threshold": float("nan")}, {"stride": 2.0}):
        args = {"stride": 4, "threshold": 0.7, "alert": 0.5, **kwargs}
        with pytest.raises(ValueError):
            describe(**args)
        with pytest.raises(ValueError):
            search(10, score=lambda idx: [0.1] * len(idx), **args)
    with pytest.raises(ValueError):
        search(0, stride=4, threshold=0.7, alert=0.5, score=lambda idx: [0.1] * len(idx))
    with pytest.raises(ValueError):
        search(10, stride=4, threshold=0.7, alert=0.5, score=lambda idx: [0.1] * len(idx), batch=0)
    block = describe(stride=8, threshold=0.7, alert=0.5)
    assert block["strategy"] == "coarse_alert_windows" and (block["stride"], block["threshold"], block["alert"]) == (8, 0.7, 0.5)
    assert "alert" in block["rule"] and "window" in block["rule"]


def test_verify_replays_the_search_on_the_recorded_frames_and_refuses_a_record_missing_one():
    from autofly_ue5.dataset.sparse import replay, verify

    rng = np.random.default_rng(4)
    for _ in range(50):
        n = int(rng.integers(2, 200))
        scores = _random_episode(rng, n)
        first, visited = replay(scores, stride=8, threshold=0.7, alert=0.5)
        assert verify(n, dict(visited), stride=8, threshold=0.7, alert=0.5) == first
        extra = {**visited, n - 1: scores[n - 1]}
        assert verify(n, extra, stride=8, threshold=0.7, alert=0.5) == first, "frames the search did not need are fine"
        needed = sorted(visited)[len(visited) // 2]
        short = {i: s for i, s in visited.items() if i != needed}
        with pytest.raises(ValueError, match=f"record {needed}"):
            verify(n, short, stride=8, threshold=0.7, alert=0.5)
    with pytest.raises(ValueError):
        verify(5, {0: 0.1, 7: 0.9}, stride=4, threshold=0.7, alert=0.5)  # a record index outside the episode


def test_the_modules_the_detector_venv_imports_use_the_standard_library_only():
    """scripts/detect_targets.py runs in runs/tools/gdino_venv, which has torch and transformers but none of the project's
    pins, and imports the search and the cache format from here so that the two stages cannot disagree. Anything beyond
    the standard library in these files (or the package __init__s they pass through) would break the detector."""
    files = ["autofly_ue5/__init__.py", "autofly_ue5/dataset/__init__.py", "autofly_ue5/dataset/sparse.py",
             "autofly_ue5/dataset/detections.py"]
    allowed_local = {"autofly_ue5", "autofly_ue5.dataset", "autofly_ue5.dataset.sparse", "autofly_ue5.dataset.detections"}
    for rel in files:
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] if node.level == 0 else ["autofly_ue5.dataset." + (node.module or "")]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                assert name in allowed_local or top in sys.stdlib_module_names or top == "__future__", f"{rel} imports {name}"


# ------------------------------------------------------------------------------------------------------------------
# autofly_ue5/dataset/detections.py: the appendable cache
# ------------------------------------------------------------------------------------------------------------------


def _header(**overrides):
    from autofly_ue5.dataset.detections import FORMAT

    return {"kind": "header", "format": FORMAT, "dataset": "pilot", "dataset_created": "2026-10-06 12:00:00", "score": "t",
            "detector": {"model": "fake", "revision": "0"}, "sparse": None, **overrides}


def _lines(*objs) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def test_the_cache_reads_a_header_runs_and_episodes_and_sets_aside_a_torn_last_line():
    from autofly_ue5.dataset.detections import FORMAT, parse

    run = {"kind": "run", "started": "2026-10-08 10:00:00", "device": "cpu", "batch": 4, "episodes_total": 3}
    e0 = {"kind": "episode", "id": "e0", "query": "red cube", "prompt": "red cube.", "records": 2, "scores": [0.1, 0.8],
          "boxes": [[0.5, 0.5, 0.1, 0.1]] * 2, "frames_scored": 2, "seconds": 1.5}
    e1 = {**e0, "id": "e1", "seconds": 2.5}
    blob = _lines(_header(), run, e0, e1, {"kind": "run_end", "finished": "2026-10-08 10:01:00"})
    parsed = parse(blob)
    doc = parsed.doc
    assert doc["format"] == FORMAT and doc["dataset"] == "pilot" and doc["sparse"] is None
    assert list(doc["episodes"]) == ["e0", "e1"] and doc["episodes"]["e1"]["scores"] == [0.1, 0.8]
    assert "kind" not in doc["episodes"]["e0"] and "id" not in doc["episodes"]["e0"]
    (r,) = doc["runs"]
    assert r["episodes_scored"] == 2 and r["frames_scored"] == 4 and r["seconds"] == 4.0 and r["finished"] == "2026-10-08 10:01:00"
    assert doc["episodes_total"] == 3 and doc["complete"] is False
    assert parsed.complete_bytes == len(blob) and parsed.torn is False

    torn = blob + json.dumps({**e0, "id": "e2"}).encode()[:40]
    parsed = parse(torn)
    assert parsed.torn is True and parsed.complete_bytes == len(blob) and list(parsed.doc["episodes"]) == ["e0", "e1"]


def test_the_cache_refuses_a_corrupt_line_a_repeated_episode_and_a_missing_header():
    from autofly_ue5.dataset.detections import parse

    e0 = {"kind": "episode", "id": "e0", "query": "q", "prompt": "q.", "records": 1, "scores": [0.1], "boxes": [[0, 0, 0, 0]]}
    with pytest.raises(ValueError, match="line 2"):
        parse(_lines(_header()) + b"{not json\n" + _lines(e0))
    with pytest.raises(ValueError, match="e0"):
        parse(_lines(_header(), e0, e0))
    with pytest.raises(ValueError, match="header"):
        parse(_lines(e0))
    with pytest.raises(ValueError, match="header"):
        parse(b"")
    with pytest.raises(ValueError, match="kind"):
        parse(_lines(_header(), {"kind": "mystery"}))


def test_the_cache_reads_the_pilot_s_whole_document_form_too():
    from autofly_ue5.dataset.detections import FORMAT_V1, parse

    v1 = {"format": FORMAT_V1, "dataset": "pilot", "dataset_created": "x", "score": "t", "detector": {"model": "fake"},
          "episodes_total": 1, "complete": True, "runs": [],
          "episodes": {"e0": {"query": "q", "prompt": "q.", "scores": [0.1], "boxes": [[0, 0, 0, 0]]}}}
    parsed = parse(json.dumps(v1).encode() + b"\n")
    assert parsed.doc["format"] == FORMAT_V1 and parsed.doc["sparse"] is None and parsed.doc["episodes"] == v1["episodes"]
    assert parse(json.dumps(v1, indent=2).encode()).doc["episodes"] == v1["episodes"], "any JSON layout of the old form"


# ------------------------------------------------------------------------------------------------------------------
# scripts/sparse_search_study.py: the sparse search replayed on dense scores
# ------------------------------------------------------------------------------------------------------------------

INSTRUCTION = "go through and avoid the white pillars or other obstacles to reach the orange cylinder"


def _study_store(tmp_path: Path, episodes: dict[str, list[float]]) -> Path:
    """A store with a manifest and a dense detections file in the pilot's (whole-document) form; no frames: the study
    reads scores only."""
    from autofly_ue5.dataset.detections import FORMAT_V1

    root = tmp_path / "store"
    root.mkdir()
    manifest = {"name": "pilot", "created": "2026-10-06 12:00:00",
                "episodes": [{"id": eid, "steps": len(s), "target_name": "orange cylinder", "instruction": INSTRUCTION,
                              "path": f"episodes/{eid}", "scene": "s01"} for eid, s in episodes.items()]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    doc = {"format": FORMAT_V1, "dataset": "pilot", "dataset_created": manifest["created"], "score": "t",
           "detector": {"model": "fake", "revision": "0"}, "episodes_total": len(episodes), "complete": True, "runs": [],
           "episodes": {eid: {"query": "orange cylinder", "prompt": "orange cylinder.", "scores": s,
                              "boxes": [[0.5, 0.5, 0.1, 0.1]] * len(s)} for eid, s in episodes.items()}}
    (root / "detections.json").write_text(json.dumps(doc) + "\n")
    return root


def _flicker_store(tmp_path):
    steady = [0.3] * 30 + [0.85] * 10                       # first detection 30, a probe of every grid
    flicker = [0.3] * 40
    flicker[5], flicker[8] = 0.71, 0.62                     # first detection 5, a one-record flicker
    flicker[30:] = [0.85] * 10
    never = [0.4] * 25                                      # never above 0.7
    return _study_store(tmp_path, {"steady": steady, "flicker": flicker, "never": never})


def test_the_study_replays_each_strategy_on_the_dense_scores_and_counts_what_it_changes(tmp_path):
    from autofly_ue5.dataset.detections import parse
    from autofly_ue5.dataset.rebalance import build_rebalance
    from scripts.sparse_search_study import study

    root = _flicker_store(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    dense = parse((root / "detections.json").read_bytes()).doc
    result = study(manifest, dense, threshold=0.7, strides=[1, 8], alerts=[0.7, 0.5])
    reference = build_rebalance(manifest, dense, threshold=0.7)
    assert result["dense"]["p0"] == reference["p0"] and result["dense"]["weights"] == reference["weights"]
    assert result["dense"]["first_detection"] == {"steady": 30, "flicker": 5, "never": None}
    assert result["dense"]["calls_total"] == 105, "the dense mode scores every record"
    rows = {(r["stride"], r["alert"]): r for r in result["rows"]}
    exact = rows[(1, 0.7)]
    assert exact["strategy"] == "coarse_fine" and exact["episodes_differing"] == 0 and exact["differences"] == []
    assert exact["calls_total"] == 31 + 6 + 25, "stride 1 stops at the first detection"
    assert exact["p0"] == reference["p0"] and exact["label_change_fraction"] == 0.0

    plain = rows[(8, 0.7)]   # probes 0, 8, 16, 24, 32, 39: the flicker at 5 is missed, the steady run found at 30
    assert plain["episodes_differing"] == 1
    assert plain["differences"] == [{"id": "flicker", "dense": 5, "sparse": 30, "delta": 25, "dense_score": 0.71}]
    assert plain["label_change_fraction"] == pytest.approx(25 / 105)
    counts = [30 + 30 + 25, 10 + 10]                        # phase records with the flicker found at 30
    assert plain["p0"] == pytest.approx([counts[0] / 105, counts[1] / 105])
    assert plain["weights"] == pytest.approx([0.5 / (counts[0] / 105), 0.5 / (counts[1] / 105)])
    assert plain["p0_delta"] == pytest.approx(plain["p0"][0] - reference["p0"][0])
    # Over the eight grid positions only a probe landing on record 5 itself catches the flicker
    assert plain["offset_misses"] == {"flicker": 7}
    assert plain["differing_over_offsets_per_100"] == pytest.approx(100 * 7 / (8 * 3))

    alert = rows[(8, 0.5)]
    assert alert["strategy"] == "coarse_alert_windows" and alert["episodes_differing"] == 0
    assert alert["calls_max"] < 40 and alert["calls_mean"] < exact["calls_mean"]
    for row in result["rows"]:
        assert 0.0 <= row["differing_over_offsets_per_100"] <= 100.0
        assert row["calls_mean_batch4"] >= row["calls_mean"]


def test_the_study_cli_writes_the_table_and_names_the_cheapest_row_that_meets_the_rule(tmp_path):
    from scripts.sparse_search_study import main

    root = _flicker_store(tmp_path)
    out = tmp_path / "study" / "sparse_study.json"
    assert main(["--raw", str(root), "--out", str(out), "--strides", "1-8", "--alerts", "0.7,0.6,0.5"]) == 0
    result = json.loads(out.read_text())
    assert {(r["stride"], r["alert"]) for r in result["rows"]} == {(s, a) for s in range(1, 9) for a in (0.7, 0.6, 0.5)}
    rec = result["recommendation"]
    meeting = [r for r in result["rows"] if r["same_transition_share"] >= 0.99 and r["differing_over_offsets_per_100"] <= 0.5]
    best = min(meeting, key=lambda r: (r["calls_mean"], -r["stride"], r["alert"]))
    assert (rec["stride"], rec["alert"]) == (best["stride"], best["alert"]) and "rule" in rec
    table = out.with_suffix(".md").read_text()
    assert "| strategy |" in table and table.count("\n|") >= 1 + len(result["rows"])
    assert result["source"]["detections_sha256"] and result["threshold"] == 0.7


def test_the_live_check_compares_each_visited_frame_with_its_dense_score(tmp_path):
    from autofly_ue5.dataset.detections import FORMAT
    from autofly_ue5.dataset.sparse import describe, replay
    from scripts.sparse_search_study import main

    root = _flicker_store(tmp_path)
    dense = json.loads((root / "detections.json").read_text())
    block = describe(stride=8, threshold=0.7, alert=0.5)
    lines = [{"kind": "header", "format": FORMAT, "dataset": "pilot", "dataset_created": dense["dataset_created"],
              "score": "t", "detector": dense["detector"], "sparse": block}]
    for eid, ep in dense["episodes"].items():
        first, visited = replay(ep["scores"], stride=8, threshold=0.7, alert=0.5)
        order = sorted(visited)
        lines.append({"kind": "episode", "id": eid, "query": ep["query"], "prompt": ep["prompt"], "records": len(ep["scores"]),
                      "visited": order, "scores": [round(visited[i] + 2e-6, 6) for i in order],
                      "boxes": [ep["boxes"][i] for i in order], "first_detection": first})
    sparse_path = tmp_path / "live" / "detections.jsonl"
    sparse_path.parent.mkdir()
    sparse_path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    out = tmp_path / "live" / "check.json"
    assert main(["--raw", str(root), "--check-sparse", str(sparse_path), "--out", str(out)]) == 0
    check = json.loads(out.read_text())
    assert check["frames_compared"] == sum(len(x["visited"]) for x in lines[1:])
    assert check["score_abs_diff_max"] == pytest.approx(2e-6, abs=1e-9)
    assert check["episodes"] == 3 and check["first_detection_equal_dense"] == 3 and check["first_detection_equal_predicted"] == 3
    assert all(e["needed_frames_visited"] for e in check["per_episode"])
