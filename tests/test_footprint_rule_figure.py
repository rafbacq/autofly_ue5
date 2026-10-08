"""scripts/footprint_rule_figure.py draws the footprint rule from scripted FakeSimulator flights: it runs on a tree scene
given as files (a scene JSON and a stored layout, as runs/levels/ holds them), writes only its --out and --json, and
records that no flight reached the fake's wood."""

from __future__ import annotations

import dataclasses
import json
import os

from autofly_ue5.scenes.model import load_scene_file
from tests.test_footprint_contact import tree_scene_and_layout, tree_scene_dict


def _write_inputs(tmp_path):
    inputs = tmp_path / "in"
    inputs.mkdir()
    scene_path = inputs / "s99_test_trees.json"
    scene_path.write_text(json.dumps(tree_scene_dict()))
    _scene, layout = tree_scene_and_layout()
    layout_path = inputs / "s99.layout.json"
    layout_path.write_text(json.dumps({"scene_path": str(scene_path), "scene_sha256": load_scene_file(scene_path).sha256,
                                       "layout": dataclasses.asdict(layout)}))
    return scene_path, layout_path


def test_the_figure_and_its_numbers_are_written_and_no_flight_reaches_wood(tmp_path, monkeypatch):
    from scripts import footprint_rule_figure

    scene_path, layout_path = _write_inputs(tmp_path)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)  # a stray relative write would land here
    out, record = tmp_path / "out" / "fig.png", tmp_path / "out" / "fig.json"
    code = footprint_rule_figure.main(["--scene", str(scene_path), "--layout", str(layout_path), "--out", str(out),
                                       "--json", str(record), "--seeds", "3"])
    assert code == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = json.loads(record.read_text())
    summary = data["summary"]
    assert summary["all_checks_passed"] is True
    for run in ("task_rule", "training_margin"):
        assert summary["runs"][run]["checks"]["no_sim_endings"] is True
        assert "sim" not in summary["runs"][run]["endings"]
        assert len(data["episodes"][run]) == 3
    assert data["seeds"] == {"base": 300_000_000, "n": 3}
    written = {os.path.relpath(os.path.join(d, f), tmp_path) for d, _, files in os.walk(tmp_path) for f in files}
    assert written == {"in/s99_test_trees.json", "in/s99.layout.json", "out/fig.png", "out/fig.json"}


def test_a_layout_from_another_scene_is_refused(tmp_path):
    import pytest

    from scripts import footprint_rule_figure

    scene_path, layout_path = _write_inputs(tmp_path)
    stored = json.loads(layout_path.read_text())
    stored["scene_sha256"] = "0" * 64
    layout_path.write_text(json.dumps(stored))
    with pytest.raises(SystemExit, match="another version"):
        footprint_rule_figure.main(["--scene", str(scene_path), "--layout", str(layout_path), "--out",
                                    str(tmp_path / "f.png"), "--json", str(tmp_path / "f.json"), "--seeds", "1"])
    assert not (tmp_path / "f.png").exists() and not (tmp_path / "f.json").exists()
