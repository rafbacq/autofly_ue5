"""scripts/footprint_endings_figure.py measures where the footprint rule ended a run's evaluation flights against the
trees' actual glTF geometry: it reads a run's eval-watch records (skipping one still being written) and a stored layout,
writes only its --out and --json, and checks its own transform against validate/obstacles.py's before trusting it.
The geometry is the fetched Poly Haven models, so the test skips on a checkout without them."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path

import pytest

from autofly_ue5.paths import ASSET_MEASUREMENTS, ASSET_REGISTRY, ASSET_SOURCES
from autofly_ue5.scenes.model import load_scene_file
from tests.test_footprint_contact import tree_scene_and_layout, tree_scene_dict

GAP_M = 0.7  # the drone stopped this far outside the circle: inside the rule's 0.9 m


def _gltf_files_present(assets: set[str]) -> bool:
    """Whether the glTF behind each registry key is on disk, resolved as MeshLibrary.from_files resolves it."""
    if not (ASSET_SOURCES.exists() and ASSET_MEASUREMENTS.exists() and ASSET_REGISTRY.exists()):
        return False
    downloads = Path(json.loads(ASSET_SOURCES.read_text())["downloads_dir"])
    entries = json.loads(ASSET_REGISTRY.read_text())["assets"]
    files = {(m["asset"], m["node"]): m["file"] for m in json.loads(ASSET_MEASUREMENTS.read_text())["meshes"]}
    for key in assets:
        source = entries[key]["source"]
        if not (downloads / source["source"] / source["id"] / files[(source["id"], source["node"])]).is_file():
            return False
    return True


def _write_inputs(tmp_path: Path):
    """The test tree scene's layout as runs/levels/ stores one, and one eval-watch record whose only footprint ending is
    GAP_M outside an island_tree_02's circle (a leaning model, so its enclosing circle sits off the pivot), flying past
    it at 2 m. A record still being written (*.partial.json) sits beside it and must be skipped."""
    _scene, layout = tree_scene_and_layout()
    if not _gltf_files_present({inst.asset for inst in layout.instances}):
        pytest.skip("the fetched glTF models (downloads/) are not on this checkout")
    inputs = tmp_path / "in"
    inputs.mkdir()
    scene_path = inputs / "s99_test_trees.json"
    scene_path.write_text(json.dumps(tree_scene_dict()))
    layout_path = inputs / "s99.layout.json"
    layout_path.write_text(json.dumps({"scene_path": str(scene_path), "scene_sha256": load_scene_file(scene_path).sha256,
                                       "layout": dataclasses.asdict(layout)}))
    tree = next(inst for inst in layout.instances if inst.asset == "island_tree_02")
    x, y, yaw = tree.x + tree.radius_m + GAP_M, tree.y, math.pi / 2  # due north of the pivot, heading east
    ending = {"seed": 200_000_000, "outcome": "collision", "steps": 40, "final_pose": [x, y, -2.0, yaw],
              "collision_source": "footprint",
              "footprint_contact": {"tag": tree.tag, "asset": tree.asset, "gap_m": GAP_M, "rule": "footprint",
                                    "bearing_deg": 90.0, "drone_forward_m_s": 2.0}}
    success = {"seed": 200_000_001, "outcome": "success", "steps": 90, "final_pose": [0.0, 0.0, -2.0, 0.0],
               "collision_source": None}
    watch = tmp_path / "run" / "eval_watch"
    watch.mkdir(parents=True)
    (watch / "step_1000.json").write_text(json.dumps(
        {"checkpoints": {"step_1000": {"deterministic": {"per_episode": [ending, success]}}}}))
    (watch / "step_2000.partial.json").write_text("{ not finished")
    return layout_path, tree


def test_the_figure_and_its_numbers_are_written_and_the_transform_checks_pass(tmp_path, monkeypatch):
    from scripts import footprint_endings_figure

    layout_path, tree = _write_inputs(tmp_path)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)  # a stray relative write would land here
    out, record_path = tmp_path / "out" / "fig.png", tmp_path / "out" / "fig.json"
    code = footprint_endings_figure.main(["--run-root", str(tmp_path / "run"), "--layout", str(layout_path),
                                          "--out", str(out), "--json", str(record_path)])
    assert code == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = json.loads(record_path.read_text())
    assert data["eval_watch"]["records"] == ["step_1000.json"] and data["eval_watch"]["flights"] == 2
    assert data["checks"]["passed"] is True and data["checks"]["window_depths_agree"] is True
    assert data["checks"]["round_trip_max_error_m"] < 1e-9
    assert data["checks"]["recorded_gap_is_final_gap_max_diff_m"] < 1e-6
    (row,) = data["endings"]
    assert row["tag"] == tree.tag and row["circle_gap_final_m"] == pytest.approx(GAP_M, abs=1e-4)
    # the circle holds every corner of the model, so nothing can be nearer than the drone's gap to it
    for key in ("clearance_column", "clearance_slab", "clearance_3d"):
        assert row[key]["all"] >= GAP_M - 1e-3, key
    assert row["clearance_final"]["all"]["min"] is not None and row["clearance_final"]["all"]["min"] >= GAP_M - 1e-3
    assert len(data["trees"]) == len(tree_scene_and_layout()[1].instances)
    for t in data["trees"]:  # the enclosing circle never exceeds the pivot circle it tightens
        assert t["mec_all"]["radius_m"] <= t["radius_m"] + 1e-3 and t["area_reduction"] >= -1e-3
    leaning = data["assets"]["island_tree_02"]["unit_scale"]["mec_all"]
    assert leaning["centre_offset_m"] > 0.5, "island_tree_02 leans: its enclosing circle sits off the pivot"
    written = {os.path.relpath(os.path.join(d, f), tmp_path) for d, _, files in os.walk(tmp_path) for f in files}
    assert written == {"in/s99_test_trees.json", "in/s99.layout.json", "run/eval_watch/step_1000.json",
                       "run/eval_watch/step_2000.partial.json", "out/fig.png", "out/fig.json"}


def test_a_run_without_eval_watch_records_is_refused(tmp_path):
    from scripts import footprint_endings_figure

    layout_path, _tree = _write_inputs(tmp_path)
    for p in (tmp_path / "run" / "eval_watch").iterdir():
        if not p.name.endswith(".partial.json"):
            p.unlink()
    with pytest.raises(SystemExit, match="no eval-watch records"):
        footprint_endings_figure.main(["--run-root", str(tmp_path / "run"), "--layout", str(layout_path),
                                       "--out", str(tmp_path / "f.png"), "--json", str(tmp_path / "f.json")])
    assert not (tmp_path / "f.png").exists() and not (tmp_path / "f.json").exists()
