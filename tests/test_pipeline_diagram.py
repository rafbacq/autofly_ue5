"""scripts/pipeline_diagram.py: the system overview for the final README, drawn with matplotlib. Its counts are read from
the repository (the scene files' splits, the target pool) and the code's own constants, never typed into the figure; its
eight stages appear in reading order; and no line of a box's text runs outside its box at the size it is saved at."""

from __future__ import annotations

import json

import pytest

PNG = b"\x89PNG\r\n\x1a\n"
STAGES = ["Scenes", "Expert pilots", "Collector", "Raw stores", "Merged dataset", "Rebalancing", "Next: a VLA",
          "Evaluation harness"]


def _repository(tmp_path, train=("s01", "s02", "s03"), test_seen=("s05r",), targets=7):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    for scene_id in train:
        (scenes / f"{scene_id}_x.json").write_text(json.dumps({"id": scene_id, "split": "train"}))
    for scene_id in test_seen:
        (scenes / f"{scene_id}_x.json").write_text(json.dumps({"id": scene_id, "split": "test_seen"}))
    (scenes / "s99_unseen.json").write_text(json.dumps({"id": "s99", "split": "test_unseen"}))
    pool = tmp_path / "targets.json"
    pool.write_text(json.dumps({"version": 1, "targets": [
        {"name": f"t{i}", "category": ("vehicle", "human", "geometry")[i % 3], "seen": i % 4 != 0}
        for i in range(targets)]}))
    return scenes, pool


def test_the_counts_come_from_the_scene_files_the_pool_and_the_code(tmp_path):
    from autofly_ue5.dataset.merge import DEFAULT_NAME, DEFAULT_VAL_FRACTION
    from autofly_ue5.dataset.rebalance import DEFAULT_THRESHOLD
    from scripts.pipeline_diagram import facts

    scenes, pool = _repository(tmp_path)
    f = facts(scenes_dir=scenes, targets_path=pool)
    assert f["training_scenes"] == ["s01", "s02", "s03"] and f["test_seen_scenes"] == ["s05r"]
    assert f["targets"] == 7 and f["target_categories"] == 3 and f["unseen_targets"] == 2
    assert f["dataset_name"] == DEFAULT_NAME and f["val_fraction"] == DEFAULT_VAL_FRACTION
    assert f["detection_threshold"] == DEFAULT_THRESHOLD
    assert f["image"] == "RGB 256x256" and f["action"] == "action[3]" and f["state"] == "state[9]"
    assert f["conditions"] == ["seen", "test_seen", "unseen_targets"]


def test_the_repository_s_own_counts(tmp_path):
    from scripts.pipeline_diagram import facts

    f = facts()
    assert len(f["training_scenes"]) == 9 and f["test_seen_scenes"] == ["s05r", "s06r"]
    assert f["targets"] == 60


def test_every_stage_is_drawn_in_reading_order_with_the_counts_it_was_given(tmp_path):
    from scripts.pipeline_diagram import facts, render

    scenes, pool = _repository(tmp_path)
    fig, boxes = render(facts(scenes_dir=scenes, targets_path=pool))
    assert [b["title"] for b in boxes] == STAGES
    scene_text = " ".join(boxes[0]["lines"])
    assert "3 training scenes" in scene_text and "1 test_seen layout" in scene_text and "7-object target pool" in scene_text
    assert boxes[6]["next"] is True and all(not b["next"] for i, b in enumerate(boxes) if i != 6)


def test_no_line_of_text_leaves_its_box():
    from scripts.pipeline_diagram import DPI, facts, render

    fig, boxes = render(facts())
    fig.set_dpi(DPI)
    renderer = fig.canvas.get_renderer()
    for box in boxes:
        frame = box["patch"].get_window_extent(renderer)
        for text in box["artists"]:
            extent = text.get_window_extent(renderer)
            assert frame.x0 <= extent.x0 and extent.x1 <= frame.x1, f"{box['title']}: {text.get_text()!r} runs out sideways"
            assert frame.y0 <= extent.y0 and extent.y1 <= frame.y1, f"{box['title']}: {text.get_text()!r} runs out of the box"


def test_the_cli_writes_the_figure(tmp_path, capsys):
    from scripts.pipeline_diagram import main

    out = tmp_path / "figures" / "pipeline.png"
    assert main(["--out", str(out)]) == 0
    assert out.read_bytes()[:8] == PNG and "pipeline.png" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "pipeline.gif")])
