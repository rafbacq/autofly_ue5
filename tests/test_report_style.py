"""autofly_ue5/report/style.py: the report figures' shared look. scripts/dataset_figures.py keeps its own copy of the same
values (it came first), and the two must not drift apart: a reader sees every figure in final_results/ side by side."""

from __future__ import annotations


def test_the_shared_style_is_the_one_dataset_figures_draws_with():
    from autofly_ue5.report import style
    from scripts import dataset_figures

    assert style.SLOTS == dataset_figures.SLOTS
    for name in ("INK", "INK_2", "MUTED", "GRID", "AXIS", "SURFACE", "STYLE", "PAPER_PER"):
        assert getattr(style, name) == getattr(dataset_figures, name), name
    for scene_id in ("s01", "s01d", "s05", "s05r", "s09"):
        assert style.scene_label(scene_id) == dataset_figures.scene_label(scene_id)


def test_a_scene_is_labelled_by_its_file_name_and_an_unknown_one_by_its_id():
    from autofly_ue5.report.style import scene_label

    assert scene_label("s01") == "s01 white pillars", "s01d_moving_pillars.json must not match s01's glob"
    assert scene_label("s05") == "s05 standing stones" and scene_label("s05r") == "s05r standing stones"
    assert scene_label("s99") == "s99"
