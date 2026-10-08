"""scripts/layout_figure.py draws stored scene layouts: on s02's committed layout it writes its PNG and its numbers."""

from __future__ import annotations

import json

import pytest

from autofly_ue5.paths import RUNS_DIR


def test_the_figure_and_its_numbers_are_written_for_s02(tmp_path):
    from scripts import layout_figure

    levels = RUNS_DIR / "levels"
    if not (levels / "s02.layout.json").is_file():
        pytest.skip("runs/levels/s02.layout.json is not on this host")
    out, record = tmp_path / "out" / "fig.png", tmp_path / "out" / "fig.json"
    assert layout_figure.main(["--scenes", "s02", "--levels", str(levels), "--out", str(out),
                               "--json", str(record)]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    (scene,) = json.loads(record.read_text())["scenes"]
    assert scene["id"] == "s02"
    assert scene["instances"] == sum(scene["per_asset"].values())
    assert set(scene["crossing_edge_max"]) == {"x_min", "x_max", "y_min", "y_max"}
