"""Expand a scene file into a layout, check reachability, and write `<id>.layout.json` (+ `<id>.level.json` if it passes).

The library form of `scripts/build_scenes.py`, so the test suite can build the s01 layout on a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.level_spec import layout_to_level_spec
from autofly_ue5.scenes.model import SceneFileError, load_registry, load_scene_file
from autofly_ue5.scenes.reachability import check_reachability


def build_scene(scene_path: Path, out_dir: Path) -> dict:
    """Returns a JSON-able summary; the level spec is written only when the layout passes reachability."""
    scene = load_scene_file(scene_path)
    if scene.level is not None:
        raise SceneFileError(f"{scene_path}: scene {scene.id} reuses {scene.level}'s level (spec §6.1) and gets no level "
                             f"of its own; build {scene.level} instead")
    registry = load_registry()
    layout = generate_layout(scene, registry)
    reach = check_reachability(layout, scene.start_band, scene.target_band)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_out = out_dir / f"{scene.id}.layout.json"
    layout_out.write_text(json.dumps({"scene_path": str(scene_path), "scene_sha256": scene.sha256,
                                      "layout": layout.to_json(), "reachability": reach.to_json()}, indent=2))
    summary = {"scene": scene.id, "instances": len(layout.instances), "reachability": reach.to_json(), "layout": str(layout_out)}
    if reach.ok:
        level_out = out_dir / f"{scene.id}.level.json"
        level_out.write_text(json.dumps(layout_to_level_spec(layout, scene, registry), indent=2))
        summary["level_spec"] = str(level_out)
    return summary
