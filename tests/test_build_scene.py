"""scenes/build.py: a layout is accepted for a level only if it passes reachability and plan 5's crossing rule (U4):
detour max <= 1.25, nothing unreachable, nothing through-unreachable. Both results are recorded in the layout file."""

from __future__ import annotations

import json

from autofly_ue5.paths import SCENES_DIR


def _scene_file(tmp_path, groups, sid="s09"):
    data = json.loads((SCENES_DIR / "s01_white_pillars.json").read_text())
    data["id"] = sid
    data["obstacle_groups"] = groups
    path = tmp_path / f"{sid}_test.json"
    path.write_text(json.dumps(data))
    return path


def test_s01_and_a_scattered_field_are_accepted_with_their_detours_recorded(tmp_path):
    from autofly_ue5.scenes.build import ACCEPTANCE, build_scene

    summary = build_scene(SCENES_DIR / "s01_white_pillars.json", tmp_path / "levels")
    assert summary["reachability"]["ok"] and summary["crossing"]["accepted"] and "level_spec" in summary
    stored = json.loads((tmp_path / "levels" / "s01.layout.json").read_text())
    assert stored["crossing"]["max"] <= ACCEPTANCE["max_detour"] and stored["crossing"]["through_unreachable"] == 0
    assert stored["crossing"]["rule"] == ACCEPTANCE


def test_a_layout_that_can_only_be_rounded_gets_no_level(tmp_path):
    from autofly_ue5.scenes.build import build_scene

    # 80 fat pillars on a tight grid: inflated by 1.4 m they fuse into a block the §6.2 rule still passes (the outer lane
    # is open), which the crossing rule must refuse
    path = _scene_file(tmp_path, [{"asset": "cylinder", "count": 80, "scale_range": {"xy": [3.6, 3.6], "z": [8.0, 8.0]},
                                   "palette": ["white"], "placement": {"type": "jittered_grid", "margin_m": 14.0, "jitter_m": 0.0}}])
    summary = build_scene(path, tmp_path / "levels")
    assert summary["reachability"]["ok"], "the lane beside the block keeps the old rule happy"
    assert not summary["crossing"]["accepted"] and "level_spec" not in summary
    assert not (tmp_path / "levels" / "s09.level.json").exists() and (tmp_path / "levels" / "s09.layout.json").exists()
