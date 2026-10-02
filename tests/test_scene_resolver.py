"""Scene ids -> scene file, base level, layout, map and default config (spec §6.1, §6.5), and the scene file's dynamic
block. One resolver replaces the id -> path guesses train.py, the gate, the renderer and the throughput script made."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from autofly_ue5.paths import RUNS_DIR, SCENES_DIR
from autofly_ue5.scenes.model import SceneFileError, load_scene_file

S01 = SCENES_DIR / "s01_white_pillars.json"

DYNAMIC = {
    "movers": {"source": "layout", "count": [8, 12], "path_movers": [2, 4], "path_corridor_m": 4.0},
    "route_kinds": ["pingpong", "orbit"],
    "speed_m_s": [0.4, 1.2],
    "pingpong_half_length_m": [1.0, 2.5],
    "orbit_radius_m": [1.0, 2.0],
    "min_gap_m": 1.0,
    "contact_m": 1.0,
    "yield_margin_m": 1.5,
    "start_keepout_m": 6.0,
    "target_keepout_m": 4.0,
    "max_path_ratio": 1.2,
}


def _dynamic_variant(**changes) -> dict:
    data = json.loads(S01.read_text())
    data.update(id="s01d", level="s01", dynamic=json.loads(json.dumps(DYNAMIC)))
    for key, value in changes.items():
        if key in DYNAMIC["movers"]:
            data["dynamic"]["movers"][key] = value
        elif key in DYNAMIC:
            data["dynamic"][key] = value
        else:
            data[key] = value
    return data


def _tree(tmp_path: Path, extra: dict[str, dict] | None = None, *, layout_sha: str | None = None) -> tuple[Path, Path]:
    """A scenes/ and levels/ pair holding s01 and its stored layout, plus `extra` scene files."""
    scenes, levels = tmp_path / "scenes", tmp_path / "levels"
    scenes.mkdir()
    levels.mkdir()
    shutil.copy(S01, scenes / S01.name)
    stored = json.loads((RUNS_DIR / "levels" / "s01.layout.json").read_text())
    if layout_sha is not None:
        stored["scene_sha256"] = layout_sha
    (levels / "s01.layout.json").write_text(json.dumps(stored))
    for name, data in (extra or {}).items():
        (scenes / name).write_text(json.dumps(data, indent=2))
    return scenes, levels


# --------------------------------------------------------------------------------------------------------
# The scene file.
# --------------------------------------------------------------------------------------------------------
def test_a_static_scene_has_no_level_and_no_dynamic_block():
    scene = load_scene_file(S01)
    assert scene.level is None and scene.dynamic is None


def test_the_dynamic_block_loads(tmp_path):
    path = tmp_path / "s01d_x.json"
    path.write_text(json.dumps(_dynamic_variant()))
    scene = load_scene_file(path)
    d = scene.dynamic
    assert scene.id == "s01d" and scene.level == "s01"
    assert d.count == (8, 12) and d.path_movers == (2, 4) and d.path_corridor_m == 4.0
    assert d.route_kinds == ("pingpong", "orbit") and d.speed_m_s == (0.4, 1.2)
    assert d.pingpong_half_length_m == (1.0, 2.5) and d.orbit_radius_m == (1.0, 2.0)
    assert (d.min_gap_m, d.contact_m, d.yield_margin_m) == (1.0, 1.0, 1.5)
    assert (d.start_keepout_m, d.target_keepout_m, d.max_path_ratio) == (6.0, 4.0, 1.2)


@pytest.mark.parametrize("scene_id, ok", [("s01d", True), ("s06r", True), ("s01", True), ("s01x", False),
                                           ("s01dd", False), ("s1d", False)])
def test_scene_ids_may_end_in_r_or_d(tmp_path, scene_id, ok):
    path = tmp_path / "scene.json"
    data = _dynamic_variant(id=scene_id) if scene_id.endswith("d") else dict(json.loads(S01.read_text()), id=scene_id)
    path.write_text(json.dumps(data))
    if ok:
        assert load_scene_file(path).id == scene_id
    else:
        with pytest.raises(SceneFileError):
            load_scene_file(path)


def test_a_level_cannot_name_a_dynamic_scene(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(_dynamic_variant(level="s01d")))
    with pytest.raises(SceneFileError, match="level"):
        load_scene_file(path)


def test_a_mover_fast_enough_to_teleport_into_the_drone_is_refused_at_load(tmp_path):
    # contact_m - max_speed * dt must exceed the rotor-tip half-span (0.48 m): 1.0 - 3.0 * 0.2 = 0.4 does not.
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(_dynamic_variant(speed_m_s=[0.4, 3.0])))
    with pytest.raises(SceneFileError, match="invariant"):
        load_scene_file(path)


@pytest.mark.parametrize("field, value, reason", [
    ("count", [12, 8], "count: minimum 12"), ("speed_m_s", [1.2, 0.4], "speed_m_s: minimum 1.2"),
    ("path_movers", [2, 9], "path_movers"), ("route_kinds", ["zigzag"], "zigzag"),
    ("max_path_ratio", 0.9, "max_path_ratio"), ("source", "spawn", "source"),
])
def test_inconsistent_dynamic_blocks_are_refused(tmp_path, field, value, reason):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(_dynamic_variant(**{field: value})))
    with pytest.raises(SceneFileError, match=reason) as err:
        load_scene_file(path)
    assert "Additional properties" not in str(err.value), "refused for the wrong reason"


# --------------------------------------------------------------------------------------------------------
# The resolver.
# --------------------------------------------------------------------------------------------------------
def test_s01_resolves_to_its_own_level():
    from autofly_ue5.scenes.resolve import resolve_scene

    r = resolve_scene("s01")
    assert r.scene.id == "s01" and r.base_id == "s01" and r.base_scene.id == "s01"
    assert len(r.layout.instances) == 80 and r.layout.scene_id == "s01"
    assert r.map_path == "/Game/AutoFly/Maps/S01" and r.default_scene_config == "scene_autofly_s01.jsonc"
    stored = RUNS_DIR / "levels" / "s01.layout.json"
    assert r.layout_path == stored and r.layout_sha256 == hashlib.sha256(stored.read_bytes()).hexdigest()


def test_a_dynamic_scene_resolves_to_the_level_it_reuses(tmp_path):
    from autofly_ue5.scenes.resolve import resolve_scene

    scenes, levels = _tree(tmp_path, {"s01d_moving_pillars.json": _dynamic_variant()})
    r = resolve_scene("s01d", scenes_dir=scenes, levels_dir=levels)
    assert r.scene.id == "s01d" and r.scene.dynamic is not None
    assert r.base_id == "s01" and r.base_scene.id == "s01"
    assert r.map_path == "/Game/AutoFly/Maps/S01" and r.default_scene_config == "scene_autofly_s01.jsonc"
    assert r.layout == resolve_scene("s01", scenes_dir=scenes, levels_dir=levels).layout


def test_an_unknown_scene_is_a_clear_error(tmp_path):
    from autofly_ue5.scenes.resolve import resolve_scene

    scenes, levels = _tree(tmp_path)
    with pytest.raises(FileNotFoundError, match="s99"):
        resolve_scene("s99", scenes_dir=scenes, levels_dir=levels)


def test_a_layout_built_from_another_version_of_the_scene_is_refused(tmp_path):
    from autofly_ue5.scenes.resolve import resolve_scene

    scenes, levels = _tree(tmp_path, layout_sha="0" * 64)
    with pytest.raises(SceneFileError, match="rebuild"):
        resolve_scene("s01", scenes_dir=scenes, levels_dir=levels)


def test_a_missing_layout_says_to_build_the_level(tmp_path):
    from autofly_ue5.scenes.resolve import resolve_scene

    scenes, levels = _tree(tmp_path)
    (levels / "s01.layout.json").unlink()
    with pytest.raises(FileNotFoundError, match="build"):
        resolve_scene("s01", scenes_dir=scenes, levels_dir=levels)


@pytest.mark.parametrize("field, value", [("seed", 7), ("altitude_band", [1.0, 4.0]), ("ground", "grass")])
def test_a_scene_reusing_a_level_must_match_its_static_fields(tmp_path, field, value):
    from autofly_ue5.scenes.resolve import resolve_scene

    scenes, levels = _tree(tmp_path, {"s01d_moving_pillars.json": _dynamic_variant(**{field: value})})
    with pytest.raises(SceneFileError, match=field):
        resolve_scene("s01d", scenes_dir=scenes, levels_dir=levels)


def test_two_files_claiming_one_id_are_refused(tmp_path):
    from autofly_ue5.scenes.resolve import resolve_scene

    scenes, levels = _tree(tmp_path, {"s01_copy.json": json.loads(S01.read_text())})
    with pytest.raises(SceneFileError, match="s01"):
        resolve_scene("s01", scenes_dir=scenes, levels_dir=levels)


def test_a_file_whose_id_disagrees_with_its_name_is_refused(tmp_path):
    from autofly_ue5.scenes.resolve import resolve_scene

    scenes, levels = _tree(tmp_path, {"s02_trees.json": json.loads(S01.read_text())})
    with pytest.raises(SceneFileError, match="s02_trees.json"):
        resolve_scene("s02", scenes_dir=scenes, levels_dir=levels)


def test_no_level_is_built_for_a_scene_that_reuses_one(tmp_path):
    from autofly_ue5.scenes.build import build_scene

    path = tmp_path / "s01d_moving_pillars.json"
    path.write_text(json.dumps(_dynamic_variant()))
    with pytest.raises(SceneFileError, match="reuses"):
        build_scene(path, tmp_path / "levels")
    assert not (tmp_path / "levels").exists()


def test_an_episode_is_labelled_with_the_scene_it_belongs_to(tmp_path):
    import numpy as np

    from autofly_ue5.expert.episode import sample_setup
    from autofly_ue5.scenes.resolve import resolve_scene

    scenes, levels = _tree(tmp_path, {"s01d_moving_pillars.json": _dynamic_variant()})
    r = resolve_scene("s01d", scenes_dir=scenes, levels_dir=levels)
    assert sample_setup(r.scene, r.layout, np.random.default_rng(3)).scene_id == "s01d"


def test_only_a_dynamic_scene_lets_its_simulator_move_anything(tmp_path):
    from autofly_ue5.scenes.resolve import resolve_scene

    scenes, levels = _tree(tmp_path, {"s01d_moving_pillars.json": _dynamic_variant()})
    assert resolve_scene("s01", scenes_dir=scenes, levels_dir=levels).movable_objects == ()
    tags = resolve_scene("s01d", scenes_dir=scenes, levels_dir=levels).movable_objects
    assert len(tags) == 80 and tags[0] == "obs_0000" and tags[-1] == "obs_0079"
