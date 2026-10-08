import json

import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.model import SceneFileError, load_registry, load_scene_file

S01 = SCENES_DIR / "s01_white_pillars.json"


def test_load_s01():
    scene = load_scene_file(S01)
    assert scene.id == "s01" and scene.split == "train" and scene.seed == 1001
    assert (scene.bounds.width, scene.bounds.height) == (70.0, 70.0)
    assert scene.start_band == (2.0, 6.0) and scene.target_band == (0.0, 3.0) and scene.altitude_band == (1.0, 3.0)
    group = scene.obstacle_groups[0]
    assert group.asset == "cylinder" and group.count == 80 and group.palette == ("white",)
    assert group.scale_xy == (0.8, 1.2) and group.scale_z == (8.0, 12.0)
    assert group.placement == {"type": "jittered_grid", "margin_m": 8.0, "jitter_m": 1.0}
    assert len(scene.sha256) == 64 and scene.instruction_obstacle == "white pillars"


def _variant(tmp_path, mutate):
    data = json.loads(S01.read_text())
    mutate(data)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(data))
    return path


def test_schema_rejects_missing_field(tmp_path):
    with pytest.raises(SceneFileError, match="instruction_obstacle"):
        load_scene_file(_variant(tmp_path, lambda d: d.pop("instruction_obstacle")))


def test_schema_rejects_unknown_placement(tmp_path):
    def mutate(d):
        d["obstacle_groups"][0]["placement"] = {"type": "spiral", "margin_m": 1.0}
    with pytest.raises(SceneFileError):
        load_scene_file(_variant(tmp_path, mutate))


def test_inverted_range_is_rejected(tmp_path):
    with pytest.raises(SceneFileError, match="start_band"):
        load_scene_file(_variant(tmp_path, lambda d: d.update(start_band=[6.0, 2.0])))


def test_inverted_bounds_are_rejected(tmp_path):
    with pytest.raises(SceneFileError, match="bounds"):
        load_scene_file(_variant(tmp_path, lambda d: d["bounds"].update(x_min=40.0)))


def test_off_centre_or_resized_bounds_are_rejected(tmp_path):
    with pytest.raises(SceneFileError, match="70"):
        load_scene_file(_variant(tmp_path, lambda d: d["bounds"].update(x_min=-30.0, x_max=40.0)))
    with pytest.raises(SceneFileError, match="70"):
        load_scene_file(_variant(tmp_path, lambda d: d.update(bounds={"x_min": -50.0, "x_max": 50.0, "y_min": -50.0, "y_max": 50.0})))


def test_load_registry():
    registry = load_registry()
    cyl = registry.assets["cylinder"]
    assert cyl.ue_path == "/Engine/BasicShapes/Cylinder" and cyl.base_size_m == (1.0, 1.0, 1.0)
    assert cyl.pivot == "center" and cyl.footprint == "circle" and cyl.seen is None
    measured = cyl.measured_extent_cm_at_unit_scale  # null until Task 14 Step 11 writes the live measurement
    assert measured is None or (len(measured) == 3 and all(abs(v - 50.0) <= 1.0 for v in measured))
    white = registry.materials["white"]
    assert white.kind == "color_instance" and white.parameter == "Color" and white.rgba == (0.9, 0.9, 0.9, 1.0)
    assert registry.materials["grid"].parent is None


# --------------------------------------------------------------------------------------------------------
# spawn_clearance_m (2026-10-08): where a scene's episodes may begin, not what its level is.
# --------------------------------------------------------------------------------------------------------
def test_a_scene_without_a_spawn_clearance_leaves_the_samplers_own():
    assert load_scene_file(S01).spawn_clearance_m is None


def test_a_scene_may_keep_its_starts_targets_and_distractors_farther_out(tmp_path):
    assert load_scene_file(_variant(tmp_path, lambda d: d.update(spawn_clearance_m=1.7))).spawn_clearance_m == 1.7


@pytest.mark.parametrize("value", [1.39, 0.5, 0.0, -1.0])
def test_a_spawn_clearance_below_the_layouts_acceptance_is_refused(tmp_path, value):
    # A layout is accepted only at the sampler's 1.4 m inflation: a smaller clearance would start episodes in space no
    # check ever cleared.
    with pytest.raises(SceneFileError, match=r"spawn_clearance_m.*1\.4"):
        load_scene_file(_variant(tmp_path, lambda d: d.update(spawn_clearance_m=value)))


def test_the_schemas_floor_is_the_samplers_own_clearance():
    from autofly_ue5.expert.episode import spawn_clearance_m
    from autofly_ue5.scenes.model import SCHEMA_PATH

    field = json.loads(SCHEMA_PATH.read_text())["properties"]["spawn_clearance_m"]
    assert field["minimum"] == spawn_clearance_m() and "sampler" in field["description"]


def test_a_spawn_clearance_leaves_the_layout_and_the_level_as_they_are(tmp_path):
    # Only the scene file's hash changes, which the stored layout and level spec record: a scene that adds the field
    # regenerates its layout file (scripts/build_scenes.py) with the same instances.
    from autofly_ue5.scenes.generate import generate_layout
    from autofly_ue5.scenes.level_spec import layout_to_level_spec

    registry = load_registry()
    plain, wide = load_scene_file(S01), load_scene_file(_variant(tmp_path, lambda d: d.update(spawn_clearance_m=1.7)))
    layout = generate_layout(plain, registry)
    assert generate_layout(wide, registry) == layout and wide.sha256 != plain.sha256
    assert ({**layout_to_level_spec(layout, wide, registry), "scene_sha256": None}
            == {**layout_to_level_spec(layout, plain, registry), "scene_sha256": None})
