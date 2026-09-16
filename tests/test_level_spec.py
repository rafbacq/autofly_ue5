import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.level_spec import GAME_MODE_CLASS, SUNSKY_CLASS, layout_to_level_spec, map_path_for
from autofly_ue5.scenes.model import Bounds, Instance, Layout, load_registry, load_scene_file

SCENE = load_scene_file(SCENES_DIR / "s01_white_pillars.json")
REGISTRY = load_registry()


def test_map_path():
    assert map_path_for("s01") == "/Game/AutoFly/Maps/S01"


def test_single_pillar_converts_to_ue_centimetres():
    pillar = Instance(tag="obs_0000", asset="cylinder", x=1.0, y=-2.0, z_center=-5.0, yaw=0.0, scale=(1.0, 1.0, 10.0),
                      material="white", radius_m=0.5, height_m=10.0)
    spec = layout_to_level_spec(Layout("s01", 1001, Bounds(-35, 35, -35, 35), (pillar,)), SCENE, REGISTRY)
    actor = spec["actors"][0]
    assert actor["location_cm"] == [100.0, -200.0, 500.0]
    assert actor["mesh"] == "/Engine/BasicShapes/Cylinder" and actor["yaw_deg"] == 0.0
    assert actor["scale"] == [1.0, 1.0, 10.0] and actor["expected_extent_cm"] == [50.0, 50.0, 500.0]
    assert actor["material"] == "white"


def test_ground_top_is_at_ue_zero():
    spec = layout_to_level_spec(Layout("s01", 1001, SCENE.bounds, ()), SCENE, REGISTRY)
    ground = spec["ground"]
    assert ground["tag"] == "AF_Ground" and ground["mesh"] == "/Engine/BasicShapes/Cube"
    assert ground["location_cm"] == pytest.approx([0.0, 0.0, -10.0])
    assert ground["expected_extent_cm"] == pytest.approx([10000.0, 10000.0, 10.0])
    assert ground["location_cm"][2] + ground["expected_extent_cm"][2] == pytest.approx(0.0)
    assert ground["material"] == "grid"


def test_full_s01_spec():
    layout = generate_layout(SCENE, REGISTRY)
    spec = layout_to_level_spec(layout, SCENE, REGISTRY)
    assert spec["map_path"] == "/Game/AutoFly/Maps/S01" and spec["scene_sha256"] == SCENE.sha256
    assert spec["game_mode_class"] == GAME_MODE_CLASS and spec["sunsky_class"] == SUNSKY_CLASS
    assert len(spec["actors"]) == 80 and spec["actors"][79]["tag"] == "obs_0079"
    names = {m["name"] for m in spec["materials"]}
    assert names == {"grid", "white"}
    white = next(m for m in spec["materials"] if m["name"] == "white")
    assert white["ue_path"] == "/Game/AutoFly/Materials/MI_White" and white["rgba"] == [0.9, 0.9, 0.9, 1.0]
    for actor, inst in zip(spec["actors"], layout.instances):
        assert actor["location_cm"][2] == pytest.approx(inst.height_m * 50.0, abs=0.01)  # base on the ground
