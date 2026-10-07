import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.level_spec import (
    EXPOSURE_APPLY_PHYSICAL_CAMERA,
    EXPOSURE_BIAS_EV,
    EXPOSURE_TAG,
    GAME_MODE_CLASS,
    SUNSKY_CLASS,
    layout_to_level_spec,
    map_path_for,
)
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
    # what the builder checks on the component's local bounds: pivot- and yaw-independent, at unit scale
    assert actor["expected_local_bounds_cm"] == {"origin": [0.0, 0.0, 0.0], "extent": [50.0, 50.0, 50.0]}


def test_an_imported_base_pivot_mesh_carries_its_local_bounds_origin_and_stands_on_the_ground():
    import dataclasses

    from autofly_ue5.scenes.model import AssetEntry

    tree = AssetEntry(name="tree", ue_path="/Game/AutoFly/Assets/tree/tree", base_size_m=(2.8, 2.6, 3.5), pivot="base",
                      footprint="circle", category="nature", role="obstacle", seen=None, measured_extent_cm_at_unit_scale=(140.0, 130.0, 175.0),
                      profile_step_m=0.25, radius_profile_m=(1.4,) * 14, ground_radius_m=1.4, bounds_origin_cm=(5.0, -12.0, 175.0))
    registry = dataclasses.replace(REGISTRY, assets={**REGISTRY.assets, "tree": tree})
    inst = Instance(tag="obs_0000", asset="tree", x=1.0, y=-2.0, z_center=0.0, yaw=0.5, scale=(1.2, 1.2, 1.2), material="white",
                    radius_m=1.68, height_m=4.2)
    actor = layout_to_level_spec(Layout("s02", 1, Bounds(-35, 35, -35, 35), (inst,)), SCENE, registry)["actors"][0]
    assert actor["location_cm"] == [100.0, -200.0, 0.0], "a base pivot stands on the ground: the actor is placed at z = 0"
    assert actor["expected_local_bounds_cm"] == {"origin": [5.0, -12.0, 175.0], "extent": [140.0, 130.0, 175.0]}
    assert actor["expected_extent_cm"] == pytest.approx([168.0, 156.0, 210.0])


def test_ground_top_is_at_ue_zero():
    spec = layout_to_level_spec(Layout("s01", 1001, SCENE.bounds, ()), SCENE, REGISTRY)
    ground = spec["ground"]
    assert ground["tag"] == "AF_Ground" and ground["mesh"] == "/Engine/BasicShapes/Cube"
    assert ground["location_cm"] == pytest.approx([0.0, 0.0, -10.0])
    assert ground["expected_extent_cm"] == pytest.approx([10000.0, 10000.0, 10.0])
    assert ground["location_cm"][2] + ground["expected_extent_cm"][2] == pytest.approx(0.0)
    assert ground["material"] == "grid"


def test_exposure_block_is_manual_and_deterministic():
    # R16: fixed manual exposure, carried as data in the level spec (not a magic number in the UE script),
    # so every scene the builder produces shares the same deterministic-exposure mechanism.
    spec = layout_to_level_spec(Layout("s01", 1001, SCENE.bounds, ()), SCENE, REGISTRY)
    exposure = spec["exposure"]
    assert exposure["tag"] == EXPOSURE_TAG == "AF_Exposure"
    assert exposure["method"] == "manual"
    assert exposure["bias_ev"] == EXPOSURE_BIAS_EV == -11.0
    assert exposure["apply_physical_camera_exposure"] is EXPOSURE_APPLY_PHYSICAL_CAMERA is False


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
