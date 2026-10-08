import math

import numpy as np
import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import Bounds, load_registry, load_scene_file
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import Pose
from autofly_ue5.validate.geometry import choose_depth_probes, pillars_from_layout_json
from autofly_ue5.validate.live_m1 import (
    S01_POSES,
    check_crash,
    check_depth,
    check_map,
    check_one_step,
    check_rgb_changes,
    check_tracking,
    scene_poses,
)
from autofly_ue5.validate.obstacles import choose_probes, obstacles_from_layout_json

SCENE = load_scene_file(SCENES_DIR / "s01_white_pillars.json")
REGISTRY = load_registry()
LAYOUT_JSON = {"layout": generate_layout(SCENE, REGISTRY).to_json()}
OBSTACLES = obstacles_from_layout_json(LAYOUT_JSON, REGISTRY)


def _probes():
    return choose_probes(OBSTACLES, SCENE.bounds)


def test_the_poses_derived_from_s01_s_bounds_are_m1_s_constants():
    # M1's gate stood at OPEN_POSE (-31, -25, -2, pi/2), PILLAR_VIEW (-31, 0, -2, 0), EMPTY_VIEW (-31, 0, -2, pi),
    # SPAWN_POSE (-31, -10, -2, 0) and SPAWN_VIEW (-31, -13, -2, pi/2); the same scene must still be checked from them
    assert S01_POSES.open == Pose(-31.0, -25.0, -2.0, math.pi / 2)
    assert S01_POSES.field_view == Pose(-31.0, 0.0, -2.0, 0.0) and S01_POSES.empty_view == Pose(-31.0, 0.0, -2.0, math.pi)
    assert S01_POSES.spawn == Pose(-31.0, -10.0, -2.0, 0.0) and S01_POSES.spawn_view == Pose(-31.0, -13.0, -2.0, math.pi / 2)
    assert scene_poses(SCENE.bounds) == S01_POSES


def test_depth_probes_exist_for_the_generated_s01_layout_and_match_m1_s():
    probes = _probes()
    assert [p.distance_m for p in probes] == [3.0, 6.0, 12.0]
    assert all(abs(p.expected_depth_m - p.distance_m) < 0.02 for p in probes)  # the window's off-axis pixels see a round pillar slightly farther
    # the general chooser picks what M1's cylinder-only chooser picked: same pillars, same poses
    old = choose_depth_probes(pillars_from_layout_json(LAYOUT_JSON), SCENE.bounds)
    assert [(p.tag, p.pose) for p in probes] == [(p.tag, p.pose) for p in old]


def test_s01_s_probes_are_those_of_the_fallbacks_before_the_six_metre_one():
    # The 6 m fallback (s06r, 2026-10-08) is tried only where 6 m offers no probe: s01's probes stay M1's, pose for pose.
    before = choose_probes(OBSTACLES, SCENE.bounds, fallback_distances_m={12.0: (10.0, 8.0)})
    assert [(p.tag, p.distance_m, p.pose) for p in _probes()] == [(p.tag, p.distance_m, p.pose) for p in before]


def test_gate_checks_pass_on_the_fake_simulator(tmp_path):
    sim = FakeSimulator(obstacles=[(o.x, o.y, o.radius_m) for o in OBSTACLES], image_size=256, render_depth=True)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    probes = _probes()
    assert check_rgb_changes(sim, S01_POSES)["pass"] is True
    depth = check_depth(sim, probes, tmp_path / "frames")
    assert depth["pass"] is True, depth
    for row in depth["probes"]:
        assert row["error_m"] < 0.01 and row["hfov_consistent"] and row["width_px"] == row["expected_width_px"]
        assert row["inlier_share"] == 1.0 and row["solid"] and row["pixels"] == {"solid": 16, "interior": 16, "card": 0, "empty": 0}
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == sorted(
        f"probe_{k}_{p.tag}_{p.distance_m:g}m_{kind}.png" for k, p in enumerate(probes) for kind in ("rgb", "depth_mm"))
    crash = check_crash(sim, probes[1], S01_POSES)
    assert crash["pass"] is True and crash["reset_after_crash"]["depth"]["mask_agreement"] == 1.0
    assert abs(crash["travel_to_impact_m"] - crash["expected_travel_m"]) <= 0.5
    one_step = check_one_step(sim, S01_POSES)
    assert one_step["pass"] is True and all(r["frame_times_match"] for r in one_step["rows"])
    assert check_tracking(sim, S01_POSES)["pass"] is True


def test_map_check_reads_every_actor_s_box():
    # the fake names its cylinders obstacle_<i>, all one height; a layout of such pillars under those tags is "loaded"
    pillars = obstacles_from_layout_json({"layout": {"instances": [
        {"tag": f"obstacle_{i}", "asset": "cylinder", "x": -20.0 + 5.0 * i, "y": 3.0 * i, "z_center": -5.0, "yaw": 0.0,
         "scale": [1.0, 1.0, 10.0], "material": "white", "radius_m": 0.5, "height_m": 10.0} for i in range(5)]}}, REGISTRY)
    sim = FakeSimulator(obstacles=[(o.x, o.y, o.radius_m) for o in pillars], image_size=32, obstacle_height_m=10.0,
                        scene_objects={"AF_Ground": (Pose(0.0, 0.0, 0.5, 0.0), 0.5)})
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    result = check_map(sim, pillars)
    assert result["pass"] is True and result["found"] == result["checked"] == 5
    assert result["max_center_error_m"] < 1e-9 and result["max_size_error_m"] < 1e-9 and result["ground_top_z_ned"] == 0.0
    shifted = [type(o)(**{**vars(o), "x": o.x + 0.1}) for o in pillars[:1]] + pillars[1:]
    assert check_map(sim, shifted)["pass"] is False and check_map(sim, shifted)["max_center_error_m"] == pytest.approx(0.1)
    missing = check_map(sim, OBSTACLES[:5])  # obs_0000... are not actors the fake knows
    assert missing["pass"] is False and missing["found"] == 0


def test_a_cards_actor_is_found_without_a_box():
    import dataclasses

    from autofly_ue5.scenes.model import AssetEntry, AssetPart
    from autofly_ue5.validate.obstacles import MeshLibrary, bbox_error, obstacle_from_instance

    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    entry = AssetEntry(name="tree", ue_path="/Game/AutoFly/Assets/tree/tree", base_size_m=(1.0, 1.0, 1.0), pivot="base", footprint="circle",
                       category="nature", role="obstacle", seen=None, measured_extent_cm_at_unit_scale=None, bounds_origin_cm=(0.0, 0.0, 50.0),
                       parts={"solid": AssetPart("/Game/AutoFly/Assets/tree/tree", (1.0, 1.0, 1.0), (0.0, 0.0, 50.0)),
                              "cards": AssetPart("/Game/AutoFly/Assets/tree/Cards/tree_cards", (1.0, 1.0, 0.5), (0.0, 0.0, 75.0))})
    reg = dataclasses.replace(REGISTRY, assets={**REGISTRY.assets, "tree": entry})
    obs = obstacle_from_instance({"tag": "obs_0000", "asset": "tree", "x": 0.0, "y": 0.0, "z_center": 0.0, "yaw": 0.0, "scale": [1, 1, 1],
                                  "material": "mesh", "radius_m": 0.5, "height_m": 1.0}, reg, MeshLibrary.from_arrays({"tree": (v, np.array([[0, 1, 2], [0, 2, 3]]))}))
    row = bbox_error(obs, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), part="cards")
    assert row == {"found": True, "box_unavailable": True, "reason": row["reason"]}
    assert bbox_error(obs, ((0.0, 0.0, -0.75), (1.0, 1.0, 0.5)), part="cards")["center_error_m"] == pytest.approx(0.0, abs=1e-9)


def test_probe_choice_on_a_scene_of_low_boxes_skips_what_the_camera_overflies():
    low = obstacles_from_layout_json({"layout": {"instances": [
        {"tag": "obs_0000", "asset": "crate", "x": 0.0, "y": 0.0, "z_center": -0.9, "yaw": 0.4, "scale": [1.0, 1.0, 1.8], "material": "wood",
         "radius_m": 0.7071, "height_m": 1.8},
        {"tag": "obs_0001", "asset": "crate", "x": 0.0, "y": 10.0, "z_center": -2.0, "yaw": 0.4, "scale": [1.0, 1.0, 4.0], "material": "wood",
         "radius_m": 0.7071, "height_m": 4.0}]}}, REGISTRY)
    probes = choose_probes(low, Bounds(-35, 35, -35, 35), distances=(3.0, 6.0), lower_altitudes_m=())
    assert [p.tag for p in probes] == ["obs_0001", "obs_0001"]
    assert all(p.expected_depth_m >= p.distance_m for p in probes), "a box face is never nearer than its footprint circle"
    with pytest.raises(ValueError):
        choose_probes(low[:1], Bounds(-35, 35, -35, 35), distances=(3.0,), lower_altitudes_m=())
    assert choose_probes(low[:1], Bounds(-35, 35, -35, 35), distances=(3.0,))[0].pose.z == -1.5, "else it is probed lower in the band"
