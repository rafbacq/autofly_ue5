"""autofly_ue5/validate/obstacles.py: one geometry for every obstacle kind the live checks meet (plan 5, C8). The world
bounding box is the transformed local box (what UE reports), the surface distance along a camera ray is exact for
cylinders and yawed boxes and ray-cast against the glTF triangles for imported meshes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from autofly_ue5.sim.types import Pose


def _registry():
    from autofly_ue5.scenes.model import load_registry

    return load_registry()


def _instance(**kw):
    base = {"tag": "obs_0000", "asset": "cylinder", "x": 0.0, "y": 0.0, "z_center": -5.0, "yaw": 0.0, "scale": [1.0, 1.0, 10.0],
            "material": "white", "radius_m": 0.5, "height_m": 10.0}
    base.update(kw)
    return base


def test_a_cylinder_s_box_and_surface_match_the_old_formulas():
    from autofly_ue5.validate.obstacles import obstacle_from_instance

    obs = obstacle_from_instance(_instance(x=-24.1, y=3.2), _registry())
    centre, size = obs.world_bbox()
    assert centre == pytest.approx([-24.1, 3.2, -5.0]) and size == pytest.approx([1.0, 1.0, 10.0])
    assert obs.kind == "cylinder" and obs.base_altitude_m == pytest.approx(0.0) and obs.top_altitude_m == pytest.approx(10.0)
    d = obs.surface_distance((-30.0, 3.2, -2.0), (1.0, 0.0, 0.0))
    assert d == pytest.approx(30.0 - 24.1 - 0.5)
    assert obs.surface_distance((-30.0, 3.2, -12.0), (1.0, 0.0, 0.0)) is None, "a ray above the cylinder's top misses it"


def test_a_yawed_box_s_bounds_and_faces():
    from autofly_ue5.validate.obstacles import obstacle_from_instance

    reg = _registry()
    inst = _instance(asset="crate", x=10.0, y=0.0, z_center=-0.6, yaw=math.pi / 4, scale=[1.2, 1.2, 1.2],
                     radius_m=round(math.hypot(1.2, 1.2) / 2, 4), height_m=1.2)
    obs = obstacle_from_instance(inst, reg)
    centre, size = obs.world_bbox()
    assert obs.kind == "box" and centre == pytest.approx([10.0, 0.0, -0.6])
    assert size[0] == pytest.approx(1.2 * math.sqrt(2)) and size[1] == pytest.approx(1.2 * math.sqrt(2)) and size[2] == pytest.approx(1.2)
    # head-on at 45 deg yaw the ray meets the corner (half-diagonal 0.8485 from the centre); at yaw 0 it would meet a face
    assert obs.surface_distance((0.0, 0.0, -0.6), (1.0, 0.0, 0.0)) == pytest.approx(10.0 - 1.2 * math.sqrt(2) / 2)
    flat = obstacle_from_instance({**inst, "yaw": 0.0}, reg)
    assert flat.surface_distance((0.0, 0.0, -0.6), (1.0, 0.0, 0.0)) == pytest.approx(10.0 - 0.6)
    assert flat.surface_distance((0.0, 2.0, -0.6), (1.0, 0.0, 0.0)) is None, "a ray beside the box misses it"
    assert obs.base_altitude_m == pytest.approx(0.0) and obs.top_altitude_m == pytest.approx(1.2)


def _box_mesh_gltf_frame(half: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """A closed box in glTF's frame (x right, y up, z towards the viewer), standing on y = 0, centred on the pivot."""
    x0, x1, y0, y1, z0, z1 = -half, half, 0.0, height, -half, half
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (3, 2, 6), (3, 6, 7), (1, 5, 6), (1, 6, 2), (0, 3, 7), (0, 7, 4)]
    return v, np.array(faces)


def test_a_mesh_is_placed_with_ue_s_axis_mapping_and_ray_cast_against_its_triangles():
    import dataclasses

    from autofly_ue5.scenes.model import AssetEntry
    from autofly_ue5.validate.obstacles import MeshLibrary, obstacle_from_instance

    # a 2 x 2 m box 3 m tall whose glTF pivot is offset: the box spans x in [1, 3] (so its centre is 2 m along glTF x)
    v, f = _box_mesh_gltf_frame(1.0, 3.0)
    v[:, 0] += 2.0
    v[:, 2] += 0.5  # and 0.5 m along glTF z, which UE makes +y
    entry = AssetEntry(name="block", ue_path="/Game/AutoFly/Assets/block/block", base_size_m=(2.0, 2.0, 3.0), pivot="base",
                       footprint="circle", category="nature", role="obstacle", seen=None, measured_extent_cm_at_unit_scale=(100.0, 100.0, 150.0),
                       profile_step_m=0.25, radius_profile_m=(3.6,) * 12, ground_radius_m=3.6, bounds_origin_cm=(200.0, 50.0, 150.0))
    reg = _registry()
    reg = dataclasses.replace(reg, assets={**reg.assets, "block": entry})
    library = MeshLibrary.from_arrays({"block": (v, f)})
    inst = _instance(asset="block", x=10.0, y=0.0, z_center=0.0, yaw=0.0, scale=[1.0, 1.0, 1.0], radius_m=3.6, height_m=3.0)
    obs = obstacle_from_instance(inst, reg, library)
    assert obs.kind == "mesh"
    centre, size = obs.world_bbox()
    # UE local origin (2.0, 0.5, 1.5) m -> at yaw 0 the world centre is the actor (10, 0, ground) plus that, z up -> NED z down
    assert centre == pytest.approx([12.0, 0.5, -1.5]) and size == pytest.approx([2.0, 2.0, 3.0])
    # a horizontal ray along +x at 2 m altitude, aimed at the box's y: it meets the near face at x = 11
    assert obs.surface_distance((0.0, 0.5, -2.0), (1.0, 0.0, 0.0)) == pytest.approx(11.0)
    assert obs.surface_distance((0.0, 0.5, -3.5), (1.0, 0.0, 0.0)) is None, "above the 3 m top"
    assert obs.surface_distance((0.0, 3.0, -2.0), (1.0, 0.0, 0.0)) is None, "beside it"
    # yawed by 90 deg about the actor: the offset (2, 0.5) turns to (-0.5, 2); the box now spans x in [9.5-1, 9.5+1]
    turned = obstacle_from_instance({**inst, "yaw": math.pi / 2}, reg, library)
    centre, _ = turned.world_bbox()
    assert centre == pytest.approx([9.5, 2.0, -1.5])
    assert turned.surface_distance((0.0, 2.0, -2.0), (1.0, 0.0, 0.0)) == pytest.approx(8.5)
    # scaled x2: the offset doubles too, the box is 4 x 4 x 6
    big = obstacle_from_instance({**inst, "scale": [2.0, 2.0, 2.0], "height_m": 6.0}, reg, library)
    centre, size = big.world_bbox()
    assert centre == pytest.approx([14.0, 1.0, -3.0]) and size == pytest.approx([4.0, 4.0, 6.0])
    assert big.top_altitude_m == pytest.approx(6.0)


def test_probes_skip_obstacles_the_camera_flies_over_and_aim_by_footprint(tmp_path):
    from autofly_ue5.scenes.model import Bounds
    from autofly_ue5.validate.obstacles import choose_probes, obstacle_from_instance

    reg = _registry()
    low = obstacle_from_instance(_instance(tag="obs_0000", asset="crate", x=0.0, y=0.0, z_center=-0.9, scale=[1.0, 1.0, 1.8],
                                           radius_m=0.7071, height_m=1.8), reg)  # a 1.8 m stack: under a 2 m camera
    tall = obstacle_from_instance(_instance(tag="obs_0001", asset="cylinder", x=10.0, y=10.0), reg)
    probes = choose_probes([low, tall], Bounds(-35, 35, -35, 35), distances=(3.0, 6.0), altitude_m=2.0, lower_altitudes_m=())
    assert [p.obstacle.tag for p in probes] == ["obs_0001", "obs_0001"]
    for p in probes:
        assert p.expected_depth_m == pytest.approx(p.distance_m, abs=0.01), "the standoff puts the surface at the requested distance"
        assert p.window_spread_m < 0.01 and p.solid and p.hit_fraction == 1.0 and p.pose.z == -2.0
    # with the band's lower altitudes allowed, the low stack is probed at 1.5 m instead (its box face, 0.2 m past the circle)
    lower = choose_probes([low, tall], Bounds(-35, 35, -35, 35), distances=(3.0,), altitude_m=2.0)
    assert lower[0].obstacle.tag == "obs_0000" and lower[0].pose.z == -1.5
    assert lower[0].expected_depth_m == pytest.approx(3.0 + 0.7071 - 0.5, abs=0.01)


@pytest.mark.parametrize("half_width_m, expected_m", [(6.5, 5.0), (5.8, 4.0)])
def test_a_six_metre_probe_falls_back_to_five_then_four_metres(half_width_m, expected_m):
    # s06r's live check (2026-10-08) found no unobstructed probe at 6 m, and only the 12 m probe had fallbacks. One pillar
    # in a small field: a 6 m probe would stand 6.9 m from its axis, outside the 2 m edge margin on all 8 bearings (4.88 m
    # along a diagonal against a 4.5 m limit), while 5 m stands 4.17 m out along a diagonal and 4 m 3.46 m.
    from autofly_ue5.scenes.model import Bounds
    from autofly_ue5.validate.obstacles import choose_probes, obstacle_from_instance

    pillar = obstacle_from_instance(_instance(), _registry())
    bounds = Bounds(-half_width_m, half_width_m, -half_width_m, half_width_m)
    (probe,) = choose_probes([pillar], bounds, distances=(6.0,), lower_altitudes_m=())
    assert probe.distance_m == expected_m and probe.expected_depth_m == pytest.approx(expected_m, abs=0.02)
    with pytest.raises(ValueError, match=r"6\.0 m \(nor at \(5\.0, 4\.0\)\)"):
        choose_probes([pillar], Bounds(-5.0, 5.0, -5.0, 5.0), distances=(6.0,), lower_altitudes_m=())


def test_pixel_rays_follow_the_image_axes():
    from autofly_ue5.validate.obstacles import pixel_rays

    # facing east (yaw pi/2): right is south (-x), down is +z (NED); the far right column and the bottom row
    right_col = pixel_rays(Pose(0, 0, -2, math.pi / 2), slice(128, 129), slice(255, 256))[0]
    assert right_col[0] < -0.9 and right_col[1] == pytest.approx(1.0) and abs(right_col[2]) < 0.005
    bottom = pixel_rays(Pose(0, 0, -2, math.pi / 2), slice(255, 256), slice(128, 129))[0]
    assert bottom[2] > 0.9 and bottom[1] == pytest.approx(1.0)
    # pixel 128's centre is half a pixel right of and below the optical axis (a 256-wide image has no centre pixel)
    centre = pixel_rays(Pose(0, 0, -2, 0.0), slice(128, 129), slice(128, 129))[0]
    assert centre == pytest.approx((1.0, 0.5 / 128, 0.5 / 128))


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.4])
def test_a_point_projects_onto_the_pixel_whose_ray_reaches_it(yaw):
    # project_point is pixel_rays' inverse (one focal length, one set of axes): a point at planar depth d along a pixel's
    # ray lands in that pixel, half a pixel from its edges, at depth d; for a 64-pixel frame too.
    from autofly_ue5.validate.obstacles import camera_origin, pixel_rays, project_point

    pose = Pose(3.0, -4.0, -2.0, yaw)
    origin = camera_origin(pose)
    for size in (256, 64):
        for r, c in ((0, 0), (128 % size, 37), (size - 1, size - 1), (40, size - 3)):
            (ray,) = pixel_rays(pose, slice(r, r + 1), slice(c, c + 1), image_size=size)
            for d in (0.5, 7.0, 60.0):
                point = tuple(o + d * v for o, v in zip(origin, ray))
                row, col, depth = project_point(pose, point, image_size=size)
                assert (row, col) == pytest.approx((r + 0.5, c + 0.5), abs=1e-9) and depth == pytest.approx(d)
    _row, _col, depth = project_point(pose, (pose.x - 5 * math.cos(yaw), pose.y - 5 * math.sin(yaw), -2.0))
    assert depth < 0, "behind the camera"


def test_a_box_s_interval_along_a_ray_is_its_near_and_far_faces():
    from autofly_ue5.validate.obstacles import Box

    box = Box(tag="b", x=10.0, y=0.0, radius_m=1.0, z_center=-1.0, yaw=0.0, half=(0.5, 1.0, 1.0))
    assert box.surface_interval((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)) == pytest.approx((9.5, 10.5))
    assert box.surface_distance((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)) == pytest.approx(9.5)
    assert box.surface_interval((0.0, 3.0, -1.0), (1.0, 0.0, 0.0)) is None, "beside it"
    assert box.surface_interval((20.0, 0.0, -1.0), (1.0, 0.0, 0.0)) is None, "behind the ray's origin"
    turned = Box(tag="b", x=10.0, y=0.0, radius_m=1.0, z_center=-1.0, yaw=math.pi / 2, half=(0.5, 1.0, 1.0))
    assert turned.surface_interval((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)) == pytest.approx((9.0, 11.0))


def test_row_silhouette_of_a_cylinder_matches_m1_s_pinhole_width():
    from autofly_ue5.validate.geometry import contiguous_width, pillar_width_px
    from autofly_ue5.validate.obstacles import obstacle_from_instance, row_silhouette

    obs = obstacle_from_instance(_instance(x=6.9, y=0.0, radius_m=0.5364, scale=[1.0728, 1.0728, 10.0]), _registry())
    pose = Pose(0.0, 0.0, -2.0, 0.0)  # camera 0.4 m ahead: axis distance 6.5 m
    mask = row_silhouette(obs, pose, near_m=6.5 + 0.3)
    assert abs(contiguous_width(mask) - pillar_width_px(6.5, 0.5364)) <= 1.0


def test_a_real_imported_tree_stands_where_its_measurement_says():
    import json

    from autofly_ue5.paths import ASSET_MEASUREMENTS, ASSET_SOURCES
    from autofly_ue5.validate.obstacles import MeshLibrary, obstacle_from_instance

    if not ASSET_SOURCES.exists() or not ASSET_MEASUREMENTS.exists():
        pytest.skip("no fetched assets on this checkout")
    downloads = Path(json.loads(ASSET_SOURCES.read_text())["downloads_dir"])
    if not downloads.exists():
        pytest.skip("downloads are not present")
    library = MeshLibrary.from_files({"quiver_tree_01"})
    reg = _registry()
    measured = next(m for m in json.loads(ASSET_MEASUREMENTS.read_text())["meshes"] if m["asset"] == "quiver_tree_01")
    inst = _instance(asset="quiver_tree_01", x=0.0, y=0.0, z_center=0.0, yaw=0.7, scale=[1.0, 1.0, 1.0],
                     radius_m=measured["band_radius_m"], height_m=measured["height_m"])
    obs = obstacle_from_instance(inst, reg, library)
    # the registry's UE bounds reproduce the glTF extents (its roots reach 4 cm below the pivot; height_m is top - base)
    assert obs.top_altitude_m == pytest.approx(measured["top_m"], abs=0.001)
    assert obs.base_altitude_m == pytest.approx(measured["base_m"], abs=0.001)
    # a horizontal ray at 1.5 m, aimed at the pivot from 6 m away on each side, meets the trunk (radius 0.09-0.13 m in
    # that slice of the measured profile) just short of 6 m; at 2 m the tree has forked and two sides see through
    for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
        d = obs.surface_distance((-6 * math.cos(yaw), -6 * math.sin(yaw), -1.5), (math.cos(yaw), math.sin(yaw), 0.0))
        assert d is not None and 6.0 - 0.15 <= d <= 6.0, (yaw, d)
    assert obs.surface_distance((-6.0, 0.0, -(measured["height_m"] + 0.5)), (1.0, 0.0, 0.0)) is None, "above the crown"


def test_a_tree_with_leaf_cards_is_probed_on_its_clear_trunk_pixels():
    import dataclasses

    from autofly_ue5.scenes.model import AssetEntry, Bounds
    from autofly_ue5.validate.obstacles import WIDE_WINDOW, MeshLibrary, choose_probes, obstacle_from_instance, window_depths

    # a 0.4 m square trunk 4 m tall, and one leaf card (a 2 x 1 m quad, alphaMode BLEND) hanging 0.8 m west of its
    # upper half (glTF x = UE x = NED x at yaw 0; glTF y is up), so a camera west of the tree at 2.5 m sees the card first
    tv, tf = _box_mesh_gltf_frame(0.2, 4.0)
    tv[:, 0] += 3.0  # trunk centre 3 m along glTF x
    card = np.array([[2.0, 2.0, -1.0], [2.0, 2.0, 1.0], [2.0, 3.0, 1.0], [2.0, 3.0, -1.0]], float)  # at x = 2, y in [2, 3]
    cf = np.array([[0, 1, 2], [0, 2, 3]])
    v = np.concatenate([tv, card]); f = np.concatenate([tf, cf + len(tv)])
    opaque = np.concatenate([np.ones(len(tf), bool), np.zeros(len(cf), bool)])
    entry = AssetEntry(name="cardtree", ue_path="/Game/AutoFly/Assets/cardtree/cardtree", base_size_m=(1.4, 2.0, 4.0), pivot="base",
                       footprint="circle", category="nature", role="obstacle", seen=None, measured_extent_cm_at_unit_scale=(70.0, 100.0, 200.0),
                       profile_step_m=0.25, radius_profile_m=(3.4,) * 16, ground_radius_m=3.4, bounds_origin_cm=(260.0, 0.0, 200.0))
    reg = _registry()
    reg = dataclasses.replace(reg, assets={**reg.assets, "cardtree": entry})
    library = MeshLibrary.from_arrays({"cardtree": (v, f, opaque)})
    inst = _instance(asset="cardtree", x=0.0, y=0.0, z_center=0.0, yaw=0.0, scale=[1.0, 1.0, 1.0], radius_m=3.4, height_m=4.0)
    obs = obstacle_from_instance(inst, reg, library)
    assert obs.has_cards
    # from 6 m west of the pivot at 2.5 m altitude the card (x = 2) is in front of the trunk face (x = 2.8)
    pose = Pose(-6.0, 0.0, -2.5, 0.0)
    anything = window_depths([obs], pose)
    solid_only = window_depths([obs], pose, opaque_only=True)
    assert anything.min() == pytest.approx(2.0 + 6.0 - 0.4, abs=0.01) and solid_only.min() == pytest.approx(2.8 + 6.0 - 0.4, abs=0.01)
    # at 1.5 m altitude the card is above the window: the trunk is in the clear
    assert window_depths([obs], Pose(-6.0, 0.0, -1.5, 0.0)).min() == pytest.approx(2.8 + 6.0 - 0.4, abs=0.01)
    # the chooser's first bearing (from the west, yaw 0) sees the card first on every window pixel: no solid pixel there
    west = Pose(-(3.4 + 3.0 + 0.4), 0.0, -2.5, 0.0)
    any_w, opaque_w = window_depths([obs], west, WIDE_WINDOW), window_depths([obs], west, WIDE_WINDOW, opaque_only=True)
    assert (np.isfinite(opaque_w) & (opaque_w <= any_w + 1e-3)).sum() == 0 and np.isfinite(any_w).all()
    # so it turns to another bearing at the same altitude, where the trunk is in the clear, before it tries lower
    p = choose_probes([obs], Bounds(-35, 35, -35, 35), distances=(3.0,), altitude_m=2.5, lower_altitudes_m=(1.5,))[0]
    assert not p.solid and p.pose.z == -2.5 and p.pose.yaw != 0.0
    assert p.interior_pixels.sum() >= 12 and p.hit_fraction == pytest.approx(p.solid_pixels.mean())
    assert np.all(p.predicted_opaque_m[p.solid_pixels] <= p.predicted_m[p.solid_pixels] + 1e-3)
    # and the depth it expects is the trunk's face from where it stands, 3 m plus the margin between circle and trunk
    face = {math.pi / 2: 0.2, -math.pi / 2: 0.2, math.pi: 0.2}.get(round(p.pose.yaw, 6), None)
    if face is not None:
        assert p.expected_depth_m == pytest.approx(3.0 + 3.4 - face, abs=0.02)


def test_a_mesh_with_cards_reports_its_two_actors_boxes():
    import dataclasses

    from autofly_ue5.scenes.model import AssetEntry, AssetPart
    from autofly_ue5.validate.obstacles import MeshLibrary, bbox_error, obstacle_from_instance

    v, f = _box_mesh_gltf_frame(1.0, 3.0)
    entry = AssetEntry(name="tree", ue_path="/Game/AutoFly/Assets/tree/tree", base_size_m=(2.8, 2.6, 3.5), pivot="base",
                       footprint="circle", category="nature", role="obstacle", seen=None, measured_extent_cm_at_unit_scale=(140.0, 130.0, 175.0),
                       profile_step_m=0.25, radius_profile_m=(1.4,) * 14, ground_radius_m=1.4, bounds_origin_cm=(10.0, 0.0, 175.0),
                       parts={"solid": AssetPart("/Game/AutoFly/Assets/tree/tree", (2.6, 2.4, 3.4), (0.0, 5.0, 170.0)),
                              "cards": AssetPart("/Game/AutoFly/Assets/tree/Cards/tree_cards", (2.8, 2.6, 2.0), (10.0, 0.0, 250.0))})
    reg = dataclasses.replace(_registry(), assets={**_registry().assets, "tree": entry})
    obs = obstacle_from_instance(_instance(asset="tree", x=10.0, y=0.0, z_center=0.0, yaw=0.0, scale=[1.0, 1.0, 1.0], radius_m=1.4, height_m=3.5),
                                 reg, MeshLibrary.from_arrays({"tree": (v, f)}))
    assert obs.cards_tag == "obs_0000_cards"
    centre, size = obs.world_bbox()
    assert centre == pytest.approx([10.0, 0.05, -1.7]) and size == pytest.approx([2.6, 2.4, 3.4]), "the tag's actor is the solid part"
    centre, size = obs.cards_world_bbox()
    assert centre == pytest.approx([10.1, 0.0, -2.5]) and size == pytest.approx([2.8, 2.6, 2.0])
    centre, size = obs.whole_world_bbox()
    assert centre == pytest.approx([10.1, 0.0, -1.75]) and size == pytest.approx([2.8, 2.6, 3.5])
    assert obs.top_altitude_m == pytest.approx(3.5), "the camera's view of it is the whole model's"
    assert bbox_error(obs, ((10.1, 0.0, -2.5), (2.8, 2.6, 2.0)), part="cards")["center_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert bbox_error(obs, ((10.1, 0.0, -2.5), (2.8, 2.6, 2.0)))["size_error_m"] == pytest.approx(1.4)
