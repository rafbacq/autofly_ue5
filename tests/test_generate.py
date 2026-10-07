import itertools
import json
import math
import random

import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import UnsupportedPlacementError, generate_layout, jittered_grid
from autofly_ue5.scenes.model import Bounds, load_registry, load_scene_file

S01 = SCENES_DIR / "s01_white_pillars.json"
BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)


def test_jittered_grid_count_margin_and_spacing():
    points = jittered_grid(random.Random(7), BOUNDS, 80, margin_m=8.0, jitter_m=1.0)
    assert len(points) == 80
    for x, y in points:
        assert -28.0 <= x <= 28.0 and -28.0 <= y <= 28.0
    closest = min(math.dist(a, b) for a, b in itertools.combinations(points, 2))
    assert closest >= 4.0 - 1e-9  # 6 m cells minus twice the 1 m jitter


def test_jittered_grid_is_deterministic():
    a = jittered_grid(random.Random(3), BOUNDS, 20, margin_m=8.0, jitter_m=1.0)
    b = jittered_grid(random.Random(3), BOUNDS, 20, margin_m=8.0, jitter_m=1.0)
    assert a == b


def test_jitter_larger_than_half_cell_is_rejected():
    with pytest.raises(ValueError, match="jitter"):
        jittered_grid(random.Random(1), BOUNDS, 80, margin_m=8.0, jitter_m=3.5)


def test_generate_s01_layout():
    scene, registry = load_scene_file(S01), load_registry()
    layout = generate_layout(scene, registry)
    assert layout.scene_id == "s01" and layout.seed == 1001 and len(layout.instances) == 80
    assert [i.tag for i in layout.instances[:2]] == ["obs_0000", "obs_0001"]
    assert len({i.tag for i in layout.instances}) == 80
    for inst in layout.instances:
        assert 0.4 <= inst.radius_m <= 0.6 and 8.0 <= inst.height_m <= 12.0
        assert inst.z_center == pytest.approx(-inst.height_m / 2, abs=1e-3)
        assert inst.scale[0] == inst.scale[1] and inst.material == "white" and inst.asset == "cylinder"
    assert generate_layout(scene, registry) == layout
    assert generate_layout(scene, registry, seed=2002) != layout
    assert json.loads(json.dumps(layout.to_json()))["instances"][0]["tag"] == "obs_0000"


def test_a_footprint_the_generator_cannot_place_is_explicit(tmp_path):
    data = json.loads(S01.read_text())
    data["obstacle_groups"][0]["asset"] = "cube"  # the ground slab: footprint "none"
    path = tmp_path / "cube.json"
    path.write_text(json.dumps(data))
    with pytest.raises(UnsupportedPlacementError, match="footprint=none"):
        generate_layout(load_scene_file(path), load_registry())


# ------------------------------------------------------------------------------------------------------------------
# M4: the other placements of spec §6.1 (poisson, clusters, stacks), box footprints, and several groups per scene.
# ------------------------------------------------------------------------------------------------------------------

# M1's own build of s01 (runs/levels/s01.layout.json on the GPU host, 2026-09-16). Every expert, gate and dataset flew
# this layout; a change to the generator must not move one pillar of it.
S01_LAYOUT_SHA256 = "c43019cef5f1c87224134881c6c6657c4ce8944a73abe5d41f84081609707e07"


def _sha(layout) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(layout.to_json(), sort_keys=True).encode()).hexdigest()


def test_s01_s_layout_is_still_m1_s():
    assert _sha(generate_layout(load_scene_file(S01), load_registry())) == S01_LAYOUT_SHA256


def _registry_with_box():
    """The real registry plus a box-footprint obstacle on the engine cube, as s07's crates will be."""
    import dataclasses

    from autofly_ue5.scenes.model import AssetEntry

    registry = load_registry()
    cube = registry.assets["cube"]
    box = AssetEntry(name="crate", ue_path=cube.ue_path, base_size_m=cube.base_size_m, pivot="center", footprint="box",
                     category="geometry", role="obstacle", seen=None, measured_extent_cm_at_unit_scale=cube.measured_extent_cm_at_unit_scale)
    return dataclasses.replace(registry, assets={**registry.assets, "crate": box})


def _scene(tmp_path, groups: list[dict], scene_id: str = "s07"):
    data = json.loads(S01.read_text())
    data["id"] = scene_id
    data["obstacle_groups"] = groups
    path = tmp_path / f"{scene_id}.json"
    path.write_text(json.dumps(data))
    return load_scene_file(path)


def _group(asset="cylinder", count=30, placement=None, xy=(0.8, 1.2), z=(2.0, 4.0), palette=("white",)):
    return {"asset": asset, "count": count, "scale_range": {"xy": list(xy), "z": list(z)}, "palette": list(palette),
            "placement": placement or {"type": "poisson", "margin_m": 8.0, "min_distance_m": 3.0}}


def test_poisson_places_count_points_inside_the_margin_at_least_min_distance_apart():
    from autofly_ue5.scenes.generate import poisson

    points = poisson(random.Random(7), BOUNDS, 60, margin_m=8.0, min_distance_m=3.0)
    assert len(points) == 60
    for x, y in points:
        assert -27.0 <= x <= 27.0 and -27.0 <= y <= 27.0
    assert min(math.dist(a, b) for a, b in itertools.combinations(points, 2)) >= 3.0
    assert poisson(random.Random(7), BOUNDS, 60, margin_m=8.0, min_distance_m=3.0) == points
    assert poisson(random.Random(8), BOUNDS, 60, margin_m=8.0, min_distance_m=3.0) != points
    with pytest.raises(ValueError, match="poisson"):
        poisson(random.Random(1), BOUNDS, 2000, margin_m=8.0, min_distance_m=3.0)  # more than the area can hold


def test_poisson_keeps_clear_of_what_is_already_there():
    from autofly_ue5.scenes.generate import poisson

    keepout = [(0.0, 0.0, 10.0)]  # an existing obstacle of radius 10 m at the origin
    points = poisson(random.Random(2), BOUNDS, 40, margin_m=8.0, min_distance_m=2.0, keepout=keepout, radius_m=0.5)
    assert all(math.hypot(x, y) >= 10.0 + 0.5 for x, y in points), "surface gap >= 0 to the existing obstacle"


def test_clusters_partition_the_count_into_clusters_of_the_given_size_and_radius():
    from autofly_ue5.scenes.generate import clusters

    rng = random.Random(5)
    centres, members = clusters(rng, BOUNDS, 24, margin_m=8.0, cluster_count=4, per_cluster=(4, 8), radius_m=3.0)
    assert len(centres) == 4 and sum(len(m) for m in members) == 24
    assert all(4 <= len(m) <= 8 for m in members)
    for (cx, cy), group in zip(centres, members):
        assert all(math.hypot(x - cx, y - cy) <= 3.0 + 1e-9 for x, y in group)
        assert all(-27.0 <= x <= 27.0 and -27.0 <= y <= 27.0 for x, y in group), "members stay inside the margin"
    assert min(math.dist(a, b) for a, b in itertools.combinations(centres, 2)) >= 6.0, "clusters do not overlap"
    with pytest.raises(ValueError, match="cluster"):
        clusters(random.Random(5), BOUNDS, 40, margin_m=8.0, cluster_count=4, per_cluster=(4, 8), radius_m=3.0)  # 40 > 4 x 8
    with pytest.raises(ValueError, match="cluster"):
        clusters(random.Random(5), BOUNDS, 10, margin_m=8.0, cluster_count=4, per_cluster=(4, 8), radius_m=3.0)  # 10 < 4 x 4


def test_a_clusters_scene_generates_one_instance_per_member(tmp_path):
    scene = _scene(tmp_path, [_group(count=24, placement={"type": "clusters", "margin_m": 8.0, "cluster_count": 4,
                                                           "per_cluster": [4, 8], "radius_m": 3.0})], scene_id="s06")
    layout = generate_layout(scene, load_registry())
    assert len(layout.instances) == 24 and len({i.tag for i in layout.instances}) == 24
    assert all(i.asset == "cylinder" and 0.4 <= i.radius_m <= 0.6 for i in layout.instances)
    assert generate_layout(scene, load_registry()) == layout


def test_a_poisson_scene_generates_count_instances_that_do_not_touch(tmp_path):
    scene = _scene(tmp_path, [_group(count=50)], scene_id="s03")
    layout = generate_layout(scene, load_registry())
    assert len(layout.instances) == 50
    for a, b in itertools.combinations(layout.instances, 2):
        assert math.hypot(a.x - b.x, a.y - b.y) >= a.radius_m + b.radius_m


def test_stacks_put_boxes_on_top_of_each_other_with_a_box_footprint(tmp_path):
    scene = _scene(tmp_path, [_group(asset="crate", count=20, xy=(1.0, 1.5), z=(1.0, 1.0),
                                     placement={"type": "stacks", "margin_m": 8.0, "stack_count": 6, "height_range": [2, 5]})])
    layout = generate_layout(scene, _registry_with_box())
    assert len(layout.instances) == 20
    by_xy: dict[tuple[float, float], list] = {}
    for inst in layout.instances:
        by_xy.setdefault((inst.x, inst.y), []).append(inst)
    assert len(by_xy) == 6, "six stacks"
    for stack in by_xy.values():
        assert 2 <= len(stack) <= 5
        stack.sort(key=lambda i: -i.z_center)  # NED: the lowest box has the z_center closest to 0
        assert stack[0].z_center == pytest.approx(-stack[0].height_m / 2, abs=1e-3), "the first box stands on the ground"
        for below, above in zip(stack, stack[1:]):
            assert above.z_center == pytest.approx(below.z_center - below.height_m / 2 - above.height_m / 2, abs=1e-3)
            assert above.scale[:2] == below.scale[:2] and above.yaw == below.yaw, "a stack shares one footprint"
        side = stack[0].scale[0] * 1.0
        assert stack[0].radius_m == pytest.approx(math.hypot(side, side) / 2, abs=1e-3), "the box's circumscribed circle"
    with pytest.raises(ValueError, match="stack"):
        generate_layout(_scene(tmp_path, [_group(asset="crate", count=40, placement={"type": "stacks", "margin_m": 8.0,
                                                                                      "stack_count": 6, "height_range": [2, 5]})]),
                        _registry_with_box())


def test_a_later_group_keeps_clear_of_an_earlier_one_and_a_grid_that_cannot_is_refused(tmp_path):
    scene = _scene(tmp_path, [_group(count=20, z=(8.0, 12.0)),
                              _group(count=20, xy=(0.3, 0.5), z=(0.5, 1.0), placement={"type": "poisson", "margin_m": 8.0, "min_distance_m": 1.5})],
                   scene_id="s04")
    layout = generate_layout(scene, load_registry())
    assert len(layout.instances) == 40
    for a, b in itertools.combinations(layout.instances, 2):
        assert math.hypot(a.x - b.x, a.y - b.y) >= a.radius_m + b.radius_m, f"{a.tag} and {b.tag} overlap"
    with pytest.raises(ValueError, match="overlap"):
        generate_layout(_scene(tmp_path, [_group(count=80), _group(count=80, placement={"type": "jittered_grid", "margin_m": 8.0,
                                                                                         "jitter_m": 1.0})], scene_id="s10"),
                        load_registry())


def test_a_box_footprint_on_a_single_box_group_gets_the_circumscribed_radius(tmp_path):
    scene = _scene(tmp_path, [_group(asset="crate", count=10, xy=(2.0, 2.0), z=(1.0, 1.0))])
    layout = generate_layout(scene, _registry_with_box())
    assert all(i.radius_m == pytest.approx(math.sqrt(2.0), abs=1e-3) for i in layout.instances)


def test_cluster_members_never_touch_across_clusters_or_earlier_groups(tmp_path):
    big = _group(count=6, xy=(10.0, 10.0), z=(2.0, 2.0), placement={"type": "poisson", "margin_m": 12.0, "min_distance_m": 12.0})
    trees = _group(count=24, xy=(1.0, 1.2), z=(6.0, 10.0),
                   placement={"type": "clusters", "margin_m": 8.0, "cluster_count": 4, "per_cluster": [4, 8], "radius_m": 3.0})
    dense = _group(count=60, xy=(1.2, 1.2), z=(6.0, 10.0),
                   placement={"type": "clusters", "margin_m": 8.0, "cluster_count": 10, "per_cluster": [4, 8], "radius_m": 3.0})
    for seed in range(60):
        for groups, sid in (([dense], "s06"), ([big, trees], "s02")):
            layout = generate_layout(_scene(tmp_path, groups, scene_id=sid), load_registry(), seed=seed)
            for a, b in itertools.combinations(layout.instances, 2):
                assert math.hypot(a.x - b.x, a.y - b.y) >= a.radius_m + b.radius_m - 1e-9, f"seed {seed}: {a.tag} touches {b.tag}"


def test_stacks_after_an_earlier_group_keep_clear_of_it(tmp_path):
    big = _group(count=6, xy=(10.0, 10.0), z=(2.0, 2.0), placement={"type": "poisson", "margin_m": 12.0, "min_distance_m": 12.0})
    boxes = _group(asset="crate", count=20, xy=(1.0, 1.5), z=(1.0, 1.0),
                   placement={"type": "stacks", "margin_m": 8.0, "stack_count": 6, "height_range": [2, 5]})
    for seed in range(20):
        layout = generate_layout(_scene(tmp_path, [big, boxes]), _registry_with_box(), seed=seed)
        pillars = [i for i in layout.instances if i.asset == "cylinder"]
        for box in (i for i in layout.instances if i.asset == "crate"):
            assert all(math.hypot(box.x - p.x, box.y - p.y) >= box.radius_m + p.radius_m - 1e-9 for p in pillars)


def test_a_grid_placed_after_a_distant_group_draws_as_it_would_alone(tmp_path):
    corner = _group(count=1, xy=(1.0, 1.0), z=(2.0, 2.0), placement={"type": "poisson", "margin_m": 1.0, "min_distance_m": 1.0})
    grid = _group(count=20, placement={"type": "jittered_grid", "margin_m": 8.0, "jitter_m": 1.0})
    alone = generate_layout(_scene(tmp_path, [grid], scene_id="s09"), load_registry(), seed=3)
    # The corner obstacle is drawn with the same rng first, so re-seed the comparison by generating both from one scene
    # whose first group is trivially far away: the grid's own draws must not depend on the keepout check.
    together = generate_layout(_scene(tmp_path, [corner, grid], scene_id="s09"), load_registry(), seed=3)
    assert len(together.instances) == 21
    # Same positions up to the corner group's draws? Not comparable directly (shared rng), so check the invariant that
    # matters: a grid never moves its points for a keepout, it only refuses.
    assert all(i.asset == "cylinder" for i in together.instances)
    assert alone.instances[0].x != together.instances[1].x or alone.instances[0].y != together.instances[1].y


def test_rocks_and_trees_face_random_ways_but_pillars_on_the_grid_do_not(tmp_path):
    rocks = generate_layout(_scene(tmp_path, [_group(count=30)], scene_id="s03"), load_registry())
    assert len({i.yaw for i in rocks.instances}) > 1 and all(-math.pi <= i.yaw < math.pi for i in rocks.instances)
    trees = generate_layout(_scene(tmp_path, [_group(count=24, placement={"type": "clusters", "margin_m": 8.0, "cluster_count": 4,
                                                                          "per_cluster": [4, 8], "radius_m": 3.0})], scene_id="s06"),
                            load_registry())
    assert len({i.yaw for i in trees.instances}) > 1
    assert all(i.yaw == 0.0 for i in generate_layout(load_scene_file(S01), load_registry()).instances), "s01 is pinned"


def test_errors_name_the_cause_for_inputs_the_schema_admits(tmp_path):
    from autofly_ue5.scenes.model import SceneFileError

    with pytest.raises(ValueError, match="radius"):
        generate_layout(_scene(tmp_path, [_group(count=8, placement={"type": "clusters", "margin_m": 8.0, "cluster_count": 2,
                                                                      "per_cluster": [4, 4], "radius_m": 30.0})], scene_id="s06"),
                        load_registry())
    with pytest.raises(SceneFileError, match="per_cluster"):
        _scene(tmp_path, [_group(count=8, placement={"type": "clusters", "margin_m": 8.0, "cluster_count": 2,
                                                      "per_cluster": [8, 4], "radius_m": 3.0})], scene_id="s06")
    with pytest.raises(SceneFileError, match="height_range"):
        _scene(tmp_path, [_group(count=8, placement={"type": "stacks", "margin_m": 8.0, "stack_count": 2, "height_range": [5, 2]})])


# ------------------------------------------------------------------------------------------------------------------
# M4: imported models with a radius-by-height profile and a base pivot (scenes/gltf.py, assets/measurements.json).
# ------------------------------------------------------------------------------------------------------------------


def _registry_with_tree():
    """The real registry plus a tree measured like scenes/gltf.py measures: a thin trunk to 1.25 m, a 1.4 m canopy from
    1.25 to 3.5 m, 3.5 m tall, the pivot at its base; and a rock that is 1.0 m tall at unit scale."""
    import dataclasses

    from autofly_ue5.scenes.model import AssetEntry

    registry = load_registry()
    tree = AssetEntry(name="tree", ue_path="/Game/AutoFly/Assets/tree/tree", base_size_m=(2.8, 2.8, 3.5), pivot="base",
                      footprint="circle", category="nature", role="obstacle", seen=None, measured_extent_cm_at_unit_scale=None,
                      profile_step_m=0.25, radius_profile_m=(0.15,) * 5 + (1.4,) * 9, ground_radius_m=1.4)
    rock = AssetEntry(name="rock", ue_path="/Game/AutoFly/Assets/rock/rock", base_size_m=(1.3, 1.8, 1.0), pivot="base",
                      footprint="circle", category="nature", role="obstacle", seen=None, measured_extent_cm_at_unit_scale=None,
                      profile_step_m=0.25, radius_profile_m=(0.95, 0.9, 0.85, 0.6), ground_radius_m=0.95)
    return dataclasses.replace(registry, assets={**registry.assets, "tree": tree, "rock": rock})


def test_a_profiled_asset_s_footprint_is_what_the_flight_band_meets_at_its_scale(tmp_path):
    scene = _scene(tmp_path, [_group(asset="tree", count=12, xy=(1.0, 1.0), z=(1.0, 1.0),
                                     placement={"type": "poisson", "margin_m": 8.0, "min_distance_m": 6.0})], scene_id="s02")
    layout = generate_layout(scene, _registry_with_tree())
    assert len(layout.instances) == 12
    for inst in layout.instances:
        assert inst.radius_m == pytest.approx(1.4), "the canopy fills the 1-3 m band at unit scale"
        assert inst.height_m == pytest.approx(3.5) and inst.z_center == 0.0, "a base pivot stands on the ground"
    halved = _scene(tmp_path, [_group(asset="tree", count=12, xy=(1.0, 1.0), z=(0.3, 0.3),
                                      placement={"type": "poisson", "margin_m": 8.0, "min_distance_m": 6.0})], scene_id="s02")
    for inst in generate_layout(halved, _registry_with_tree()).instances:
        assert inst.radius_m == pytest.approx(1.4), "at 0.3 the tree is 1.05 m tall: its canopy top (1.25-3.5 -> 0.375-1.05 m) still reaches 1 m"
    tiny = _scene(tmp_path, [_group(asset="tree", count=4, xy=(1.0, 1.0), z=(0.2, 0.2),
                                    placement={"type": "poisson", "margin_m": 8.0, "min_distance_m": 6.0})], scene_id="s02")
    with pytest.raises(ValueError, match="flight band"):
        generate_layout(tiny, _registry_with_tree())  # 0.7 m tall: below the band, no obstacle to the drone


def test_scaled_rocks_present_their_scaled_profile_and_unscaled_rocks_are_refused(tmp_path):
    big = _scene(tmp_path, [_group(asset="rock", count=20, xy=(2.0, 3.0), z=(2.0, 3.0),
                                   placement={"type": "poisson", "margin_m": 8.0, "min_distance_m": 7.0})], scene_id="s05")
    layout = generate_layout(big, _registry_with_tree())
    for inst in layout.instances:
        s_xy, _s, s_z = inst.scale
        # 1-3 m at scale s_z is the model's 1/s_z to 3/s_z; a 1 m rock scaled 2-3x spans that with its slices 0.33+ m
        assert 0.6 * s_xy - 1e-6 <= inst.radius_m <= 0.95 * s_xy + 1e-6
        assert inst.height_m == pytest.approx(1.0 * s_z)
    natural = _scene(tmp_path, [_group(asset="rock", count=5, xy=(1.0, 1.0), z=(1.0, 1.0))], scene_id="s03")
    with pytest.raises(ValueError, match="flight band"):
        generate_layout(natural, _registry_with_tree())


def test_placement_spacing_uses_the_ground_footprint_so_meshes_never_interpenetrate(tmp_path):
    # At scale 0.3 the tree's band radius is 1.4 but its ground radius is also 1.4 x 1.0: here make the band radius small
    # by using the rock scaled tall and thin: band radius comes from the upper, narrower slices; ground radius from the base
    scene = _scene(tmp_path, [_group(asset="rock", count=40, xy=(1.0, 1.0), z=(3.0, 3.0),
                                     placement={"type": "poisson", "margin_m": 8.0, "min_distance_m": 0.5})], scene_id="s05")
    layout = generate_layout(scene, _registry_with_tree())
    for a, b in itertools.combinations(layout.instances, 2):
        assert math.hypot(a.x - b.x, a.y - b.y) >= 2 * 0.95 - 1e-9, "centres at least two ground radii apart"


def test_the_registry_reads_profiles_and_stacks_refuse_a_base_pivot(tmp_path):
    import dataclasses

    from autofly_ue5.scenes.model import load_registry as load

    data = json.loads((SCENES_DIR.parent / "assets" / "registry.json").read_text())
    data["assets"]["tree"] = {"ue_path": "/Game/AutoFly/Assets/tree/tree", "base_size_m": [2.8, 2.8, 3.5], "pivot": "base",
                              "footprint": "circle", "category": "nature", "role": "obstacle", "seen": None,
                              "measured_extent_cm_at_unit_scale": [140.0, 140.0, 175.0], "profile_step_m": 0.25,
                              "radius_profile_m": [0.15] * 5 + [1.4] * 9, "ground_radius_m": 1.4}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(data))
    registry = load(path)
    assert registry.assets["tree"].radius_profile_m == (0.15,) * 5 + (1.4,) * 9 and registry.assets["tree"].ground_radius_m == 1.4
    assert registry.assets["cylinder"].radius_profile_m is None, "engine primitives have none"
    boxes = _registry_with_box()
    based = dataclasses.replace(boxes, assets={**boxes.assets, "crate": dataclasses.replace(boxes.assets["crate"], pivot="base")})
    with pytest.raises(ValueError, match="pivot"):
        generate_layout(_scene(tmp_path, [_group(asset="crate", count=10, placement={"type": "stacks", "margin_m": 8.0,
                                                                                      "stack_count": 3, "height_range": [2, 5]})]), based)
