"""scripts/dataset_figures.py draws what a list of raw stores holds (composition, kept-rates, trajectory maps,
action/state distributions, lengths and PER, first frames, target close-ups) and writes every number it draws."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tests.synthetic_stores import Flight, instance, legacy_target, pool_target, write_layout, write_store

PNG = b"\x89PNG\r\n\x1a\n"
TREE = instance("obs_0000", "island_tree_02", 1.1312, -4.2525, scale=(0.9451, 0.9451, 1.0577), yaw=-2.9664,
                radius_m=3.1341, height_m=3.6032, z_center=0.0, material="mesh")
QUIVER = instance("obs_0021", "quiver_tree_01", -1.2636, -8.7103, scale=(0.9337, 0.9337, 1.0137), yaw=-3.0221,
                  radius_m=0.6274, height_m=4.0, z_center=0.0, material="mesh")


def _pool(tmp_path):
    def entry(name, category, seen=True, status="imported", **extra):
        return {"name": name, "category": category, "seen": seen, "asset": "x", "scale": [1, 1, 1], "material": None,
                "status": status, "source": None, "height_m": 1.0, **extra}
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"version": 1, "targets": [
        entry("orange cylinder", "geometry", status="primitive"), entry("red sedan", "vehicle", status="placeholder",
                                                                        stand_in=True),
        entry("beige sofa", "furniture"), entry("green suitcase", "equipment"),
        entry("grey armchair", "furniture")]}))  # drawable, never kept
    return path


def _world(tmp_path):
    """Four stores over s01, s01d (s01's level, with movers), s02 (trees: outlines) and s07 (stacked crates: boxes)."""
    data = tmp_path / "data"
    sofa, sedan = pool_target("beige sofa", "furniture", stand_in=False), pool_target("red sedan", "vehicle", stand_in=True)
    ladder = pool_target("wooden ladder", "equipment", seen=False)  # in no pool file and unmarked
    suitcase = pool_target("green suitcase", "equipment", stand_in=False)
    stores = [
        write_store(data, "s01_c1", "s01", [
            Flight(1, 6, legacy_target(), start=(-30.0, -10.0), target_xy=(30.0, -10.0)),
            Flight(2, 8, sedan, start=(-30.0, 0.0), target_xy=(30.0, 5.0), template=1),
            Flight(3, 7, sofa, start=(-30.0, 10.0), target_xy=(30.0, 12.0), bearing_end_rad=0.3)],
            rejects=[(Flight(4, 3, sofa), "collision", "sim")]),
        write_store(data, "s01d_c1", "s01d", [
            Flight(1, 5, sofa, movers=[{"tag": "obs_0001"}], bearing_end_rad=0.2),
            Flight(2, 9, sedan, movers=[{"tag": "obs_0001"}, {"tag": "obs_0000"}])]),
        write_store(data, "s02_c1", "s02", [
            Flight(1, 5, sofa, bearing_end_rad=0.05, colour=(40, 200, 90), extra={"prewarm_steps": 5,
                   "first_frame_visibility": {"status": "visible"}}),
            Flight(2, 6, sofa, bearing_end_rad=-0.02, extra={"first_frame_visibility": {"status": "occluded"}}),
            Flight(3, 7, ladder), Flight(4, 8, suitcase, start=(0.0, -30.0), target_xy=(0.0, 30.0))],
            rejects=[(Flight(5, 3, sofa), "collision", "footprint"), (Flight(6, 3, sofa), "collision", "footprint"),
                     (Flight(7, 3, sofa), "out_of_bounds", None)]),
        write_store(data, "s07_c1", "s07", [Flight(1, 4, suitcase, bearing_end_rad=-0.4)]),
    ]
    levels = tmp_path / "levels"
    write_layout(levels, "s01", [instance("obs_0000", "cylinder", 0.0, 20.0), instance("obs_0001", "cylinder", 10.0, -20.0)])
    write_layout(levels, "s02", [TREE, QUIVER])
    crate = dict(asset="crate", scale=(1.0, 1.0, 1.0), yaw=0.3, radius_m=0.7071, height_m=1.0, material="wood")
    write_layout(levels, "s07", [instance("obs_0000", x=5.0, y=5.0, z_center=-0.5, **crate),
                                 instance("obs_0001", x=5.0, y=5.0, z_center=-1.5, **crate)])
    return stores, levels


def test_every_figure_is_written_and_every_number_drawn_is_in_the_json(tmp_path):
    from scripts import dataset_figures

    stores, levels = _world(tmp_path)
    out = tmp_path / "figures"
    record = tmp_path / "s01_c1.record.json"
    record.write_text(json.dumps({"dataset": {"name": "s01_c1"}, "collection": {
        "kept": 3, "attempted": 4, "fault_retries": 2, "outcomes": {"success": 3, "collision": 1}}}))
    assert dataset_figures.main(["--stores", *map(str, stores), "--out-dir", str(out), "--levels", str(levels),
                                 "--targets", str(_pool(tmp_path)), "--record", f"s01_c1={record}",
                                 "--records-dir", str(tmp_path / "no_records")]) == 0
    summary = json.loads((out / "dataset_figures.json").read_text())
    expected = {"composition.png", "kept_rates.png", "actions_states.png", "lengths_per.png", "targets_closeup.png",
                *(f"trajectories_{s}.png" for s in ("s01", "s01d", "s02", "s07")),
                *(f"first_frames_{s}.png" for s in ("s01", "s01d", "s02", "s07"))}
    assert set(summary["files"]) == expected
    for name in expected:
        assert (out / name).read_bytes()[:8] == PNG, name

    comp = summary["composition"]
    assert (comp["episodes"], comp["records"]) == (10, 6 + 8 + 7 + 5 + 9 + 5 + 6 + 7 + 8 + 4)
    assert comp["by_scene"] == {"s01": 3, "s01d": 2, "s02": 4, "s07": 1}
    assert comp["by_category"] == {"vehicle": 2, "geometry": 1, "furniture": 4, "equipment": 3, "human": 0, "other": 0}
    assert comp["by_target"] == {"beige sofa": 4, "green suitcase": 2, "red sedan": 2, "orange cylinder": 1,
                                 "wooden ladder": 1}
    assert list(comp["by_target"]) == ["beige sofa", "green suitcase", "red sedan", "orange cylinder", "wooden ladder"]
    assert comp["seen"] == {"seen": 9, "unseen": 1, "unknown": 0}
    assert comp["stand_in"] == {"stand-in": 2, "not a stand-in": 7, "unmarked": 1}
    assert comp["filled_from_pool"] == {"category": 1, "seen": 1, "stand_in": 1, "status": 1}
    assert comp["pool_targets_never_kept"] == ["grey armchair"]

    rates = {r["store"]: r for r in summary["kept_rates"]["stores"]}
    s02 = rates["s02_c1"]
    assert (s02["kept"], s02["flown"], s02["rejected"], s02["source"]) == (4, 7, {"collision": 2, "out_of_bounds": 1},
                                                                           "rejects")
    assert s02["kept_rate"] == pytest.approx(4 / 7, abs=1e-4) and s02["collision_sources"] == {"footprint": 2}
    assert s02["wilson_95"] == pytest.approx([0.2505, 0.8418], abs=1e-6)
    assert rates["s01_c1"]["source"] == "rejects+record" and rates["s01_c1"]["fault_retries"] == 2
    assert rates["s01d_c1"]["flown"] == 2 and rates["s01d_c1"]["rejected"] == {}
    assert {s["scene"]: s["flown"] for s in summary["kept_rates"]["scenes"]} == {"s01": 4, "s01d": 2, "s02": 7, "s07": 1}

    traj = summary["trajectories"]
    assert traj["s02"]["obstacles"] == {"outline": 2} and traj["s02"]["kept"] == 4 and traj["s02"]["rejected_endings"] == 3
    assert traj["s07"]["obstacles"] == {"box": 1}, "a stack of crates is one box seen from above"
    assert traj["s01"]["obstacles"] == {"circle": 2} and traj["s01d"]["layout_scene"] == "s01"
    assert traj["s01d"]["moving_pillars"] == {"moved_in_any_episode": 2, "of": 2}

    dims = summary["actions_states"]["scenes"]["s02"]
    assert dims["action[0]"]["median"] == pytest.approx(2.0) and dims["state[8]"]["median"] == pytest.approx(2.0)
    assert dims["state[6]"]["min"] == pytest.approx(0.0) and summary["actions_states"]["records"]["s02"] == 26
    assert summary["actions_states"]["dimensions"]["state[0]"]["privileged"] is True

    per = summary["lengths_per"]["scenes"]
    assert per["s01"]["records"]["median"] == 7 and per["s01"]["per_interval"][0] <= per["s01"]["per_interval"][1]
    assert per["s01d"]["per_interval"][1] is None and "movers" in per["s01d"]["note"]
    assert per["s01d"]["per_straight"]["n"] == 2 and per["s01"]["per_physical"]["n"] == 3

    assert summary["first_frames"]["s02"]["episodes"] == ["s02_1", "s02_4", "s02_3", "s02_2"]
    assert summary["first_frames"]["s02"]["first_frame_visibility"] == {"visible": 1, "occluded": 1, "not recorded": 2}
    assert summary["first_frames"]["s01"]["first_frame_visibility"] is None, "no episode of s01 carries the check"
    closeups = summary["targets_closeup"]["targets"]
    assert closeups["beige sofa"]["episode"] == "s02_2" and closeups["green suitcase"]["episode"] == "s02_4"
    assert [t["category"] for t in closeups.values()] == sorted(
        (t["category"] for t in closeups.values()), key=["vehicle", "geometry", "furniture", "equipment", "human",
                                                         "other", "unknown"].index)


def test_a_scene_without_a_layout_is_mapped_without_obstacles_and_keeps_only_the_straight_line_bound(tmp_path):
    from scripts import dataset_figures

    store = write_store(tmp_path / "data", "s03_c1", "s03", [Flight(1, 5, pool_target("beige sofa", "furniture"))])
    out = tmp_path / "figures"
    assert dataset_figures.main(["--stores", str(store), "--out-dir", str(out), "--levels", str(tmp_path / "none"),
                                 "--records-dir", str(tmp_path / "none")]) == 0
    summary = json.loads((out / "dataset_figures.json").read_text())
    assert summary["trajectories"]["s03"]["obstacles"] == {} and summary["trajectories"]["s03"]["layout"] is None
    per = summary["lengths_per"]["scenes"]["s03"]
    assert per["per_interval"][1] is None and per["per_interval"][0] is not None and "no layout" in per["note"]
    assert any(n.startswith("s03: no layout") for n in summary["notes"])
    assert (out / "trajectories_s03.png").read_bytes()[:8] == PNG


def test_first_frames_take_targets_in_turn_and_stop_at_the_limit(tmp_path):
    from autofly_ue5.report.stores import load_store
    from scripts.dataset_figures import pick_first_frames

    sofa, chair = pool_target("beige sofa", "furniture"), pool_target("grey armchair", "furniture")
    root = write_store(tmp_path / "data", "s02_c1", "s02", [Flight(i, 3, sofa if i < 5 else chair) for i in range(1, 7)])
    picked = pick_first_frames(load_store(root).episodes, 3)
    assert [e.id for e in picked] == ["s02_1", "s02_5", "s02_2"]


def test_an_outline_runs_along_the_field_s_zero_set_inside_the_footprint_circle():
    from autofly_ue5.paths import ASSET_FOOTPRINTS
    from autofly_ue5.scenes.footprint_field import CELL_M, FootprintFields, to_local
    from autofly_ue5.scenes.model import Instance
    from scripts.dataset_figures import outline_polygons, polygon_area

    inst = Instance(**{**TREE, "scale": tuple(TREE["scale"])})
    field = FootprintFields.load(["island_tree_02"], ASSET_FOOTPRINTS)["island_tree_02"]
    polygons = outline_polygons(field, inst)
    assert polygons, "a tree has an outline"
    marked = float((field.values <= CELL_M * math.sqrt(2) / 2).sum()) * (CELL_M * inst.scale[0]) ** 2
    assert sum(polygon_area(p) for p in polygons) >= 0.8 * marked
    for p in polygons:
        assert p.shape[1] == 2
        assert np.hypot(p[:, 0] - inst.x, p[:, 1] - inst.y).max() <= inst.radius_m + 0.05, "inside the circle"
        assert field.distance(to_local(inst, p)).max() <= CELL_M, "on the outline (a cell at unit scale)"


def test_a_box_is_its_scaled_footprint_turned_by_its_yaw():
    from autofly_ue5.scenes.model import Instance, load_registry
    from scripts.dataset_figures import box_corners

    inst = Instance(tag="b", asset="crate", x=2.0, y=3.0, z_center=-0.5, yaw=math.pi / 6, scale=(2.0, 1.0, 1.0),
                    material="wood", radius_m=1.12, height_m=1.0)
    corners = box_corners(inst, load_registry().assets["crate"])
    c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
    expected = [(2.0 + c * dx - s * dy, 3.0 + s * dx + c * dy) for dx, dy in ((1, 0.5), (-1, 0.5), (-1, -0.5), (1, -0.5))]
    assert np.allclose(sorted(map(tuple, corners)), sorted(expected))


def test_a_mesh_without_an_outline_field_is_drawn_as_its_circle(tmp_path):
    from autofly_ue5.scenes.resolve import resolve_scene
    from autofly_ue5.scenes.model import load_registry
    from scripts.dataset_figures import obstacle_shapes

    levels = tmp_path / "levels"
    write_layout(levels, "s02", [TREE, QUIVER])
    shapes, notes = obstacle_shapes(resolve_scene("s02", levels_dir=levels).layout, load_registry(), tmp_path / "none")
    assert [s.kind for s in shapes] == ["circle", "circle"] and any("no outline field" in n for n in notes)
    # A field measures at one horizontal scale; an instance stretched unevenly cannot be outlined, so it is its circle.
    write_layout(levels, "s02", [TREE, {**QUIVER, "scale": [0.9, 1.4, 1.0]}])
    from autofly_ue5.paths import ASSET_FOOTPRINTS

    shapes, notes = obstacle_shapes(resolve_scene("s02", levels_dir=levels).layout, load_registry(), ASSET_FOOTPRINTS)
    assert [s.kind for s in shapes] == ["outline", "circle"] and any("obs_0021" in n for n in notes)
