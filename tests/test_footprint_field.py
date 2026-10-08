"""Each mesh obstacle's outline as a distance field (autofly_ue5/scenes/footprint_field.py, 2026-10-08 afternoon).

On s02 the footprint circle (about the pivot, the model's widest reach as radius) ended 82 of 220 evaluation flights a
median 2.70 m from any geometry: the expert, which sees only depth, was stopped by an invisible bulge of the circle on
the side away from a leaning tree's crown. The rule now measures to the model's outline: the horizontal projection of
all its geometry (wood and leaf cards, every height, a column), as a distance field per asset at unit scale in the
mesh's glTF horizontal plane. The field must never overstate the clearance (it is a safety rule) and must stay within a
few centimetres of it (it is the task's rule). These tests hold it to an analytic distance on synthetic models written
as glTF files, to Mesh._to_gltf's transform, and to a brute-force distance on the real s02 models when they are on disk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from autofly_ue5.scenes.model import Instance

# ----------------------------------------------------------------------------------------------------------------------
# Synthetic models, written as glTF 2.0 files the live checks' loader reads (MeshLibrary.from_files).
# ----------------------------------------------------------------------------------------------------------------------
L_RECTS = ((0.0, 2.0, 0.0, 0.5), (0.0, 0.5, 0.0, 1.5))  # the L's two bars, (x0, x1, z0, z1) in glTF x and z
TRUNK = (-0.05, 0.05, -0.05, 0.05, 0.0, 2.5)  # x0, x1, z0, z1, y0, y1: a thin trunk standing on the pivot
CROWN = (0.5, 1.5, -0.5, 0.5, 2.0, 3.5)  # a crown box offset 1 m in x, at 2-3.5 m: the lean


def box_triangles(x0, x1, z0, z1, y0, y1) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box in glTF's frame (y up): 8 vertices, 12 triangles."""
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float64)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int64)
    return v, f


def l_prism() -> tuple[np.ndarray, np.ndarray]:
    """An L-shaped vertical prism (two boxes, 0-3 m tall): its projection is the L of L_RECTS."""
    parts = [box_triangles(x0, x1, z0, z1, 0.0, 3.0) for x0, x1, z0, z1 in L_RECTS]
    v = np.concatenate([p[0] for p in parts])
    f = np.concatenate([parts[0][1], parts[1][1] + len(parts[0][0])])
    return v, f


def write_gltf(folder: Path, name: str, parts: list[tuple[np.ndarray, np.ndarray, bool]]) -> Path:
    """`parts`: (vertices, triangles, opaque) per primitive; a non-opaque one is an alpha-blended card material, as
    Poly Haven's leaves are. One node named `name` holding one mesh, its buffer in an external .bin."""
    folder.mkdir(parents=True, exist_ok=True)
    blob, accessors, views, prims, materials = bytearray(), [], [], [], []
    for v, f, opaque in parts:
        for data, kind, comp in ((np.asarray(v, "<f4"), "VEC3", 5126), (np.asarray(f, "<u4").ravel(), "SCALAR", 5125)):
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": data.nbytes})
            acc = {"bufferView": len(views) - 1, "componentType": comp, "count": len(data), "type": kind}
            if kind == "VEC3":
                acc.update(min=data.min(axis=0).tolist(), max=data.max(axis=0).tolist())
            accessors.append(acc)
            blob += data.tobytes()
        materials.append({"name": f"m{len(materials)}", "alphaMode": "OPAQUE" if opaque else "BLEND"})
        prims.append({"attributes": {"POSITION": len(accessors) - 2}, "indices": len(accessors) - 1,
                      "material": len(materials) - 1})
    (folder / f"{name}.bin").write_bytes(bytes(blob))
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": name, "mesh": 0}],
           "meshes": [{"name": name, "primitives": prims}], "materials": materials, "accessors": accessors,
           "bufferViews": views, "buffers": [{"uri": f"{name}.bin", "byteLength": len(blob)}]}
    path = folder / f"{name}.gltf"
    path.write_text(json.dumps(doc))
    return path


def synthetic_library(tmp_path: Path, models: dict[str, list]) -> dict[str, Path]:
    """Registry, sources and measurements files in tmp_path naming each synthetic model as MeshLibrary.from_files
    resolves a real one (downloads/<source>/<id>/<file>, the node by name). Returns the three paths."""
    downloads = tmp_path / "downloads"
    registry, meshes = {"assets": {}}, []
    for key, parts in models.items():
        path = write_gltf(downloads / "synthetic" / key, key, parts)
        registry["assets"][key] = {"source": {"source": "synthetic", "id": key, "node": key}}
        meshes.append({"asset": key, "node": key, "file": path.name})
    paths = {"registry_path": tmp_path / "registry.json", "sources_path": tmp_path / "sources.json",
             "measurements_path": tmp_path / "measurements.json"}
    paths["registry_path"].write_text(json.dumps(registry))
    paths["sources_path"].write_text(json.dumps({"downloads_dir": str(downloads), "assets": []}))
    paths["measurements_path"].write_text(json.dumps({"meshes": meshes}))
    return paths


def leaning_tree_parts() -> list:
    """A thin opaque trunk on the pivot and a crown of leaf cards (alpha-blended) offset 1 m, at 2-3.5 m."""
    return [(*box_triangles(*TRUNK), True), (*box_triangles(*CROWN), False)]


def synthetic_field(tmp_path: Path, key: str, parts: list):
    from autofly_ue5.scenes.footprint_field import build_field, outline_triangles
    from autofly_ue5.validate.obstacles import MeshLibrary

    library = MeshLibrary.from_files({key}, **synthetic_library(tmp_path, {key: parts}))
    return build_field(outline_triangles(library[key]))


def rect_distance(px, pz, rect) -> np.ndarray:
    x0, x1, z0, z1 = rect
    dx = np.maximum(np.maximum(x0 - px, px - x1), 0.0)
    dz = np.maximum(np.maximum(z0 - pz, pz - z1), 0.0)
    return np.hypot(dx, dz)


def union_distance(points: np.ndarray, rects) -> np.ndarray:
    return np.min([rect_distance(points[:, 0], points[:, 1], r) for r in rects], axis=0)


# ----------------------------------------------------------------------------------------------------------------------
# The field against analytic distances.
# ----------------------------------------------------------------------------------------------------------------------
def test_an_l_shaped_prism_field_is_the_distance_to_its_projection_and_never_more(tmp_path):
    field = synthetic_field(tmp_path, "l_prism", [(*l_prism(), True)])
    rng = np.random.default_rng(0)
    points = rng.uniform([-2.5, -2.5], [4.5, 4.0], size=(4000, 2))
    exact = union_distance(points, L_RECTS)
    got = field.distance(points)
    outside = exact > 0.02  # on the outline and inside it both read ~0 (the rule ends 0.9 m out)
    assert outside.sum() > 3000
    err = got[outside] - exact[outside]
    assert err.min() >= -0.03, f"understated by {-err.min():.4f} m"
    assert err.max() <= 1e-3, f"overstated by {err.max():.4f} m: the field must never claim more clearance"
    inside = union_distance(points, [(x0 + 0.03, x1 - 0.03, z0 + 0.03, z1 - 0.03) for x0, x1, z0, z1 in L_RECTS]) == 0
    assert np.all(got[inside] == 0.0), "inside the projection the clearance is zero"


def test_edges_on_fine_cell_boundaries_are_never_overstated(tmp_path):
    # The raster marks the fine cell an edge lies in, and the distance transform measures to that cell's centre: an
    # edge on the cell's far side from a point is then up to half a fine cell (3.1 mm) nearer than the transform says,
    # and a corner up to half a fine diagonal (4.4 mm). build_field subtracts half a fine diagonal for exactly this.
    # The layout's own lowest corner always falls on a fine centre (the grid starts MARGIN_M before it), which is why
    # the L's edges never showed it: here a small anchor sets the grid, and a square's sides sit 1 um inside fine-cell
    # boundaries (left and bottom just above one, right and top just below), the worst place, so every side's marked
    # cells lie on its far side from the free space. Points facing a side (not a corner, where bilinear reading of the
    # distance's curvature adds its own fraction of a millimetre) must never read more than their distance.
    from autofly_ue5.scenes.footprint_field import CELL_M, MARGIN_M, SUPERSAMPLE

    fine = CELL_M / SUPERSAMPLE
    anchor = (-2.0, -1.95, -2.0, -1.95)  # its lowest corner sets the grid, 2.4 m or more from every point read below
    boundary = lambda i: anchor[0] - MARGIN_M + (i + 0.5) * fine  # noqa: E731  (the fine cells' boundaries)
    x0, x1 = boundary(880) + 1e-6, boundary(1040) - 1e-6  # 0.503126 and 1.503124 m
    assert x0 == pytest.approx(0.503126) and x1 == pytest.approx(1.503124)
    square = (x0, x1, x0, x1)
    field = synthetic_field(tmp_path, "aligned", [(*box_triangles(*anchor, 0.0, 3.0), True),
                                                  (*box_triangles(*square, 0.0, 3.0), True)])
    assert field.origin == pytest.approx((anchor[0] - MARGIN_M, anchor[2] - MARGIN_M)), "the anchor sets the grid"
    rng = np.random.default_rng(9)
    span = rng.uniform(x0 + 0.05, x1 - 0.05, size=600)  # along a side, clear of the corners' curvature
    out = rng.uniform(0.05, 1.0, size=600)  # how far out from that side
    points = np.concatenate([np.stack([x0 - out, span], axis=1)[:150], np.stack([x1 + out, span], axis=1)[150:300],
                             np.stack([span, x0 - out], axis=1)[300:450], np.stack([span, x1 + out], axis=1)[450:]])
    exact = union_distance(points, [anchor, square])
    assert np.allclose(exact, out), "each point faces one side of the square, `out` from it"
    err = field.distance(points) - exact
    assert err.max() <= 0.0, f"overstated by {err.max() * 1000:.2f} mm: an edge on a cell boundary reads too far"
    assert err.min() >= -0.004, f"understated by {-err.min() * 1000:.2f} mm"


def test_a_leaning_trees_field_counts_its_crown_as_a_column_and_its_cards_as_geometry(tmp_path):
    # The crown is 2-3.5 m up and offset 1 m; a drone at 1.5 m beside it is under it, and after an ending the reset
    # climbs straight up, so the crown counts at every height (the column). It is leaf cards (alpha-blended): it still
    # counts, as everything the depth camera can see does.
    field = synthetic_field(tmp_path, "lean", leaning_tree_parts())
    rects = [TRUNK[:4], CROWN[:4]]
    rng = np.random.default_rng(1)
    points = rng.uniform([-3.0, -3.0], [4.0, 3.0], size=(3000, 2))
    exact = union_distance(points, rects)
    got = field.distance(points)
    outside = exact > 0.02
    err = got[outside] - exact[outside]
    assert err.min() >= -0.03 and err.max() <= 1e-3, (err.min(), err.max())
    # under the crown (x = 1, z = 0) the clearance is zero, as it is on the trunk
    assert field.distance(np.array([[1.0, 0.0], [0.0, 0.0]])).tolist() == [0.0, 0.0]
    # on the side away from the lean the outline is the trunk: a point 0.9 m from its face is 0.9 m clear
    away = field.distance(np.array([[-0.95, 0.0]]))[0]
    assert away == pytest.approx(0.9, abs=0.03) and away <= 0.9 + 1e-3


def test_geometry_below_the_ground_is_not_part_of_the_outline(tmp_path):
    # Poly Haven's rocks sit up to 10 cm into the ground (boulder_01 to -0.074 m, sand_rocks_small_01 to -0.104 m),
    # and their buried part reaches 6-17 mm past the widest slice above the ground, the footprint circle the layouts
    # were generated with. The drone cannot be under the ground, so the outline is the projection of what stands above
    # it: a buried footing is dropped and a triangle crossing the ground counts only above it.
    from autofly_ue5.scenes.footprint_field import clip_above_ground

    column = box_triangles(-0.2, 0.2, -0.2, 0.2, 0.0, 2.0)
    footing = box_triangles(-1.0, 1.0, -1.0, 1.0, -0.5, -0.1)
    slanted = (np.array([[0.2, 0.5, 0.0], [1.2, -0.5, 0.0], [0.2, -0.5, 0.6]]), np.array([[0, 1, 2]]))
    field = synthetic_field(tmp_path, "sunk", [(*column, True), (*footing, True), (*slanted, True)])
    # Above the ground the slanted triangle is (0.2, 0), (0.7, 0), (0.2, 0.3) in glTF x, z (cut half way down the edges
    # from its one corner above). The first two points lie over the buried footing, the third over the slanted
    # triangle's buried part; the last inside its part above the ground.
    points = np.array([[-0.6, 0.0], [0.0, -0.8], [1.0, 0.0], [0.4, 0.1]])
    expected = [0.4, 0.6, 0.3, 0.0]
    got = field.distance(points)
    assert got == pytest.approx(expected, abs=0.012) and np.all(got <= np.array(expected) + 1e-3), got
    corners = np.concatenate([column[0][column[1]], footing[0][footing[1]], slanted[0][slanted[1]]])
    clipped = clip_above_ground(corners)
    assert clipped[:, :, 1].min() >= 0.0 and len(clipped) == 12 + 1, "the footing dropped, the slanted one cut to one"
    assert np.hypot(clipped[:, :, 0], clipped[:, :, 2]).max() == pytest.approx(0.7, abs=1e-12)


def test_the_field_reads_infinity_beyond_its_grid_and_states_its_reach(tmp_path):
    from autofly_ue5.scenes.footprint_field import CELL_M, MARGIN_M

    field = synthetic_field(tmp_path, "l_prism", [(*l_prism(), True)])
    assert field.reach_m == pytest.approx(MARGIN_M - CELL_M / 2)
    far = field.distance(np.array([[2.0 + MARGIN_M + 0.5, 0.25], [-MARGIN_M - 0.2, 0.0]]))
    assert np.all(np.isinf(far))
    near_edge = field.distance(np.array([[2.0 + field.reach_m - 1e-6, 0.25]]))[0]
    assert np.isfinite(near_edge) and near_edge == pytest.approx(field.reach_m, abs=0.03)


def test_a_non_finite_point_reads_as_beyond_every_field_never_an_error(tmp_path):
    # Non-finite kinematics are the backend's to refuse, not the rule's: a NaN or infinite position reads as +inf (no
    # contact), as the field reads any point off its grid, and never raises from the swept step's sample count (a bare
    # ValueError there would end a 12 h run, MEMORY.md).
    from autofly_ue5.scenes.footprint_field import outline_distance

    field = synthetic_field(tmp_path, "l_prism", [(*l_prism(), True)])
    inst = Instance("t", "l_prism", 0.0, 0.0, 0.0, 0.0, (1.0, 1.0, 1.0), "mesh", 2.5, 3.0)
    nan, inf = math.nan, math.inf
    for bad in ((nan, 0.0), (inf, 0.0), (0.0, -inf), (nan, nan)):
        assert outline_distance(field, inst, bad) == inf
        assert outline_distance(field, inst, bad, (3.0, 3.0)) == inf
        assert outline_distance(field, inst, (3.0, 3.0), bad) == inf
    assert np.isinf(field.distance(np.array([[nan, 0.25], [inf, 0.25], [0.25, -inf]]))).all()
    assert outline_distance(field, inst, (2.5, 0.25), (2.6, 0.25)) == pytest.approx(0.5, abs=0.015), "finite: as ever"


def test_a_damaged_field_file_is_refused_as_a_value_error_naming_it(tmp_path):
    # A ValueError, so the trainer's checks before it claims a run root report "refusing to start" (a BadZipFile or a
    # KeyError escaped them as a traceback), with the file to rebuild.
    import zipfile

    from autofly_ue5.scenes.footprint_field import FootprintField

    path = tmp_path / "lean.npz"
    synthetic_field(tmp_path, "lean", leaning_tree_parts()).save(path)
    whole = path.read_bytes()
    for damaged in (whole[: len(whole) // 2], whole[:100], b""):
        path.write_bytes(damaged)
        with pytest.raises(ValueError, match="lean.npz.*scripts/footprint_fields.py"):
            FootprintField.load(path)
    with zipfile.ZipFile(path, "w") as archive:  # a readable archive without the field's members
        archive.writestr("other.npy", b"")
    with pytest.raises(ValueError, match="lean.npz"):
        FootprintField.load(path)
    path.unlink()
    with pytest.raises(ValueError, match="lean.npz"):
        FootprintField.load(path)


def test_a_damaged_manifest_is_refused_as_a_value_error_naming_it(tmp_path):
    from autofly_ue5.scenes.footprint_field import FootprintFields

    (tmp_path / "manifest.json").write_text("{ cut short")
    with pytest.raises(ValueError, match="manifest.json"):
        FootprintFields.load({"lean"}, tmp_path)


def test_a_field_round_trips_through_its_file_and_keeps_its_content_hash(tmp_path):
    from autofly_ue5.scenes.footprint_field import FootprintField

    field = synthetic_field(tmp_path, "lean", leaning_tree_parts())
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    field.save(first)
    field.save(second)
    assert first.read_bytes() == second.read_bytes(), "the file is byte-reproducible: its sha256 means its content"
    back = FootprintField.load(first)
    assert back.content_sha256() == field.content_sha256()
    assert np.array_equal(back.values, field.values) and back.origin == field.origin
    assert back.values.dtype == np.float32 and np.load(first)["values"].dtype == np.float16


def test_float16_storage_rounds_down_so_a_stored_clearance_is_never_more_than_the_computed_one(tmp_path):
    from autofly_ue5.scenes.footprint_field import round_down_float16

    values = np.random.default_rng(2).uniform(0.0, 7.0, size=10_000).astype(np.float32)
    stored = round_down_float16(values)
    assert stored.dtype == np.float16
    assert np.all(stored.astype(np.float32) <= values)
    assert np.all(values - stored.astype(np.float32) <= 0.004)


# ----------------------------------------------------------------------------------------------------------------------
# The transform: world NED into the mesh's glTF horizontal plane, exactly as validate/obstacles.Mesh._to_gltf does.
# ----------------------------------------------------------------------------------------------------------------------
def _mesh_for(inst: Instance):
    from autofly_ue5.validate.obstacles import Mesh, TriangleSet

    v, f = box_triangles(*TRUNK)
    return Mesh(tag=inst.tag, x=inst.x, y=inst.y, radius_m=inst.radius_m, actor_z=inst.z_center, yaw=inst.yaw,
                scale=inst.scale, bounds_centre_local_m=(0.0, 0.0, 1.0), bounds_half_local_m=(1.0, 1.0, 1.0),
                triangles=TriangleSet(v, f))


def test_the_transform_agrees_with_mesh_to_gltf_for_any_yaw_and_scale():
    from autofly_ue5.scenes.footprint_field import to_local

    rng = np.random.default_rng(3)
    for _ in range(50):
        s_xy = float(rng.uniform(0.5, 3.0))
        inst = Instance("t", "lean", float(rng.uniform(-30, 30)), float(rng.uniform(-30, 30)), 0.0,
                        float(rng.uniform(-math.pi, math.pi)), (s_xy, s_xy, float(rng.uniform(0.5, 3.0))), "mesh", 2.0, 3.0)
        mesh = _mesh_for(inst)
        world = rng.uniform(-40, 40, size=(20, 2))
        expected = np.array([mesh._to_gltf((x, y, -1.7), point=True) for x, y in world])[:, [0, 2]]
        assert np.abs(to_local(inst, world) - expected).max() < 1e-9


def test_outline_distance_scales_and_rotates_with_the_instance(tmp_path):
    # The same world geometry seen through an instance at a yaw and a scale: the distance is the field's times the
    # horizontal scale, measured where the transform puts the point.
    from autofly_ue5.scenes.footprint_field import outline_distance

    field = synthetic_field(tmp_path, "l_prism", [(*l_prism(), True)])
    inst = Instance("t", "l_prism", 10.0, -5.0, 0.0, 0.7, (1.6, 1.6, 1.0), "mesh", 5.0, 3.0)
    rng = np.random.default_rng(4)
    for _ in range(200):
        local = rng.uniform([-1.5, -1.5], [3.5, 3.0])
        c, s = math.cos(inst.yaw), math.sin(inst.yaw)
        lx, ly = local * inst.scale[0]
        world = (inst.x + c * lx - s * ly, inst.y + s * lx + c * ly)
        exact = float(union_distance(local[None, :], L_RECTS)[0]) * inst.scale[0]
        got = outline_distance(field, inst, world, world)
        if exact > 0.05:
            assert -0.03 * 1.6 <= got - exact <= 2e-3, (local, got, exact)


def test_the_swept_segment_is_sampled_between_its_ends(tmp_path):
    # A step that passes the L's corner closer than either end must report the closer pass.
    from autofly_ue5.scenes.footprint_field import outline_distance

    field = synthetic_field(tmp_path, "l_prism", [(*l_prism(), True)])
    inst = Instance("t", "l_prism", 0.0, 0.0, 0.0, 0.0, (1.0, 1.0, 1.0), "mesh", 2.5, 3.0)
    a, b = (2.4, -0.3), (2.4, 0.9)  # 0.4 m clear of the bar's end at the segment's middle, ~0.5 m at both ends
    both_ends = min(outline_distance(field, inst, a, a), outline_distance(field, inst, b, b))
    swept = outline_distance(field, inst, a, b)
    assert both_ends > 0.45 and swept == pytest.approx(0.4, abs=0.03) and swept <= 0.4 + 2e-3


def test_an_instance_with_unequal_horizontal_scales_is_refused(tmp_path):
    from autofly_ue5.scenes.footprint_field import outline_distance

    field = synthetic_field(tmp_path, "l_prism", [(*l_prism(), True)])
    inst = Instance("t", "l_prism", 0.0, 0.0, 0.0, 0.0, (1.0, 1.2, 1.0), "mesh", 2.5, 3.0)
    with pytest.raises(ValueError, match="scale"):
        outline_distance(field, inst, (5.0, 5.0), (5.0, 5.0))


# ----------------------------------------------------------------------------------------------------------------------
# The brute force the real models are checked with, itself checked against the analytic distance.
# ----------------------------------------------------------------------------------------------------------------------
def test_the_brute_force_distance_is_exact_on_the_l(tmp_path):
    from autofly_ue5.scenes.footprint_field import exact_distance

    v, f = l_prism()
    tri = v[f][:, :, [0, 2]]
    points = np.random.default_rng(5).uniform([-2.0, -2.0], [4.0, 3.5], size=(500, 2))
    assert np.abs(exact_distance(tri, points) - union_distance(points, L_RECTS)).max() < 1e-9


def test_the_brute_force_distance_agrees_with_a_scan_of_every_triangle_on_many_small_ones():
    # The bucketed search must find what scanning every triangle finds: thousands of leaf-sized triangles, some long
    # slivers spanning many buckets, points inside, beside and far from them.
    from autofly_ue5.scenes.footprint_field import exact_distance, triangle_distances

    rng = np.random.default_rng(7)
    centres = rng.uniform(-2.0, 2.0, size=(3000, 1, 2))
    tri = centres + rng.normal(0.0, 0.04, size=(3000, 3, 2))
    tri[:50, 1] += rng.uniform(-1.5, 1.5, size=(50, 2))  # slivers
    points = np.concatenate([rng.uniform(-4.0, 4.0, size=(150, 2)), tri[::100, 0] + 0.01])
    scanned = np.array([triangle_distances(tri, p).min() for p in points])
    assert np.abs(exact_distance(tri, points) - scanned).max() < 1e-12
    assert (scanned == 0).any() and (scanned > 0.5).any()


# ----------------------------------------------------------------------------------------------------------------------
# The real s02 models, when the fetched glTFs are on this checkout.
# ----------------------------------------------------------------------------------------------------------------------
def _s02_species():
    from autofly_ue5.scenes.resolve import resolve_scene
    from tests.test_footprint_endings_figure import _gltf_files_present

    try:
        layout = resolve_scene("s02").layout
    except (FileNotFoundError, ValueError) as err:
        pytest.skip(f"s02's stored layout is not on this checkout ({err})")
    first = {}
    for inst in layout.instances:
        first.setdefault(inst.asset, inst)
    if not _gltf_files_present(set(first)):
        pytest.skip("the fetched glTF models (downloads/) are not on this checkout")
    return first


def test_on_the_real_s02_models_the_field_matches_a_brute_force_distance_to_every_triangle():
    # 200 random world points around one instance of each species: the field (through the instance's transform) within
    # 0.035 m of the exact distance to the projected triangles (corners, edges and interiors), and never above it (the
    # measured maximum is -0.6 mm: every point read nearer than it is). Points on or inside the outline (both ~0) are
    # not where the rule decides anything, and points beyond the field's reach (3 m at unit scale) read +inf by design
    # (test_the_field_reads_infinity_beyond_its_grid).
    from autofly_ue5.scenes.footprint_field import build_field, exact_distance, outline_distance, outline_triangles, to_world
    from autofly_ue5.validate.obstacles import MeshLibrary

    species = _s02_species()
    library = MeshLibrary.from_files(set(species))
    rng = np.random.default_rng(6)
    for asset, inst in sorted(species.items()):
        projected = outline_triangles(library[asset])
        field = build_field(projected)
        world_tri = to_world(inst, projected.reshape(-1, 2)).reshape(-1, 3, 2)
        box = inst.radius_m + 1.5
        candidates = np.array([inst.x, inst.y]) + rng.uniform(-box, box, size=(400, 2))
        exact = exact_distance(world_tri, candidates)
        keep = np.flatnonzero((exact > 0.05) & (exact <= 2.5))[:200]
        assert len(keep) == 200, (asset, len(keep))
        errors = np.array([outline_distance(field, inst, p, p) for p in candidates[keep]]) - exact[keep]
        assert errors.min() >= -0.035, (asset, errors.min())
        assert errors.max() <= 0.0, (asset, errors.max())
