"""autofly_ue5/scenes/gltf.py: read a glTF model's vertices straight from its buffers and measure what the scene generator
and the registry need: height, extents, and the footprint of whatever lies in the drone's 1-3 m flight band."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest


def _write_gltf(path, nodes_meshes: list[tuple[str, tuple, dict | None]], alpha_modes: dict[str, str] | None = None):
    """A minimal glTF 2.0 file: one node per (name, (vertices (n, 3) in glTF's Y-up metres, triangles (t, 3)), node
    extras) with its own mesh of one indexed triangle-list primitive, everything in one .bin. `alpha_modes` gives a
    node's primitive a material of that alphaMode (BLEND, MASK); the others get none."""
    blob = b""
    buffer_views, accessors, meshes, nodes, materials = [], [], [], [], []
    for i, (name, (vertices, triangles), node) in enumerate(nodes_meshes):
        vertices = np.asarray(vertices, dtype="<f4")
        indices = np.asarray(triangles, dtype="<u4").ravel()
        for data in (vertices.tobytes(), indices.tobytes()):
            buffer_views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
            blob += data
        accessors.append({"bufferView": 2 * i, "componentType": 5126, "count": len(vertices), "type": "VEC3",
                          "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()})
        accessors.append({"bufferView": 2 * i + 1, "componentType": 5125, "count": len(indices), "type": "SCALAR"})
        primitive = {"attributes": {"POSITION": 2 * i}, "indices": 2 * i + 1, "mode": 4}
        if alpha_modes and name in alpha_modes:
            materials.append({"name": f"{name}_material", "alphaMode": alpha_modes[name], "doubleSided": True})
            primitive["material"] = len(materials) - 1
        meshes.append({"name": f"mesh_{name}", "primitives": [primitive]})
        nodes.append({"name": name, "mesh": i, **(node or {})})
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
            "meshes": meshes, "accessors": accessors, "bufferViews": buffer_views, "materials": materials,
            "buffers": [{"uri": "model.bin", "byteLength": len(blob)}]}
    path.write_text(json.dumps(gltf))
    (path.parent / "model.bin").write_bytes(blob)


def _column(radius: float, height: float, n: int = 64, base: float = 0.0, shift_x: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A vertical cylinder's side: two rings of `n` vertices at y = base and y = base + height (glTF: y up), joined by
    2n triangles, like a low-poly trunk whose only vertices are at its ends."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(angles) + shift_x, np.zeros(n), radius * np.sin(angles)], axis=1)
    vertices = np.concatenate([ring + [0, base, 0], ring + [0, base + height, 0]])
    i = np.arange(n)
    j = (i + 1) % n
    triangles = np.concatenate([np.stack([i, n + i, j], axis=1), np.stack([j, n + i, n + j], axis=1)])
    return vertices, triangles


def _join(*parts: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    vertices, triangles, offset = [], [], 0
    for v, t in parts:
        vertices.append(v)
        triangles.append(np.asarray(t) + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(triangles)


def test_vertices_are_read_per_node_with_the_node_s_transform_applied(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes

    _write_gltf(tmp_path / "model.gltf", [("trunk", _column(0.2, 4.0), None),
                                          ("moved", _column(0.5, 1.0), {"translation": [10.0, 0.0, 0.0], "scale": [2.0, 2.0, 2.0]})])
    meshes = load_meshes(tmp_path / "model.gltf")
    assert [m.name for m in meshes] == ["trunk", "moved"]
    assert meshes[0].vertices.shape == (128, 3) and meshes[0].triangles.shape == (128, 3) and meshes[0].vertices[:, 1].max() == pytest.approx(4.0)
    assert meshes[1].vertices[:, 0].max() == pytest.approx(10.0 + 1.0), "scale 2 then translation 10 on a 0.5 m radius"
    assert meshes[1].local_vertices[:, 0].max() == pytest.approx(0.5), "and the mesh's own frame is kept too"
    assert meshes[1].transformed is True and meshes[0].transformed is False


def test_a_tree_s_flight_band_footprint_is_the_canopy_not_the_trunk(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes, measure

    trunk = _column(0.15, 5.0)  # 5 m tall, 0.15 m radius: its only vertices are at the base and the top
    canopy = _column(1.4, 2.0, base=1.8)  # a 1.4 m canopy from 1.8 to 3.8 m: inside the band from 1.8 to 3 m
    _write_gltf(tmp_path / "tree.gltf", [("tree", _join(trunk, canopy), None)])
    (mesh,) = load_meshes(tmp_path / "tree.gltf")
    m = measure(mesh, band_m=(1.0, 3.0))
    assert m["height_m"] == pytest.approx(5.0) and m["base_m"] == pytest.approx(0.0)
    assert m["extent_m"] == pytest.approx([2.8, 5.0, 2.8], abs=1e-6), "(x, y up, z) extents in glTF's frame"
    assert m["band_radius_m"] == pytest.approx(1.4), "the widest reach from the pivot axis between 1 and 3 m"
    assert m["band_centre_offset_m"] == pytest.approx(0.0, abs=1e-6)
    assert 0 < m["band_vertex_share"] < 1
    low = measure(mesh, band_m=(0.5, 1.5))
    assert low["band_radius_m"] == pytest.approx(0.15), "below the canopy only the trunk is in the band: its edges cross the planes"


def test_a_leaning_tree_s_footprint_is_centred_where_its_mass_is(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes, measure

    canopy = _column(1.0, 1.0, base=1.5, shift_x=1.2)  # a 1 m canopy whose centre leans 1.2 m to +x, on a 0.1 m trunk
    _write_gltf(tmp_path / "lean.gltf", [("lean", _join(_column(0.1, 3.0), canopy), None)])
    (mesh,) = load_meshes(tmp_path / "lean.gltf")
    m = measure(mesh, band_m=(1.0, 3.0))
    assert m["band_radius_m"] == pytest.approx(2.2), "measured from the pivot axis: the far edge of the canopy"
    # the band's mass spans the trunk (x = -0.1) to the canopy's far edge (x = 2.2): centred at 1.05, reaching 1.15
    assert m["band_centre_offset_m"] == pytest.approx(1.05, abs=0.02)
    assert m["band_radius_about_centre_m"] == pytest.approx(1.15, abs=0.02)


def test_a_model_with_nothing_in_the_band_says_so(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes, measure

    _write_gltf(tmp_path / "low.gltf", [("rock", _column(0.6, 0.5), None)])
    (mesh,) = load_meshes(tmp_path / "low.gltf")
    m = measure(mesh, band_m=(1.0, 3.0))
    assert m["band_radius_m"] is None and m["band_vertex_share"] == 0.0 and m["height_m"] == pytest.approx(0.5)


def test_unsupported_accessors_are_refused_not_misread(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes

    _write_gltf(tmp_path / "m.gltf", [("a", _column(0.2, 1.0), None)])
    g = json.loads((tmp_path / "m.gltf").read_text())
    g["accessors"][0]["componentType"] = 5123  # unsigned short positions: not a glTF POSITION type we handle
    (tmp_path / "m.gltf").write_text(json.dumps(g))
    with pytest.raises(ValueError, match="componentType"):
        load_meshes(tmp_path / "m.gltf")


def test_the_measurement_script_records_every_model_of_the_sources(tmp_path):
    from scripts.measure_assets import measure_sources

    (tmp_path / "downloads" / "polyhaven" / "demo").mkdir(parents=True)
    _write_gltf(tmp_path / "downloads" / "polyhaven" / "demo" / "demo_2k.gltf", [("demo_a", _column(0.3, 2.0), None),
                                                                                  ("demo_b", _column(0.4, 0.4), None)])
    sources = {"format": "autofly_ue5_asset_sources/1", "downloads_dir": str(tmp_path / "downloads"),
               "assets": [{"id": "demo", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s02"],
                           "files": [{"path": "demo_2k.gltf", "sha256": "x"}]},
                          {"id": "Grass004", "source": "ambientcg", "kind": "material", "resolution": "2K-JPG", "use": ["s02"], "files": []}]}
    (tmp_path / "sources.json").write_text(json.dumps(sources))
    report = measure_sources(tmp_path / "sources.json", band_m=(1.0, 3.0))
    assert [m["asset"] for m in report["meshes"]] == ["demo", "demo"] and [m["node"] for m in report["meshes"]] == ["demo_a", "demo_b"]
    assert report["meshes"][0]["band_radius_m"] == pytest.approx(0.3) and report["meshes"][1]["band_radius_m"] is None
    assert report["band_m"] == [1.0, 3.0]
    # every node's parts: all of these are opaque, so the solid part is the whole and there are no cards
    assert report["meshes"][0]["parts"]["solid"]["extent_m"] == report["meshes"][0]["extent_m"] and report["meshes"][0]["parts"]["cards"] is None


def test_part_extents_follow_the_material_opacity(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes, part_extents

    _write_gltf(tmp_path / "m.gltf", [("trunk", _column(0.2, 3.0), None), ("leaves", _column(1.0, 1.0, base=2.0), None)],
                alpha_modes={"leaves": "BLEND"})
    by_name = {m.name: m for m in load_meshes(tmp_path / "m.gltf")}
    solid = part_extents(by_name["trunk"], opaque=True)
    assert solid["extent_m"] == pytest.approx([0.4, 3.0, 0.4], abs=1e-3) and solid["base_m"] == 0.0 and solid["triangles"] == 128
    assert part_extents(by_name["trunk"], opaque=False) is None
    cards = part_extents(by_name["leaves"], opaque=False)
    assert cards["extent_m"] == pytest.approx([2.0, 1.0, 2.0], abs=1e-3) and cards["base_m"] == pytest.approx(2.0) and cards["top_m"] == pytest.approx(3.0)
    assert part_extents(by_name["leaves"], opaque=True) is None


def test_the_radius_profile_gives_the_footprint_at_any_scale(tmp_path):
    from autofly_ue5.scenes.gltf import PROFILE_STEP_M, band_radius_from_profile, load_meshes, measure

    trunk = _column(0.15, 5.0)
    canopy = _column(1.4, 2.0, base=1.8)
    _write_gltf(tmp_path / "tree.gltf", [("tree", _join(trunk, canopy), None)])
    (mesh,) = load_meshes(tmp_path / "tree.gltf")
    m = measure(mesh)
    profile = m["radius_profile_m"]
    assert PROFILE_STEP_M == 0.25 and len(profile) == 20, "one entry per 0.25 m slice from the base to the 5 m top"
    assert profile[0] == pytest.approx(0.15) and profile[4] == pytest.approx(0.15), "trunk only below 1.8 m"
    assert profile[8] == pytest.approx(1.4) and profile[15] == pytest.approx(1.4), "canopy from 1.8 to 3.8 m"
    assert profile[19] == pytest.approx(0.15), "trunk only above the canopy"
    # the flight band 1-3 m on the model at unit scale: the canopy; scaled to 0.4, the whole tree is 2 m tall and the
    # band 1-3 m maps to 2.5-7.5 m of the model, which holds only the canopy top and the trunk
    assert band_radius_from_profile(profile, PROFILE_STEP_M, s_xy=1.0, s_z=1.0, band_m=(1.0, 3.0)) == pytest.approx(1.4)
    assert band_radius_from_profile(profile, PROFILE_STEP_M, s_xy=2.0, s_z=1.0, band_m=(1.0, 3.0)) == pytest.approx(2.8)
    assert band_radius_from_profile(profile, PROFILE_STEP_M, s_xy=1.0, s_z=0.4, band_m=(1.0, 3.0)) == pytest.approx(1.4)
    assert band_radius_from_profile(profile, PROFILE_STEP_M, s_xy=1.0, s_z=0.3, band_m=(1.0, 3.0)) == pytest.approx(1.4), \
        "at 0.3 the band is the model's 3.33-10 m: the canopy top (to 3.8 m) is still in it"
    assert band_radius_from_profile(profile, PROFILE_STEP_M, s_xy=1.0, s_z=0.25, band_m=(1.0, 3.0)) == pytest.approx(0.15), \
        "at 0.25 the band is the model's 4-12 m: above the canopy, only the trunk's top metre"
    assert band_radius_from_profile(profile, PROFILE_STEP_M, s_xy=1.0, s_z=0.1, band_m=(1.0, 3.0)) is None, "0.5 m tall: below the band"
    rock = measure(load_meshes(tmp_path / "tree.gltf")[0])  # reuse: a 1 m 'rock' is a scaled-down tree for this purpose
    assert band_radius_from_profile(rock["radius_profile_m"], PROFILE_STEP_M, s_xy=3.0, s_z=0.6, band_m=(1.0, 3.0)) == pytest.approx(4.2), \
        "scaled 3x wide and to 3 m tall: the canopy (1.8-3.8 m -> 1.08-2.28 m) is in the band at 3 x 1.4"


def test_each_triangle_knows_whether_its_material_is_opaque(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes

    trunk = _column(0.2, 3.0)
    leaves = _column(1.0, 1.0, base=2.0)
    _write_gltf(tmp_path / "model.gltf", [("trunk", trunk, None), ("leaves", leaves, None)], alpha_modes={"leaves": "BLEND"})
    by_name = {m.name: m for m in load_meshes(tmp_path / "model.gltf")}
    assert by_name["trunk"].triangle_opaque.all() and len(by_name["trunk"].triangle_opaque) == len(trunk[1])
    assert not by_name["leaves"].triangle_opaque.any() and len(by_name["leaves"].triangle_opaque) == len(leaves[1])


def test_a_gltf_splits_into_its_solid_and_card_primitives(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes, split_by_opacity

    src = tmp_path / "src"
    src.mkdir()
    trunk = _column(0.2, 3.0)
    leaves = _column(1.0, 1.0, base=2.0)
    rock = _column(0.5, 0.8)
    # one node whose mesh holds a trunk and a leaf primitive, one all-rock node (no cards at all)
    _write_gltf(src / "tree.gltf", [("tree", trunk, None), ("rock", rock, None)], alpha_modes={"tree": "OPAQUE"})
    gltf = json.loads((src / "tree.gltf").read_text())
    # give the tree mesh a second, BLEND primitive sharing the file's buffers: the leaves
    import numpy as np
    blob = (src / "model.bin").read_bytes()
    lv = np.asarray(leaves[0], dtype="<f4"); li = np.asarray(leaves[1], dtype="<u4").ravel()
    gltf["bufferViews"] += [{"buffer": 0, "byteOffset": len(blob), "byteLength": lv.nbytes},
                            {"buffer": 0, "byteOffset": len(blob) + lv.nbytes, "byteLength": li.nbytes}]
    blob += lv.tobytes() + li.tobytes()
    gltf["buffers"][0]["byteLength"] = len(blob)
    gltf["accessors"] += [{"bufferView": len(gltf["bufferViews"]) - 2, "componentType": 5126, "count": len(lv), "type": "VEC3",
                           "min": lv.min(axis=0).tolist(), "max": lv.max(axis=0).tolist()},
                          {"bufferView": len(gltf["bufferViews"]) - 1, "componentType": 5125, "count": len(li), "type": "SCALAR"}]
    gltf["materials"].append({"name": "leaves", "alphaMode": "BLEND", "doubleSided": True})
    gltf["meshes"][0]["primitives"].append({"attributes": {"POSITION": len(gltf["accessors"]) - 2}, "indices": len(gltf["accessors"]) - 1,
                                            "mode": 4, "material": len(gltf["materials"]) - 1})
    gltf["images"] = [{"uri": "textures/leaf.png"}]
    (src / "textures").mkdir()
    (src / "textures" / "leaf.png").write_bytes(b"not really a png")
    (src / "tree.gltf").write_text(json.dumps(gltf))
    (src / "model.bin").write_bytes(blob)

    out = tmp_path / "split"
    parts = split_by_opacity(src / "tree.gltf", out)
    assert set(parts) == {"solid", "cards"} and parts["solid"].parent == out and parts["cards"].parent == out
    solid = {m.name: m for m in load_meshes(parts["solid"])}
    cards = {m.name: m for m in load_meshes(parts["cards"])}
    # the solid file keeps both nodes (trunk, rock) with only opaque triangles; the cards file keeps the tree node only
    assert set(solid) == {"tree", "rock"} and set(cards) == {"tree"}
    assert len(solid["tree"].triangles) == len(trunk[1]) and solid["tree"].triangle_opaque.all()
    assert len(cards["tree"].triangles) == len(leaves[1]) and not cards["tree"].triangle_opaque.any()
    assert cards["tree"].local_vertices[:, 1].min() == pytest.approx(2.0), "the leaves start at 2 m"
    # the split files reference the original buffers and images by relative name, present next to them
    assert (out / "model.bin").exists() and (out / "textures" / "leaf.png").exists()
    assert json.loads(parts["cards"].read_text())["meshes"][0]["name"] == "mesh_tree", "mesh names survive, so Interchange names the import the same way"
    # a file without cards yields no cards part (and its solid part is the whole file)
    rock_dir = tmp_path / "rock_src"
    rock_dir.mkdir()
    _write_gltf(rock_dir / "rock.gltf", [("rock", rock, None)])
    rock_parts = split_by_opacity(rock_dir / "rock.gltf", tmp_path / "split3")
    assert rock_parts["cards"] is None and len(load_meshes(rock_parts["solid"])) == 1


def _to_glb(gltf_path, glb_path) -> None:
    """Pack a .gltf + its one .bin into the binary container (GLB): a 12-byte header, a JSON chunk and a BIN chunk, each
    padded to 4 bytes, the buffer carrying no uri. Kenney's kits ship their models this way (plan 5, C5)."""
    import struct

    gltf = json.loads(gltf_path.read_text())
    blob = (gltf_path.parent / gltf["buffers"][0]["uri"]).read_bytes()
    del gltf["buffers"][0]["uri"]
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    blob += b"\0" * (-len(blob) % 4)
    body = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(blob), b"BIN\0") + blob
    glb_path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def test_a_glb_is_read_like_the_gltf_it_packs(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes

    _write_gltf(tmp_path / "model.gltf", [("trunk", _column(0.2, 4.0), None),
                                          ("moved", _column(0.5, 1.0), {"translation": [10.0, 0.0, 0.0]})],
                alpha_modes={"moved": "BLEND"})
    _to_glb(tmp_path / "model.gltf", tmp_path / "model.glb")
    (tmp_path / "model.bin").unlink()  # the GLB carries its own buffer
    meshes = load_meshes(tmp_path / "model.glb")
    assert [m.name for m in meshes] == ["trunk", "moved"]
    assert meshes[0].vertices[:, 1].max() == pytest.approx(4.0) and meshes[1].vertices[:, 0].max() == pytest.approx(10.5)
    assert meshes[0].triangle_opaque.all() and not meshes[1].triangle_opaque.any()


def test_a_file_that_is_not_a_glb_is_refused(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes

    (tmp_path / "fake.glb").write_bytes(b"PK\x03\x04 not a glTF container at all")
    with pytest.raises(ValueError, match="GLB"):
        load_meshes(tmp_path / "fake.glb")


def test_a_whole_model_is_measured_in_the_scene_frame(tmp_path):
    """A low-poly car is a body node plus wheel nodes placed by their translations: as a target it is one object, so its
    size is that of every node together, in the file's own frame."""
    from autofly_ue5.scenes.gltf import combine, load_meshes, measure

    _write_gltf(tmp_path / "car.gltf", [("body", _column(1.0, 1.0, base=0.3), None),
                                        ("wheel", _column(0.3, 0.2), {"translation": [0.0, 0.0, 1.5]})])
    whole = combine(load_meshes(tmp_path / "car.gltf"), name="car")
    m = measure(whole)
    assert whole.name == "car" and len(whole.triangles) == 256 and whole.transformed is False
    assert m["extent_m"] == pytest.approx([2.0, 1.3, 2.8], abs=1e-3), "z from the body's -1.0 to the wheel's 1.5 + 0.3"
    assert m["height_m"] == pytest.approx(1.3) and m["base_m"] == pytest.approx(0.0)


def test_the_measurement_script_measures_targets_whole_and_leaves_the_per_node_records_alone(tmp_path):
    """A target is one object: a kit model (Kenney's GLBs: a body and its wheels, a torso and its limbs) or a Poly Haven
    model for the target pool gets one record of all its nodes together, under `models`. `meshes` stays one record per
    node of a Poly Haven model, exactly as before: `scripts/import_assets.py` imports every one of those as a static mesh,
    so a whole-file record there would be imported as a node."""
    from scripts.measure_assets import measure_sources

    downloads = tmp_path / "downloads"
    kit = downloads / "kenney" / "kenney_demo-kit" / "Models" / "GLB format"
    kit.mkdir(parents=True)
    _write_gltf(kit / "car.gltf", [("body", _column(1.0, 1.0, base=0.3), None),
                                   ("wheel", _column(0.3, 0.2), {"translation": [0.0, 0.0, 1.5]})])
    _to_glb(kit / "car.gltf", kit / "car.glb")
    (downloads / "polyhaven" / "sofa").mkdir(parents=True)
    _write_gltf(downloads / "polyhaven" / "sofa" / "sofa_1k.gltf", [("frame", _column(1.0, 0.8), None),
                                                                     ("cushion", _column(0.5, 0.2, base=0.4), None)])
    (downloads / "polyhaven" / "rock").mkdir(parents=True)
    _write_gltf(downloads / "polyhaven" / "rock" / "rock_2k.gltf", [("rock_a", _column(0.6, 1.5), None)])
    sources = {"format": "autofly_ue5_asset_sources/1", "downloads_dir": str(downloads), "assets": [
        {"id": "kenney_demo-kit", "source": "kenney", "kind": "model_pack", "use": ["targets"],
         "files": [{"path": "kenney_demo-kit.zip", "sha256": "z"}],
         "models": [{"name": "car", "path": "Models/GLB format/car.glb", "sha256": "c"}]},
        {"id": "sofa", "source": "polyhaven", "kind": "model", "use": ["targets"], "files": [{"path": "sofa_1k.gltf", "sha256": "s"}]},
        {"id": "rock", "source": "polyhaven", "kind": "model", "use": ["s03"], "files": [{"path": "rock_2k.gltf", "sha256": "r"}]}]}
    (tmp_path / "sources.json").write_text(json.dumps(sources))
    report = measure_sources(tmp_path / "sources.json", band_m=(1.0, 3.0))
    assert [(m["asset"], m["node"]) for m in report["meshes"]] == [("sofa", "frame"), ("sofa", "cushion"), ("rock", "rock_a")]
    assert [(m["asset"], m["model"], m["file"], m["nodes"]) for m in report["models"]] == [
        ("kenney_demo-kit", "car", "Models/GLB format/car.glb", ["body", "wheel"]), ("sofa", "sofa", "sofa_1k.gltf", ["frame", "cushion"])]
    car, sofa = report["models"]
    assert car["extent_m"] == pytest.approx([2.0, 1.3, 2.8], abs=1e-3) and car["height_m"] == pytest.approx(1.3)
    assert sofa["height_m"] == pytest.approx(0.8) and sofa["extent_m"] == pytest.approx([2.0, 0.8, 2.0], abs=1e-3)
