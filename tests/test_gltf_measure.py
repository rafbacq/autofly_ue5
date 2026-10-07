"""autofly_ue5/scenes/gltf.py: read a glTF model's vertices straight from its buffers and measure what the scene generator
and the registry need: height, extents, and the footprint of whatever lies in the drone's 1-3 m flight band."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest


def _write_gltf(path, nodes_meshes: list[tuple[str, tuple, dict | None]]):
    """A minimal glTF 2.0 file: one node per (name, (vertices (n, 3) in glTF's Y-up metres, triangles (t, 3)), node
    extras) with its own mesh of one indexed triangle-list primitive, everything in one .bin."""
    blob = b""
    buffer_views, accessors, meshes, nodes = [], [], [], []
    for i, (name, (vertices, triangles), node) in enumerate(nodes_meshes):
        vertices = np.asarray(vertices, dtype="<f4")
        indices = np.asarray(triangles, dtype="<u4").ravel()
        for data in (vertices.tobytes(), indices.tobytes()):
            buffer_views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
            blob += data
        accessors.append({"bufferView": 2 * i, "componentType": 5126, "count": len(vertices), "type": "VEC3",
                          "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()})
        accessors.append({"bufferView": 2 * i + 1, "componentType": 5125, "count": len(indices), "type": "SCALAR"})
        meshes.append({"name": f"mesh_{name}", "primitives": [{"attributes": {"POSITION": 2 * i}, "indices": 2 * i + 1, "mode": 4}]})
        nodes.append({"name": name, "mesh": i, **(node or {})})
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
            "meshes": meshes, "accessors": accessors, "bufferViews": buffer_views,
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
