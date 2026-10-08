"""scenes/gltf.py merge_to_single_mesh: a target is spawned as ONE static mesh (Project AirSim's spawn places one asset),
so every target model is written as a single-node, single-mesh glTF with its node transforms baked into the vertices
(plan 5 C6). Checked here on small synthetic files: vertex and triangle counts kept, bounds equal to the original's
transformed geometry (shifted so the pivot is the base centre), normals and tangents turned with the geometry, mirrored
nodes re-wound, skins posed, a node subset selected, and the materials made importable (alpha BLEND -> MASK, unlit -> lit)."""

from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

CUBE_V = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (0.0, 1.0) for z in (-0.25, 0.25)], dtype=np.float32)
CUBE_T = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                   [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.uint16)
TRI_V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
TRI_T = np.array([[0, 1, 2]], dtype=np.uint16)


class _Buf:
    def __init__(self):
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, array: np.ndarray, type_: str, component: int, *, minmax: bool = False, normalized: bool = False) -> int:
        while len(self.blob) % 4:
            self.blob += b"\0"
        data = np.ascontiguousarray(array).tobytes()
        self.views.append({"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)})
        self.blob += data
        acc = {"bufferView": len(self.views) - 1, "componentType": component, "count": int(len(array)), "type": type_}
        if minmax:
            acc["min"], acc["max"] = array.min(axis=0).tolist(), array.max(axis=0).tolist()
        if normalized:
            acc["normalized"] = True
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def raw_view(self, data: bytes) -> int:
        while len(self.blob) % 4:
            self.blob += b"\0"
        self.views.append({"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)})
        self.blob += data
        return len(self.views) - 1


def _normals(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-vertex normals from the faces (enough for the tests: a direction to turn with the node)."""
    n = np.zeros_like(v, dtype=np.float64)
    for a, b, c in t:
        n[[a, b, c]] += np.cross(v[b] - v[a], v[c] - v[a])
    return (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)


def _mesh(buf: _Buf, v, t, material=None, *, uv=True, tangent=False, extra=None) -> dict:
    attrs = {"POSITION": buf.add(v, "VEC3", 5126, minmax=True), "NORMAL": buf.add(_normals(v, t), "VEC3", 5126)}
    if uv:
        attrs["TEXCOORD_0"] = buf.add((v[:, [0, 2]] * 255 % 255).astype(np.uint8), "VEC2", 5121, normalized=True)
    if tangent:
        attrs["TANGENT"] = buf.add(np.tile(np.array([1, 0, 0, 1], np.float32), (len(v), 1)), "VEC4", 5126)
    attrs.update(extra or {})
    prim = {"attributes": attrs, "indices": buf.add(t.reshape(-1), "SCALAR", 5123)}
    if material is not None:
        prim["material"] = material
    return prim


def _quat_y(deg: float) -> list[float]:
    h = math.radians(deg) / 2.0
    return [0.0, math.sin(h), 0.0, math.cos(h)]


def write_scene(tmp_path: Path, *, glb: bool = False, unlit: bool = False) -> Path:
    """A root (translated) with two children: "chair_seat" (a box, turned 90 deg about y and scaled 2) and "chair_glass" (a
    triangle with an alpha-BLEND, transmissive material); and a separate root "other_box" (translated far away)."""
    folder = tmp_path / "src"
    (folder / "textures").mkdir(parents=True, exist_ok=True)
    (folder / "textures" / "wood.png").write_bytes(b"\x89PNG wood")
    (folder / "textures" / "glass.png").write_bytes(b"\x89PNG glass")
    (folder / "textures" / "unused.png").write_bytes(b"\x89PNG unused")
    buf = _Buf()
    seat = _mesh(buf, CUBE_V, CUBE_T, 0, tangent=True)
    glass = _mesh(buf, TRI_V, TRI_T, 1)
    other = _mesh(buf, CUBE_V, CUBE_T, 2)
    images = [{"uri": "textures/wood.png"}, {"uri": "textures/glass.png"}, {"uri": "textures/unused.png"}]
    if glb:  # the wood image travels inside the GLB, as a bufferView
        images[0] = {"bufferView": buf.raw_view(b"\x89PNG embedded wood"), "mimeType": "image/png"}
    wood = {"name": "wood", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
    if unlit:
        wood["extensions"] = {"KHR_materials_unlit": {}}
    doc = {
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": [0, 3]}],
        "nodes": [{"name": "chair", "translation": [1.0, 0.5, 0.0], "children": [1, 2]},
                  {"name": "chair_seat", "mesh": 0, "rotation": _quat_y(90.0), "scale": [2.0, 2.0, 2.0]},
                  {"name": "chair_glass", "mesh": 1, "translation": [0.0, 1.0, 0.0]},
                  {"name": "other_box", "mesh": 2, "translation": [10.0, 0.0, 0.0]}],
        "meshes": [{"name": "Cube.001", "primitives": [seat]}, {"name": "Plane", "primitives": [glass]},
                   {"name": "Cube.002", "primitives": [other]}],
        "materials": [wood,
                      {"name": "glass", "alphaMode": "BLEND", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}},
                       "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}}},
                      {"name": "other", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}}}],
        "textures": [{"source": 0, "sampler": 0}, {"source": 1, "sampler": 0}, {"source": 2}],
        "samplers": [{"minFilter": 9729}],
        "images": images,
        "extensionsUsed": ["KHR_materials_transmission"] + (["KHR_materials_unlit"] if unlit else []),
        "accessors": buf.accessors, "bufferViews": buf.views,
    }
    if unlit:
        doc["extensionsRequired"] = ["KHR_materials_unlit"]
    if glb:
        doc["buffers"] = [{"byteLength": len(buf.blob)}]
        text = json.dumps(doc).encode()
        text += b" " * (-len(text) % 4)
        binary = bytes(buf.blob) + b"\0" * (-len(buf.blob) % 4)
        path = folder / "chair.glb"
        path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(text) + 8 + len(binary))
                         + struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary)
        return path
    (folder / "chair.bin").write_bytes(bytes(buf.blob))
    doc["buffers"] = [{"uri": "chair.bin", "byteLength": len(buf.blob)}]
    path = folder / "chair.gltf"
    path.write_text(json.dumps(doc))
    return path


def _bounds(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vertices.min(axis=0), vertices.max(axis=0)


def test_a_whole_model_becomes_one_node_one_mesh_with_every_vertex_and_its_transforms_baked(tmp_path):
    from autofly_ue5.scenes.gltf import combine, load_meshes, merge_to_single_mesh

    src = write_scene(tmp_path)
    original = combine(load_meshes(src), name="orig")
    summary = merge_to_single_mesh(src, tmp_path / "out", name="tgt_chair")
    out = Path(summary["path"])
    assert out == tmp_path / "out" / "tgt_chair.gltf" and (tmp_path / "out" / "tgt_chair.bin").is_file()
    doc = json.loads(out.read_text())
    assert [n["name"] for n in doc["nodes"]] == ["tgt_chair"] and doc["scenes"] == [{"name": "tgt_chair", "nodes": [0]}]
    assert not any(k in doc["nodes"][0] for k in ("translation", "rotation", "scale", "matrix", "children"))
    assert [m["name"] for m in doc["meshes"]] == ["tgt_chair"], "Interchange names a single mesh after the glTF mesh"
    (merged,) = load_meshes(out)
    assert merged.transformed is False
    assert len(merged.vertices) == len(original.vertices) == summary["vertices"] == 8 + 3 + 8
    assert len(merged.triangles) == len(original.triangles) == summary["triangles"] == 12 + 1 + 12
    # bounds: the original's world geometry, moved so the pivot is the base centre (x-z centre at 0, lowest point at 0)
    lo, hi = _bounds(original.vertices)
    shift = np.array(summary["translation_m"])
    assert shift == pytest.approx([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2], abs=1e-6)
    mlo, mhi = _bounds(merged.vertices)
    assert mlo == pytest.approx(lo + shift, abs=1e-5) and mhi == pytest.approx(hi + shift, abs=1e-5)
    assert mlo[1] == pytest.approx(0.0, abs=1e-6) and (mlo[0] + mhi[0]) == pytest.approx(0.0, abs=1e-5)
    assert summary["extent_m"] == pytest.approx((hi - lo).tolist(), abs=1e-4)


def test_normals_and_tangents_turn_with_the_node_and_stay_unit(tmp_path):
    from autofly_ue5.scenes.gltf import _load_document, _read_accessor, merge_to_single_mesh

    src = write_scene(tmp_path)
    summary = merge_to_single_mesh(src, tmp_path / "out", name="tgt_chair", nodes=["chair_seat"])
    doc, buffers = _load_document(Path(summary["path"]))
    (prim,) = doc["meshes"][0]["primitives"]
    normals = _read_accessor(doc, buffers, prim["attributes"]["NORMAL"])
    assert np.linalg.norm(normals, axis=1) == pytest.approx(np.ones(len(normals)), abs=1e-5)
    # the seat is turned +90 deg about y: its +x tangent now points to -z
    tangents = _read_accessor(doc, buffers, prim["attributes"]["TANGENT"])
    assert tangents[:, :3] == pytest.approx(np.tile([0.0, 0.0, -1.0], (len(tangents), 1)), abs=1e-6)
    assert set(tangents[:, 3]) == {1.0}
    positions = _read_accessor(doc, buffers, prim["attributes"]["POSITION"])
    # a 1 x 1 x 0.5 box scaled 2 and turned 90 deg: 1 m along x, 2 m tall, 2 m along z
    assert (positions.max(axis=0) - positions.min(axis=0)) == pytest.approx([1.0, 2.0, 2.0], abs=1e-5)
    # every face's winding still agrees with its vertices' normals (outward)
    tris = _read_accessor(doc, buffers, prim["indices"]).reshape(-1, 3)
    for a, b, c in tris:
        face = np.cross(positions[b] - positions[a], positions[c] - positions[a])
        assert face @ normals[a] > 0


def test_a_mirrored_node_is_rewound_so_its_faces_still_face_out(tmp_path):
    from autofly_ue5.scenes.gltf import _load_document, _read_accessor, merge_to_single_mesh

    src = write_scene(tmp_path)
    doc = json.loads(src.read_text())
    doc["nodes"][1]["scale"] = [-1.0, 1.0, 1.0]
    doc["nodes"][1].pop("rotation")
    src.write_text(json.dumps(doc))
    summary = merge_to_single_mesh(src, tmp_path / "out", name="tgt_m", nodes=["chair_seat"])
    out, buffers = _load_document(Path(summary["path"]))
    (prim,) = out["meshes"][0]["primitives"]
    p = _read_accessor(out, buffers, prim["attributes"]["POSITION"])
    n = _read_accessor(out, buffers, prim["attributes"]["NORMAL"])
    t = _read_accessor(out, buffers, prim["attributes"]["TANGENT"])
    for a, b, c in _read_accessor(out, buffers, prim["indices"]).reshape(-1, 3):
        assert np.cross(p[b] - p[a], p[c] - p[a]) @ n[a] > 0
    assert set(t[:, 3]) == {-1.0}, "a mirror flips the bitangent's sign"


def test_a_node_subset_takes_only_those_nodes_and_only_the_materials_and_images_they_use(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes, merge_to_single_mesh

    src = write_scene(tmp_path)
    summary = merge_to_single_mesh(src, tmp_path / "out", name="tgt_chair", nodes=["chair_seat", "chair_glass"])
    assert summary["nodes"] == ["chair_seat", "chair_glass"] and summary["vertices"] == 11 and summary["triangles"] == 13
    doc = json.loads(Path(summary["path"]).read_text())
    assert [m["name"] for m in doc["materials"]] == ["wood", "glass"]
    assert [i["uri"] for i in doc["images"]] == ["textures/wood.png", "textures/glass.png"]
    assert (tmp_path / "out" / "textures" / "wood.png").read_bytes() == b"\x89PNG wood"
    assert not (tmp_path / "out" / "textures" / "unused.png").exists()
    assert sorted(p["material"] for p in doc["meshes"][0]["primitives"]) == [0, 1]
    (merged,) = load_meshes(Path(summary["path"]))
    assert len(merged.vertices) == 11
    with pytest.raises(ValueError, match="no_such_node"):
        merge_to_single_mesh(src, tmp_path / "out2", name="tgt_x", nodes=["chair_seat", "no_such_node"])


def test_alpha_blend_becomes_masked_and_loses_what_only_a_translucent_material_supports(tmp_path):
    """A translucent material writes no depth (s02's leaves, 2026-10-07), and a transmission master is translucent by
    construction: the merged file asks Interchange for a masked material from the start."""
    from autofly_ue5.scenes.gltf import merge_to_single_mesh

    summary = merge_to_single_mesh(write_scene(tmp_path), tmp_path / "out", name="tgt_chair")
    doc = json.loads(Path(summary["path"]).read_text())
    glass = next(m for m in doc["materials"] if m["name"] == "glass")
    assert glass["alphaMode"] == "MASK" and glass["alphaCutoff"] == 0.5 and "extensions" not in glass
    assert "KHR_materials_transmission" not in doc.get("extensionsUsed", [])
    assert summary["rewrites"] == [{"material": "glass", "change": "alphaMode BLEND -> MASK", "alpha_cutoff": 0.5,
                                    "removed_extensions": ["KHR_materials_transmission"]}]
    assert summary["non_opaque_triangle_share"] == pytest.approx(1 / 25)


def test_a_glb_with_an_embedded_image_and_an_unlit_material_is_written_as_a_lit_gltf_with_image_files(tmp_path):
    """Kenney's characters are KHR_materials_unlit: under the scenes' manual exposure (-11 EV for a physical sun) an unlit
    surface's emissive colour renders near black, so the merge makes it lit and non-metallic (glTF's default
    metallicFactor is 1, a dark mirror once lit)."""
    from autofly_ue5.scenes.gltf import merge_to_single_mesh

    src = write_scene(tmp_path, glb=True, unlit=True)
    summary = merge_to_single_mesh(src, tmp_path / "out", name="tgt_chair", nodes=["chair_seat"])
    doc = json.loads(Path(summary["path"]).read_text())
    (wood,) = doc["materials"]
    assert "extensions" not in wood and wood["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    assert "extensionsRequired" not in doc and "KHR_materials_unlit" not in doc.get("extensionsUsed", [])
    (image,) = doc["images"]
    assert "bufferView" not in image and (tmp_path / "out" / image["uri"]).read_bytes() == b"\x89PNG embedded wood"
    assert {"material": "wood", "change": "KHR_materials_unlit -> lit", "metallic_factor": 0.0} in summary["rewrites"]


def test_a_skinned_mesh_is_posed_by_its_joints_and_its_own_node_transform_is_ignored(tmp_path):
    """glTF 2.0 §3.7.3: a skinned mesh's vertices are placed by its joints' world transforms times the inverse bind
    matrices; the transform of the node that carries it does not apply. Kenney's mini characters are skinned."""
    from autofly_ue5.scenes.gltf import load_meshes, merge_to_single_mesh

    folder = tmp_path / "skin"
    folder.mkdir()
    buf = _Buf()
    joints = np.array([[0, 0, 0, 0]] * 4 + [[1, 0, 0, 0]] * 4, dtype=np.uint8)
    weights = np.array([[1, 0, 0, 0]] * 8, dtype=np.float32)
    prim = _mesh(buf, CUBE_V, CUBE_T, None, uv=False, extra={
        "JOINTS_0": buf.add(joints, "VEC4", 5121), "WEIGHTS_0": buf.add(weights, "VEC4", 5126)})
    # joint 0 at the origin, joint 1 bound at y = 0 but posed 1 m higher: the vertices it carries rise by 1 m
    ibm = np.stack([np.eye(4), np.eye(4)]).astype(np.float32).transpose(0, 2, 1).reshape(2, 16)
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 3]}],
           "nodes": [{"name": "root", "children": [1, 2]}, {"name": "hips"}, {"name": "head", "translation": [0.0, 1.0, 0.0]},
                     {"name": "body-mesh", "mesh": 0, "skin": 0, "translation": [100.0, 0.0, 0.0]}],
           "skins": [{"joints": [1, 2], "inverseBindMatrices": buf.add(ibm, "MAT4", 5126)}],
           "meshes": [{"name": "body", "primitives": [prim]}], "accessors": buf.accessors, "bufferViews": buf.views,
           "buffers": [{"uri": "s.bin", "byteLength": len(buf.blob)}],
           "animations": [{"channels": [], "samplers": []}]}
    (folder / "s.bin").write_bytes(bytes(buf.blob))
    (folder / "s.gltf").write_text(json.dumps(doc))
    summary = merge_to_single_mesh(folder / "s.gltf", tmp_path / "out", name="tgt_person", recentre=False)
    (merged,) = load_meshes(Path(summary["path"]))
    expected = CUBE_V.astype(np.float64).copy()
    expected[4:, 1] += 1.0
    assert merged.vertices == pytest.approx(expected, abs=1e-6)
    out = json.loads(Path(summary["path"]).read_text())
    attrs = out["meshes"][0]["primitives"][0]["attributes"]
    assert "JOINTS_0" not in attrs and "WEIGHTS_0" not in attrs and "skins" not in out and "animations" not in out
    assert summary["skinned_nodes"] == ["body-mesh"] and summary["translation_m"] == [0.0, 0.0, 0.0]


def test_a_primitive_that_is_not_triangles_is_refused(tmp_path):
    from autofly_ue5.scenes.gltf import merge_to_single_mesh

    src = write_scene(tmp_path)
    doc = json.loads(src.read_text())
    doc["meshes"][1]["primitives"][0]["mode"] = 1  # LINES
    src.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="mode 1"):
        merge_to_single_mesh(src, tmp_path / "out", name="tgt_chair")


def test_an_image_uri_that_leaves_the_model_s_folder_is_refused(tmp_path):
    """The downloads are untrusted data: a URI must not make the merge read or write outside its folders."""
    from autofly_ue5.scenes.gltf import merge_to_single_mesh

    src = write_scene(tmp_path)
    doc = json.loads(src.read_text())
    doc["images"][0]["uri"] = "../../escape.png"
    src.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="escape.png"):
        merge_to_single_mesh(src, tmp_path / "out", name="tgt_chair", nodes=["chair_seat"])


def test_a_data_uri_buffer_is_read(tmp_path):
    from autofly_ue5.scenes.gltf import merge_to_single_mesh

    src = write_scene(tmp_path)
    doc = json.loads(src.read_text())
    blob = (src.parent / "chair.bin").read_bytes()
    doc["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
    src.write_text(json.dumps(doc))
    assert merge_to_single_mesh(src, tmp_path / "out", name="tgt_chair")["vertices"] == 19


KENNEY_SEDAN = Path(__file__).resolve().parents[1] / "downloads" / "kenney" / "kenney_car-kit" / "Models" / "GLB format" / "sedan.glb"


@pytest.mark.skipif(not KENNEY_SEDAN.is_file(), reason="downloads/ is git-ignored; present on the GPU host")
def test_a_real_kenney_car_merges_to_its_measured_size(tmp_path):
    from autofly_ue5.scenes.gltf import combine, load_meshes, merge_to_single_mesh

    original = combine(load_meshes(KENNEY_SEDAN), name="sedan")
    summary = merge_to_single_mesh(KENNEY_SEDAN, tmp_path / "out", name="tgt_sedan")
    (merged,) = load_meshes(Path(summary["path"]))
    assert len(merged.vertices) == len(original.vertices) and len(merged.triangles) == len(original.triangles)
    extent = original.vertices.max(axis=0) - original.vertices.min(axis=0)
    assert merged.vertices.max(axis=0) - merged.vertices.min(axis=0) == pytest.approx(extent, abs=1e-5)
    doc = json.loads(Path(summary["path"]).read_text())
    assert (tmp_path / "out" / doc["images"][0]["uri"]).is_file()


def _skinned_file(tmp_path: Path, *, head_translation, ibm_head: np.ndarray, weights: np.ndarray) -> Path:
    folder = tmp_path / "skin2"
    folder.mkdir(exist_ok=True)
    buf = _Buf()
    joints = np.array([[0, 0, 0, 0]] * 4 + [[1, 0, 0, 0]] * 4, dtype=np.uint8)
    prim = _mesh(buf, CUBE_V, CUBE_T, None, uv=False, extra={
        "JOINTS_0": buf.add(joints, "VEC4", 5121), "WEIGHTS_0": buf.add(weights.astype(np.float32), "VEC4", 5126)})
    ibm = np.stack([np.eye(4), ibm_head]).astype(np.float32).transpose(0, 2, 1).reshape(2, 16)  # glTF: column-major
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 3]}],
           "nodes": [{"name": "root", "children": [1, 2]}, {"name": "hips"}, {"name": "head", "translation": head_translation},
                     {"name": "body-mesh", "mesh": 0, "skin": 0}],
           "skins": [{"joints": [1, 2], "inverseBindMatrices": buf.add(ibm, "MAT4", 5126)}],
           "meshes": [{"name": "body", "primitives": [prim]}], "accessors": buf.accessors, "bufferViews": buf.views,
           "buffers": [{"uri": "s.bin", "byteLength": len(buf.blob)}]}
    (folder / "s.bin").write_bytes(bytes(buf.blob))
    (folder / "s.gltf").write_text(json.dumps(doc))
    return folder / "s.gltf"


def test_inverse_bind_matrices_are_read_column_major_so_a_bind_pose_rest_pose_leaves_the_mesh_as_modelled(tmp_path):
    """Kenney's joints sit where they were bound: joint world x inverse bind = identity, so the merged mesh is the mesh as
    modelled. Read row-major, the bind translation would land in the bottom row and wreck every vertex."""
    from autofly_ue5.scenes.gltf import load_meshes, merge_to_single_mesh

    unbind = np.eye(4)
    unbind[:3, 3] = [0.0, -1.0, 0.0]
    src = _skinned_file(tmp_path, head_translation=[0.0, 1.0, 0.0], ibm_head=unbind,
                        weights=np.array([[1, 0, 0, 0]] * 8))
    summary = merge_to_single_mesh(src, tmp_path / "out", name="tgt_p", recentre=False)
    (merged,) = load_meshes(Path(summary["path"]))
    assert merged.vertices == pytest.approx(CUBE_V.astype(np.float64), abs=1e-6)


def test_a_vertex_without_weights_stays_where_it_was_modelled(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes, merge_to_single_mesh

    weights = np.array([[1, 0, 0, 0]] * 8, dtype=float)
    weights[7] = 0.0
    src = _skinned_file(tmp_path, head_translation=[0.0, 1.0, 0.0], ibm_head=np.eye(4), weights=weights)
    summary = merge_to_single_mesh(src, tmp_path / "out", name="tgt_p", recentre=False)
    (merged,) = load_meshes(Path(summary["path"]))
    assert merged.vertices[7] == pytest.approx(CUBE_V[7].astype(np.float64), abs=1e-6)
    assert merged.vertices[6] == pytest.approx(CUBE_V[6] + np.array([0.0, 1.0, 0.0]), abs=1e-6)
