"""Read a glTF 2.0 model's triangles straight from its buffers and measure what the scene generator needs (plan 5, C3).

The registry's footprint is not the mesh's bounding box: a tree's obstacle footprint at the drone's 1-3 m altitude is
its trunk plus whatever canopy reaches down into that band, and a leaning tree's is off its pivot (spec §6.4, the M4
asset survey). The editor reports bounding boxes; the band needs the geometry, which the glTF holds in plain float32
buffers, so it is measured here, offline, before anything is imported. glTF is Y-up in metres with the model's pivot at
its origin (Poly Haven plants and rocks stand on y = 0); UE's importer turns that into Z-up centimetres, and the editor
build (C3) checks the overall bounds against these numbers.

Some Poly Haven "assets" are sets of variants laid side by side as separate nodes (searsia_lucida_a..f): each node is
measured on its own, in the mesh's own frame, because each becomes its own static mesh. A low-poly kit model (Kenney's
cars, characters and animals, the target pool's stand-ins) is the opposite: one object built of parts (a body and its
wheels, a torso and its limbs), so `combine` measures it whole, in the file's frame. Those kits ship binary glTF (GLB),
which `load_meshes` reads too.
"""

from __future__ import annotations

import base64
import json
import shutil
import struct
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLOAT32 = 5126
PROFILE_STEP_M = 0.25  # the radius-by-height profile's slice: fine enough for a 1-3 m band at any scale a scene uses
INDEX_TYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}
TRIANGLES = 4


@dataclass(frozen=True)
class MeshGeometry:
    name: str  # the node's name
    mesh_name: str
    local_vertices: np.ndarray  # (n, 3) float64 metres, the mesh's own frame
    vertices: np.ndarray  # (n, 3) with the node's world transform applied
    triangles: np.ndarray  # (t, 3) vertex indices into either array
    transformed: bool  # whether the node's world transform is not the identity
    # (t,) per triangle: whether its material renders opaque (alphaMode OPAQUE, or no material). Poly Haven's trees
    # carry their leaves as alpha-blended cards (alphaMode BLEND), which the simulator renders alpha-tested once imported
    # as Masked: a ray through such a card may see through it, a ray onto an opaque triangle may not (2026-10-07).
    triangle_opaque: np.ndarray | None = None


def _accessor_array(gltf: dict, buffers: list[bytes], index: int, *, expect_float: bool) -> np.ndarray:
    acc = gltf["accessors"][index]
    if "sparse" in acc:
        raise ValueError("sparse accessors are not supported")
    view = gltf["bufferViews"][acc["bufferView"]]
    n_comp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    if expect_float:
        if acc["componentType"] != FLOAT32:
            raise ValueError(f"POSITION componentType {acc['componentType']} is not float32 ({FLOAT32})")
        dtype = np.dtype("<f4")
    else:
        if acc["componentType"] not in INDEX_TYPES:
            raise ValueError(f"index componentType {acc['componentType']} is not one of {sorted(INDEX_TYPES)}")
        dtype = np.dtype(INDEX_TYPES[acc["componentType"]]).newbyteorder("<")
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * n_comp)
    data = buffers[view["buffer"]]
    count = acc["count"]
    if stride == dtype.itemsize * n_comp:
        out = np.frombuffer(data, dtype=dtype, count=count * n_comp, offset=start).reshape(count, n_comp)
    else:
        rows = [np.frombuffer(data, dtype=dtype, count=n_comp, offset=start + i * stride) for i in range(count)]
        out = np.stack(rows) if rows else np.zeros((0, n_comp), dtype)
    return out.astype(np.float64 if expect_float else np.int64)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T  # glTF stores column-major
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=float)
    qx, qy, qz, qw = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=float)
    rot = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot * s[None, :]
    m[:3, 3] = t
    return m


GLB_MAGIC, GLB_JSON, GLB_BIN = b"glTF", b"JSON", b"BIN\0"


def _read_glb(path: Path) -> tuple[dict, bytes | None]:
    """The JSON chunk and the BIN chunk (None if absent) of a binary glTF: a 12-byte header (magic, version 2, length),
    then chunks of (length, type, data) padded to 4 bytes (glTF 2.0 spec, section 4.4)."""
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != GLB_MAGIC:
        raise ValueError(f"{path}: not a GLB (binary glTF) file: magic {data[:4]!r}")
    _magic, version, length = struct.unpack_from("<4sII", data, 0)
    if version != 2 or length > len(data):
        raise ValueError(f"{path}: GLB version {version}, length {length} of {len(data)} bytes: not a glTF 2.0 container")
    gltf, binary, offset = None, None, 12
    while offset + 8 <= length:
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        chunk = data[offset + 8: offset + 8 + chunk_length]
        if chunk_type == GLB_JSON and gltf is None:
            gltf = json.loads(chunk)
        elif chunk_type == GLB_BIN and binary is None:
            binary = chunk
        offset += 8 + chunk_length
    if gltf is None:
        raise ValueError(f"{path}: GLB without a JSON chunk")
    return gltf, binary


def load_meshes(path: Path) -> list[MeshGeometry]:
    """Every node that carries a mesh, in scene order, with its triangles and vertices (local and world). A `.glb` is
    read from its own chunks: its first buffer, without a uri, is the BIN chunk."""
    path = Path(path)
    binary = None
    if path.suffix.lower() == ".glb":
        gltf, binary = _read_glb(path)
    else:
        gltf = json.loads(path.read_text())
    buffers = []
    for i, b in enumerate(gltf.get("buffers", [])):
        uri = b.get("uri")
        if uri is None and i == 0 and binary is not None:
            buffers.append(binary)
            continue
        if uri is None or uri.startswith("data:"):
            raise ValueError(f"{path}: only external .bin buffers (or a GLB's own BIN chunk) are supported")
        buffers.append((path.parent / uri).read_bytes())
    scene = gltf["scenes"][gltf.get("scene", 0)]
    out: list[MeshGeometry] = []

    def visit(index: int, parent: np.ndarray) -> None:
        node = gltf["nodes"][index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            mesh = gltf["meshes"][node["mesh"]]
            local_parts, tri_parts, opaque_parts, offset = [], [], [], 0
            for prim in mesh["primitives"]:
                if prim.get("mode", TRIANGLES) != TRIANGLES or "POSITION" not in prim["attributes"]:
                    continue
                positions = _accessor_array(gltf, buffers, prim["attributes"]["POSITION"], expect_float=True)
                if "indices" in prim:
                    indices = _accessor_array(gltf, buffers, prim["indices"], expect_float=False).ravel()
                else:
                    indices = np.arange(len(positions))
                tris = indices[: len(indices) - len(indices) % 3].reshape(-1, 3) + offset
                material = gltf["materials"][prim["material"]] if "material" in prim else {}
                local_parts.append(positions)
                tri_parts.append(tris)
                opaque_parts.append(np.full(len(tris), material.get("alphaMode", "OPAQUE") == "OPAQUE"))
                offset += len(positions)
            if local_parts:
                local = np.concatenate(local_parts)
                tris = np.concatenate(tri_parts) if tri_parts else np.zeros((0, 3), np.int64)
                opaque = np.concatenate(opaque_parts) if opaque_parts else np.zeros(0, bool)
                transformed = not np.allclose(world, np.eye(4))
                vertices = (local @ world[:3, :3].T) + world[:3, 3] if transformed else local.copy()
                out.append(MeshGeometry(name=node.get("name", f"node_{index}"), mesh_name=mesh.get("name", f"mesh_{node['mesh']}"),
                                        local_vertices=local, vertices=vertices, triangles=tris, transformed=transformed,
                                        triangle_opaque=opaque))
        for child in node.get("children", []):
            visit(child, world)

    for root in scene.get("nodes", []):
        visit(root, np.eye(4))
    return out


def combine(meshes: list[MeshGeometry], *, name: str) -> MeshGeometry:
    """All of a file's nodes as one mesh in the file's (scene) frame: each node's world-transformed vertices become the
    combined mesh's own. What a target from a low-poly kit is -- one object of several parts -- and what its spawn scale
    and height are taken from."""
    if not meshes:
        raise ValueError(f"{name}: no mesh to combine")
    vertices, triangles, opaque, offset = [], [], [], 0
    for m in meshes:
        vertices.append(m.vertices)
        triangles.append(m.triangles + offset)
        opaque.append(m.triangle_opaque if m.triangle_opaque is not None else np.ones(len(m.triangles), bool))
        offset += len(m.vertices)
    v = np.concatenate(vertices)
    return MeshGeometry(name=name, mesh_name=name, local_vertices=v, vertices=v.copy(), triangles=np.concatenate(triangles),
                        transformed=False, triangle_opaque=np.concatenate(opaque))


def _band_points(vertices: np.ndarray, triangles: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Points of the surface inside the band lo <= y < hi: the vertices there, plus where each triangle edge crosses
    the band's two planes, so a trunk with no vertex between its base and its crown is still seen."""
    inside = vertices[(vertices[:, 1] >= lo) & (vertices[:, 1] < hi)]
    crossings = []
    if len(triangles):
        edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
        for plane in (lo, hi):
            ya, yb = a[:, 1], b[:, 1]
            spans = ((ya - plane) * (yb - plane) < 0)
            if spans.any():
                t = ((plane - ya[spans]) / (yb[spans] - ya[spans]))[:, None]
                crossings.append(a[spans] + t * (b[spans] - a[spans]))
    return np.concatenate([inside] + crossings) if crossings else inside


def band_radius_from_profile(profile: list[float], step_m: float, *, s_xy: float, s_z: float,
                             band_m: tuple[float, float]) -> float | None:
    """The footprint radius an instance scaled by (s_xy, s_z) presents to a drone flying in `band_m`: the widest slice of
    the unit-scale profile whose scaled height overlaps the band, times s_xy. None when the scaled model has no surface
    in the band (it is below the drone, or the band falls in a gap of the model)."""
    lo, hi = band_m
    radii = [r for i, r in enumerate(profile) if r is not None and (i + 1) * step_m * s_z > lo and i * step_m * s_z < hi]
    return max(radii) * s_xy if radii else None


def radius_profile(mesh: MeshGeometry, step_m: float = PROFILE_STEP_M) -> list[float | None]:
    """Max horizontal reach from the pivot axis per `step_m` slice of height from the base (y = 0, where the actor is
    placed) to the top; None for a slice with no surface (a gap, or below a floating base)."""
    top = float(mesh.local_vertices[:, 1].max())
    profile: list[float | None] = []
    k = 0
    while k * step_m < top:
        pts = _band_points(mesh.local_vertices, mesh.triangles, k * step_m, (k + 1) * step_m)
        profile.append(round(float(np.hypot(pts[:, 0], pts[:, 2]).max()), 4) if len(pts) else None)
        k += 1
    return profile


def measure(mesh: MeshGeometry, *, band_m: tuple[float, float] = (1.0, 3.0)) -> dict:
    """Height, extents and the flight-band footprint of a mesh in its own frame (glTF: y up, the pivot at the origin)."""
    v = mesh.local_vertices
    lo, hi = band_m
    band = _band_points(v, mesh.triangles, lo, hi)
    result = {
        "vertices": int(len(v)),
        "triangles": int(len(mesh.triangles)),
        "extent_m": [round(float(e), 4) for e in (v.max(axis=0) - v.min(axis=0))],
        "base_m": round(float(v[:, 1].min()), 4),
        "top_m": round(float(v[:, 1].max()), 4),
        "height_m": round(float(v[:, 1].max() - v[:, 1].min()), 4),
        "pivot_offset_xz_m": [round(float(c), 4) for c in ((v[:, [0, 2]].max(axis=0) + v[:, [0, 2]].min(axis=0)) / 2.0)],
        "band_m": [lo, hi],
        "band_vertex_share": round(float(((v[:, 1] >= lo) & (v[:, 1] < hi)).mean()), 4) if len(v) else 0.0,
        "band_radius_m": None,
        "band_centre_offset_m": None,
        "band_radius_about_centre_m": None,
        "profile_step_m": PROFILE_STEP_M,
        "radius_profile_m": radius_profile(mesh),
        "ground_radius_m": round(float(np.hypot(v[:, 0], v[:, 2]).max()), 4),
    }
    if len(band):
        xz = band[:, [0, 2]]
        centre = (xz.max(axis=0) + xz.min(axis=0)) / 2.0
        result["band_radius_m"] = round(float(np.hypot(xz[:, 0], xz[:, 1]).max()), 4)
        result["band_centre_offset_m"] = round(float(np.hypot(*centre)), 4)
        result["band_radius_about_centre_m"] = round(float(np.hypot(*(xz - centre).T).max()), 4)
    return result


def part_extents(mesh: MeshGeometry, *, opaque: bool) -> dict | None:
    """The box of the triangles whose material is opaque (or not): what UE's bounds of the split part should read.
    None when the mesh has no such triangles."""
    flags = mesh.triangle_opaque if mesh.triangle_opaque is not None else np.ones(len(mesh.triangles), bool)
    tris = mesh.triangles[flags == opaque]
    if len(tris) == 0:
        return None
    v = mesh.local_vertices[np.unique(tris)]
    return {"extent_m": [round(float(e), 4) for e in (v.max(axis=0) - v.min(axis=0))],
            "base_m": round(float(v[:, 1].min()), 4), "top_m": round(float(v[:, 1].max()), 4),
            "centre_m": [round(float(c), 4) for c in (v.max(axis=0) + v.min(axis=0)) / 2.0], "triangles": int(len(tris))}


def split_by_opacity(path: Path, out_dir: Path) -> dict[str, Path | None]:
    """Write two copies of a glTF next to each other in `out_dir`: `<stem>_solid.gltf` holding the primitives whose
    material is opaque, `<stem>_cards.gltf` the others (alphaMode BLEND or MASK: leaves, twigs). Meshes left without a
    primitive are dropped along with the nodes that carried them; node and mesh names are kept, so Interchange names
    the imported meshes as it would have for the whole file. Buffers and images are shared: every URI the file names is
    linked (or copied) into `out_dir`. Returns the two paths; "cards" is None for a file without any.

    Why: imported as one static mesh, a tree's leaf cards were part of its complex collision, and a drone that crashed
    into a canopy wedged among them beyond any teleport (s02's live checks, 2026-10-07). The level builder spawns the
    cards part without collision; the depth camera still sees it."""
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gltf = json.loads(path.read_text())
    materials = gltf.get("materials", [])

    def is_opaque(prim: dict) -> bool:
        return "material" not in prim or materials[prim["material"]].get("alphaMode", "OPAQUE") == "OPAQUE"

    def filtered(keep_opaque: bool) -> dict | None:
        doc = json.loads(json.dumps(gltf))
        kept_meshes: dict[int, int] = {}
        meshes = []
        for i, mesh in enumerate(doc.get("meshes", [])):
            prims = [p for p in mesh.get("primitives", []) if is_opaque(p) == keep_opaque]
            if prims:
                kept_meshes[i] = len(meshes)
                meshes.append({**mesh, "primitives": prims})
        if not meshes:
            return None
        doc["meshes"] = meshes
        # nodes: drop a mesh reference that is gone, drop a node left with nothing (no mesh, no children), remap indices
        nodes = doc.get("nodes", [])
        for node in nodes:
            if "mesh" in node:
                if node["mesh"] in kept_meshes:
                    node["mesh"] = kept_meshes[node["mesh"]]
                else:
                    del node["mesh"]
        alive = [i for i, n in enumerate(nodes) if "mesh" in n or n.get("children")]
        # a node kept only for children all of which are gone is dropped too (one pass suffices for Poly Haven's flat files)
        remap = {old: new for new, old in enumerate(alive)}
        new_nodes = []
        for i in alive:
            n = dict(nodes[i])
            if "children" in n:
                n["children"] = [remap[c] for c in n["children"] if c in remap]
                if not n["children"]:
                    del n["children"]
            new_nodes.append(n)
        doc["nodes"] = new_nodes
        for scene in doc.get("scenes", []):
            scene["nodes"] = [remap[c] for c in scene.get("nodes", []) if c in remap]
        return doc

    for section in ("buffers", "images"):
        for item in gltf.get(section, []):
            uri = item.get("uri")
            if uri and not uri.startswith("data:"):
                target = out_dir / uri
                target.parent.mkdir(parents=True, exist_ok=True)  # Poly Haven keeps images under textures/
                if not target.exists():
                    source = (path.parent / uri).resolve()
                    try:
                        target.symlink_to(source)
                    except OSError:
                        target.write_bytes(source.read_bytes())
    out: dict[str, Path | None] = {}
    for name, keep_opaque in (("solid", True), ("cards", False)):
        doc = filtered(keep_opaque)
        if doc is None:
            out[name] = None
            continue
        target = out_dir / f"{path.stem}_{name}.gltf"
        target.write_text(json.dumps(doc))
        out[name] = target
    if out["solid"] is None:
        raise ValueError(f"{path}: no opaque primitive at all")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# One model as one mesh (plan 5 C6): a target is spawned at runtime as ONE static mesh -- Project AirSim's SpawnObject
# places one asset by its short name -- while a kit model is a body and its wheels, a character its limbs, a Poly Haven
# cabinet its doors: several nodes, each in its own frame. The obstacle import keeps nodes apart on purpose (a variant set
# is several meshes), so a target is first written as a single-node, single-mesh glTF with every node's transform baked
# into its vertices, and Interchange imports that as one static mesh.
# ---------------------------------------------------------------------------------------------------------------------

COMPONENT_DTYPES = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
NORMALIZED_DIVISORS = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
KEPT_ATTRIBUTES = ("POSITION", "NORMAL", "TANGENT", "TEXCOORD_", "COLOR_")  # skin weights go once the skin is posed
# What only a translucent material does: once BLEND becomes MASK these would ask Interchange for a translucent master
# (transmission, volume: ThinTranslucent) or an untested masked variant (clear coat) for a few glass triangles.
BLEND_ONLY_EXTENSIONS = ("KHR_materials_transmission", "KHR_materials_volume", "KHR_materials_clearcoat")
MERGE_GENERATOR = "autofly_ue5 scenes/gltf.py merge_to_single_mesh"
IMAGE_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/ktx2": ".ktx2"}


def _decode_data_uri(uri: str) -> bytes:
    header, _, payload = uri.partition(",")
    if not header.endswith(";base64"):
        raise ValueError(f"only base64 data URIs are supported, got {header[:40]!r}")
    return base64.b64decode(payload)


def _load_document(path: Path) -> tuple[dict, list[bytes]]:
    """A glTF's JSON and its buffers (a .glb's BIN chunk, external .bin files, or base64 data URIs)."""
    path = Path(path)
    binary = None
    if path.suffix.lower() == ".glb":
        doc, binary = _read_glb(path)
    else:
        doc = json.loads(path.read_text())
    buffers = []
    for i, b in enumerate(doc.get("buffers", [])):
        uri = b.get("uri")
        if uri is None:
            if i != 0 or binary is None:
                raise ValueError(f"{path}: buffer {i} has no uri and is not a GLB's BIN chunk")
            buffers.append(binary)
        elif uri.startswith("data:"):
            buffers.append(_decode_data_uri(uri))
        else:
            buffers.append(_inside(path.parent, uri).read_bytes())
    return doc, buffers


def _inside(folder: Path, uri: str) -> Path:
    """`uri` resolved under `folder`, refusing one that leaves it (an absolute path, `..`): the downloads are untrusted."""
    relative = Path(urllib.parse.unquote(uri))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"URI {uri!r} leaves the model's folder")
    return folder / relative


def _read_accessor(doc: dict, buffers: list[bytes], index: int) -> np.ndarray:
    """Any accessor as a (count, components) array: float64 for floats and normalized integers (glTF 2.0 §3.11), int64
    for plain integers. Strided views are read row by row."""
    acc = doc["accessors"][index]
    if "sparse" in acc:
        raise ValueError(f"accessor {index}: sparse accessors are not supported")
    n_comp = TYPE_COMPONENTS[acc["type"]]
    component = acc["componentType"]
    dtype = np.dtype(COMPONENT_DTYPES[component])
    count = acc["count"]
    if "bufferView" not in acc:
        out = np.zeros((count, n_comp), dtype)
    else:
        view = doc["bufferViews"][acc["bufferView"]]
        data = buffers[view["buffer"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        element = dtype.itemsize * n_comp
        stride = view.get("byteStride") or element
        if stride == element:
            out = np.frombuffer(data, dtype=dtype, count=count * n_comp, offset=start).reshape(count, n_comp)
        else:
            rows = np.frombuffer(data, dtype=np.uint8, count=stride * (count - 1) + element, offset=start) if count else np.zeros(0, np.uint8)
            raw = np.lib.stride_tricks.as_strided(rows, shape=(count, element), strides=(stride, 1)).copy()
            out = raw.view(dtype).reshape(count, n_comp)
    if component == FLOAT32:
        return out.astype(np.float64)
    if acc.get("normalized"):
        return np.maximum(out.astype(np.float64) / NORMALIZED_DIVISORS[component], -1.0)
    return out.astype(np.int64)


def _world_matrices(doc: dict) -> dict[int, np.ndarray]:
    scene = doc["scenes"][doc.get("scene", 0)]
    world: dict[int, np.ndarray] = {}

    def visit(index: int, parent: np.ndarray) -> None:
        world[index] = parent @ _node_matrix(doc["nodes"][index])
        for child in doc["nodes"][index].get("children", []):
            visit(child, world[index])

    for root in scene.get("nodes", []):
        visit(root, np.eye(4))
    return world


def _skin_matrices(doc: dict, buffers: list[bytes], skin: dict, world: dict[int, np.ndarray]) -> np.ndarray:
    """(joints, 4, 4): each joint's world transform times its inverse bind matrix (glTF 2.0 §3.7.3)."""
    n = len(skin["joints"])
    if "inverseBindMatrices" in skin:
        ibm = _read_accessor(doc, buffers, skin["inverseBindMatrices"]).reshape(n, 4, 4).transpose(0, 2, 1)  # column-major
    else:
        ibm = np.tile(np.eye(4), (n, 1, 1))
    return np.stack([world[j] @ ibm[k] for k, j in enumerate(skin["joints"])])


def _transform_attributes(attrs: dict[str, np.ndarray], linear: np.ndarray, offset: np.ndarray) -> tuple[dict, np.ndarray]:
    """Positions, normals and tangents under (linear, offset), each (3, 3) and (3,) or per vertex (n, 3, 3) and (n, 3).
    Returns the new attributes and each vertex's determinant sign (a mirror re-winds its triangles)."""
    out = dict(attrs)
    per_vertex = linear.ndim == 3
    p = attrs["POSITION"]
    out["POSITION"] = (np.einsum("vij,vj->vi", linear, p) if per_vertex else p @ linear.T) + offset
    det = np.linalg.det(linear)
    if "NORMAL" in attrs:
        inv_t = np.linalg.inv(linear).transpose(0, 2, 1) if per_vertex else np.linalg.inv(linear).T
        n = np.einsum("vij,vj->vi", inv_t, attrs["NORMAL"]) if per_vertex else attrs["NORMAL"] @ inv_t.T
        out["NORMAL"] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if "TANGENT" in attrs:
        t = attrs["TANGENT"]
        xyz = np.einsum("vij,vj->vi", linear, t[:, :3]) if per_vertex else t[:, :3] @ linear.T
        xyz = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12)
        w = t[:, 3:4] * np.sign(det).reshape(-1, 1) if per_vertex else t[:, 3:4] * np.sign(det)
        out["TANGENT"] = np.concatenate([xyz, w], axis=1)
    signs = np.sign(det) if per_vertex else np.full(len(p), np.sign(det))
    return out, signs


def _texture_refs(obj):
    """Every {"index": i, ...} texture reference under a material (baseColorTexture, normalTexture, extension textures)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict) and key.endswith("Texture") and "index" in value:
                yield value
            yield from _texture_refs(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _texture_refs(value)


def _extensions_used(obj) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "extensions" and isinstance(value, dict):
                found.update(value)
            found |= _extensions_used(value)
    elif isinstance(obj, list):
        for value in obj:
            found |= _extensions_used(value)
    return found


def merge_to_single_mesh(path: Path, out_dir: Path, *, name: str, nodes: list[str] | None = None, recentre: bool = True,
                         alpha_cutoff: float = 0.5) -> dict:
    """Write `<out_dir>/<name>.gltf` (+ `.bin`, + the images it uses, copied) holding one node and one mesh, both named
    `name`, made of every mesh node of the file (or only those named in `nodes`) with its world transform baked in.

    - Every primitive is kept, grouped by material and attribute set (one primitive per group); POSITION, NORMAL (by the
      inverse transpose), TANGENT (by the linear part; its handedness sign flips under a mirror), TEXCOORD_n and COLOR_n
      are carried, and a mirrored node's triangles are re-wound so they still face out.
    - A skinned mesh is posed by its joints (the rest pose of the file's node transforms), its own node transform ignored
      as glTF requires; skins, joints, weights, animations and morph targets are not written: the result is static.
    - `recentre`: the vertices are moved so the model's lowest point is at y = 0 and its x-z box centre on the origin,
      the "base" pivot a spawn stands on the ground at the sampled point; the shift is returned.
    - Materials are made importable as an opaque-or-masked static mesh: alpha BLEND becomes MASK at `alpha_cutoff`
      (a translucent surface writes no depth; s02 2026-10-07) and loses the extensions only a translucent material has;
      KHR_materials_unlit is dropped and the material made non-metallic (an unlit surface's emissive colour renders near
      black under the scenes' -11 EV exposure, and glTF's default metallicFactor is 1). Each change is returned.
    - Only the materials, textures, samplers and images the kept primitives use are written.

    Returns a summary: the paths, the node names merged, vertex and triangle counts, the shift, the extents (glTF x, y
    up, z), the material rewrites, the share of triangles whose material was not opaque, and what was left out."""
    path = Path(path)
    out_dir = Path(out_dir)
    doc, buffers = _load_document(path)
    world = _world_matrices(doc)
    mesh_nodes = [i for i in world if "mesh" in doc["nodes"][i]]
    names = [doc["nodes"][i].get("name", f"node_{i}") for i in mesh_nodes]
    if nodes is not None:
        missing = [n for n in nodes if n not in names]
        if missing:
            raise ValueError(f"{path}: nodes {missing} not found among the mesh nodes {names}")
        selected = [i for i, n in zip(mesh_nodes, names) if n in nodes]
    else:
        selected = mesh_nodes
    if not selected:
        raise ValueError(f"{path}: no mesh node to merge")
    materials_in = doc.get("materials", [])
    groups: dict[tuple, dict] = {}
    dropped: set[str] = set()
    skinned: list[str] = []
    morph = False
    non_opaque = 0
    for index in selected:
        node = doc["nodes"][index]
        mesh = doc["meshes"][node["mesh"]]
        joint_matrices = None
        if "skin" in node:
            joint_matrices = _skin_matrices(doc, buffers, doc["skins"][node["skin"]], world)
            skinned.append(node.get("name", f"node_{index}"))
        for k, prim in enumerate(mesh["primitives"]):
            mode = prim.get("mode", TRIANGLES)
            if mode != TRIANGLES:
                raise ValueError(f"{path}: mesh {mesh.get('name')!r} primitive {k} has mode {mode}; only triangles (4) "
                                 f"make a static mesh")
            morph = morph or "targets" in prim
            attrs = {}
            for attr, acc in prim["attributes"].items():
                if attr.startswith(KEPT_ATTRIBUTES):
                    attrs[attr] = _read_accessor(doc, buffers, acc)
                elif not attr.startswith(("JOINTS_", "WEIGHTS_")):
                    dropped.add(attr)
            count = len(attrs["POSITION"])
            indices = (_read_accessor(doc, buffers, prim["indices"]).ravel() if "indices" in prim
                       else np.arange(count, dtype=np.int64))
            tris = indices[: len(indices) - len(indices) % 3].reshape(-1, 3)
            if joint_matrices is not None:
                weights = np.zeros((count, len(joint_matrices)))
                s = 0
                while f"JOINTS_{s}" in prim["attributes"]:
                    j = _read_accessor(doc, buffers, prim["attributes"][f"JOINTS_{s}"])
                    w = _read_accessor(doc, buffers, prim["attributes"][f"WEIGHTS_{s}"])
                    np.add.at(weights, (np.repeat(np.arange(count), j.shape[1]), j.ravel()), w.ravel())
                    s += 1
                total = weights.sum(axis=1, keepdims=True)
                weights = np.where(total > 0, weights / np.where(total > 0, total, 1.0), 0.0)
                m = np.einsum("vj,jab->vab", weights, joint_matrices)
                m[total[:, 0] <= 0] = np.eye(4)  # no weight at all: the vertex stays as modelled (a zero matrix is singular)
                attrs, signs = _transform_attributes(attrs, m[:, :3, :3], m[:, :3, 3])
            else:
                w = world[index]
                attrs, signs = _transform_attributes(attrs, w[:3, :3], w[:3, 3])
            flip = signs[tris[:, 0]] < 0
            tris = np.where(flip[:, None], tris[:, [0, 2, 1]], tris)
            material = prim.get("material")
            if material is not None and materials_in[material].get("alphaMode", "OPAQUE") != "OPAQUE":
                non_opaque += len(tris)
            key = (material if material is not None else -1, tuple(sorted((a, v.shape[1]) for a, v in attrs.items())))
            group = groups.setdefault(key, {"material": material, "attrs": {a: [] for a in attrs}, "tris": [], "count": 0})
            for a, v in attrs.items():
                group["attrs"][a].append(v)
            group["tris"].append(tris + group["count"])
            group["count"] += count
    all_positions = np.concatenate([np.concatenate(g["attrs"]["POSITION"]) for g in groups.values()])
    lo, hi = all_positions.min(axis=0), all_positions.max(axis=0)
    shift = np.array([-(lo[0] + hi[0]) / 2.0, -lo[1], -(lo[2] + hi[2]) / 2.0]) if recentre else np.zeros(3)

    # materials, textures, samplers, images: only what the kept primitives use, in order of first use
    material_map: dict[int, int] = {}
    for group in groups.values():
        if group["material"] is not None and group["material"] not in material_map:
            material_map[group["material"]] = len(material_map)
    materials_out = [json.loads(json.dumps(materials_in[i])) for i in material_map]
    texture_map: dict[int, int] = {}
    for material in materials_out:
        for ref in _texture_refs(material):
            texture_map.setdefault(ref["index"], len(texture_map))
            ref["index"] = texture_map[ref["index"]]
    textures_out = [json.loads(json.dumps(doc["textures"][i])) for i in texture_map]
    sampler_map: dict[int, int] = {}
    image_map: dict[int, int] = {}
    for texture in textures_out:
        if "sampler" in texture:
            texture["sampler"] = sampler_map.setdefault(texture["sampler"], len(sampler_map))
        if "source" in texture:
            texture["source"] = image_map.setdefault(texture["source"], len(image_map))
        for ext in (texture.get("extensions") or {}).values():
            if isinstance(ext, dict) and "source" in ext:
                ext["source"] = image_map.setdefault(ext["source"], len(image_map))
    samplers_out = [dict(doc["samplers"][i]) for i in sampler_map]
    out_dir.mkdir(parents=True, exist_ok=True)
    images_out = []
    for k, i in enumerate(image_map):
        image = dict(doc["images"][i])
        uri = image.get("uri")
        if uri is not None and not uri.startswith("data:"):
            target = _inside(out_dir, uri)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_inside(path.parent, uri), target)
        else:
            if uri is not None:
                data = _decode_data_uri(uri)
                mime = uri[5:].split(";")[0]
            else:
                view = doc["bufferViews"][image["bufferView"]]
                data = buffers[view["buffer"]][view.get("byteOffset", 0): view.get("byteOffset", 0) + view["byteLength"]]
                mime = image.get("mimeType", "")
            uri = f"images/{name}_{k}{IMAGE_EXTENSIONS.get(mime, '.bin')}"
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / uri).write_bytes(data)
            image.pop("bufferView", None)
            image["uri"] = uri
        images_out.append(image)

    rewrites = []
    for material in materials_out:
        label = material.get("name", "material")
        extensions = material.get("extensions") or {}
        if material.get("alphaMode") == "BLEND":
            material["alphaMode"] = "MASK"
            material["alphaCutoff"] = float(alpha_cutoff)
            removed = [e for e in BLEND_ONLY_EXTENSIONS if e in extensions]
            for e in removed:
                extensions.pop(e)
            rewrites.append({"material": label, "change": "alphaMode BLEND -> MASK", "alpha_cutoff": float(alpha_cutoff),
                             "removed_extensions": removed})
        if "KHR_materials_unlit" in extensions:
            extensions.pop("KHR_materials_unlit")
            pbr = material.setdefault("pbrMetallicRoughness", {})
            pbr.setdefault("metallicFactor", 0.0)
            rewrites.append({"material": label, "change": "KHR_materials_unlit -> lit", "metallic_factor": pbr["metallicFactor"]})
        if "extensions" in material and not material["extensions"]:
            del material["extensions"]

    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add(array: np.ndarray, dtype: str, type_: str, component: int, target: int, minmax: bool = False) -> int:
        while len(blob) % 4:
            blob.extend(b"\0")
        data = np.ascontiguousarray(array, dtype=dtype)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": data.nbytes, "target": target})
        blob.extend(data.tobytes())
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": int(len(data)), "type": type_}
        if minmax:
            acc["min"] = [float(v) for v in data.min(axis=0)]
            acc["max"] = [float(v) for v in data.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    primitives = []
    vertices = triangles = 0
    for group in groups.values():
        attributes = {}
        for attr, parts in group["attrs"].items():
            values = np.concatenate(parts)
            if attr == "POSITION":
                values = values + shift
            n_comp = values.shape[1]
            attributes[attr] = add(values, "<f4", {2: "VEC2", 3: "VEC3", 4: "VEC4"}[n_comp], FLOAT32, 34962,
                                   minmax=attr == "POSITION")
        tris = np.concatenate(group["tris"])
        prim = {"attributes": attributes, "indices": add(tris.reshape(-1), "<u4", "SCALAR", 5125, 34963), "mode": TRIANGLES}
        if group["material"] is not None:
            prim["material"] = material_map[group["material"]]
        primitives.append(prim)
        vertices += group["count"]
        triangles += len(tris)

    out = {"asset": {"version": "2.0", "generator": MERGE_GENERATOR}, "scene": 0, "scenes": [{"name": name, "nodes": [0]}],
           "nodes": [{"name": name, "mesh": 0}], "meshes": [{"name": name, "primitives": primitives}]}
    for key, value in (("materials", materials_out), ("textures", textures_out), ("samplers", samplers_out), ("images", images_out)):
        if value:
            out[key] = value
    used = sorted(_extensions_used({k: out.get(k, []) for k in ("materials", "textures", "samplers", "images")}))
    if used:
        out["extensionsUsed"] = used
    required = sorted(set(doc.get("extensionsRequired", [])) & set(used))
    if required:
        out["extensionsRequired"] = required
    out.update(accessors=accessors, bufferViews=views, buffers=[{"uri": f"{name}.bin", "byteLength": len(blob)}])
    (out_dir / f"{name}.bin").write_bytes(bytes(blob))
    gltf_path = out_dir / f"{name}.gltf"
    gltf_path.write_text(json.dumps(out, indent=1) + "\n")
    extent = hi - lo
    return {"path": str(gltf_path), "name": name, "source": str(path),
            "nodes": [doc["nodes"][i].get("name", f"node_{i}") for i in selected],
            "vertices": int(vertices), "triangles": int(triangles), "primitives": len(primitives),
            "materials": [m.get("name") for m in materials_out], "images": [i["uri"] for i in images_out],
            "translation_m": [round(float(v), 6) for v in shift], "extent_m": [round(float(v), 6) for v in extent],
            "rewrites": rewrites, "non_opaque_triangle_share": round(non_opaque / max(triangles, 1), 6),
            "skinned_nodes": skinned, "dropped_attributes": sorted(dropped),
            "left_out": {"animations": len(doc.get("animations", [])), "skins": len(doc.get("skins", [])), "morph_targets": morph}}
