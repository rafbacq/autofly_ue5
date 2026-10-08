"""Read a glTF 2.0 model's triangles straight from its buffers and measure what the scene generator needs (plan 5, C3).

The registry's footprint is not the mesh's bounding box: a tree's obstacle footprint at the drone's 1-3 m altitude is
its trunk plus whatever canopy reaches down into that band, and a leaning tree's is off its pivot (spec §6.4, the M4
asset survey). The editor reports bounding boxes; the band needs the geometry, which the glTF holds in plain float32
buffers, so it is measured here, offline, before anything is imported. glTF is Y-up in metres with the model's pivot at
its origin (Poly Haven plants and rocks stand on y = 0); UE's importer turns that into Z-up centimetres, and the editor
build (C3) checks the overall bounds against these numbers.

Some Poly Haven "assets" are sets of variants laid side by side as separate nodes (searsia_lucida_a..f): each node is
measured on its own, in the mesh's own frame, because each becomes its own static mesh.
"""

from __future__ import annotations

import json
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


def load_meshes(path: Path) -> list[MeshGeometry]:
    """Every node that carries a mesh, in scene order, with its triangles and vertices (local and world)."""
    path = Path(path)
    gltf = json.loads(path.read_text())
    buffers = []
    for b in gltf.get("buffers", []):
        uri = b.get("uri")
        if uri is None or uri.startswith("data:"):
            raise ValueError(f"{path}: only external .bin buffers are supported")
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
