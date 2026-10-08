"""A mesh obstacle's outline as a distance field: what the footprint rule measures to since 2026-10-08 (afternoon).

The rule (expert/footprint_contact.py) ends an episode when the drone's swept step comes within its contact distance of
a mesh obstacle. Until now "of" meant its footprint circle (0.9 m), about the model's pivot with its widest reach as
radius; now it means its outline (1.05 m, for the drone's measured collision reach). On s02
that circle ended 82 of s02_r1's 220 deterministic evaluation flights a median 2.70 m from any geometry, wood or leaf
cards, and never closer than 0.94 m; on the three leaning species the median was 3.0 m, the tree was in the camera's view
at 8 of 82 endings, and 49 of 62 were on the side away from the lean
(results/2026-10-07-project-status/figures/s02_footprint_endings.json). An expert that sees only depth was being
stopped by an invisible bulge of the circle. So the distance is now to the model's outline: the horizontal projection of
all its geometry, wood and leaf cards, at every height. A column, because after an ending the reset climbs straight up
from where the drone stopped: nothing at any height may be within reach above it, which is the safety the circle gave.

The outline lies inside the circle (the circle's radius is the projection's widest reach from the pivot), so every
layout, the sampler's 1.4 m clearance, reachability, the crossing detour and the built levels stay as they are and stay
conservative; only the runtime rule changes.

Frame. validate/obstacles.Mesh._to_gltf takes a world NED point into the mesh's glTF frame (the actor's x and y undone,
then its yaw, then its scale; UE's x, y, z read as glTF's x, z, y), the frame the live depth checks were verified in to
millimetres (docs/gates/m4_s02_gate_run4.json). Its horizontal plane is glTF (x, z) at unit scale; `to_local` is that
mapping for many points at once, held to _to_gltf to 1e-9 by tests/test_footprint_field.py. A field is per asset, in
that plane: one file serves every instance, through the instance's x, y, yaw and horizontal scale.

Build (`outline_triangles`, `build_field`). The asset's triangles, opaque and cards alike, from the loader the live
checks use (MeshLibrary), are cut at the ground (the buried part of a rock is not where a drone can be, see
`clip_above_ground`), projected onto glTF (x, z) and rasterised exactly: every cell a triangle edge passes through (a supercover: each
end's cell, and at every grid-line crossing both cells it separates) and every cell whose centre lies inside a triangle.
So every cell holding geometry is marked and every marked cell holds some (2026-10-08: cv2.fillPoly with several
polygons XORs where they overlap, 127 pixels against 155 for two overlapping triangles, and its sub-pixel LINE_8 lines
start half a pixel off the true line; either would break the guarantee below). cv2.distanceTransform (DIST_L2,
DIST_MASK_PRECISE: the exact Euclidean transform, 2e-6 px off a brute force) then gives each free cell centre its
distance to the nearest marked centre, from which half a cell diagonal is subtracted: the geometry in the nearest cell
lies within that of its centre, so the field never overstates the clearance at a centre, and understates it by at most a
cell diagonal. The raster is SUPERSAMPLE times finer than the stored grid (CELL_M), placed so every stored centre is a
fine centre: the bound is then 0.0088 m instead of 0.035 m, for a few seconds of build per two-million-triangle tree.
Stored values are rounded down to float16 (never up). Points are read by bilinear interpolation between centres (which
can exceed the distance by at most CELL_M^2 / 4r at a distance r: 0.15 mm at the outline's 1.05 m), and a swept step is
sampled every SEGMENT_STEP_M at unit scale (3.1 mm; in the world, times the instance's horizontal scale); beyond the
grid's reach (MARGIN_M from the projection) a field reads +inf, as does a non-finite point.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import os
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autofly_ue5.paths import ASSET_FOOTPRINTS

CELL_M = 0.025  # the stored grid: a cell is 2.5 cm at unit scale
MARGIN_M = 3.0  # how far the grid reaches beyond the projection's bounding box, at unit scale
SUPERSAMPLE = 4  # the raster is this much finer than the stored grid (even, so stored centres are fine centres)
SEGMENT_STEP_M = CELL_M / 8  # a swept step is sampled this often at unit scale: it can read at most 2.2 mm long there,
# 2.2 mm times the instance's horizontal scale in the world (7 mm at s04's 3.17), inside the outline invariant's 1 cm
FIELD_VERSION = 1
MANIFEST = "manifest.json"
MANIFEST_FORMAT = "autofly_ue5_footprint_fields/1"
FRAME = ("glTF x (columns) and glTF z (rows) at unit scale, the horizontal plane of validate/obstacles.Mesh._to_gltf: "
         "UE x = glTF x, UE y = glTF z; origin = the corner of cell (0, 0)")
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # a fixed member time: the same field writes the same bytes
_TRIANGLE_CHUNK = 250_000


# ----------------------------------------------------------------------------------------------------------------------
# Triangles in the mesh's horizontal plane.
# ----------------------------------------------------------------------------------------------------------------------
def triangle_corners(triangles) -> np.ndarray:
    """A validate/obstacles.TriangleSet's triangles as (m, 3, 3) corners in the glTF frame (float64)."""
    v0 = np.asarray(triangles.v0, dtype=np.float64)
    return np.stack([v0, v0 + triangles.e1, v0 + triangles.e2], axis=1)


def _ground_crossing(p: np.ndarray, q: np.ndarray, ground_y: float) -> np.ndarray:
    t = (ground_y - p[:, 1]) / (q[:, 1] - p[:, 1])
    return p + t[:, None] * (q - p)


def clip_above_ground(corners: np.ndarray, ground_y: float = 0.0) -> np.ndarray:
    """The parts of (m, 3, 3) glTF triangles at or above the ground (glTF y >= ground_y; a base-pivot model stands on
    its origin): a triangle wholly below is dropped, one crossing the ground is cut there (to one triangle, or to a quad
    split in two). The drone cannot be under the ground, and Poly Haven's rocks sit up to 10 cm into it: their buried
    part reaches 6-17 mm past the widest slice above it, the footprint circle every layout was generated with
    (scenes/gltf.py profiles from y = 0 up), so an outline that kept it would not lie inside the circle."""
    c = np.asarray(corners, dtype=np.float64)
    above = c[:, :, 1] >= ground_y
    n_above = above.sum(axis=1)
    parts = [c[n_above == 3]]
    for n in (1, 2):
        sel = c[n_above == n]
        if not len(sel):
            continue
        rows = np.arange(len(sel))
        # rotate so the odd one out comes first: the one corner above (n = 1) or the one below (n = 2)
        k = np.argmax(above[n_above == n], axis=1) if n == 1 else np.argmin(above[n_above == n], axis=1)
        a, b, d = sel[rows, k], sel[rows, (k + 1) % 3], sel[rows, (k + 2) % 3]
        ab, ad = _ground_crossing(a, b, ground_y), _ground_crossing(a, d, ground_y)
        if n == 1:
            parts.append(np.stack([a, ab, ad], axis=1))
        else:  # a is below: the part above is the quad b, d, ad, ab
            parts += [np.stack([b, d, ad], axis=1), np.stack([b, ad, ab], axis=1)]
    return np.concatenate(parts) if parts else np.zeros((0, 3, 3))


def project_xz(corners: np.ndarray) -> np.ndarray:
    """(m, 3, 3) glTF corners onto the horizontal plane: (m, 3, 2) as (glTF x, glTF z). Height is dropped: a column."""
    return np.ascontiguousarray(np.asarray(corners, dtype=np.float64)[:, :, [0, 2]])


def outline_triangles(triangles) -> np.ndarray:
    """A TriangleSet's outline geometry: every triangle, opaque and cards alike, above the ground, projected onto the
    glTF horizontal plane. What a field is built from and checked against."""
    return project_xz(clip_above_ground(triangle_corners(triangles)))


# ----------------------------------------------------------------------------------------------------------------------
# The field.
# ----------------------------------------------------------------------------------------------------------------------
def round_down_float16(values) -> np.ndarray:
    """float16 values never above `values` (round to nearest, then one step down where that went up)."""
    exact = np.asarray(values, dtype=np.float64)
    half = exact.astype(np.float16)
    over = half.astype(np.float64) > exact
    half[over] = np.nextafter(half[over], np.float16(-np.inf))
    return half


@dataclass(frozen=True, eq=False)
class FootprintField:
    """An asset's outline clearance at unit scale: `values[j, i]` is the distance (m) from the centre of cell (i, j),
    (origin + (i + 0.5, j + 0.5) * cell_m in glTF x and z), to the projection of the model's geometry; 0 on it."""

    values: np.ndarray  # (nz, nx) float32, each exactly a float16
    origin: tuple[float, float]
    cell_m: float = CELL_M
    margin_m: float = MARGIN_M

    @property
    def reach_m(self) -> float:
        """Every point within this distance of the outline lies between the grid's centres, where the field is read."""
        return self.margin_m - self.cell_m / 2

    @functools.cached_property
    def pivot_reach_m(self) -> float:
        """A bound on how far the outline reaches from the pivot (the glTF origin), read from the field alone: every
        outline point lies within half a cell diagonal of a centre, whose value is then at most that (the field never
        overstates), so the farthest such centre plus half a diagonal holds the whole outline."""
        half = self.cell_m * math.sqrt(2.0) / 2.0
        rows, cols = np.nonzero(self.values <= half)
        x = self.origin[0] + (cols + 0.5) * self.cell_m
        z = self.origin[1] + (rows + 0.5) * self.cell_m
        return float(np.hypot(x, z).max()) + half

    def distance(self, local) -> np.ndarray:
        """The clearance at points (n, 2) or (2,) in the glTF horizontal plane at unit scale, by bilinear interpolation
        between cell centres; +inf beyond the grid (farther than `reach_m` from the outline)."""
        p = np.atleast_2d(np.asarray(local, dtype=np.float64))
        nz, nx = self.values.shape
        u = (p[:, 0] - self.origin[0]) / self.cell_m - 0.5
        v = (p[:, 1] - self.origin[1]) / self.cell_m - 0.5
        inside = (u >= 0.0) & (u <= nx - 1) & (v >= 0.0) & (v <= nz - 1)
        out = np.full(len(p), np.inf)
        if inside.any():
            ui, vi = u[inside], v[inside]
            i0 = np.minimum(np.floor(ui).astype(np.int64), nx - 2)
            j0 = np.minimum(np.floor(vi).astype(np.int64), nz - 2)
            fu, fv = ui - i0, vi - j0
            val = self.values
            out[inside] = ((1.0 - fu) * (1.0 - fv) * val[j0, i0] + fu * (1.0 - fv) * val[j0, i0 + 1]
                           + (1.0 - fu) * fv * val[j0 + 1, i0] + fu * fv * val[j0 + 1, i0 + 1])
        return out

    def content_sha256(self) -> str:
        """What the rule's behaviour depends on: the grid and its values (not the file's bytes, nor the code's
        version). A run's identity names it, so only a change of this asset's field refuses a resume."""
        h = hashlib.sha256(b"autofly_ue5 footprint field")
        h.update(np.asarray(self.values.shape, dtype=np.int64).tobytes())
        h.update(np.asarray([*self.origin, self.cell_m, self.margin_m], dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.values.astype(np.float16)).tobytes())
        return h.hexdigest()

    def save(self, path: Path) -> None:
        """A .npz written byte-reproducibly (fixed member times, fixed order), so its sha256 stands for its content.
        Written beside and renamed into place."""
        path = Path(path)
        members = {"values": self.values.astype(np.float16), "origin": np.asarray(self.origin, dtype=np.float64),
                   "cell_m": np.float64(self.cell_m), "margin_m": np.float64(self.margin_m),
                   "field_version": np.int64(FIELD_VERSION)}
        tmp = path.with_name(f".{path.name}.tmp")
        with zipfile.ZipFile(tmp, "w") as archive:
            for name, array in members.items():
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asanyarray(array), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, buffer.getvalue(), compresslevel=9)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> FootprintField:
        """The field in `path`. A file that is missing, damaged (a truncated archive, a missing member) or of another
        field version is refused as a ValueError naming it, which the trainer's checks before it claims a run root
        report as a refusal, never as a traceback."""
        try:
            with np.load(Path(path), allow_pickle=False) as data:
                version = int(data["field_version"])
                values = data["values"].astype(np.float32)
                origin = tuple(float(v) for v in data["origin"])
                cell_m, margin_m = float(data["cell_m"]), float(data["margin_m"])
        except Exception as err:  # OSError, zipfile.BadZipFile, KeyError, EOFError, numpy's ValueErrors: unreadable
            raise ValueError(f"{path}: unreadable footprint field ({type(err).__name__}: {err}): rebuild it with "
                             f"scripts/footprint_fields.py") from err
        if version != FIELD_VERSION:
            raise ValueError(f"{path}: footprint field version {version}, this code reads {FIELD_VERSION}: rebuild it "
                             f"with scripts/footprint_fields.py")
        return cls(values=values, origin=origin, cell_m=cell_m, margin_m=margin_m)


def _mark(flat: np.ndarray, nx: int, nz: int, cols: np.ndarray, rows: np.ndarray) -> None:
    if len(cols) and (cols.min() < 0 or cols.max() >= nx or rows.min() < 0 or rows.max() >= nz):
        raise AssertionError("geometry outside its own grid: the margin must hold every triangle")
    flat[rows * nx + cols] = True


def _mark_segments(flat: np.ndarray, nx: int, nz: int, a: np.ndarray, b: np.ndarray) -> None:
    """Every cell the segments a -> b (rows of (u, v) in cells: cell (i, j) spans [i, i + 1) x [j, j + 1)) pass through:
    each end's cell, and at each crossing of a grid line both cells it separates."""
    for end in (a, b):
        cell = np.floor(end).astype(np.int64)
        _mark(flat, nx, nz, cell[:, 0], cell[:, 1])
    d = b - a
    for axis in (0, 1):
        lo, hi = np.minimum(a[:, axis], b[:, axis]), np.maximum(a[:, axis], b[:, axis])
        first = np.floor(lo).astype(np.int64) + 1
        count = np.maximum(np.floor(hi).astype(np.int64) - first + 1, 0)
        total = int(count.sum())
        if not total:
            continue
        seg = np.repeat(np.arange(len(a)), count)
        k = first[seg] + (np.arange(total) - np.repeat(np.cumsum(count) - count, count))
        t = (k - a[seg, axis]) / d[seg, axis]
        other = 1 - axis
        w = np.floor(a[seg, other] + t * d[seg, other]).astype(np.int64)
        if axis == 0:
            _mark(flat, nx, nz, k - 1, w)
            _mark(flat, nx, nz, k, w)
        else:
            _mark(flat, nx, nz, w, k - 1)
            _mark(flat, nx, nz, w, k)


def _edge(p: np.ndarray, q: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (q[:, 0] - p[:, 0]) * (z - p[:, 1]) - (q[:, 1] - p[:, 1]) * (x - p[:, 0])


def _fill_interiors(flat: np.ndarray, nx: int, nz: int, tri: np.ndarray) -> None:
    """Every cell whose centre (i + 0.5, j + 0.5) lies inside (or on) a triangle (rows of (3, 2) in cells)."""
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    i0, i1 = np.ceil(lo[:, 0] - 0.5).astype(np.int64), np.floor(hi[:, 0] - 0.5).astype(np.int64)
    j0, j1 = np.ceil(lo[:, 1] - 0.5).astype(np.int64), np.floor(hi[:, 1] - 0.5).astype(np.int64)
    ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
    count = ni * nj
    sel = np.flatnonzero(count)
    if not len(sel):
        return
    tri, i0, j0, ni, count = tri[sel], i0[sel], j0[sel], ni[sel], count[sel]
    seg = np.repeat(np.arange(len(sel)), count)
    local = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    ci, cj = i0[seg] + local % ni[seg], j0[seg] + local // ni[seg]
    x, z = ci + 0.5, cj + 0.5
    p0, p1, p2 = tri[seg, 0], tri[seg, 1], tri[seg, 2]
    e0, e1, e2 = _edge(p0, p1, x, z), _edge(p1, p2, x, z), _edge(p2, p0, x, z)
    inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
    _mark(flat, nx, nz, ci[inside], cj[inside])


def build_field(tri_xz: np.ndarray, *, cell_m: float = CELL_M, margin_m: float = MARGIN_M,
                supersample: int = SUPERSAMPLE) -> FootprintField:
    """The field of projected triangles (m, 3, 2) in the glTF horizontal plane at unit scale (see the module's doc)."""
    import cv2

    tri = np.asarray(tri_xz, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 2) or not len(tri):
        raise ValueError(f"expected projected triangles of shape (m, 3, 2) with m > 0, got {tri.shape}")
    if supersample < 2 or supersample % 2:
        raise ValueError(f"supersample must be even and at least 2, got {supersample}")
    lo = tri.reshape(-1, 2).min(axis=0) - margin_m
    hi = tri.reshape(-1, 2).max(axis=0) + margin_m
    nx, nz = (int(math.ceil((hi[k] - lo[k]) / cell_m)) for k in (0, 1))
    fine = cell_m / supersample
    # fine centre s * k + s / 2 - 1 is stored centre k: the fine grid starts half a fine cell after the stored one
    fine_origin = lo + fine / 2
    fnx, fnz = nx * supersample, nz * supersample
    flat = np.zeros(fnx * fnz, dtype=bool)
    for start in range(0, len(tri), _TRIANGLE_CHUNK):
        t = (tri[start:start + _TRIANGLE_CHUNK] - fine_origin) / fine
        _mark_segments(flat, fnx, fnz, t.reshape(-1, 2), t[:, [1, 2, 0]].reshape(-1, 2))
        _fill_interiors(flat, fnx, fnz, t)
    free = np.where(flat.reshape(fnz, fnx), 0, 255).astype(np.uint8)
    dt = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    m = supersample // 2 - 1
    at_centres = dt[m::supersample, m::supersample].astype(np.float64)
    if at_centres.shape != (nz, nx):
        raise AssertionError(f"stored grid {at_centres.shape} != {(nz, nx)}")
    values = np.maximum(at_centres * fine - fine * math.sqrt(2.0) / 2.0, 0.0)
    return FootprintField(values=round_down_float16(values).astype(np.float32), origin=(float(lo[0]), float(lo[1])),
                          cell_m=float(cell_m), margin_m=float(margin_m))


# ----------------------------------------------------------------------------------------------------------------------
# An instance's transform, and the clearance of a swept step.
# ----------------------------------------------------------------------------------------------------------------------
def horizontal_scale(inst) -> float:
    """The instance's one horizontal scale: a field measured at unit scale is a world distance times it only when x and
    y scale alike, as every generated layout does (scale = (s_xy, s_xy, s_z)). Anything else is refused."""
    sx, sy = float(inst.scale[0]), float(inst.scale[1])
    if not (sx > 0.0 and abs(sx - sy) <= 1e-9 * max(sx, sy)):
        raise ValueError(f"instance {inst.tag} ({inst.asset}) has horizontal scale {sx} x {sy}: an outline field needs "
                         f"one horizontal scale (scale[0] == scale[1])")
    return sx


def to_local(inst, xy) -> np.ndarray:
    """World NED points (n, 2) into the instance's mesh frame at unit scale, as (glTF x, glTF z): Mesh._to_gltf's mapping
    (translation, then yaw, then scale undone)."""
    p = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    s = horizontal_scale(inst)
    c, sn = math.cos(inst.yaw), math.sin(inst.yaw)
    dx, dy = p[:, 0] - inst.x, p[:, 1] - inst.y
    return np.stack([(c * dx + sn * dy) / s, (-sn * dx + c * dy) / s], axis=1)


def to_world(inst, local) -> np.ndarray:
    """The inverse of `to_local`: (glTF x, glTF z) at unit scale to world NED (x, y)."""
    p = np.atleast_2d(np.asarray(local, dtype=np.float64)) * horizontal_scale(inst)
    c, sn = math.cos(inst.yaw), math.sin(inst.yaw)
    return np.stack([inst.x + c * p[:, 0] - sn * p[:, 1], inst.y + sn * p[:, 0] + c * p[:, 1]], axis=1)


def segment_samples(length_local_m: float) -> int:
    return 1 if length_local_m <= 0.0 else int(math.ceil(length_local_m / SEGMENT_STEP_M)) + 1


def outline_distance(field: FootprintField, inst, a_xy, b_xy=None) -> float:
    """The world clearance (m) of the swept step a_xy -> b_xy from the instance's outline: the field sampled every
    SEGMENT_STEP_M of the step (at unit scale) through the instance's transform, its minimum, times the horizontal scale.
    +inf when the whole step is beyond the field's reach."""
    s = horizontal_scale(inst)
    a = np.asarray(a_xy, dtype=np.float64)
    b = a if b_xy is None else np.asarray(b_xy, dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        # Non-finite kinematics are the backend's to refuse, not the rule's to judge. Every sample of such a step is
        # non-finite (a + t (b - a)), and the field reads a non-finite point as off its grid: +inf, no contact. Said
        # here, before the sample count, which cannot be taken of a NaN length (a bare ValueError would end a run).
        return math.inf
    t = np.linspace(0.0, 1.0, segment_samples(float(np.hypot(*(b - a))) / s))[:, None]
    return float(field.distance(to_local(inst, a + t * (b - a))).min()) * s


# ----------------------------------------------------------------------------------------------------------------------
# A brute-force distance, for checking a field against its triangles.
# ----------------------------------------------------------------------------------------------------------------------
def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v, w = b - a, p - a
    vv = np.einsum("ij,ij->i", v, v)
    t = np.clip(np.einsum("ij,ij->i", w, v) / np.where(vv > 0.0, vv, 1.0), 0.0, 1.0)
    return np.hypot(w[:, 0] - t * v[:, 0], w[:, 1] - t * v[:, 1])


def triangle_distances(tri: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Each triangle's distance from the point p (0 on or inside it): its nearest edge, unless p is inside."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    d = np.minimum(np.minimum(_segment_distance(p, a, b), _segment_distance(p, b, c)), _segment_distance(p, c, a))
    x, z = np.full(len(tri), p[0]), np.full(len(tri), p[1])
    e0, e1, e2 = _edge(a, b, x, z), _edge(b, c, x, z), _edge(c, a, x, z)
    area = _edge(a, b, c[:, 0], c[:, 1])
    inside = (area != 0.0) & (((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0)))
    return np.where(inside, 0.0, d)


def exact_distance(tri_xz: np.ndarray, points, *, bucket_m: float = 0.1) -> np.ndarray:
    """Each point's distance to the union of the triangles (m, 3, 2): exact, by brute force over the triangles that can
    be nearest. Each triangle is listed in every bucket (a `bucket_m` grid) its box overlaps; a point visits the non-empty
    buckets nearest first and stops at the first whose own distance is no less than the best found, since a triangle's
    nearest point lies in one of its buckets."""
    tri = np.asarray(tri_xz, dtype=np.float64)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    origin = lo.min(axis=0)
    width = int(np.floor((hi.max(axis=0)[0] - origin[0]) / bucket_m)) + 1
    b0 = np.floor((lo - origin) / bucket_m).astype(np.int64)
    b1 = np.floor((hi - origin) / bucket_m).astype(np.int64)
    ni, nj = b1[:, 0] - b0[:, 0] + 1, b1[:, 1] - b0[:, 1] + 1
    count = ni * nj
    owner = np.repeat(np.arange(len(tri)), count)
    local = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    key = (b0[owner, 1] + local // ni[owner]) * width + b0[owner, 0] + local % ni[owner]
    order = np.argsort(key, kind="stable")
    key, owner = key[order], owner[order]
    keys, first = np.unique(key, return_index=True)
    last = np.append(first[1:], len(key))
    rect_lo = origin + np.stack([keys % width, keys // width], axis=1) * bucket_m
    rect_hi = rect_lo + bucket_m
    out = np.empty(len(pts))
    for n, p in enumerate(pts):
        gap = np.maximum(np.maximum(rect_lo - p, p - rect_hi), 0.0)
        bound = np.hypot(gap[:, 0], gap[:, 1])
        best = math.inf
        for b in np.argsort(bound, kind="stable"):
            if bound[b] >= best:
                break
            best = min(best, float(triangle_distances(tri[owner[first[b]:last[b]]], p).min()))
        out[n] = best
    return out


# ----------------------------------------------------------------------------------------------------------------------
# The fields on disk: assets/footprints/<asset>.npz and their manifest.
# ----------------------------------------------------------------------------------------------------------------------
def read_manifest(root: Path = ASSET_FOOTPRINTS) -> dict:
    path = Path(root) / MANIFEST
    if not path.is_file():
        raise ValueError(f"no footprint fields at {root} ({MANIFEST} is missing): build them with "
                         f"scripts/footprint_fields.py")
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise ValueError(f"{path}: unreadable ({type(err).__name__}: {err}): rebuild it with scripts/footprint_fields.py") from err
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a footprint field manifest")
    if data.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{path}: format {data.get('format')!r}, expected {MANIFEST_FORMAT!r}")
    return data


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_LOADED: dict[tuple[str, str, str], FootprintField] = {}  # (root, asset, field_sha256): one copy per process


class FootprintFields(Mapping[str, FootprintField]):
    """The outline fields of a set of assets, read once (per process, per field) from one directory's manifest."""

    def __init__(self, fields: Mapping[str, FootprintField], *, manifest_sha256: str | None = None) -> None:
        self._fields = dict(fields)
        self._digest = {asset: field.content_sha256() for asset, field in sorted(self._fields.items())}
        self.manifest_sha256 = manifest_sha256

    @classmethod
    def load(cls, assets: Iterable[str], root: Path = ASSET_FOOTPRINTS) -> FootprintFields:
        """The fields of `assets`. An asset without one, a field of another version, or a file whose content is not the
        manifest's is refused (ValueError): the outline rule never falls back to anything."""
        root = Path(root)
        manifest = read_manifest(root)
        entries = manifest.get("fields", {})
        wanted = sorted(set(assets))
        missing = [asset for asset in wanted if asset not in entries]
        if missing:
            raise ValueError(f"no footprint field for {', '.join(missing)} in {root / MANIFEST}: the outline rule cannot "
                             f"measure {'it' if len(missing) == 1 else 'them'} (scripts/footprint_fields.py --only "
                             f"{' '.join(missing)})")
        fields = {}
        for asset in wanted:
            entry = entries[asset]
            if entry.get("field_version") != FIELD_VERSION:
                raise ValueError(f"{root / MANIFEST}: {asset}'s field is version {entry.get('field_version')}, this code "
                                 f"reads {FIELD_VERSION}: rebuild it with scripts/footprint_fields.py --only {asset}")
            key = (str(root.resolve()), asset, entry["field_sha256"])
            if key not in _LOADED:
                field = FootprintField.load(root / entry["file"])
                if field.content_sha256() != entry["field_sha256"]:
                    raise ValueError(f"{root / entry['file']}: its content is not the one {MANIFEST} names "
                                     f"(field_sha256): rebuild with scripts/footprint_fields.py --only {asset}")
                _LOADED[key] = field
            fields[asset] = _LOADED[key]
        return cls(fields, manifest_sha256=file_sha256(root / MANIFEST))

    def digest(self) -> dict[str, str]:
        """{asset: field content sha256}: what a run's identity and a record name."""
        return dict(self._digest)

    def __getitem__(self, asset: str) -> FootprintField:
        return self._fields[asset]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)
