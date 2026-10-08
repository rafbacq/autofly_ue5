"""One geometry for every obstacle kind the live checks meet (plan 5, C8): engine cylinders (s01's pillars, s09's
poles), yawed engine cubes (s07's stacks) and imported glTF meshes (s02's trees, the rocks to come).

The M1 checks (`validate/live_m1.py`, `sim/check_map.py`) were written for centred cylinders: a world box of (2r, 2r, h)
about the layout's (x, y) and a ray-circle distance for the depth at the image centre. Neither holds for a cube at a
random yaw (UE reports the axis-aligned box of the *rotated* local box) or a tree whose pivot is its trunk base while
its bounds centre sits in the canopy, 0.9 m away. Here each kind answers the same two questions: what world-axis box
does UE report for it, and where along a camera ray does its surface lie. The box follows UE's own rule (the local
bounds box, scaled, yawed, re-fitted to the axes); the surface distance is exact for cylinders and boxes and ray-cast
against the glTF triangles for meshes, in the mesh's own frame so a two-million-triangle tree is never transformed.

Frames: layout and poses are NED metres (x north, y east, z down, yaw north toward east). UE's local mesh axes are glTF's
with y and z swapped (UE x = glTF x, UE y = glTF z, UE z = glTF y: the registry's `bounds_origin_cm`, read from the
imported meshes, matched the glTF pivot offsets that way to 1 mm on 2026-10-07). Rays carry an *unnormalised*
direction: with the direction (1, a, b) in the camera's frame the ray parameter at the hit is the planar depth the
simulator's depth camera reports, so no per-pixel cosine is needed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autofly_ue5.frames import wrap_pi
from autofly_ue5.paths import ASSET_MEASUREMENTS, ASSET_REGISTRY, ASSET_SOURCES
from autofly_ue5.scenes.model import AssetEntry, AssetRegistry, Bounds
from autofly_ue5.sim.types import Pose
from autofly_ue5.validate.geometry import CAMERA_OFFSET_M

Vec3 = tuple[float, float, float]
IMAGE_SIZE = 256
HFOV_DEG = 90.0
# the depth window the checks read: 4 x 4 pixels about the image centre (live_m1.check_depth, since M1)
WINDOW = (slice(126, 130), slice(126, 130))
EPS = 1e-9


def _rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


def _aabb_of_yawed_box(centre_local: Vec3, half_local: Vec3, scale: Vec3, yaw: float, actor_ned: Vec3) -> tuple[list[float], list[float]]:
    """UE's world-axis box of an actor's local bounds box: scaled, yawed about the actor, translated, re-fitted to the
    axes (FBox::TransformBy on the eight corners). `centre_local`/`half_local` are in UE's local axes, metres, z up."""
    cx, cy, cz = (centre_local[i] * scale[i] for i in range(3))
    hx, hy, hz = (half_local[i] * scale[i] for i in range(3))
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    wx, wy = _rotate_xy(cx, cy, yaw)
    ax, ay, az = actor_ned
    centre = [ax + wx, ay + wy, az - cz]  # z up (UE) -> z down (NED)
    size = [2 * (c * hx + s * hy), 2 * (s * hx + c * hy), 2 * hz]
    return centre, size


@dataclass(frozen=True)
class Cylinder:
    """An engine cylinder with its pivot at its centre: the layout's radius and height are the geometry."""

    tag: str
    x: float
    y: float
    radius_m: float
    z_center: float
    height_m: float
    kind: str = "cylinder"

    def world_bbox(self) -> tuple[list[float], list[float]]:
        return [self.x, self.y, self.z_center], [2 * self.radius_m, 2 * self.radius_m, self.height_m]

    @property
    def top_altitude_m(self) -> float:
        return -(self.z_center - self.height_m / 2)

    @property
    def base_altitude_m(self) -> float:
        return -(self.z_center + self.height_m / 2)

    def surface_distance(self, origin: Vec3, direction: Vec3) -> float | None:
        ox, oy, oz = origin
        dx, dy, dz = direction
        fx, fy = ox - self.x, oy - self.y
        z_lo, z_hi = self.z_center - self.height_m / 2, self.z_center + self.height_m / 2
        hits = []
        a = dx * dx + dy * dy
        if a > EPS:
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - self.radius_m**2
            disc = b * b - a * c
            if disc >= 0:
                root = math.sqrt(disc)
                for t in ((-b - root) / a, (-b + root) / a):
                    if t > 0 and z_lo <= oz + t * dz <= z_hi:
                        hits.append(t)
        if abs(dz) > EPS:  # the caps
            for z in (z_lo, z_hi):
                t = (z - oz) / dz
                if t > 0 and math.hypot(fx + t * dx, fy + t * dy) <= self.radius_m:
                    hits.append(t)
        return min(hits) if hits else None

    def surface_distances(self, origin: Vec3, directions, *, opaque_only: bool = False) -> np.ndarray:
        return _distances_one_by_one(self, origin, directions)


@dataclass(frozen=True)
class Box:
    """An engine cube with its pivot at its centre, at a yaw: half-extents are half the actor's scale."""

    tag: str
    x: float
    y: float
    radius_m: float
    z_center: float
    yaw: float
    half: Vec3
    kind: str = "box"

    def world_bbox(self) -> tuple[list[float], list[float]]:
        return _aabb_of_yawed_box((0.0, 0.0, 0.0), self.half, (1.0, 1.0, 1.0), self.yaw, (self.x, self.y, self.z_center))

    @property
    def top_altitude_m(self) -> float:
        return -(self.z_center - self.half[2])

    @property
    def base_altitude_m(self) -> float:
        return -(self.z_center + self.half[2])

    def surface_distance(self, origin: Vec3, direction: Vec3) -> float | None:
        # slab test in the box's own frame (yaw undone about the centre; z needs no rotation)
        ox, oy = _rotate_xy(origin[0] - self.x, origin[1] - self.y, -self.yaw)
        dx, dy = _rotate_xy(direction[0], direction[1], -self.yaw)
        o = (ox, oy, origin[2] - self.z_center)
        d = (dx, dy, direction[2])
        t_near, t_far = 0.0, math.inf
        for i in range(3):
            if abs(d[i]) < EPS:
                if abs(o[i]) > self.half[i]:
                    return None
                continue
            t1, t2 = (-self.half[i] - o[i]) / d[i], (self.half[i] - o[i]) / d[i]
            t_near, t_far = max(t_near, min(t1, t2)), min(t_far, max(t1, t2))
            if t_near > t_far:
                return None
        return t_near if t_near > 0 else None

    def surface_distances(self, origin: Vec3, directions, *, opaque_only: bool = False) -> np.ndarray:
        return _distances_one_by_one(self, origin, directions)


def _distances_one_by_one(obstacle, origin: Vec3, directions) -> np.ndarray:
    return np.array([math.inf if (t := obstacle.surface_distance(origin, tuple(d))) is None else t for d in directions])


class TriangleSet:
    """A mesh's triangles in glTF's frame, with the per-triangle height range for the ray prefilter and whether each
    renders opaque (an alpha-tested leaf card may let a ray through; `opaque` None means all do)."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, opaque: np.ndarray | None = None):
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("vertices must be (n, 3) and faces (m, 3)")
        self.opaque = np.ones(len(f), bool) if opaque is None else np.asarray(opaque, bool)
        if len(self.opaque) != len(f):
            raise ValueError("opaque must have one flag per face")
        tri = v[f]  # (m, 3, 3)
        self.v0 = np.ascontiguousarray(tri[:, 0], dtype=np.float32)
        self.e1 = np.ascontiguousarray(tri[:, 1] - tri[:, 0], dtype=np.float32)
        self.e2 = np.ascontiguousarray(tri[:, 2] - tri[:, 0], dtype=np.float32)
        # per-triangle extents on each glTF axis, for the ray-bundle prefilter
        self.lo = tri.min(axis=1).astype(np.float32)  # (m, 3)
        self.hi = tri.max(axis=1).astype(np.float32)
        self.y_lo, self.y_hi = self.lo[:, 1], self.hi[:, 1]
        self.count = int(len(f))

    def first_hits(self, origin: np.ndarray, directions: np.ndarray, *, t_max: float = 200.0, opaque_only: bool = False) -> np.ndarray:
        """For each ray from `origin` along a row of `directions`, the smallest positive parameter at which it meets a
        triangle (Moller-Trumbore, both faces; only opaque ones with `opaque_only`), inf where it meets none. Only
        triangles whose box overlaps the bundle's between t = 0 and t_max (the box of the origin and the rays' far
        ends) are tested: a 16 x 16 window at 6 m is a corridor 0.8 m wide and 0.8 m tall through a 4 m canopy, and
        the test runs on the trunk's triangles, not the tree's million. Rays go in batches of 16 against triangle
        chunks of 250k so the (rays x triangles) temporaries stay around 50 MB."""
        o = np.asarray(origin, dtype=np.float32)
        dirs = np.atleast_2d(np.asarray(directions, dtype=np.float32))
        ends = np.concatenate([o[None, :], o[None, :] + t_max * dirs])  # (r + 1, 3)
        lo, hi = ends.min(axis=0), ends.max(axis=0)
        keep = np.all((self.hi >= lo) & (self.lo <= hi), axis=1)
        if opaque_only:
            keep &= self.opaque
        best = np.full(len(dirs), np.inf)
        if not keep.any():
            return best
        idx = np.flatnonzero(keep)
        for tri in np.array_split(idx, max(1, len(idx) // 250_000 + 1)):
            v0, e1, e2 = self.v0[tri], self.e1[tri], self.e2[tri]
            tvec = o - v0  # (m, 3), the same for every ray
            for r0 in range(0, len(dirs), 16):
                d = dirs[r0:r0 + 16]  # (r, 3)
                pvec = np.cross(d[:, None, :], e2[None, :, :])  # (r, m, 3)
                det = np.einsum("mj,rmj->rm", e1, pvec)
                parallel = np.abs(det) < 1e-12
                det = np.where(parallel, 1.0, det)
                u = np.einsum("mj,rmj->rm", tvec, pvec) / det
                qvec = np.cross(tvec, e1)  # (m, 3)
                v = np.einsum("rj,mj->rm", d, qvec) / det
                tt = np.einsum("mj,mj->m", e2, qvec)[None, :] / det
                ok = (~parallel) & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (tt > 1e-6) & (tt < t_max)
                hit = np.where(ok, tt, np.inf).min(axis=1)
                best[r0:r0 + 16] = np.minimum(best[r0:r0 + 16], hit)
        return best

    def first_hit(self, origin: np.ndarray, direction: np.ndarray, *, t_max: float = 200.0, opaque_only: bool = False) -> float | None:
        t = float(self.first_hits(origin, np.asarray(direction, dtype=np.float32)[None, :], t_max=t_max, opaque_only=opaque_only)[0])
        return t if math.isfinite(t) else None


@dataclass(frozen=True)
class Mesh:
    """An imported model standing on its pivot (base pivot), placed at an actor location, yaw and scale. The actor
    under the instance's tag is the model's solid part (wood, rock: the collision mesh); a model with alpha cards has a
    second actor, <tag>_cards, at the same transform and without collision (level_spec.py). The whole model's box
    (bounds_*) decides where the camera can look at it; each actor's own box is what the simulator reports for it."""

    tag: str
    x: float
    y: float
    radius_m: float
    actor_z: float  # NED z of the actor (the ground top for a base pivot)
    yaw: float
    scale: Vec3  # UE actor scale (x, y, z)
    bounds_centre_local_m: Vec3  # the whole model, UE local axes, from the registry's bounds_origin_cm
    bounds_half_local_m: Vec3  # half of base_size_m
    triangles: TriangleSet
    solid_centre_local_m: Vec3 | None = None  # the solid part's own box; None: the whole is the solid
    solid_half_local_m: Vec3 | None = None
    cards_centre_local_m: Vec3 | None = None  # the cards part's own box; None: no cards actor
    cards_half_local_m: Vec3 | None = None
    kind: str = "mesh"

    def world_bbox(self) -> tuple[list[float], list[float]]:
        centre = self.solid_centre_local_m if self.solid_centre_local_m is not None else self.bounds_centre_local_m
        half = self.solid_half_local_m if self.solid_half_local_m is not None else self.bounds_half_local_m
        return _aabb_of_yawed_box(centre, half, self.scale, self.yaw, (self.x, self.y, self.actor_z))

    def whole_world_bbox(self) -> tuple[list[float], list[float]]:
        return _aabb_of_yawed_box(self.bounds_centre_local_m, self.bounds_half_local_m, self.scale, self.yaw, (self.x, self.y, self.actor_z))

    @property
    def cards_tag(self) -> str | None:
        return f"{self.tag}_cards" if self.cards_centre_local_m is not None else None

    def cards_world_bbox(self) -> tuple[list[float], list[float]]:
        return _aabb_of_yawed_box(self.cards_centre_local_m, self.cards_half_local_m, self.scale, self.yaw, (self.x, self.y, self.actor_z))

    @property
    def top_altitude_m(self) -> float:
        return -self.actor_z + (self.bounds_centre_local_m[2] + self.bounds_half_local_m[2]) * self.scale[2]

    @property
    def base_altitude_m(self) -> float:
        return -self.actor_z + (self.bounds_centre_local_m[2] - self.bounds_half_local_m[2]) * self.scale[2]

    def _to_gltf(self, p: Vec3, *, point: bool) -> np.ndarray:
        """A world NED point (or direction) into the mesh's glTF frame: z up, actor undone, yaw undone, scale undone,
        UE's (x, y, z) read as glTF's (x, z, y). Directions skip the translation; the ray parameter is unchanged."""
        x, y, z_up = p[0], p[1], -p[2]
        if point:
            x, y, z_up = x - self.x, y - self.y, z_up + self.actor_z
        x, y = _rotate_xy(x, y, -self.yaw)
        ux, uy, uz = x / self.scale[0], y / self.scale[1], z_up / self.scale[2]
        return np.array([ux, uz, uy])

    def _t_max(self, origin: Vec3) -> float:
        # the rays need not reach past the mesh's far side: a tight bound keeps the height prefilter to the band they
        # cross (a 12 m probe's window rays rise 0.14 m over their length, a 200 m horizon would rise 2.3 m)
        centre, size = self.whole_world_bbox()
        return math.dist(origin, centre) + math.hypot(*size) / 2 + 0.5

    def surface_distance(self, origin: Vec3, direction: Vec3) -> float | None:
        return self.triangles.first_hit(self._to_gltf(origin, point=True), self._to_gltf(direction, point=False), t_max=self._t_max(origin))

    def surface_distances(self, origin: Vec3, directions, *, opaque_only: bool = False) -> np.ndarray:
        local = np.array([self._to_gltf(tuple(d), point=False) for d in directions])
        return self.triangles.first_hits(self._to_gltf(origin, point=True), local, t_max=self._t_max(origin), opaque_only=opaque_only)

    @property
    def has_cards(self) -> bool:
        return not self.triangles.opaque.all()


Obstacle = Cylinder | Box | Mesh


class MeshLibrary:
    """The glTF triangles of the imported models, by registry key, loaded once."""

    def __init__(self, triangles: dict[str, TriangleSet]):
        self._triangles = triangles

    @classmethod
    def from_arrays(cls, meshes: dict[str, tuple]) -> MeshLibrary:
        """{key: (vertices, faces[, opaque])} in glTF's frame."""
        return cls({k: TriangleSet(*arrays) for k, arrays in meshes.items()})

    @classmethod
    def from_files(cls, keys: set[str], *, registry_path: Path = ASSET_REGISTRY, sources_path: Path = ASSET_SOURCES,
                   measurements_path: Path = ASSET_MEASUREMENTS) -> MeshLibrary:
        """Load the glTF node behind each registry key in `keys`: the registry file names the source and node, the
        measurements the file (`scripts/measure_assets.py`), the sources the download directory."""
        from autofly_ue5.scenes.gltf import load_meshes

        entries = json.loads(Path(registry_path).read_text())["assets"]
        sources = json.loads(Path(sources_path).read_text())
        downloads = Path(sources["downloads_dir"])
        files = {(m["asset"], m["node"]): m["file"] for m in json.loads(Path(measurements_path).read_text())["meshes"]}
        out: dict[str, TriangleSet] = {}
        loaded: dict[Path, list] = {}
        for key in sorted(keys):
            source = entries[key].get("source")
            if not isinstance(source, dict):
                raise ValueError(f"asset {key!r} has no glTF source to load triangles from")
            path = downloads / source["source"] / source["id"] / files[(source["id"], source["node"])]
            meshes = loaded.setdefault(path, load_meshes(path))
            node = next((m for m in meshes if m.name == source["node"]), None)
            if node is None:
                raise ValueError(f"node {source['node']!r} not in {path}")
            out[key] = TriangleSet(node.local_vertices, node.triangles, node.triangle_opaque)
        return cls(out)

    def __contains__(self, key: str) -> bool:
        return key in self._triangles

    def __getitem__(self, key: str) -> TriangleSet:
        return self._triangles[key]


def obstacle_from_instance(inst: dict, registry: AssetRegistry, library: MeshLibrary | None = None) -> Obstacle:
    """The geometry of one layout instance (a dict as the layout JSON stores it), from its registry entry."""
    asset: AssetEntry = registry.assets[inst["asset"]]
    scale = tuple(float(s) for s in inst["scale"])
    common = {"tag": inst["tag"], "x": float(inst["x"]), "y": float(inst["y"]), "radius_m": float(inst["radius_m"])}
    if asset.pivot == "center" and asset.footprint == "circle":
        return Cylinder(**common, z_center=float(inst["z_center"]), height_m=float(inst["height_m"]))
    if asset.pivot == "center" and asset.footprint == "box":
        half = tuple(asset.base_size_m[i] * scale[i] / 2 for i in range(3))
        return Box(**common, z_center=float(inst["z_center"]), yaw=float(inst["yaw"]), half=half)
    if asset.pivot == "base":
        if library is None or inst["asset"] not in library:
            raise ValueError(f"asset {inst['asset']!r} is an imported mesh: a MeshLibrary holding it is required")
        origin = asset.bounds_origin_cm or (0.0, 0.0, 0.0)
        solid, cards = asset.solid_part, asset.cards_part
        return Mesh(**common, actor_z=float(inst["z_center"]), yaw=float(inst["yaw"]), scale=scale,
                    bounds_centre_local_m=tuple(c / 100.0 for c in origin), bounds_half_local_m=tuple(s / 2 for s in asset.base_size_m),
                    triangles=library[inst["asset"]],
                    solid_centre_local_m=tuple(c / 100.0 for c in solid.bounds_origin_cm), solid_half_local_m=tuple(s / 2 for s in solid.base_size_m),
                    cards_centre_local_m=tuple(c / 100.0 for c in cards.bounds_origin_cm) if cards else None,
                    cards_half_local_m=tuple(s / 2 for s in cards.base_size_m) if cards else None)
    raise ValueError(f"no geometry for pivot {asset.pivot!r} with footprint {asset.footprint!r}")


def obstacles_from_layout_json(data: dict, registry: AssetRegistry, library: MeshLibrary | None = None) -> list[Obstacle]:
    return [obstacle_from_instance(inst, registry, library) for inst in data["layout"]["instances"]]


def mesh_keys(data: dict, registry: AssetRegistry) -> set[str]:
    """The imported-mesh assets a layout uses (those a MeshLibrary must hold)."""
    return {inst["asset"] for inst in data["layout"]["instances"] if registry.assets[inst["asset"]].pivot == "base"}


def bbox_error(obstacle: Obstacle, bounds: tuple[Vec3, Vec3] | None, *, part: str = "solid") -> dict:
    """How far the simulator's world-axis box of an actor (centre, size) is from the layout's prediction: the solid
    part under the instance's tag, or a mesh's cards actor (`part="cards"`)."""
    if bounds is None:
        return {"found": False}
    got_centre, got_size = bounds
    if part == "cards" and max(abs(s) for s in got_size) < 1e-3:
        # UE's GetComponentsBoundingBox skips non-colliding components: the plugin finds the actor (so it is in the
        # level, under its tag) but can report no box for it. Its placement is the level builder's check (build time).
        return {"found": True, "box_unavailable": True, "reason": "no collision: the simulator reports no bounds for a non-colliding actor"}
    centre, size = obstacle.cards_world_bbox() if part == "cards" else obstacle.world_bbox()
    return {"found": True, "center_error_m": math.dist(got_centre, centre), "size_error_m": max(abs(a - b) for a, b in zip(got_size, size))}


def camera_origin(pose: Pose, camera_offset_m: float = CAMERA_OFFSET_M) -> Vec3:
    return (pose.x + camera_offset_m * math.cos(pose.yaw), pose.y + camera_offset_m * math.sin(pose.yaw), pose.z)


def pixel_rays(pose: Pose, rows, cols, *, image_size: int = IMAGE_SIZE, hfov_deg: float = HFOV_DEG) -> list[Vec3]:
    """Unnormalised world directions (planar depth = ray parameter) of the pixel centres in rows x cols of the front
    camera: forward along the yaw, right = yaw + 90 deg (columns increase rightward), down = +z (rows increase downward)."""
    f = (image_size / 2) / math.tan(math.radians(hfov_deg) / 2)
    fwd = (math.cos(pose.yaw), math.sin(pose.yaw), 0.0)
    right = (-math.sin(pose.yaw), math.cos(pose.yaw), 0.0)
    rays = []
    for r in range(*rows.indices(image_size)):
        b = (r + 0.5 - image_size / 2) / f
        for c in range(*cols.indices(image_size)):
            a = (c + 0.5 - image_size / 2) / f
            rays.append((fwd[0] + a * right[0], fwd[1] + a * right[1], b))
    return rays


def window_depths(obstacles: list[Obstacle], pose: Pose, window=WINDOW, *, camera_offset_m: float = CAMERA_OFFSET_M,
                  opaque_only: bool = False) -> np.ndarray:
    """The planar depth each pixel of `window` (row-major, flat) would see of `obstacles` from `pose`; inf where none is
    hit. `opaque_only` ignores a mesh's alpha-tested cards (its leaves) and sees its solid geometry behind them."""
    origin = camera_origin(pose, camera_offset_m)
    rays = np.array(pixel_rays(pose, *window))
    depths = np.full(len(rays), np.inf)
    for o in obstacles:
        depths = np.minimum(depths, o.surface_distances(origin, rays, opaque_only=opaque_only))
    return depths


def row_silhouette(obstacle: Obstacle, pose: Pose, *, row: int = IMAGE_SIZE // 2, near_m: float, camera_offset_m: float = CAMERA_OFFSET_M) -> np.ndarray:
    """Which pixels of one image row see `obstacle` nearer than `near_m`: the predicted silhouette the HFOV check
    compares with the frame's (geometry.contiguous_width about the centre column)."""
    origin = camera_origin(pose, camera_offset_m)
    depths = [obstacle.surface_distance(origin, ray) for ray in pixel_rays(pose, slice(row, row + 1), slice(0, IMAGE_SIZE))]
    return np.array([d is not None and d < near_m for d in depths])


# the wider window a foliage probe is judged over: 16 x 16 pixels about the centre, per pixel
WIDE_WINDOW = (slice(120, 136), slice(120, 136))
SOLID_KINDS = ("cylinder", "box")


@dataclass(frozen=True)
class Probe:
    """A camera pose from which a depth window should see one obstacle's surface at predicted planar depths.

    A solid obstacle (cylinder, box) is probed on the 4 x 4 centre window, all of it on the surface within
    `window_spread_m`: the depth check compares the window medians and the row silhouette, as M1 did. A mesh (a tree)
    is foliage at most heights, so it is probed on the 16 x 16 window pixel by pixel. `predicted_m` is each pixel's
    first hit on anything (the collision surface: leaf cards included), `predicted_opaque_m` its first hit on opaque
    geometry (trunk, branches, rock). A pixel whose opaque hit is also its first hit is *solid*: the camera must
    measure that depth. A pixel whose first hit is a card may measure the card (an opaque texel) or something behind
    (a transparent one), so it must only measure no nearer than the card. `expected_depth_m` is the median over the
    solid pixels; `hit_fraction` their share of the window."""

    tag: str
    distance_m: float  # the distance asked for, from the camera to the obstacle's footprint circle
    pose: Pose
    obstacle: Obstacle
    expected_depth_m: float  # the median over the solid pixels
    window_spread_m: float  # max - min over the solid pixels (a solid patch is small)
    window: tuple  # the (rows, cols) slices the arrays cover
    predicted_m: np.ndarray  # flat, row-major over `window`: first hit on anything
    predicted_opaque_m: np.ndarray  # first hit on opaque geometry only (equal to predicted_m for a solid obstacle)
    hit_fraction: float  # the share of the window's pixels that are solid

    @property
    def solid(self) -> bool:
        return self.obstacle.kind in SOLID_KINDS

    @property
    def solid_pixels(self) -> np.ndarray:
        return _solid_pixels(self.predicted_m, self.predicted_opaque_m)

    @property
    def interior_pixels(self) -> np.ndarray:
        """Solid pixels whose eight neighbours in the window are solid too: a patch of trunk or rock, not a twig one
        pixel wide (whose depth half a pixel decides). A solid obstacle's window is interior throughout."""
        if self.solid:
            return self.solid_pixels
        return _interior(self.solid_pixels, self.window)


def _solid_pixels(predicted: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    return np.isfinite(opaque) & (opaque <= predicted + 1e-3)


def _interior(mask: np.ndarray, window) -> np.ndarray:
    rows, cols = (s.stop - s.start for s in window)
    grid = np.pad(mask.reshape(rows, cols), 1, constant_values=False)
    out = np.ones((rows, cols), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= grid[1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols]
    return out.reshape(-1)


def choose_probes(
    obstacles: list[Obstacle],
    bounds: Bounds,
    distances: tuple[float, ...] = (3.0, 6.0, 12.0),
    altitude_m: float = 2.0,
    camera_offset_m: float = CAMERA_OFFSET_M,
    body_clearance_m: float = 1.5,
    ray_clearance_m: float = 0.3,
    edge_margin_m: float = 2.0,
    body_half_height_m: float = 0.5,
    max_spread_m: float = 0.3,
    lower_altitudes_m: tuple[float, ...] = (1.5, 1.2),
    min_interior_pixels: int = 12,
    fallback_distances_m: dict[float, tuple[float, ...]] | None = None,
) -> list[Probe]:
    """One probe per distance: the first obstacle (by tag), altitude (`altitude_m`, then `lower_altitudes_m`, all in
    the flight band) and bearing (8 compass points) whose camera pose lies inside the bounds and clear of every other
    obstacle at the body's altitude, where the depth window lands on the obstacle -- the whole 4 x 4 centre within
    `max_spread_m` for a solid kind, at least `min_interior_pixels` of the 16 x 16 window on a patch of opaque
    geometry (solid pixels with solid neighbours: a trunk, not twigs) with no leaf card in front for a mesh -- and
    nothing else comes within `ray_clearance_m` in front of it along those rays.

    The camera stands `distance_m` from the footprint circle, as M1's did from its pillars: for a cylinder that puts
    the surface exactly there, for a box or a mesh the surface is farther (the circle is conservative) and the probe
    records the ray-cast prediction instead of the request. Occlusion is settled by ray casting, not in plan: s07 lists
    each cube of a stack as its own actor at the same x, y, and the cubes under the one the camera looks at block
    nothing. Lower altitudes exist for the trees: at 2 m s02's quiver trees have forked and the others are canopy,
    while at 1.2-1.5 m their trunks are solid. An island tree's canopy at 2 m offered 64-71 solid pixels per window, every
    one an isolated twig pixel, and 13-20 % of them measured another branch (2026-10-07).

    `fallback_distances_m` maps a distance to the shorter ones to try when no obstacle offers a probe at it: s02's
    trunks are 0.2-0.3 m across, 2-3 pixels at 12 m, too thin for an interior pixel; the probe records the distance it
    actually stands at. Default: 12 m falls back to 10 m, then 8 m; 6 m to 5 m, then 4 m (s06r's live check, 2026-10-08,
    found no unobstructed probe at 6 m). A distance that offers a probe never reaches its fallbacks, so s01's probes are
    M1's."""
    if fallback_distances_m is None:
        fallback_distances_m = {6.0: (5.0, 4.0), 12.0: (10.0, 8.0)}

    def overlaps(o: Obstacle, lo: float, hi: float) -> bool:
        return o.base_altitude_m <= hi and o.top_altitude_m >= lo

    probes = []
    for requested in distances:
        found = None
        for distance in (requested, *fallback_distances_m.get(requested, ())):
            found = _probe_at(obstacles, bounds, distance, altitude_m, camera_offset_m, body_clearance_m, ray_clearance_m,
                              edge_margin_m, body_half_height_m, max_spread_m, lower_altitudes_m, min_interior_pixels, overlaps)
            if found is not None:
                break
        if found is None:
            raise ValueError(f"no unobstructed depth probe at {requested} m (nor at {fallback_distances_m.get(requested, ())})")
        probes.append(found)
    return probes


def _probe_at(obstacles, bounds, distance, altitude_m, camera_offset_m, body_clearance_m, ray_clearance_m, edge_margin_m,
              body_half_height_m, max_spread_m, lower_altitudes_m, min_interior_pixels, overlaps) -> Probe | None:
    """choose_probes' search for one distance (see there)."""
    found = None
    for obstacle in sorted(obstacles, key=lambda o: o.tag):
        solid = obstacle.kind in SOLID_KINDS
        others = [q for q in obstacles if q.tag != obstacle.tag]
        for altitude in (altitude_m, *lower_altitudes_m):
            if not overlaps(obstacle, altitude, altitude):
                continue  # the camera's altitude is not on it: it would look over, or under, it
            body_blockers = [q for q in others if overlaps(q, altitude - body_half_height_m, altitude + body_half_height_m)]
            for k in range(8):
                yaw = wrap_pi(k * math.pi / 4)
                ux, uy = math.cos(yaw), math.sin(yaw)
                standoff = obstacle.radius_m + distance + camera_offset_m
                x, y = obstacle.x - standoff * ux, obstacle.y - standoff * uy
                if not (bounds.x_min + edge_margin_m <= x <= bounds.x_max - edge_margin_m
                        and bounds.y_min + edge_margin_m <= y <= bounds.y_max - edge_margin_m):
                    continue
                if any(math.hypot(x - q.x, y - q.y) < q.radius_m + body_clearance_m for q in body_blockers):
                    continue
                pose = Pose(x, y, -altitude, yaw)
                centre = window_depths([obstacle], pose, WINDOW, camera_offset_m=camera_offset_m)
                if solid:
                    if not np.isfinite(centre).all() or centre.max() - centre.min() > max_spread_m:
                        continue
                    window, depths, opaque = WINDOW, centre, centre
                else:
                    # cheap first look (32 rays): the centre must already sit on solid geometry before the wide
                    # window (512 rays, seconds on a million-triangle tree) is cast
                    centre_opaque = window_depths([obstacle], pose, WINDOW, camera_offset_m=camera_offset_m, opaque_only=True)
                    if _solid_pixels(centre, centre_opaque).sum() < 12:
                        continue  # three of its four columns: a trunk two pixels wide has no interior pixel
                    window = WIDE_WINDOW
                    depths = window_depths([obstacle], pose, window, camera_offset_m=camera_offset_m)
                    opaque = window_depths([obstacle], pose, window, camera_offset_m=camera_offset_m, opaque_only=True)
                is_solid = _solid_pixels(depths, opaque)
                if not solid and _interior(is_solid, window).sum() < min_interior_pixels:
                    continue
                # whatever's footprint comes near the line of sight in plan is ray-cast for real; the rest cannot intrude
                cam = (x + camera_offset_m * ux, y + camera_offset_m * uy)
                near = (obstacle.x - obstacle.radius_m * ux, obstacle.y - obstacle.radius_m * uy)
                suspects = [q for q in others if _segment_point_distance(*cam, *near, q.x, q.y) < q.radius_m + ray_clearance_m]
                if suspects and (window_depths(suspects, pose, window, camera_offset_m=camera_offset_m) < depths + ray_clearance_m).any():
                    continue
                hits = opaque[is_solid if solid else _interior(is_solid, window)]
                found = Probe(obstacle.tag, distance, pose, obstacle, float(np.median(hits)), float(hits.max() - hits.min()),
                              window, depths, opaque, float(is_solid.mean()))
                break
            if found is not None:
                break
        if found is not None:
            break
    return found


def _segment_point_distance(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))
