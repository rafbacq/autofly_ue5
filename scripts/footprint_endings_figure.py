"""Where the footprint rule ends a tree run's evaluation flights, measured against the trees' actual geometry.

    env -u PYTHONPATH .venv/bin/python scripts/footprint_endings_figure.py --run-root runs/expert/s02_r1 \
        --layout runs/levels/s02.layout.json --out results/<date>/figures/s02_footprint_endings.png \
        --json results/<date>/figures/s02_footprint_endings.json

In a mesh scene an episode ends as a collision when the drone's swept step comes within 0.9 m of an obstacle's
footprint circle (expert/footprint_contact.py): a circle about the model's pivot whose radius is the model's widest reach
from that pivot over its whole height (scenes/generate.py). The expert flies on depth and cannot see the circle; it sees
the wood and the leaf cards. Three of s02's species lean (island_tree_02, island_tree_03, tree_small_02: canopy centres
0.65-1.07 m off the pivot, assets/measurements.json), so on the side away from the lean a pivot-centred circle reaches
well past any leaf. Two readings of s02_r1's endings lead to opposite recipes: if the drone is stopped far from anything
it can see, the rule is unobservable there and a wider training margin makes it worse; if it is stopped close to the
geometry, the margin is the lever. This measures which:

1. every footprint ending in the run's eval-watch records (deterministic flights): the horizontal clearance from the
   drone's centre to the named tree's wood alone and to wood plus leaf cards, over 720 horizontal rays at its altitude
   and 0.25 m above and below (the body's half-thickness), at the step's end and back along the heading over the 0.4 m
   a full-speed step covers, and the bearing of the nearest geometry from the heading;
2. every tree: how far its circle reaches past its geometry inside the flight band, per azimuth about the pivot, and how
   far the rule's boundary (0.9 m outside the circle) lies from that geometry; and the same for the minimum enclosing
   circle of the model's horizontal projection over its whole height, the tightest circle that still holds every slice.

The geometry is validate/obstacles.py's: the glTF triangles ray-cast in the mesh's own frame (verified live to
millimetres, docs/gates/m4_s02_gate_run4.json). Ray clearances use it directly. Extents, the enclosing circle and the
drawn slices place the triangles' corners in the world with the inverse of its transform, which is checked on every tree
before use: a round trip through Mesh._to_gltf, and window_depths against the same triangles cast in the world frame on
three endings. It launches nothing and writes only --out and --json.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from autofly_ue5.expert.footprint_contact import FOOTPRINT_CONTACT_M, footprint_obstacles  # noqa: E402
from autofly_ue5.expert.movers import FORWARD_SPEED_MAX_M_S  # noqa: E402
from autofly_ue5.frames import wrap_pi  # noqa: E402
from autofly_ue5.paths import RUNS_DIR  # noqa: E402
from autofly_ue5.scenes.model import DRONE_HALF_SPAN_M, Instance, load_registry  # noqa: E402
from autofly_ue5.sim.types import CONTROL_DT_S, Pose  # noqa: E402
from autofly_ue5.validate.obstacles import (  # noqa: E402
    WINDOW,
    Mesh,
    MeshLibrary,
    TriangleSet,
    camera_origin,
    obstacle_from_instance,
    pixel_rays,
    window_depths,
)

RAY_AZIMUTHS = 720  # horizontal rays per clearance: 0.5 deg, 1.7 cm apart at 2 m
EXTENT_AZIMUTHS = 360  # azimuth bins of a tree's extent about a centre
BODY_HALF_THICKNESS_M = 0.25  # the clearance is also taken this far above and below the drone's centre
STEP_M = FORWARD_SPEED_MAX_M_S * CONTROL_DT_S  # 0.4 m: how far back the ending step can have started
SEGMENT_BACK_M = tuple(round(k * STEP_M / 4, 3) for k in range(5))  # points along it, from the step's end back
GRID_M = 0.02  # a projection keeps one triangle corner per 2 cm cell: distances to it are within 2.8 cm
FAR_M = 1.5  # hypothesis A's line: stopped this far from any geometry, the drone cannot see what stopped it
BAND_M = (1.0, 3.0)  # the flight band every scene flies (scene altitude_band)
DEFAULT_RUN_ROOT = RUNS_DIR / "expert" / "s02_r1"
DEFAULT_LAYOUT = RUNS_DIR / "levels" / "s02.layout.json"
HEIGHTS = (("at_altitude", 0.0), ("below", -BODY_HALF_THICKNESS_M), ("above", BODY_HALF_THICKNESS_M))
_AZ = np.arange(RAY_AZIMUTHS) * (2.0 * math.pi / RAY_AZIMUTHS)
_DIRS = np.stack([np.cos(_AZ), np.sin(_AZ), np.zeros(RAY_AZIMUTHS)], axis=1)  # unit: the ray parameter is metres
_BIN_AZ = (np.arange(EXTENT_AZIMUTHS) + 0.5) * (2.0 * math.pi / EXTENT_AZIMUTHS)  # each extent bin's middle
_BIN_U = np.stack([np.cos(_BIN_AZ), np.sin(_BIN_AZ)], axis=1)


# ----------------------------------------------------------------------------------------------------------------------
# Inputs: the run's endings and the layout's trees.
# ----------------------------------------------------------------------------------------------------------------------
def _step_of(path: Path) -> int:
    match = re.fullmatch(r"step_(\d+)\.json", path.name)
    return int(match.group(1)) if match else -1


def load_endings(run_root: Path) -> tuple[list[dict], dict]:
    """Every deterministic flight of the run's eval-watch records (`*.partial.json` is a record still being written),
    and the footprint endings among them with where each one ended."""
    watch = Path(run_root) / "eval_watch"
    files = sorted((p for p in watch.glob("step_*.json") if not p.name.endswith(".partial.json")), key=_step_of)
    if not files:
        raise SystemExit(f"no eval-watch records in {watch}")
    endings: list[dict] = []
    outcomes: Counter = Counter()
    for path in files:
        record = json.loads(path.read_text())
        for name, checkpoint in sorted(record.get("checkpoints", {}).items()):
            for ep in (checkpoint.get("deterministic") or {}).get("per_episode") or []:
                outcomes[ep.get("collision_source") or ep.get("outcome") or "none"] += 1
                contact = ep.get("footprint_contact")
                if contact is None:
                    continue
                if not ep.get("final_pose"):
                    raise SystemExit(f"{path}: {name} seed {ep.get('seed')} ended at a footprint without a final pose")
                endings.append({"record": path.name, "checkpoint": name, "seed": ep.get("seed"), "steps": ep.get("steps"),
                                "final_pose": [float(v) for v in ep["final_pose"]], **contact})
    return endings, {"records": [p.name for p in files], "flights": sum(outcomes.values()),
                     "endings": dict(sorted(outcomes.items()))}


def load_trees(layout_path: Path) -> tuple[dict, str, list[dict], dict[str, Mesh], MeshLibrary]:
    """The layout's footprint obstacles (the registry decides which, as the rule does) as validate/obstacles.py's
    meshes, with the glTF triangles they are cast against."""
    blob = Path(layout_path).read_bytes()
    data = json.loads(blob)
    registry = load_registry()
    instances = [Instance(**{**raw, "scale": tuple(raw["scale"])}) for raw in data["layout"]["instances"]]
    covered = {inst.tag for inst in footprint_obstacles(instances, registry)}
    trees = [raw for raw in data["layout"]["instances"] if raw["tag"] in covered]
    if not trees:
        raise SystemExit(f"{layout_path} has no footprint obstacle: nothing for the rule to end flights at")
    library = MeshLibrary.from_files({raw["asset"] for raw in trees})
    meshes = {raw["tag"]: obstacle_from_instance(raw, registry, library) for raw in trees}
    return data, hashlib.sha256(blob).hexdigest(), trees, meshes, library


# ----------------------------------------------------------------------------------------------------------------------
# The transform, checked against validate/obstacles.py's.
# ----------------------------------------------------------------------------------------------------------------------
def corners(triangles: TriangleSet) -> np.ndarray:
    """Each triangle's three corners in the glTF frame, (3n, 3): corner k of triangle i is row k * n + i."""
    v0 = triangles.v0.astype(np.float64)
    return np.concatenate([v0, v0 + triangles.e1, v0 + triangles.e2])


def gltf_to_world(mesh: Mesh, local: np.ndarray) -> np.ndarray:
    """Points in the mesh's glTF frame (y up) into world NED: the inverse of Mesh._to_gltf (UE x = glTF x, UE y = glTF
    z, UE z = glTF y; scaled, yawed about the actor, translated; z up becomes NED's z down)."""
    local = np.asarray(local, dtype=np.float64)
    ux, uy, uz = local[:, 0] * mesh.scale[0], local[:, 2] * mesh.scale[1], local[:, 1] * mesh.scale[2]
    c, s = math.cos(mesh.yaw), math.sin(mesh.yaw)
    return np.stack([mesh.x + c * ux - s * uy, mesh.y + s * ux + c * uy, mesh.actor_z - uz], axis=1)


def round_trip_error_m(mesh: Mesh, local: np.ndarray, rng: np.random.Generator, n: int = 500) -> float:
    """The largest glTF-frame error of sample corners taken to the world here and back by Mesh._to_gltf."""
    idx = rng.choice(len(local), size=min(n, len(local)), replace=False)
    back = np.array([mesh._to_gltf(tuple(p), point=True) for p in gltf_to_world(mesh, local[idx])])
    return float(np.abs(back - local[idx]).max())


def window_depth_check(mesh: Mesh, triangles: TriangleSet, local: np.ndarray, pose: Pose) -> dict:
    """window_depths from `pose` (camera at the drone's centre) against the same triangles placed in the world here and
    cast there: equal depths mean the inverse transform puts every triangle where the live-verified one does."""
    n = triangles.count
    world = TriangleSet(gltf_to_world(mesh, local), np.arange(3 * n).reshape(3, n).T, triangles.opaque)
    origin = np.asarray(camera_origin(pose, 0.0))
    rays = np.array(pixel_rays(pose, *WINDOW))
    out = {"pose": [round(v, 4) for v in (pose.x, pose.y, pose.z, pose.yaw)], "pixels": int(len(rays))}
    for label, opaque_only in (("all", False), ("wood", True)):
        expected = window_depths([mesh], pose, WINDOW, camera_offset_m=0.0, opaque_only=opaque_only)
        got = world.first_hits(origin, rays, t_max=mesh._t_max(tuple(origin)), opaque_only=opaque_only)
        same = np.isfinite(expected) == np.isfinite(got)
        both = np.isfinite(expected) & np.isfinite(got)
        out[label] = {"hits": int(np.isfinite(expected).sum()), "hit_pattern_equal": bool(same.all()),
                      "max_abs_diff_m": float(np.abs(expected[both] - got[both]).max()) if both.any() else None,
                      "window_depth_m": float(expected.min()) if np.isfinite(expected).any() else None}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Clearances at the endings.
# ----------------------------------------------------------------------------------------------------------------------
def ray_clearance(mesh: Mesh, x: float, y: float, altitude: float, *, opaque_only: bool) -> tuple[float, float]:
    """The nearest surface over RAY_AZIMUTHS horizontal rays from (x, y) at `altitude`, and its world azimuth (inf and
    nan when no ray meets the model: nothing at that height)."""
    t = mesh.surface_distances((x, y, -altitude), _DIRS, opaque_only=opaque_only)
    i = int(np.argmin(t))
    return (float(t[i]), float(_AZ[i])) if math.isfinite(t[i]) else (math.inf, math.nan)


def _finite(v: float, nd: int = 4) -> float | None:
    return round(float(v), nd) if v is not None and math.isfinite(v) else None


def clearance_at(mesh: Mesh, x: float, y: float, altitude: float, yaw: float) -> dict:
    """Wood alone and wood plus cards, at the three heights, the minimum over them, and where the nearest lies."""
    out: dict = {}
    for label, opaque_only in (("all", False), ("wood", True)):
        per = {name: ray_clearance(mesh, x, y, altitude + dz, opaque_only=opaque_only) for name, dz in HEIGHTS}
        name, (d, az) = min(per.items(), key=lambda kv: kv[1][0])
        out[label] = {**{k: _finite(v[0]) for k, v in per.items()}, "min": _finite(d),
                      "nearest_height": name if math.isfinite(d) else None,
                      "nearest_bearing_deg": round(math.degrees(wrap_pi(az - yaw)), 1) if math.isfinite(d) else None}
    return out


def _min_distance_2d(points: np.ndarray, x: float, y: float) -> float:
    return float(np.hypot(points[:, 0] - x, points[:, 1] - y).min()) if len(points) else math.inf


def ending_record(ending: dict, mesh: Mesh, tree: dict, geom: dict, world: np.ndarray, wood: np.ndarray) -> dict:
    """One footprint ending: the circle's gap recomputed (the step's end is the swept segment's closest point when they
    agree), the ray clearances at the step's end and its estimated start, the minimum along the segment between, and
    the vertex clearances to the tree's whole column, its band column and in 3-D."""
    x, y, z, yaw = ending["final_pose"]
    altitude = -z
    hx, hy = math.cos(yaw), math.sin(yaw)
    segment = {back: clearance_at(mesh, x - back * hx, y - back * hy, altitude, yaw) for back in SEGMENT_BACK_M}
    final, start = segment[SEGMENT_BACK_M[0]], segment[SEGMENT_BACK_M[-1]]
    seg_min = {label: _finite(min((s[label]["min"] if s[label]["min"] is not None else math.inf) for s in segment.values()))
               for label in ("all", "wood")}
    pivot_d = math.hypot(x - tree["x"], y - tree["y"])
    sx, sy = x - STEP_M * hx, y - STEP_M * hy
    mec = geom["mec_all"]
    d3 = np.linalg.norm(world - np.array([x, y, z]), axis=1)
    # the whole slab the body occupies (+-0.25 m), not only the three planes the rays sample
    slab = np.abs(-world[:, 2] - altitude) <= BODY_HALF_THICKNESS_M
    return {
        "record": ending["record"], "checkpoint": ending["checkpoint"], "seed": ending["seed"], "steps": ending["steps"],
        "tag": ending["tag"], "asset": ending["asset"], "rule": ending.get("rule"),
        "recorded_gap_m": ending.get("gap_m"), "recorded_bearing_deg": ending.get("bearing_deg"),
        "drone_forward_m_s": ending.get("drone_forward_m_s"),
        "final_pose": [round(v, 4) for v in ending["final_pose"]], "altitude_m": round(altitude, 3),
        "circle_gap_final_m": round(pivot_d - tree["radius_m"], 4),
        "circle_gap_start_m": round(math.hypot(sx - tree["x"], sy - tree["y"]) - tree["radius_m"], 4),
        "pivot_bearing_from_heading_deg": round(math.degrees(wrap_pi(math.atan2(tree["y"] - y, tree["x"] - x) - yaw)), 1),
        "lean_angle_deg": round(math.degrees(abs(wrap_pi(math.atan2(y - tree["y"], x - tree["x"])
                                                         - math.atan2(mec["centre"][1] - tree["y"], mec["centre"][0] - tree["x"])))), 1)
        if mec["offset_m"] > 0.05 else None,
        "mec_gap_final_m": round(math.hypot(x - mec["centre"][0], y - mec["centre"][1]) - mec["radius_m"], 4),
        # the forward camera's 90 deg field of view: is the nearest geometry in the frame the step ended on?
        "nearest_in_forward_view": (final["all"]["nearest_bearing_deg"] is not None
                                    and abs(final["all"]["nearest_bearing_deg"]) <= 45.0),
        "clearance_final": final,
        "clearance_start": start,
        "clearance_segment_min": seg_min,
        "clearance_slab": {"all": _finite(_min_distance_2d(world[slab, :2], x, y)),
                           "wood": _finite(_min_distance_2d(world[slab & wood, :2], x, y))},
        "clearance_column": {"all": _finite(_min_distance_2d(geom["column_all"], x, y)),
                             "wood": _finite(_min_distance_2d(geom["column_wood"], x, y))},
        "clearance_band_column": {"all": _finite(_min_distance_2d(geom["band_all"], x, y)),
                                  "wood": _finite(_min_distance_2d(geom["band_wood"], x, y))},
        "clearance_3d": {"all": _finite(d3.min()), "wood": _finite(d3[wood].min()) if wood.any() else None},
    }


# ----------------------------------------------------------------------------------------------------------------------
# Each tree's extent, its circles, and where their boundaries lie.
# ----------------------------------------------------------------------------------------------------------------------
def grid_points(xy: np.ndarray, cell: float = GRID_M) -> np.ndarray:
    """One point per occupied `cell` grid cell (the first, so every kept point is geometry): a projection thinned to
    what a distance or a drawing can resolve."""
    if not len(xy):
        return np.zeros((0, 2))
    ij = np.floor(xy / cell).astype(np.int64)
    _, first = np.unique(ij[:, 0] * 4_000_037 + ij[:, 1], return_index=True)
    return xy[np.sort(first)]


def azimuth_extent(xy: np.ndarray, centre) -> np.ndarray:
    """The farthest point from `centre` in each of EXTENT_AZIMUTHS azimuth bins (0 where a bin holds none: nothing
    reaches out from the centre that way)."""
    out = np.zeros(EXTENT_AZIMUTHS)
    if len(xy):
        d = xy - np.asarray(centre)[None, :]
        az = np.arctan2(d[:, 1], d[:, 0]) % (2.0 * math.pi)
        k = np.minimum((az * EXTENT_AZIMUTHS / (2.0 * math.pi)).astype(np.int64), EXTENT_AZIMUTHS - 1)
        np.maximum.at(out, k, np.hypot(d[:, 0], d[:, 1]))
    return out


def boundary_clearance(xy: np.ndarray, centre, radius: float) -> np.ndarray:
    """For each bin's middle azimuth, the distance from the rule's boundary point (FOOTPRINT_CONTACT_M outside a circle
    of `radius` about `centre`) to the nearest of `xy`: how far the drone stopped there would be from that geometry."""
    if not len(xy):
        return np.full(EXTENT_AZIMUTHS, np.inf)
    b = np.asarray(centre)[None, :] + (radius + FOOTPRINT_CONTACT_M) * _BIN_U
    out = np.empty(EXTENT_AZIMUTHS)
    for i in range(0, EXTENT_AZIMUTHS, 45):
        chunk = b[i:i + 45]
        out[i:i + 45] = np.sqrt(((chunk[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return out


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain, counter-clockwise. Points strictly inside the polygon of the extremes in 16 directions
    cannot be on the hull, which leaves a tree's million corners at a few thousand before the Python loop."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > 64:
        ang = np.arange(16) * (2.0 * math.pi / 16)
        ext = pts[np.unique(np.argmax(pts @ np.stack([np.cos(ang), np.sin(ang)]), axis=0))]
        ext = ext[np.argsort(np.arctan2(ext[:, 1] - ext[:, 1].mean(), ext[:, 0] - ext[:, 0].mean()))]
        if len(ext) >= 3:
            inside = np.ones(len(pts), bool)
            for a, b in zip(ext, np.roll(ext, -1, axis=0)):
                inside &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) > 1e-12
            pts = pts[~inside]
    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts

    def chain(seq):
        out: list = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    seq = [tuple(p) for p in pts]
    lower, upper = chain(seq), chain(seq[::-1])
    return np.array(lower[:-1] + upper[:-1])


def min_enclosing_circle(points: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    """Welzl's randomised incremental algorithm on the convex hull (the hull's circle is the set's): centre, radius."""
    hull = convex_hull(points)
    p = [tuple(q) for q in hull[np.random.default_rng(seed).permutation(len(hull))]]

    def two(a, b):
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), math.dist(a, b) / 2

    def three(a, b, c):
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-14:  # collinear: the widest pair's circle
            return max((two(a, b), two(a, c), two(b, c)), key=lambda cr: cr[1])
        sa, sb, sc = a[0] ** 2 + a[1] ** 2, b[0] ** 2 + b[1] ** 2, c[0] ** 2 + c[1] ** 2
        centre = ((sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d,
                  (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d)
        return centre, math.dist(centre, a)

    def out(cr, q):
        return math.dist(cr[0], q) > cr[1] + 1e-9

    cr = (p[0], 0.0)
    for i in range(1, len(p)):
        if out(cr, p[i]):
            cr = (p[i], 0.0)
            for j in range(i):
                if out(cr, p[j]):
                    cr = two(p[i], p[j])
                    for k in range(j):
                        if out(cr, p[k]):
                            cr = three(p[i], p[j], p[k])
    centre, radius = np.array(cr[0]), float(cr[1])
    if np.hypot(*(hull - centre).T).max() > radius + 1e-7:
        raise AssertionError("the enclosing circle leaves a hull point outside")
    return centre, radius


def asset_circles(local: np.ndarray, wood: np.ndarray) -> dict:
    """Each model's minimum enclosing circle of its horizontal projection over the whole height, in UE's local axes at
    unit scale (UE x = glTF x, UE y = glTF z), of every corner and of the wood's, with its pivot circle for scale."""
    xy = local[:, [0, 2]]
    out = {"pivot_radius_m": float(np.hypot(xy[:, 0], xy[:, 1]).max())}
    for label, mask in (("all", slice(None)), ("wood", wood)):
        centre, radius = min_enclosing_circle(xy[mask])
        out[label] = {"centre_local": [float(c) for c in centre], "radius_m": radius}
    return out


def tree_geometry(mesh: Mesh, tree: dict, local: np.ndarray, wood: np.ndarray, circles: dict,
                  band_m: tuple[float, float]) -> tuple[dict, np.ndarray]:
    """A tree's projections (whole column and flight band, all and wood, thinned to GRID_M), its extent per azimuth
    about the pivot, its two enclosing circles in the world, and where each circle's rule boundary lies from the band's
    geometry. Returns the summary and the corners in the world (for the 3-D clearances)."""
    if abs(mesh.scale[0] - mesh.scale[1]) > 1e-9:
        raise ValueError(f"{mesh.tag}: x and y scales differ; a circle does not stay a circle")
    world = gltf_to_world(mesh, local)
    altitude = -world[:, 2]
    band = (altitude >= band_m[0]) & (altitude <= band_m[1])
    geom: dict = {
        "column_all": grid_points(world[:, :2]), "column_wood": grid_points(world[wood, :2]),
        "band_all": grid_points(world[band, :2]), "band_wood": grid_points(world[band & wood, :2]),
    }
    pivot = np.array([mesh.x, mesh.y])
    c, s = math.cos(mesh.yaw), math.sin(mesh.yaw)
    for label in ("all", "wood"):
        lx, ly = (v * mesh.scale[0] for v in circles[label]["centre_local"])
        centre = pivot + np.array([c * lx - s * ly, s * lx + c * ly])
        geom[f"mec_{label}"] = {"centre": [float(v) for v in centre], "radius_m": circles[label]["radius_m"] * mesh.scale[0],
                                "offset_m": float(math.hypot(lx, ly))}
    radius = float(tree["radius_m"])
    mec = geom["mec_all"]
    geom["extent_all"] = azimuth_extent(geom["band_all"], pivot)
    geom["extent_wood"] = azimuth_extent(geom["band_wood"], pivot)
    geom["mec_extent_all"] = azimuth_extent(geom["band_all"], mec["centre"])
    geom["boundary_all"] = boundary_clearance(geom["band_all"], pivot, radius)
    geom["boundary_wood"] = boundary_clearance(geom["band_wood"], pivot, radius)
    geom["mec_boundary_all"] = boundary_clearance(geom["band_all"], mec["centre"], mec["radius_m"])
    geom["mec_boundary_wood"] = boundary_clearance(geom["band_wood"], mec["centre"], mec["radius_m"])
    # against the whole column too: the circles are drawn round it, so its own boundary is never nearer than 0.9 m
    geom["boundary_column_all"] = boundary_clearance(geom["column_all"], pivot, radius)
    geom["mec_boundary_column_all"] = boundary_clearance(geom["column_all"], mec["centre"], mec["radius_m"])
    return geom, world


def _q(values) -> dict | None:
    """min / quartiles / max of the finite values, and how many there were (and were not)."""
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    out = {"n": int(len(v)), "n_none": int(len(values) - len(v))}
    if len(v):
        out.update({k: round(float(x), 3) for k, x in zip(("min", "q1", "median", "q3", "max"),
                                                        np.percentile(v, [0, 25, 50, 75, 100]))})
    return out


def _share(values, threshold: float) -> float | None:
    v = np.asarray(values, dtype=float)
    return round(float((v > threshold).mean()), 4) if len(v) else None


def tree_summary(tree: dict, geom: dict) -> dict:
    """What the JSON keeps per tree: the circles, the excess per azimuth (circle radius minus the band's extent, 1 deg
    bins about the circle's centre) and the boundary clearance per azimuth, with their summaries."""
    radius = float(tree["radius_m"])
    mec = geom["mec_all"]
    excess_all = radius - geom["extent_all"]
    excess_wood = radius - geom["extent_wood"]
    mec_excess = mec["radius_m"] - geom["mec_extent_all"]
    r = lambda a: [round(float(x), 3) if math.isfinite(x) else None for x in a]  # noqa: E731
    return {
        "tag": tree["tag"], "asset": tree["asset"], "x": tree["x"], "y": tree["y"], "yaw": tree["yaw"],
        "scale": tree["scale"], "radius_m": radius,
        "mec_all": {k: (round(v, 4) if isinstance(v, float) else [round(c, 4) for c in v]) for k, v in mec.items()},
        "mec_wood": {k: (round(v, 4) if isinstance(v, float) else [round(c, 4) for c in v]) for k, v in geom["mec_wood"].items()},
        "area_reduction": round(1.0 - (mec["radius_m"] / radius) ** 2, 4),
        "band_points": {"all": int(len(geom["band_all"])), "wood": int(len(geom["band_wood"]))},
        "pivot_circle": {
            "excess_all": _q(excess_all), "excess_wood": _q(excess_wood),
            "share_excess_all_gt_contact": _share(excess_all, FOOTPRINT_CONTACT_M),
            "share_excess_wood_gt_contact": _share(excess_wood, FOOTPRINT_CONTACT_M),
            "boundary_clearance_all": _q(geom["boundary_all"]), "boundary_clearance_wood": _q(geom["boundary_wood"]),
            "share_boundary_all_gt_far": _share(geom["boundary_all"], FAR_M),
            "share_boundary_wood_gt_far": _share(geom["boundary_wood"], FAR_M),
            "boundary_clearance_column_all": _q(geom["boundary_column_all"]),
        },
        "mec_circle": {
            "excess_all": _q(mec_excess), "share_excess_all_gt_contact": _share(mec_excess, FOOTPRINT_CONTACT_M),
            "boundary_clearance_all": _q(geom["mec_boundary_all"]), "boundary_clearance_wood": _q(geom["mec_boundary_wood"]),
            "share_boundary_all_gt_far": _share(geom["mec_boundary_all"], FAR_M),
            "share_boundary_wood_gt_far": _share(geom["mec_boundary_wood"], FAR_M),
            "boundary_clearance_column_all": _q(geom["mec_boundary_column_all"]),
        },
        "per_azimuth": {"bin_deg": 360 / EXTENT_AZIMUTHS, "extent_all_m": r(geom["extent_all"]),
                        "extent_wood_m": r(geom["extent_wood"]), "boundary_clearance_all_m": r(geom["boundary_all"]),
                        "boundary_clearance_wood_m": r(geom["boundary_wood"]), "mec_extent_all_m": r(geom["mec_extent_all"]),
                        "mec_boundary_clearance_all_m": r(geom["mec_boundary_all"]),
                        "mec_boundary_clearance_wood_m": r(geom["mec_boundary_wood"]),
                        "boundary_clearance_column_all_m": r(geom["boundary_column_all"]),
                        "mec_boundary_clearance_column_all_m": r(geom["mec_boundary_column_all"])},
    }


# ----------------------------------------------------------------------------------------------------------------------
# The whole measurement.
# ----------------------------------------------------------------------------------------------------------------------
def _window_check_poses(rows: list[dict]) -> list[tuple[int, Pose]]:
    """Three endings for window_depths (the nearest, the median and the farthest geometry), each with the camera at the
    drone's centre at the height of its nearest hit, facing it."""
    found = [(i, r) for i, r in enumerate(rows) if r["clearance_final"]["all"]["min"] is not None]
    if not found:
        return []
    found.sort(key=lambda ir: ir[1]["clearance_final"]["all"]["min"])
    picks = list(dict.fromkeys([found[0][0], found[len(found) // 2][0], found[-1][0]]))
    out = []
    for i in picks:
        r = rows[i]
        x, y, z, yaw = r["final_pose"]
        near = r["clearance_final"]["all"]
        dz = dict(HEIGHTS)[near["nearest_height"]]
        out.append((i, Pose(x, y, z - dz, wrap_pi(yaw + math.radians(near["nearest_bearing_deg"])))))
    return out


def asset_summary(asset: str, tree_rows: list[dict], geoms: dict, rows: list[dict], circles: dict) -> dict:
    """An asset's instances pooled: the endings' clearances, its circles, and the excess and boundary clearance over
    every instance's azimuths."""
    mine = [t for t in tree_rows if t["asset"] == asset]
    pool = lambda key: np.concatenate([geoms[t["tag"]][key] for t in mine])  # noqa: E731
    radius = np.repeat([t["radius_m"] for t in mine], EXTENT_AZIMUTHS)
    mec_r = np.repeat([t["mec_all"]["radius_m"] for t in mine], EXTENT_AZIMUTHS)
    endings = [r for r in rows if r["asset"] == asset]
    pick = lambda path: [path(r) for r in endings]  # noqa: E731
    return {
        "instances": len(mine), "endings": len(endings),
        "unit_scale": {"pivot_radius_m": round(circles["pivot_radius_m"], 4),
                       "mec_all": {"radius_m": round(circles["all"]["radius_m"], 4),
                                   "centre_offset_m": round(float(np.hypot(*circles["all"]["centre_local"])), 4)},
                       "mec_wood": {"radius_m": round(circles["wood"]["radius_m"], 4),
                                    "centre_offset_m": round(float(np.hypot(*circles["wood"]["centre_local"])), 4)}},
        "radius_m": _q([t["radius_m"] for t in mine]),
        "mec_radius_m": _q([t["mec_all"]["radius_m"] for t in mine]),
        "mec_centre_offset_m": _q([t["mec_all"]["offset_m"] for t in mine]),
        "area_reduction": _q([t["area_reduction"] for t in mine]),
        "pivot_circle": {
            "excess_all": _q(radius - pool("extent_all")), "excess_wood": _q(radius - pool("extent_wood")),
            "share_excess_all_gt_contact": _share(radius - pool("extent_all"), FOOTPRINT_CONTACT_M),
            "share_excess_wood_gt_contact": _share(radius - pool("extent_wood"), FOOTPRINT_CONTACT_M),
            "boundary_clearance_all": _q(pool("boundary_all")), "boundary_clearance_wood": _q(pool("boundary_wood")),
            "share_boundary_all_gt_far": _share(pool("boundary_all"), FAR_M),
            "share_boundary_wood_gt_far": _share(pool("boundary_wood"), FAR_M),
            "boundary_clearance_column_all": _q(pool("boundary_column_all")),
        },
        "mec_circle": {
            "excess_all": _q(mec_r - pool("mec_extent_all")),
            "share_excess_all_gt_contact": _share(mec_r - pool("mec_extent_all"), FOOTPRINT_CONTACT_M),
            "boundary_clearance_all": _q(pool("mec_boundary_all")), "boundary_clearance_wood": _q(pool("mec_boundary_wood")),
            "share_boundary_all_gt_far": _share(pool("mec_boundary_all"), FAR_M),
            "share_boundary_wood_gt_far": _share(pool("mec_boundary_wood"), FAR_M),
            "boundary_clearance_column_all": _q(pool("mec_boundary_column_all")),
        },
        "endings_nearest_in_forward_view": sum(1 for r in endings if r["nearest_in_forward_view"]),
        "endings_by_side_of_lean_deg": _bucket([r["lean_angle_deg"] for r in endings], edges=(45.0, 90.0, 135.0)),
        "endings_outside_mec_rule": sum(1 for r in endings if r["mec_gap_final_m"] > FOOTPRINT_CONTACT_M),
        "ending_clearance": {
            "final_all": _q(pick(lambda r: r["clearance_final"]["all"]["min"])),
            "final_wood": _q(pick(lambda r: r["clearance_final"]["wood"]["min"])),
            "start_all": _q(pick(lambda r: r["clearance_start"]["all"]["min"])),
            "start_wood": _q(pick(lambda r: r["clearance_start"]["wood"]["min"])),
            "segment_min_all": _q(pick(lambda r: r["clearance_segment_min"]["all"])),
            "segment_min_wood": _q(pick(lambda r: r["clearance_segment_min"]["wood"])),
            "slab_all": _q(pick(lambda r: r["clearance_slab"]["all"])),
            "slab_wood": _q(pick(lambda r: r["clearance_slab"]["wood"])),
            "column_all": _q(pick(lambda r: r["clearance_column"]["all"])),
            "band_column_all": _q(pick(lambda r: r["clearance_band_column"]["all"])),
            "nearest_3d_all": _q(pick(lambda r: r["clearance_3d"]["all"])),
            "nearest_3d_wood": _q(pick(lambda r: r["clearance_3d"]["wood"])),
            "mec_gap_final": _q(pick(lambda r: r["mec_gap_final_m"])),
        },
    }


def _bucket(values, edges=(DRONE_HALF_SPAN_M, FOOTPRINT_CONTACT_M, 1.0, FAR_M, 2.0)) -> dict:
    """How many values fall below each line, between them, above the last, or are none (no geometry at that height)."""
    out: dict = {}
    lo = 0.0
    for e in edges:
        out[f"[{lo:g}, {e:g})"] = sum(1 for v in values if v is not None and lo <= v < e)
        lo = e
    out[f">= {lo:g}"] = sum(1 for v in values if v is not None and v >= lo)
    out["none at the body's heights"] = sum(1 for v in values if v is None)
    return out


def compute(run_root: Path, layout_path: Path, band_m: tuple[float, float] = BAND_M) -> tuple[dict, dict]:
    """Everything the figure draws and the JSON keeps. Returns (record, drawing data)."""
    endings, flights = load_endings(run_root)
    data, layout_sha, trees, meshes, library = load_trees(layout_path)
    by_tag = {t["tag"]: t for t in trees}
    missing = sorted({e["tag"] for e in endings} - set(by_tag))
    if missing:
        raise SystemExit(f"endings name trees the layout does not hold as footprint obstacles: {missing}")
    assets = sorted({t["asset"] for t in trees})
    local = {a: corners(library[a]) for a in assets}
    wood = {a: np.tile(library[a].opaque, 3) for a in assets}
    circles = {a: asset_circles(local[a], wood[a]) for a in assets}
    rng = np.random.default_rng(0)
    rows: list[dict | None] = [None] * len(endings)
    at_tree = defaultdict(list)
    for i, e in enumerate(endings):
        at_tree[e["tag"]].append(i)
    geoms: dict[str, dict] = {}
    tree_rows: list[dict] = []
    round_trip = []
    for tree in trees:
        mesh, asset = meshes[tree["tag"]], tree["asset"]
        round_trip.append(round_trip_error_m(mesh, local[asset], rng))
        geom, world = tree_geometry(mesh, tree, local[asset], wood[asset], circles[asset], band_m)
        for i in at_tree[tree["tag"]]:
            rows[i] = ending_record(endings[i], mesh, by_tag[tree["tag"]], geom, world, wood[asset])
        geoms[tree["tag"]] = geom
        tree_rows.append(tree_summary(tree, geom))
        del world
    window = []
    for i, pose in _window_check_poses(rows):
        tag = rows[i]["tag"]
        asset = by_tag[tag]["asset"]
        window.append({"ending": i, "tag": tag, **window_depth_check(meshes[tag], library[asset], local[asset], pose)})
    # The vertex column holds every height, so it is never farther than the ray clearance at one height, give or take
    # the thinning (2.8 cm) and a corner standing for its triangle (half an edge: 11 cm at most on these models).
    column_excess = [r["clearance_column"]["all"] - r["clearance_final"]["all"]["at_altitude"] for r in rows
                     if r["clearance_final"]["all"]["at_altitude"] is not None]
    slab_excess = [r["clearance_slab"]["all"] - r["clearance_final"]["all"]["min"] for r in rows
                   if r["clearance_final"]["all"]["min"] is not None and r["clearance_slab"]["all"] is not None]
    checks = {
        "round_trip_max_error_m": max(round_trip),
        "window_depths": window,
        "window_depths_agree": all(w[k]["hit_pattern_equal"] and (w[k]["max_abs_diff_m"] or 0.0) < 1e-3
                                   for w in window for k in ("all", "wood")),
        "recorded_gap_is_final_gap_max_diff_m": max((abs(r["circle_gap_final_m"] - r["recorded_gap_m"]) for r in rows),
                                                    default=0.0),
        "column_minus_ray_clearance_max_m": round(max(column_excess), 4) if column_excess else None,
        "slab_minus_ray_clearance_max_m": round(max(slab_excess), 4) if slab_excess else None,
    }
    checks["passed"] = (checks["round_trip_max_error_m"] < 1e-6 and checks["window_depths_agree"]
                        and (checks["column_minus_ray_clearance_max_m"] or 0.0) < 0.15
                        and (checks["slab_minus_ray_clearance_max_m"] or 0.0) < 0.15)
    final_all = [r["clearance_final"]["all"]["min"] for r in rows]
    final_wood = [r["clearance_final"]["wood"]["min"] for r in rows]
    summary = {
        "endings": len(rows), "flights": flights["flights"],
        "endings_by_asset": dict(Counter(r["asset"] for r in rows).most_common()),
        "clearance_final_all": _q(final_all), "clearance_final_wood": _q(final_wood),
        "clearance_start_all": _q([r["clearance_start"]["all"]["min"] for r in rows]),
        "clearance_start_wood": _q([r["clearance_start"]["wood"]["min"] for r in rows]),
        "clearance_segment_min_all": _q([r["clearance_segment_min"]["all"] for r in rows]),
        "clearance_segment_min_wood": _q([r["clearance_segment_min"]["wood"] for r in rows]),
        "clearance_slab_all": _q([r["clearance_slab"]["all"] for r in rows]),
        "clearance_slab_wood": _q([r["clearance_slab"]["wood"] for r in rows]),
        "buckets_slab_all": _bucket([r["clearance_slab"]["all"] for r in rows]),
        "buckets_slab_wood": _bucket([r["clearance_slab"]["wood"] for r in rows]),
        "clearance_column_all": _q([r["clearance_column"]["all"] for r in rows]),
        "clearance_column_wood": _q([r["clearance_column"]["wood"] for r in rows]),
        "clearance_3d_all": _q([r["clearance_3d"]["all"] for r in rows]),
        "clearance_3d_wood": _q([r["clearance_3d"]["wood"] for r in rows]),
        "buckets_final_all": _bucket(final_all), "buckets_final_wood": _bucket(final_wood),
        "buckets_segment_min_wood": _bucket([r["clearance_segment_min"]["wood"] for r in rows]),
        "nearest_all_in_forward_view": sum(1 for r in rows if r["nearest_in_forward_view"]),
        "wood_within_rotor_reach": [i for i, r in enumerate(rows) if r["clearance_segment_min"]["wood"] is not None
                                    and r["clearance_segment_min"]["wood"] < DRONE_HALF_SPAN_M],
        "outside_mec_rule": sum(1 for r in rows if r["mec_gap_final_m"] > FOOTPRINT_CONTACT_M),
        "far_side_of_lean": _bucket([r["lean_angle_deg"] for r in rows], edges=(45.0, 90.0, 135.0)),
    }
    record = {
        "what": "footprint-rule endings of a run's deterministic eval-watch flights, measured against the trees' geometry",
        "run_root": str(run_root), "eval_watch": flights,
        "layout": {"path": str(layout_path), "sha256": layout_sha, "scene_id": data["layout"]["scene_id"],
                   "trees": len(trees)},
        "rule": {"contact_m": FOOTPRINT_CONTACT_M, "drone_half_span_m": DRONE_HALF_SPAN_M, "step_m": STEP_M},
        "method": {"ray_azimuths": RAY_AZIMUTHS, "heights_m": {k: v for k, v in HEIGHTS},
                   "segment_back_m": list(SEGMENT_BACK_M), "extent_azimuths": EXTENT_AZIMUTHS, "grid_m": GRID_M,
                   "band_m": list(band_m), "far_m": FAR_M,
                   "notes": ["clearances are to the named tree only, from the drone's centre; 'all' is wood plus leaf "
                             "cards (what the depth camera sees), 'wood' the opaque triangles (what collides)",
                             "'start' is the final pose moved back along its yaw by a full-speed step; the true start "
                             "and the altitude it had are not recorded",
                             "extents and boundary clearances use the band's triangle corners (with altitude in the "
                             "band) thinned to one per grid cell; excess = circle radius - extent in that azimuth bin, "
                             "0-extent bins (nothing there) count the whole radius",
                             "mec = minimum enclosing circle of the model's horizontal projection over its whole "
                             "height (all corners; 'wood' its opaque ones)"]},
        "checks": checks,
        "summary": summary,
        "assets": {a: asset_summary(a, tree_rows, geoms, rows, circles[a]) for a in assets},
        "endings": rows,
        "trees": tree_rows,
    }
    return record, {"rows": rows, "trees": by_tag, "geoms": geoms, "meshes": meshes, "local": local, "wood": wood,
                    "summary": summary}


# ----------------------------------------------------------------------------------------------------------------------
# Drawing. Maps are top-down in NED: x (north) up the page, y (east) across it.
# ----------------------------------------------------------------------------------------------------------------------
TREE_FILL, TREE_EDGE, WOOD = "#b7e4a7", "#4f8a3c", "#8c510a"  # wood + leaf cards, its edge, wood alone
SLICE = "#173f12"  # the geometry within the body's half-thickness of the drone's altitude
RULE, MEC_COLOUR, ENDING, DRONE = "#333333", "#2a78d6", "#d62828", "#3a5a8c"
FAR_SHADE, MUTED = "#eef1f4", "#555555"
DRAW_GRID_M = 0.04  # the drawn projections: one point per 4 cm cell


def _asset_order(record: dict) -> list[str]:
    """Most off-centre first: the enclosing circle's centre offset from the pivot at unit scale (the lean)."""
    assets = record["assets"]
    return sorted(assets, key=lambda a: -assets[a]["unit_scale"]["mec_all"]["centre_offset_m"])


def _lean(record: dict, asset: str) -> float:
    return record["assets"][asset]["unit_scale"]["mec_all"]["centre_offset_m"]


def _histogram(ax, rows: list[dict]) -> None:
    all_v = [r["clearance_final"]["all"]["min"] for r in rows]
    slab = [r["clearance_slab"]["all"] for r in rows if r["clearance_slab"]["all"] is not None]
    wood_v = [r["clearance_final"]["wood"]["min"] for r in rows]
    finite = [v for v in all_v + wood_v if v is not None]
    hi = max(4.5, math.ceil(max(finite, default=0.0) * 2) / 2 + 0.2)
    bins = np.arange(0.0, hi + 1e-9, 0.1)
    ax.axvspan(FAR_M, hi, color=FAR_SHADE, zorder=0)
    counts, _, _ = ax.hist([v for v in all_v if v is not None], bins=bins, color=TREE_EDGE, edgecolor="white",
                           linewidth=0.6, zorder=2, label="to wood or leaf cards (camera sees)")
    wc, _, _ = ax.hist([v for v in wood_v if v is not None], bins=bins, histtype="step", color=WOOD, linewidth=1.6,
                       zorder=3, label="to wood alone (collides)")
    x_none = hi + 0.35
    n_all, n_wood = all_v.count(None), wood_v.count(None)
    ax.bar(x_none, n_all, width=0.1, color=TREE_EDGE, zorder=2)
    ax.bar(x_none, n_wood, width=0.1, fill=False, edgecolor=WOOD, linewidth=1.6, zorder=3)
    top = max(counts.max(initial=0), wc.max(initial=0), n_all, n_wood, 1)
    ax.set_ylim(0, top * 1.6)
    for x, style, text in ((DRONE_HALF_SPAN_M, (0, (1, 1.5)), f"rotor reach {DRONE_HALF_SPAN_M:g} m"),
                           (FOOTPRINT_CONTACT_M, (0, (4, 2)), f"rule {FOOTPRINT_CONTACT_M:g} m")):
        ax.axvline(x, color=RULE, linestyle=style, linewidth=1.2, zorder=4)
        ax.text(x - 0.03, 0.03, text, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="bottom",
                fontsize=7, color=RULE)
    near = sum(1 for v in all_v if v is not None and v < 1.0)
    far = sum(1 for v in all_v if v is not None and v >= FAR_M)
    slab_note = f"\nwhole ±0.25 m slab: median {np.median(slab):.2f}, min {min(slab):.2f} m" if slab else ""
    ax.text(hi - 0.05, 0.97, f"> {FAR_M:g} m from anything: {far} of {len(rows)}\nwithin 1 m: {near}\nnothing at its "
            f"heights: {n_all}{slab_note}", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=7.5,
            color="#222222")
    ticks = list(np.arange(0.0, hi + 1e-9, 0.5))
    ax.set_xticks(ticks + [x_none], [f"{t:g}" for t in ticks] + ["none"])
    ax.set_xlim(0, x_none + 0.15)
    ax.set_xlabel("clearance at the ending (m): nearest over the drone's altitude and ±0.25 m", fontsize=8)
    ax.set_ylabel("endings", fontsize=8.5)
    ax.set_title(f"(a) How far the drone was from the tree when the rule ended the flight ({len(rows)} endings)",
                 fontsize=9.5, loc="left")
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95, borderaxespad=0.3)


def _asset_strip(ax, rows: list[dict], record: dict, order: list[str]) -> None:
    """Each asset's endings: clearance to wood or cards (filled) and to wood alone (hollow), one row per asset."""
    rng = np.random.default_rng(1)
    hi = 4.6
    ax.axvspan(FAR_M, hi + 0.5, color=FAR_SHADE, zorder=0)
    for x, style in ((DRONE_HALF_SPAN_M, (0, (1, 1.5))), (FOOTPRINT_CONTACT_M, (0, (4, 2)))):
        ax.axvline(x, color=RULE, linestyle=style, linewidth=1.0, zorder=1)
    labels = []
    for k, asset in enumerate(order):
        mine = [r for r in rows if r["asset"] == asset]
        jitter = rng.uniform(-0.12, 0.12, len(mine))
        for label, dy, kw in (("all", -0.16, dict(s=16, color=TREE_EDGE, edgecolors="white", linewidths=0.4)),
                              ("wood", 0.16, dict(s=14, facecolors="none", edgecolors=WOOD, linewidths=1.1, marker="s"))):
            vals = [r["clearance_final"][label]["min"] for r in mine]
            ax.scatter([v if v is not None else hi for v in vals], k + dy + jitter, zorder=3, **kw)
        vals = [r["clearance_final"]["all"]["min"] for r in mine if r["clearance_final"]["all"]["min"] is not None]
        med = float(np.median(vals)) if vals else math.nan
        if vals:
            ax.plot([med, med], [k - 0.38, k + 0.38], color="#111111", linewidth=1.4, zorder=4)
        labels.append(f"{asset} (lean {_lean(record, asset):.2f})\n{len(mine)} endings, med {med:.1f} m")
    ax.set_yticks(range(len(order)), labels, fontsize=6.8)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, hi + 0.2)
    ax.set_xticks(list(np.arange(0, 4.6, 1.0)) + [hi], [f"{t:g}" for t in np.arange(0, 4.6, 1.0)] + ["none"])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("clearance at the ending (m): ● wood or cards, □ wood alone, bar = median", fontsize=8)
    ax.set_title("(a) per asset; lean = enclosing circle's centre off the pivot", fontsize=9.5, loc="left")


def _boundary_ranges(ax, record: dict, order: list[str]) -> None:
    """Where each circle's rule boundary lies from the band's geometry (wood or cards), over every instance's azimuths:
    the pivot circle the rule uses against the minimum enclosing circle."""
    for k, asset in enumerate(order):
        a = record["assets"][asset]
        for dy, key, colour, label in ((-0.15, "pivot_circle", TREE_EDGE, "pivot circle (the rule today)"),
                                       (0.15, "mec_circle", MEC_COLOUR, "minimum enclosing circle")):
            q = a[key]["boundary_clearance_all"]
            ax.plot([q["min"], q["max"]], [k + dy] * 2, color=colour, linewidth=0.8, zorder=2)
            ax.plot([q["q1"], q["q3"]], [k + dy] * 2, color=colour, linewidth=4.0, solid_capstyle="butt", zorder=3,
                    label=label if k == 0 else None)
            ax.plot(q["median"], k + dy, "o", color="white", markeredgecolor=colour, markersize=4.5, zorder=4)
            ax.text(3.75, k + dy, f"{100 * a[key]['share_boundary_all_gt_far']:.0f}%", fontsize=6.5, va="center",
                    ha="right", color="#222222")
    ax.axvline(FOOTPRINT_CONTACT_M, color=RULE, linestyle=(0, (4, 2)), linewidth=1.0)
    ax.axvline(FAR_M, color=RULE, linestyle=(0, (1, 1.5)), linewidth=1.0)
    ax.text(3.75, -0.55, f"share > {FAR_M:g} m", fontsize=6.5, ha="right", va="bottom", color="#222222")
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.set_ylim(len(order) - 0.5, -1.0)
    ax.set_xlim(0.6, 3.8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("boundary to the nearest band geometry (m)", fontsize=8)
    ax.set_title("(a) how far each circle's boundary lies from the leaves", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.36, 0.0), framealpha=0.95)


def _tree_layers(ax, tree: dict, geom: dict, *, slice_xy: np.ndarray | None = None) -> None:
    """A tree top-down: its band projection (wood or cards, wood), the pivot circle and its rule boundary, the minimum
    enclosing circle and its boundary."""
    allp, woodp = grid_points(geom["band_all"], DRAW_GRID_M), grid_points(geom["band_wood"], DRAW_GRID_M)
    # the leaves as a halo under the wood, so where they reach past it shows round its edge
    ax.scatter(allp[:, 1], allp[:, 0], s=3.0, color=TREE_EDGE, alpha=0.35, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(woodp[:, 1], woodp[:, 0], s=0.6, color=WOOD, alpha=0.9, linewidths=0, rasterized=True, zorder=2)
    if slice_xy is not None and len(slice_xy):
        ax.scatter(slice_xy[:, 1], slice_xy[:, 0], s=1.2, color=SLICE, linewidths=0, rasterized=True, zorder=3)
    r = float(tree["radius_m"])
    ax.add_patch(plt.Circle((tree["y"], tree["x"]), r, fill=False, edgecolor=TREE_EDGE, linewidth=1.3, zorder=4))
    ax.add_patch(plt.Circle((tree["y"], tree["x"]), r + FOOTPRINT_CONTACT_M, fill=False, edgecolor=RULE,
                            linestyle=(0, (4, 2)), linewidth=1.1, zorder=4))
    mec = geom["mec_all"]
    ax.add_patch(plt.Circle((mec["centre"][1], mec["centre"][0]), mec["radius_m"], fill=False, edgecolor=MEC_COLOUR,
                            linewidth=1.1, zorder=4))
    ax.add_patch(plt.Circle((mec["centre"][1], mec["centre"][0]), mec["radius_m"] + FOOTPRINT_CONTACT_M, fill=False,
                            edgecolor=MEC_COLOUR, linestyle=(0, (4, 2)), linewidth=0.8, zorder=4))
    ax.plot(tree["y"], tree["x"], "o", color="black", markersize=3, zorder=5)
    ax.plot(mec["centre"][1], mec["centre"][0], "+", color=MEC_COLOUR, markersize=6, markeredgewidth=1.3, zorder=5)


def _square(ax, cx: float, cy: float, half: float) -> None:
    ax.set_xlim(cy - half, cy + half)
    ax.set_ylim(cx - half, cx + half)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6.5)


def _asset_panel(ax, asset: str, record: dict, drawing: dict) -> None:
    """One instance of `asset` (the one the most flights ended at), with those endings."""
    rows = drawing["rows"]
    tags = [t["tag"] for t in record["trees"] if t["asset"] == asset]
    counts = Counter(r["tag"] for r in rows if r["asset"] == asset)
    tag = max(tags, key=lambda t: (counts.get(t, 0), -tags.index(t)))
    tree, geom = drawing["trees"][tag], drawing["geoms"][tag]
    _tree_layers(ax, tree, geom)
    for r in rows:
        if r["tag"] == tag:
            ax.plot(r["final_pose"][1], r["final_pose"][0], "x", color=ENDING, markersize=5, markeredgewidth=1.4, zorder=6)
    row = next(t for t in record["trees"] if t["tag"] == tag)
    a = record["assets"][asset]
    _square(ax, tree["x"], tree["y"], tree["radius_m"] + FOOTPRINT_CONTACT_M + 0.35)
    ax.set_title(f"{asset} ({tag}, {counts.get(tag, 0)} endings)\ncircle {tree['radius_m']:.2f} m → enclosing "
                 f"{row['mec_all']['radius_m']:.2f} m\ncentre {row['mec_all']['offset_m']:.2f} m off pivot, area "
                 f"−{100 * row['area_reduction']:.0f}%\nboundary > {FAR_M:g} m from leaves: "
                 f"{100 * a['pivot_circle']['share_boundary_all_gt_far']:.0f}% → "
                 f"{100 * a['mec_circle']['share_boundary_all_gt_far']:.0f}%", fontsize=7)


def _close_up(ax, row: dict, drawing: dict, band_m: tuple[float, float]) -> None:
    """One ending: the tree's band projection, the slice at the drone's altitude, the circles, the drone and its last
    step, and the line to the nearest geometry."""
    tag = row["tag"]
    tree, geom, mesh = drawing["trees"][tag], drawing["geoms"][tag], drawing["meshes"][tag]
    x, y, z, yaw = row["final_pose"]
    altitude = -z
    world = gltf_to_world(mesh, drawing["local"][tree["asset"]])
    near = np.abs(-world[:, 2] - altitude) <= BODY_HALF_THICKNESS_M
    _tree_layers(ax, tree, geom, slice_xy=grid_points(world[near, :2], DRAW_GRID_M))
    del world
    ax.add_patch(plt.Circle((y, x), DRONE_HALF_SPAN_M, fill=False, edgecolor=DRONE, linewidth=1.5, zorder=7))
    hx, hy = math.cos(yaw), math.sin(yaw)
    ax.plot([y - STEP_M * hy, y], [x - STEP_M * hx, x], color=DRONE, linewidth=2.6, solid_capstyle="round", zorder=7)
    ax.annotate("", (y + 0.9 * hy, x + 0.9 * hx), (y, x), zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=DRONE, linewidth=1.3, mutation_scale=9))
    best = row["clearance_final"]["all"]
    if best["min"] is not None:
        az = yaw + math.radians(best["nearest_bearing_deg"])
        hit = (x + best["min"] * math.cos(az), y + best["min"] * math.sin(az))
        ax.plot([y, hit[1]], [x, hit[0]], color=ENDING, linewidth=1.0, linestyle=(0, (2, 1.5)), zorder=8)
        ax.plot(hit[1], hit[0], "o", color=ENDING, markersize=3.5, zorder=8)
        ax.text((y + hit[1]) / 2, (x + hit[0]) / 2, f" {best['min']:.2f} m", fontsize=7, color=ENDING, zorder=9,
                ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8))
    _square(ax, tree["x"], tree["y"], tree["radius_m"] + FOOTPRINT_CONTACT_M + 0.35)
    wood = row["clearance_final"]["wood"]["min"]
    ax.set_title(f"{tag} {tree['asset']}: {row['checkpoint']}, seed {row['seed']}\nat {altitude:.2f} m, "
                 f"{row['circle_gap_final_m']:.2f} m from the circle; nearest leaf {best['min'] or float('nan'):.2f} m,"
                 f"\nwood {wood if wood is not None else float('nan'):.2f} m, bearing "
                 f"{best['nearest_bearing_deg'] if best['nearest_bearing_deg'] is not None else float('nan'):+.0f}° "
                 f"from the heading", fontsize=7.2)


def draw(out: Path, record: dict, drawing: dict, band_m: tuple[float, float]) -> None:
    rows, s = drawing["rows"], record["summary"]
    order = _asset_order(record)
    fig = plt.figure(figsize=(16, 10))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.12, 1.18], hspace=0.42, left=0.045, right=0.99, top=0.875,
                             bottom=0.075)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.2, 1.0, 0.95], wspace=0.45)
    _histogram(fig.add_subplot(top[0]), rows)
    _asset_strip(fig.add_subplot(top[1]), rows, record, order)
    _boundary_ranges(fig.add_subplot(top[2]), record, order)
    mid = outer[1].subgridspec(1, len(order), wspace=0.32)
    for k, asset in enumerate(order):
        ax = fig.add_subplot(mid[k])
        _asset_panel(ax, asset, record, drawing)
        if k == 0:
            ax.set_ylabel("(b) one instance per asset\nx (m, north)", fontsize=8)
    worst = sorted((r for r in rows if r["clearance_final"]["all"]["min"] is not None),
                   key=lambda r: -r["clearance_final"]["all"]["min"])[:4]
    bottom = outer[2].subgridspec(1, max(len(worst), 1), wspace=0.18)
    for k, row in enumerate(worst):
        ax = fig.add_subplot(bottom[k])
        _close_up(ax, row, drawing, band_m)
        if k == 0:
            ax.set_ylabel("(c) the 4 endings farthest from the tree\nx (m, north)", fontsize=8)
    lean = [a for a in order if _lean(record, a) >= 0.3]
    lean_rows = [r for r in rows if r["asset"] in lean]
    other_rows = [r for r in rows if r["asset"] not in lean]
    med = lambda rs: float(np.median([r["clearance_final"]["all"]["min"] for r in rs  # noqa: E731
                                      if r["clearance_final"]["all"]["min"] is not None] or [math.nan]))
    fig.suptitle(f"Footprint-rule endings on {record['layout']['scene_id']} ({Path(record['run_root']).name}): "
                 f"{s['endings']} of {s['flights']} deterministic evaluation flights, measured against the trees' "
                 f"actual geometry", fontsize=13, y=0.988)
    fig.text(0.5, 0.952, f"Leaning species ({', '.join(lean)}): {len(lean_rows)} endings, median {med(lean_rows):.2f} m "
             f"from the nearest leaf or wood at the drone's altitude. Upright species: {len(other_rows)} endings, median "
             f"{med(other_rows):.2f} m.", ha="center", fontsize=9.5, color="#222222")
    fig.text(0.5, 0.934, f"No ending had wood within the rotor reach ({DRONE_HALF_SPAN_M:g} m). {s['outside_mec_rule']} of "
             f"{s['endings']} endings lie more than {FOOTPRINT_CONTACT_M:g} m outside the minimum enclosing circle: a rule "
             f"on that circle would not have ended them there.", ha="center", fontsize=9.5, color="#222222")
    fig.text(0.5, 0.916, "Clearances: validate/obstacles.py's glTF ray caster, 720 horizontal rays from the drone's "
             "centre at its altitude and ±0.25 m, to the tree the ending names. Maps: NED, x north up, y east "
             f"across; band = {band_m[0]:g}-{band_m[1]:g} m altitude.", ha="center", fontsize=8, color=MUTED)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=TREE_EDGE, alpha=0.6, markersize=5,
               label=f"wood + leaf cards in the band ({band_m[0]:g}-{band_m[1]:g} m)"),
        Line2D([], [], marker="o", linestyle="none", color=WOOD, markersize=5, label="wood in the band"),
        Line2D([], [], marker="o", linestyle="none", color=SLICE, markersize=5, label="within ±0.25 m of the drone (c)"),
        Line2D([], [], color=TREE_EDGE, linewidth=1.3, label="footprint circle (radius_m, about the pivot •)"),
        Line2D([], [], color=RULE, linestyle=(0, (4, 2)), label=f"rule boundary (+{FOOTPRINT_CONTACT_M:g} m)"),
        Line2D([], [], color=MEC_COLOUR, linewidth=1.1, label="minimum enclosing circle (+ centre), dashed: its boundary"),
        Line2D([], [], color=ENDING, marker="x", linestyle="none", markersize=6, markeredgewidth=1.4,
               label="where a flight ended"),
        Line2D([], [], color=DRONE, linewidth=1.5, marker="o", markerfacecolor="none", markersize=9,
               label=f"drone: rotor span, heading, last {STEP_M:g} m"),
        Line2D([], [], color=ENDING, linestyle=(0, (2, 1.5)), marker="o", markersize=3.5, label="nearest geometry"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def dumps(record: dict) -> str:
    """Indented JSON with each list of plain numbers kept on one line (the per-azimuth arrays are 360 long)."""
    text = json.dumps(record, indent=1, allow_nan=False)
    number = r"(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|null|true|false)"
    return re.sub(rf"\[\n\s*({number}(?:,\n\s*{number})*)\n\s*\]",
                  lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="training run whose eval_watch/ to read")
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT, help="stored layout (runs/levels/<id>.layout.json)")
    ap.add_argument("--out", type=Path, required=True, help="PNG to write")
    ap.add_argument("--json", type=Path, required=True, help="every number the figure draws")
    ap.add_argument("--band", type=float, nargs=2, default=BAND_M, metavar=("LO", "HI"), help="flight band (m)")
    args = ap.parse_args(argv)
    if not args.band[0] < args.band[1]:
        ap.error(f"--band must be increasing, got {args.band}")
    band = (float(args.band[0]), float(args.band[1]))
    record, drawing = compute(args.run_root, args.layout, band)
    if not record["endings"]:
        print(f"no footprint ending in {args.run_root}/eval_watch: nothing to measure", file=sys.stderr)
        return 2
    record["figure"] = str(args.out)
    draw(args.out, record, drawing, band)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    text = dumps(record)
    if json.loads(text) != json.loads(json.dumps(record)):
        raise AssertionError("the compact JSON does not read back as the record")
    args.json.write_text(text + "\n")
    s = record["summary"]
    print(f"{s['endings']} footprint endings of {s['flights']} flights; clearance to wood or cards at the ending "
          f"{s['clearance_final_all']}; to wood {s['clearance_final_wood']}; checks passed: {record['checks']['passed']}")
    print(f"wrote {args.out} and {args.json}")
    return 0 if record["checks"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
