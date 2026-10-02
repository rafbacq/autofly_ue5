"""Episode setup sampling (spec §9, steps 1-3): where an episode starts, where its target and distractors go.

`sample_setup` is pure with respect to the simulator -- it only reads the scene/layout and draws from an
`np.random.Generator` -- so a seed reproduces exactly and M3's dataset collector can import this module
unchanged: nothing here takes a training-specific argument or touches an RL type. `apply_setup` and
`clear_setup` are the only functions that touch a `Simulator`, and they are a thin, reversible spawn/destroy
pair so a failed episode can always be cleaned up.

The scene's start/target bands are measured as distance to the NEAREST of the four edges (see
`reachability.band_mask`/`edge_distances`), so they are rings, not sides. A start and target drawn
independently from their rings can therefore land on the same side of the scene and produce a one-metre
"episode". This module avoids that by drawing an edge first (for both start and target, with the target
forced to the opposite edge) and then a position along that specific edge's band -- see `_sample_band_point`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autofly_ue5.scenes.model import Bounds, Instance, Layout, SceneFile, load_registry
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.types import ObjectNotFoundError, Pose

EDGES = ("x_min", "x_max", "y_min", "y_max")
OPPOSITE = {"x_min": "x_max", "x_max": "x_min", "y_min": "y_max", "y_max": "y_min"}

# Spec §9.3: instructions are copied verbatim from the real released episodes, misspelling included -- the
# student model must see the same text distribution the released AutoFly dataset was trained on. Do not
# "fix" `avioding`.
INSTRUCTION_TEMPLATES = (
    "go through and avoid the {obstacle} or other obstacles to reach the {target}",
    "advance to {target} while avioding {obstacle} and other obstacles",
)

DRONE_RADIUS_M = 0.4
CLEARANCE_M = 1.0

# Task 3's reward/termination rules (autofly_ue5/expert/reward.py) treat the altitude band as a *hard*
# boundary with zero margin: `classify()` returns OUT_OF_BOUNDS the instant altitude leaves [lo, hi], no
# grace region. With v_z in [-1, 1] m/s over a CONTROL_DT_S = 0.2 s step, one step covers 0.2 m. A start
# drawn from within centimetres of either edge of the band could therefore end the episode on its very
# first downward step, through no fault of the policy -- early training would be dominated by terminations
# the agent could not have avoided. So the START altitude is drawn from the band's *interior*, inset by
# this margin at each end; the band itself -- where the drone MAY FLY, and where termination is checked --
# stays the full spec range (spec unchanged, reward.py unchanged). Only where an episode MAY BEGIN is
# narrowed. Derived from a constant, not hard-coded per scene, because other scenes have other bands.
START_ALTITUDE_MARGIN_M = 0.3

N_DISTRACTORS_DEFAULT = (3, 5)
DISTRACTOR_SPACING_M = 4.0
MAX_REJECTION_TRIES = 200

# A distractor closer than this to the drone's start pose can spawn on top of it: `spawn_clearance_m()`
# only keeps a point clear of the layout's own OBSTACLES, not of the start point itself, so nothing
# previously stopped a distractor from landing inside collision range of the drone. That makes the
# episode unwinnable from step 0 regardless of the policy, which silently eats into the M2 gate's
# success-rate budget (measured: about 1 seed in 200 without this keepout, some under 0.1 m). 8 m sits
# outside any spawn/collision risk and outside the 5 m success radius, while removing only about 6% of
# the ~280 m perimeter ring, so episode variety is essentially unchanged. This is a different radius from
# DISTRACTOR_SPACING_M -- the two must stay separate (see the distractor loop in sample_setup), not share
# one spacing value.
START_KEEPOUT_M = 8.0

# Target/distractor reference height: the object's centre sits 1 m above the ground (NED z = -1.0), and a
# scale of (1, 1, 2) stands a 2 m-tall cylinder on the ground given the registry's 1 m unit-scale cylinder.
TARGET_Z_NED = -1.0
TARGET_SCALE = (1.0, 1.0, 2.0)


class EpisodeSetupError(RuntimeError):
    """Rejection sampling could not place a point within `MAX_REJECTION_TRIES` tries."""


def spawn_clearance_m() -> float:
    """Minimum surface-to-surface gap a sampled point must keep from every obstacle.

    This is exactly the inflation `reachability.check_reachability` validated the scene's layout with
    (`drone_radius_m + clearance_m`), so a point this sampler accepts is flyable by the same definition
    the scene was already checked against.
    """
    return DRONE_RADIUS_M + CLEARANCE_M


@dataclass(frozen=True)
class EpisodeSetup:
    scene_id: str
    seed: int
    start: Pose
    target_xy_z: tuple[float, float, float]
    target_scale: tuple[float, float, float]
    distractors: tuple[tuple[float, float, float], ...]
    instruction: str
    start_edge: str
    target_edge: str
    # Not part of the brief's field list: apply_setup's pinned signature is (sim, setup) only, with no
    # SceneFile in reach, so it has no other way to learn which material real obstacles use (needed so
    # distractors are painted with "the obstacle palette's material", per Task 4's brief). Carrying the
    # one material name here keeps apply_setup scene-independent without adding a parameter that would
    # break the brief's own round-trip test call `apply_setup(sim, setup)`. See the task report.
    obstacle_material: str


def _nearest_obstacle_gap(x: float, y: float, instances: tuple[Instance, ...]) -> float:
    """Distance from (x, y) to the nearest obstacle SURFACE, not centre; +inf with no obstacles."""
    if not instances:
        return math.inf
    return min(math.hypot(x - inst.x, y - inst.y) - inst.radius_m for inst in instances)


def _perpendicular_range(lo_bound: float, hi_bound: float, inset: float) -> tuple[float, float]:
    """The usable range for the axis perpendicular to a sampled edge, inset by `inset` at each end.

    Without this inset, a point drawn near a corner could be closer to a DIFFERENT edge than the one it
    was drawn for, since the band is measured to the nearest edge (see the module docstring). Insetting
    the perpendicular coordinate by the band's own lower bound guarantees the point's distance to every
    other edge is at least that lower bound too, so its nearest-edge distance is governed by the edge it
    was actually drawn for.
    """
    lo, hi = lo_bound + inset, hi_bound - inset
    if lo > hi:
        # The inset would collapse the range to nothing -- only possible for a band whose lower bound
        # exceeds half the scene's own width, which no scene in this project has. Fall back to the full
        # range rather than raising on a case that cannot occur for a real scene.
        return lo_bound, hi_bound
    return lo, hi


def _edge_point(bounds: Bounds, edge: str, distance: float, perpendicular: float) -> tuple[float, float]:
    if edge == "x_min":
        return bounds.x_min + distance, perpendicular
    if edge == "x_max":
        return bounds.x_max - distance, perpendicular
    if edge == "y_min":
        return perpendicular, bounds.y_min + distance
    return perpendicular, bounds.y_max - distance  # "y_max"


def _sample_band_point(bounds: Bounds, edge: str, band: tuple[float, float], rng: np.random.Generator) -> tuple[float, float]:
    """A point whose distance to `edge` is drawn uniformly from `band`, inset from the perpendicular
    edges so its distance to the NEAREST edge also lands in `band` (see `_perpendicular_range`)."""
    lo, hi = band
    distance = float(rng.uniform(lo, hi))
    if edge in ("x_min", "x_max"):
        perp_lo, perp_hi = _perpendicular_range(bounds.y_min, bounds.y_max, lo)
    else:
        perp_lo, perp_hi = _perpendicular_range(bounds.x_min, bounds.x_max, lo)
    perpendicular = float(rng.uniform(perp_lo, perp_hi))
    return _edge_point(bounds, edge, distance, perpendicular)


Exclusion = tuple[tuple[tuple[float, float], ...], float]  # (points, min_distance_from_each)


def _place_with_clearance(
    candidate_fn, instances: tuple[Instance, ...], clearance: float,
    exclusions: tuple[Exclusion, ...],
    rng: np.random.Generator, *, what: str,
) -> tuple[float, float]:
    """Draw candidates from `candidate_fn(rng)` until one clears every obstacle by `clearance` and every
    point in every `(points, min_distance)` pair in `exclusions` by that pair's own `min_distance`.

    Each exclusion set carries its own radius rather than one shared spacing value, because the radii
    genuinely differ: target/distractor spacing (DISTRACTOR_SPACING_M, spec §9.1) and the drone's start
    keepout (START_KEEPOUT_M) are unrelated distances, and folding both into one shared value would
    silently change one when the other is what's intended.
    """
    for _ in range(MAX_REJECTION_TRIES):
        x, y = candidate_fn(rng)
        if _nearest_obstacle_gap(x, y, instances) < clearance:
            continue
        if any(math.hypot(x - px, y - py) < min_distance for points, min_distance in exclusions for px, py in points):
            continue
        return x, y
    n_points = sum(len(points) for points, _ in exclusions)
    raise EpisodeSetupError(
        f"could not place {what} with >= {clearance} m obstacle clearance"
        + (f" and the required spacing from {n_points} existing point(s)" if n_points else "")
        + f" in {MAX_REJECTION_TRIES} tries"
    )


def sample_setup(
    scene: SceneFile, layout: Layout, rng: np.random.Generator, *, n_distractors_range: tuple[int, int] = N_DISTRACTORS_DEFAULT,
) -> EpisodeSetup:
    """Draw a full episode setup for one seed.

    Fixed draw order so a seed reproduces exactly: start edge -> start position -> start altitude ->
    start yaw -> target position on the opposite edge -> distractor count -> distractor positions ->
    instruction template.
    """
    bounds = layout.bounds
    clearance = spawn_clearance_m()

    # 1: start edge.
    start_edge = str(EDGES[int(rng.integers(len(EDGES)))])

    # 2: start position, rejection-sampled for obstacle clearance only (no other placed point exists yet).
    start_x, start_y = _place_with_clearance(
        lambda r: _sample_band_point(bounds, start_edge, scene.start_band, r),
        layout.instances, clearance, (), rng, what="the start",
    )

    # 3: start altitude, inset from the hard operating band (see START_ALTITUDE_MARGIN_M above).
    a_lo, a_hi = scene.altitude_band
    start_altitude = float(rng.uniform(a_lo + START_ALTITUDE_MARGIN_M, a_hi - START_ALTITUDE_MARGIN_M))

    # 4: start yaw, uniform over [-pi, pi) -- NOT aimed at the target (spec §9.2: "a random yaw"). The
    # expert must learn to turn; AutoFly's own a0 is itself a turn-in-place toward the target bearing, so
    # an already-aimed start would make that first turn trivial by construction.
    start_yaw = float(rng.uniform(-math.pi, math.pi))
    start = Pose(x=start_x, y=start_y, z=-start_altitude, yaw=start_yaw)

    # 5: target position, forced to the edge opposite the start (this is what makes the crossing long).
    target_edge = OPPOSITE[start_edge]
    target_x, target_y = _place_with_clearance(
        lambda r: _sample_band_point(bounds, target_edge, scene.target_band, r),
        layout.instances, clearance, (), rng, what="the target",
    )
    target_xy_z = (target_x, target_y, TARGET_Z_NED)

    # 6: distractor count.
    lo_n, hi_n = n_distractors_range
    n_distractors = int(rng.integers(lo_n, hi_n + 1))

    # 7: distractor positions -- anywhere on the target band (any edge), kept >= DISTRACTOR_SPACING_M from
    # the target and from every other distractor already placed, AND >= START_KEEPOUT_M from the drone's
    # own start pose. These are two separate exclusion sets with two separate radii -- the start is never
    # appended to `placed_xy`, so the 4 m target/distractor spacing is unaffected by the 8 m start keepout.
    placed_xy: list[tuple[float, float]] = [(target_x, target_y)]
    start_xy = ((start_x, start_y),)
    distractors: list[tuple[float, float, float]] = []

    def _distractor_candidate(r: np.random.Generator) -> tuple[float, float]:
        edge = str(EDGES[int(r.integers(len(EDGES)))])
        return _sample_band_point(bounds, edge, scene.target_band, r)

    for _ in range(n_distractors):
        dx, dy = _place_with_clearance(
            _distractor_candidate, layout.instances, clearance,
            ((tuple(placed_xy), DISTRACTOR_SPACING_M), (start_xy, START_KEEPOUT_M)),
            rng, what="a distractor",
        )
        placed_xy.append((dx, dy))
        distractors.append((dx, dy, TARGET_Z_NED))

    # 8: instruction template.
    template = str(INSTRUCTION_TEMPLATES[int(rng.integers(len(INSTRUCTION_TEMPLATES)))])
    # Decision D1: until M4 supplies the 60-target name pool, every episode names the same literal
    # target, so the instruction reads "...to reach the target". M4 replaces this with a real name drawn
    # from the split-appropriate pool.
    instruction = template.format(target="target", obstacle=scene.instruction_obstacle)

    return EpisodeSetup(
        scene_id=scene.id,  # s01d flies s01's layout but its episodes are s01d's
        seed=layout.seed,
        start=start,
        target_xy_z=target_xy_z,
        target_scale=TARGET_SCALE,
        distractors=tuple(distractors),
        instruction=instruction,
        start_edge=start_edge,
        target_edge=target_edge,
        obstacle_material=scene.obstacle_groups[0].palette[0],
    )


def _spawn_asset_name(ue_path: str) -> str:
    """The runtime spawn table (`Simulator.spawn()`'s `asset`, per `protocol.py`) keys every static mesh --
    engine content included -- by its short Unreal asset name, not by its full package path: Project
    AirSim's `WorldSimApi::asset_map_` is built from `FAssetData.AssetName`, the last path segment.
    Confirmed live (Task 7) via `world.list_assets()` against the packaged s01 binary: both
    "/Engine/BasicShapes/Cylinder" -> "Cylinder" and M1's own "/Game/Geometry/Meshes/1M_Cube" ->
    "1M_Cube" are present verbatim in the runtime spawn table."""
    return ue_path.rsplit("/", 1)[-1]


RUNTIME_SPAWNABLE_MATERIAL_KINDS = ("engine",)


def _runtime_material_path(registry, name: str | None) -> str | None:
    """The UE package path to paint a RUNTIME-spawned object with, or None to leave the mesh's own material.

    Project AirSim's `setMaterial` resolves the path with `StaticLoadObject(UMaterial::StaticClass(), ...)`
    (WorldSimApi.cpp:1019-1033) -- an EXACT base-UMaterial class filter. A `UMaterialInstanceConstant` is a
    different class, so the load returns null and `set_object_material` fails no matter how correct the path
    is. M0 measured and documented this (`sim/smoke_m0.py:44-45`, on M_Blue) but the knowledge never reached
    this module, and it cost three live blocked runs to rediscover.

    So a `color_instance` material -- which is exactly what our registry builds for the scene palette, e.g.
    "white" -> MI_White -- can never be used at runtime, even though `build_level.py` uses it happily at
    EDITOR build time. Those are different code paths with different class requirements. Rather than fail
    the spawn, return None: `Simulator.spawn()` then skips `set_object_material` entirely
    (`airsim_backend.py:247`) and the object keeps its mesh's default material. For distractors that is the
    right look anyway -- they are meant to read as more of the obstacle field, not as the target.
    """
    if name is None:
        return None
    entry = registry.materials[name]
    return entry.ue_path if entry.kind in RUNTIME_SPAWNABLE_MATERIAL_KINDS else None


def apply_setup(sim: Simulator, setup: EpisodeSetup) -> tuple[str, ...]:
    """Spawn the target then the distractors; return the actual (uniquified) names, target first.

    The target is painted "orange" if the registry has that material, else the first registered material
    that is not the one real obstacles use (`setup.obstacle_material`) -- a visually distinct colour so
    the target can be told apart from the field even before M4's asset pool lands. Distractors are left with
    the mesh's own default material: the scene's obstacle material is a `color_instance` (MI_White), which
    the runtime spawn path categorically cannot apply -- see `_runtime_material_path` for why -- and an
    unpainted cylinder reads as more of the same obstacle field, which is what a distractor is for.

    `target_material`/`setup.obstacle_material` are registry KEY names (e.g. "white"), not the full UE
    package paths `Simulator.spawn()`'s `material` parameter requires (`protocol.py`: "a package path of a
    base UMaterial, e.g. '/Game/Geometry/Materials/M_Orange'") -- this resolves each through
    `registry.materials[name].ue_path` before calling `spawn()`. Likewise the mesh: both the target and the
    distractors use the registry's "cylinder" obstacle asset, resolved to the short runtime spawn name via
    `_spawn_asset_name`, not the bare registry key. Both a bare material name and a bare/slashed asset name
    fail identically against the real backend (measured live, Task 7: `set_object_material` rejects a
    string that is not a real package path) but succeed silently against `FakeSimulator`, which is why
    `FakeSimulator.spawn()` now enforces the same contract `spawn()`'s docstring always documented.

    If a `spawn` partway through the loop raises, whatever was already spawned is torn down before the
    error propagates. This matters because the real backend's `spawn()` can raise AFTER the actor already
    exists server-side (WorldSimApi creates the object, then raises if `set_object_material` fails on it),
    and this function otherwise only reports names on a full, uncaught success -- so a mid-spawn failure
    would leave 1-5 actors that nothing will ever destroy (`self._spawned` in the caller stays empty,
    since this call never returns). Plan 1 named exactly this kind of actor accumulation as a trap, and a
    training run calls this tens of thousands of times, so one rare mid-spawn failure per some large
    number of episodes would otherwise leak forever.
    """
    registry = load_registry()
    if "orange" in registry.materials:
        target_material_name: str | None = "orange"
    else:
        target_material_name = next((name for name in registry.materials if name != setup.obstacle_material), None)
    target_material = _runtime_material_path(registry, target_material_name)
    obstacle_material = _runtime_material_path(registry, setup.obstacle_material)
    asset_name = _spawn_asset_name(registry.assets["cylinder"].ue_path)

    names: list[str] = []
    try:
        tx, ty, tz = setup.target_xy_z
        names.append(sim.spawn("target", asset_name, Pose(x=tx, y=ty, z=tz, yaw=0.0), setup.target_scale, target_material))
        for i, (dx, dy, dz) in enumerate(setup.distractors):
            names.append(
                sim.spawn(f"distractor_{i}", asset_name, Pose(x=dx, y=dy, z=dz, yaw=0.0), setup.target_scale, obstacle_material)
            )
    except Exception:
        clear_setup(sim, tuple(names))
        raise
    return tuple(names)


def clear_setup(sim: Simulator, names) -> None:
    """Destroy every spawned name, tolerating one that is already gone so a partially-spawned or
    already-torn-down episode can always be cleaned up without raising."""
    for name in names:
        try:
            sim.destroy(name)
        except ObjectNotFoundError:
            pass
