"""The task's contact rule for mesh obstacles (2026-10-07, plan 5 / M4 C8): the body never reaches wood.

A crash into a tree can wedge the simulator's swept Unreal actor beyond any teleport. The plugin moves the actor to
the kinematics by a collision sweep every tick, and UE lets a penetrating actor move only *out of* each contact; a body
under a branch and beside a twig has no such direction, every escape returns the same pose with a collision event, and
a wedged actor cannot fly afterwards (`docs/gates/m4_s02_wood_crash_reset_probe.json`: 2 of 4 crashes into wood stayed
wedged after eight escape attempts; `docs/decisions/2026-10-07-m4-live-checks-and-foliage.md`). Convex solids (pillars,
boxes) never wedge. So in a scene whose obstacles are imported models (trees, rocks: every obstacle the glTF measurement
gave a radius profile), physical contact is not the rule. The rule is geometric, evaluated by the environment on every
step exactly as s01d's 1 m mover rule is (`scenes/motion.first_contact`): an episode ends as a collision when the
drone's swept step passes within the shape's contact distance of an obstacle: of its outline (`OUTLINE_CONTACT_M`,
1.05 m: the task's rule since 2026-10-08 afternoon, below), or of its footprint circle (`CIRCLE_CONTACT_M`, 0.9 m), the
layout's `radius_m`, the model's widest reach from its pivot at the instance's scale, which contains every branch. The
physical collision event stays as a backstop, scored as "sim".

Each distance is chosen as spec §6.5's mover rule keeps `contact_m - max_speed * dt` above the rotor span: the rule is
checked on the swept segment *after* the step has been flown, so the ending step itself can carry the centre a full step
(2 m/s x 0.2 s = 0.4 m) past the boundary, and a full-rate turn (1 rad/s) at full speed bows a step's 0.4 m chord by
2 (1 - cos 0.1) = 0.01 m. The circle's 0.9 m keeps the rotor tips (`DRONE_HALF_SPAN_M`, 0.48 m from the centre) outside
it: 0.48 + 0.4 + 0.01 = 0.89, rounded up (the first version, 0.5 m, let the ending step reach 0.11 m from the circle on
40 of 40 scripted flights: `results/2026-10-07-project-status/figures/s02_footprint_rule.json`). But a physical
collision registers farther out than the rotor tips: 0.51 m and 0.52-0.63 m from a pillar's surface, measured
(`DRONE_COLLISION_REACH_M`, 0.63 m). The circle reaches past the model everywhere but at its widest point, and that
slack hid it; the outline has none on a trunk-side approach, so its distance keeps the measured reach outside it:
0.63 + 0.4 + 0.01 = 1.04, rounded up to 1.05. Then neither the ending step nor the reset's straight climb from where it
ended reaches the model. Both are asserted at import. The shape's distance is the task's rule: the gate scores it, the
collector rejects by it, and a training run may only add a margin on top (`training_margin_m`, the env's
`footprint_contact_margin_m`), because every s01d expert flew to the boundary it trained on and its deterministic tail
reached about 0.3 m inside it (`2026-10-06-s01d-r6-plan.md`). Starts, targets and distractors keep
`episode.spawn_clearance_m()` = 1.4 m from every circle (so from every outline inside it), and a reset may land 0.3 m
off, so the trainer refuses a margin that puts the boundary beyond 1.1 m: at most 0.05 m under the outline (0.2 m under
the circle). A scene may keep them farther out (its file's `spawn_clearance_m`, 2026-10-08;
`episode.scene_spawn_clearance_m`), and a training run may keep its own starts farther out still (train.py
`--spawn-clearance-m`, training only, so the gate's episodes stay those of the scene); either raises the ceiling by as
much: 1.9 m allows 0.55 m.

Two shapes (2026-10-08, afternoon). The circle about the pivot holds every branch, but for a leaning tree it reaches far
past anything on the side away from the lean: on s02 it ended 82 of s02_r1's 220 deterministic evaluation flights a
median 2.70 m from any wood or leaf card (3.0 m on the three leaning species), never closer than 0.94 m, with the tree in
the camera's view at 8 of the 82 (results/2026-10-07-project-status/figures/s02_footprint_endings.json). An expert that
sees only depth cannot learn a boundary it cannot see. So the task's rule is now `"outline"`: 1.05 m (above), measured
to the model's outline, the horizontal projection of all its geometry above the ground at every height (a column: after
an ending the reset climbs straight up, so nothing overhead may be within reach), read from a distance field per asset
(scenes/footprint_field.py, assets/footprints/). The outline lies inside the circle, so the layouts, the sampler's
clearance and the margin ceilings above, all drawn against circles, stay conservative. `"circle"` is kept, unchanged
(0.9 m), so s02_r1's models are evaluated under the rule they trained on. A step is first screened against a circle
about each pivot whose radius is the field's own reach (`FootprintField.pivot_reach_m`, so the screen holds whatever
radius a layout gave the instance): a step more than termination + PREFILTER_SLACK_M from it cannot be within
termination of the outline as the field reads it, and only the trees it passes near are read from their fields. An
asset without a field is refused under the outline, never measured by its circle instead.

Which shape scores a model is read from the run it trained in, never defaulted where that run names one
(`expert/run_shape.py`): the gate, the collector, the eval watch, the selection and the renders read it from the run's
identity.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from autofly_ue5.expert.movers import FORWARD_SPEED_MAX_M_S
from autofly_ue5.expert.static_contact import StaticContactBoundary
from autofly_ue5.scenes.footprint_field import FootprintFields, horizontal_scale, outline_distance
from autofly_ue5.scenes.model import (
    DRONE_COLLISION_REACH_M,
    DRONE_HALF_SPAN_M,
    AssetEntry,
    AssetRegistry,
    Instance,
    Layout,
    load_registry,
)
from autofly_ue5.sim.types import CONTROL_DT_S

CIRCLE_CONTACT_M = 0.9  # s02_r1's rule, to the footprint circle
OUTLINE_CONTACT_M = 1.05  # the task's rule, to the outline
SHAPES = ("outline", "circle")
DEFAULT_SHAPE = "outline"  # the task's rule since 2026-10-08 (afternoon), the env's and a new run's; "circle" is s02_r1's
CONTACT_M = {"outline": OUTLINE_CONTACT_M, "circle": CIRCLE_CONTACT_M}
# The screen's circle holds the outline, and the field never overstates it; it understates the outline distance by at
# most 1.3 cm at unit scale (12.6 mm measured on the 21 models), which is 3.2 cm in the world at the largest layout scale
# (3.17, s04): a step more than termination + this from the screen is beyond termination as the field reads it too.
PREFILTER_SLACK_M = 0.1
CHORD_BOW_M = 0.01  # a full-rate turning step's chord against its arc: 2 (1 - cos 0.1) m at 2 m/s, 1 rad/s, 0.2 s
STEP_M = FORWARD_SPEED_MAX_M_S * CONTROL_DT_S  # how far past the boundary the ending step can carry the centre: 0.4 m
OBSTACLE_ROLE = "obstacle"  # the registry role of what a layout may place
GROUND_TOLERANCE_M = 1e-6  # how far off the ground (z_center 0) an instance's pivot may be for its outline's ground cut

# The invariants the numbers above keep (checked at import, as scenes/model.py checks the mover rule's at load): a step
# that ends exactly at the boundary, flown at full speed straight in on a full-rate turn, leaves every rotor tip outside
# the circle, and the drone's measured collision reach outside the outline, from which the reset then climbs straight up.
if CIRCLE_CONTACT_M - STEP_M < DRONE_HALF_SPAN_M + CHORD_BOW_M:
    raise AssertionError(f"CIRCLE_CONTACT_M {CIRCLE_CONTACT_M} - {STEP_M} must be >= {DRONE_HALF_SPAN_M} + "
                         f"{CHORD_BOW_M}: a full-speed ending step would put a rotor tip inside the circle")
if OUTLINE_CONTACT_M - STEP_M - CHORD_BOW_M < DRONE_COLLISION_REACH_M:
    raise AssertionError(f"OUTLINE_CONTACT_M {OUTLINE_CONTACT_M} - {STEP_M} - {CHORD_BOW_M} must be >= "
                         f"{DRONE_COLLISION_REACH_M}: a full-speed ending step could reach the outline with the drone's "
                         f"collision")


def is_measured_obstacle(role: str | None, radius_profile_m) -> bool:
    """The one definition of a mesh obstacle, what the rule covers and scripts/footprint_fields.py builds a field for:
    an imported, measured model (its radius profile exists) in the obstacle role. Engine primitives (cylinder, cube,
    crate) have no profile and stay physical; the target pool's models are measured too, but are spawned per episode
    and never placed in a layout (`footprint_obstacles` refuses one there)."""
    return radius_profile_m is not None and role == OBSTACLE_ROLE


def is_footprint_obstacle(asset: AssetEntry) -> bool:
    """A registry entry the footprint rule covers (`is_measured_obstacle`): physical contact with it can wedge the
    actor."""
    return is_measured_obstacle(asset.role, asset.radius_profile_m)


def footprint_obstacles(instances: Iterable[Instance], registry: AssetRegistry) -> tuple[Instance, ...]:
    """The layout instances the rule covers. An asset the registry does not know is refused: the level could not have
    been built from it, and silently treating it as a primitive would let a body reach a mesh. So is a measured model
    of another role (a target): a layout places obstacles only, and the rule has no field for it. ValueErrors, so the
    trainer's checks before it claims a run root report them as refusals rather than tracebacks."""
    out = []
    for inst in instances:
        if inst.asset not in registry.assets:
            raise ValueError(f"layout instance {inst.tag} uses asset {inst.asset!r}, which is not in the asset registry")
        asset = registry.assets[inst.asset]
        if asset.radius_profile_m is not None and asset.role != OBSTACLE_ROLE:
            raise ValueError(f"layout instance {inst.tag} uses asset {inst.asset!r}, a measured model in the "
                             f"{asset.role!r} role: a layout places {OBSTACLE_ROLE}s only (the target pool's models are "
                             f"spawned per episode), and the footprint rule has no outline field for it")
        if is_footprint_obstacle(asset):
            out.append(inst)
    return tuple(out)


def has_footprint_obstacles(layout: Layout, registry: AssetRegistry | None = None) -> bool:
    """Whether the scene has a footprint rule at all: a mesh obstacle in its layout (default registry: read now)."""
    return bool(footprint_obstacles(layout.instances, registry if registry is not None else load_registry()))


def check_shape(shape: str) -> str:
    if shape not in SHAPES:
        raise ValueError(f"unknown footprint shape {shape!r} (shapes: {', '.join(SHAPES)})")
    return shape


def contact_distance_m(shape: str) -> float:
    """The task's contact distance under `shape`: what the gate scores, a run's identity records and a margin adds to."""
    return CONTACT_M[check_shape(shape)]


def fields_record(fields: FootprintFields) -> dict:
    """What a record names of the outline fields a rule measured with: each asset's field content (the identity of a
    run names the same) and the manifest it was read from."""
    return {"fields": fields.digest(), "fields_manifest_sha256": fields.manifest_sha256}


def task_contact_rules(layout: Layout, registry: AssetRegistry | None = None, *, shape: str = DEFAULT_SHAPE,
                       fields: FootprintFields | None = None) -> dict | None:
    """The task's contact rule a scene is scored under, for gate and training records: None where physical contact is
    the only rule (every pillar and box scene), else the footprint rule, how many obstacles it covers, its shape and,
    for the outline, the fields it measures with (default: read now; an asset without one is refused)."""
    check_shape(shape)
    covered = footprint_obstacles(layout.instances, registry if registry is not None else load_registry())
    if not covered:
        return None
    rule = {"contact_m": contact_distance_m(shape), "obstacles": len(covered), "shape": shape}
    if shape == "outline":
        assets = {inst.asset for inst in covered}
        rule.update(fields_record(require_fields(assets, fields)))
    return {"footprint": rule}


def require_fields(assets, fields: FootprintFields | None = None) -> FootprintFields:
    """The outline fields of `assets`: `fields` if it holds every one (else a ValueError naming those it lacks), or read
    from assets/footprints/ (FootprintFields.load refuses the same way)."""
    wanted = sorted(set(assets))
    if fields is None:
        return FootprintFields.load(wanted)
    missing = [asset for asset in wanted if asset not in fields]
    if missing:
        raise ValueError(f"no footprint field for {', '.join(missing)} in the fields given: the outline rule cannot "
                         f"measure {'it' if len(missing) == 1 else 'them'} (scripts/footprint_fields.py --only "
                         f"{' '.join(missing)})")
    return fields


class FootprintContactRule:
    """One episode's footprint obstacles and where a step among them ends the episode: within `termination_m` of an
    obstacle's outline (`shape="outline"`, the task's rule) or of its footprint circle (`"circle"`, s02_r1's).
    `contact_m`: the task's distance, by default the shape's (`contact_distance_m`)."""

    def __init__(self, obstacles: Sequence[Instance], *, contact_m: float | None = None,
                 training_margin_m: float = 0.0, shape: str = DEFAULT_SHAPE,
                 fields: FootprintFields | None = None) -> None:
        self.shape = check_shape(shape)
        contact_m = contact_distance_m(self.shape) if contact_m is None else contact_m
        if not (math.isfinite(contact_m) and contact_m > 0):
            raise ValueError(f"a footprint contact distance must be a positive number of metres, got {contact_m}")
        if not (math.isfinite(training_margin_m) and training_margin_m >= 0):
            raise ValueError(f"a footprint training margin must be a non-negative number of metres, got {training_margin_m}")
        self.contact_m = float(contact_m)  # the task's rule: what the gate scores
        self.training_margin_m = float(training_margin_m)
        self.termination_m = self.contact_m + self.training_margin_m  # where this env's episodes end
        self._instances = tuple(obstacles)
        # The circle's geometry is a static pillar boundary's: its surface distance from a point or a swept segment.
        self._boundary = StaticContactBoundary([(i.tag, i.x, i.y, i.radius_m) for i in obstacles], self.termination_m)
        self.fields: FootprintFields | None = None
        self._screen: StaticContactBoundary | None = None
        if self.shape == "outline":
            self.fields = require_fields({i.asset for i in self._instances}, fields)
            screen = []
            for inst in self._instances:
                if abs(inst.z_center) > GROUND_TOLERANCE_M:
                    raise ValueError(f"{inst.tag} ({inst.asset}): its pivot is at z_center {inst.z_center:g} m; an "
                                     f"outline is the model cut at the ground (footprint_field.clip_above_ground), so its "
                                     f"instances must stand on the ground (z_center 0)")
                field, s = self.fields[inst.asset], horizontal_scale(inst)  # refuses x and y scales that differ
                if self.termination_m + PREFILTER_SLACK_M > field.reach_m * s:
                    raise ValueError(f"{inst.tag} ({inst.asset}): its outline field reaches {field.reach_m * s:.2f} m at "
                                     f"scale {s:g}, short of the {self.termination_m + PREFILTER_SLACK_M:g} m this rule "
                                     f"reads; rebuild it with a wider margin (scenes/footprint_field.py MARGIN_M)")
                # the screen: the field's own reach, not the layout's radius (an older layout's may be the band's)
                screen.append((inst.tag, inst.x, inst.y, field.pivot_reach_m * s))
            self._screen = StaticContactBoundary(screen, self.termination_m)

    @classmethod
    def for_episode(cls, obstacles: Iterable[Instance], movers: Iterable[str], **kw) -> "FootprintContactRule":
        """The scene's footprint obstacles less the episode's movers (by tag), whose homes are vacated once they move
        and whom the mover rule covers wherever they are."""
        moving = set(movers)
        return cls([inst for inst in obstacles if inst.tag not in moving], **kw)

    @property
    def tags(self) -> tuple[str, ...]:
        return self._boundary.tags

    def _outline_gaps(self, a_xy, b_xy, indices) -> np.ndarray:
        b_xy = a_xy if b_xy is None else b_xy
        return np.array([outline_distance(self.fields[self._instances[i].asset], self._instances[i], a_xy, b_xy)
                         for i in indices], dtype=np.float64)

    def nearest(self, a_xy, b_xy) -> tuple[str, float] | None:
        """(tag, gap) of the obstacle the swept step a_xy -> b_xy came closest to, however far (None without obstacles).
        Under the circle the gap is negative once the centre is inside the circle; under the outline it is 0 on or over
        the outline, and +inf when the step is beyond every field's reach (the nearest circle's tag is named)."""
        if not self.tags:
            return None
        circle = self._boundary.gaps(a_xy, b_xy)
        if self.shape == "circle":
            i = int(np.argmin(circle))
            return self.tags[i], float(circle[i])
        gaps = self._outline_gaps(a_xy, b_xy, range(len(self.tags)))
        i = int(np.argmin(gaps)) if np.isfinite(gaps).any() else int(np.argmin(circle))
        return self.tags[i], float(gaps[i])

    def contact(self, a_xy, b_xy) -> tuple[str, float] | None:
        """`nearest`, if within termination_m: where the rule ends the step. Under the outline only the obstacles whose
        screen circle the step passes within termination_m + PREFILTER_SLACK_M are read from their fields."""
        if self.shape == "circle":
            hit = self.nearest(a_xy, b_xy)
            return hit if hit is not None and hit[1] <= self.termination_m else None
        if not self.tags:
            return None
        near = np.flatnonzero(self._screen.gaps(a_xy, b_xy) <= self.termination_m + PREFILTER_SLACK_M)
        if not len(near):
            return None
        gaps = self._outline_gaps(a_xy, b_xy, near)
        k = int(np.argmin(gaps))
        return (self.tags[near[k]], float(gaps[k])) if gaps[k] <= self.termination_m else None

    def source(self, gap_m: float) -> str:
        """Which rule ended a step at this gap: the task's ("footprint") or the run's margin ("footprint_margin")."""
        return "footprint" if gap_m <= self.contact_m else "footprint_margin"

    def nearest_gap(self, xy) -> float:
        if self.shape == "circle":
            return self._boundary.nearest_gap(xy)
        return float(self._outline_gaps(xy, xy, range(len(self.tags))).min()) if self.tags else math.inf

    def circle_gap(self, tag: str, a_xy, b_xy) -> float:
        """The swept step's gap from one obstacle's footprint circle (for records: the circle's reading of an outline
        ending)."""
        return float(self._boundary.gaps(a_xy, b_xy)[self.tags.index(tag)])

    def to_json(self) -> dict:
        out = {"contact_m": self.contact_m, "training_margin_m": self.training_margin_m,
               "termination_m": self.termination_m, "obstacles": len(self.tags), "shape": self.shape}
        if self.shape == "outline":
            out.update(fields_record(self.fields))
        return out
