"""The task's contact rule for mesh obstacles (2026-10-07, plan 5 / M4 C8): the body never reaches wood.

A crash into a tree can wedge the simulator's swept Unreal actor beyond any teleport. The plugin moves the actor to
the kinematics by a collision sweep every tick, and UE lets a penetrating actor move only *out of* each contact; a body
under a branch and beside a twig has no such direction, every escape returns the same pose with a collision event, and
a wedged actor cannot fly afterwards (`docs/gates/m4_s02_wood_crash_reset_probe.json`: 2 of 4 crashes into wood stayed
wedged after eight escape attempts; `docs/decisions/2026-10-07-m4-live-checks-and-foliage.md`). Convex solids (pillars,
boxes) never wedge. So in a scene whose obstacles are imported models (trees, rocks: every asset the glTF measurement
gave a radius profile), physical contact is not the rule. The rule is geometric, evaluated by the environment on every
step exactly as s01d's 1 m mover rule is (`scenes/motion.first_contact`): an episode ends as a collision when the
drone's swept step passes within `FOOTPRINT_CONTACT_M` of an obstacle's footprint circle, the layout's `radius_m`,
which is the model's widest reach in the flight band at the instance's scale (`scenes/gltf.band_radius_from_profile`)
and so contains every branch the band meets. The physical collision event stays as a backstop, scored as "sim".

`FOOTPRINT_CONTACT_M` is 0.9 m, for the same reason spec §6.5's mover rule keeps `contact_m - max_speed * dt` above the
rotor span: the rule is checked on the swept segment *after* the step has been flown, so the ending step itself can
carry the centre a full step (2 m/s x 0.2 s = 0.4 m) past the boundary. The circle is measured to the drone's centre,
the rotor tips reach `DRONE_HALF_SPAN_M` = 0.48 m from it, and a full-rate turn (1 rad/s) at full speed bows a step's
0.4 m chord by 2 (1 - cos 0.1) = 0.01 m: 0.48 + 0.4 + 0.01 = 0.89, rounded up. So no step, the ending one included,
puts a rotor tip inside the circle (the first version, 0.5 m, let the ending step reach 0.11 m from the circle on 40 of
40 scripted flights: `results/2026-10-07-project-status/figures/s02_footprint_rule.json`). It is the task's rule: the
gate scores it, the collector rejects by it, and a training run may only add a margin on top (`training_margin_m`,
the env's `footprint_contact_margin_m`), because every s01d expert flew to the boundary it trained on and its
deterministic tail reached about 0.3 m inside it (`2026-10-06-s01d-r6-plan.md`). Starts, targets and distractors keep
`episode.spawn_clearance_m()` = 1.4 m from every circle, and a reset may land 0.3 m off, so the trainer refuses a margin
that puts the boundary beyond 1.1 m: at most 0.2 m here.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from autofly_ue5.expert.movers import FORWARD_SPEED_MAX_M_S
from autofly_ue5.expert.static_contact import StaticContactBoundary
from autofly_ue5.scenes.model import DRONE_HALF_SPAN_M, AssetEntry, AssetRegistry, Instance, Layout, load_registry
from autofly_ue5.sim.types import CONTROL_DT_S

FOOTPRINT_CONTACT_M = 0.9
CHORD_BOW_M = 0.01  # a full-rate turning step's chord against its arc: 2 (1 - cos 0.1) m at 2 m/s, 1 rad/s, 0.2 s

# The invariant the number above keeps (checked at import, as scenes/model.py checks the mover rule's at load): a step
# that ends exactly at the boundary, flown at full speed straight in, leaves every rotor tip outside the circle.
if FOOTPRINT_CONTACT_M - FORWARD_SPEED_MAX_M_S * CONTROL_DT_S < DRONE_HALF_SPAN_M + CHORD_BOW_M:
    raise AssertionError(f"FOOTPRINT_CONTACT_M {FOOTPRINT_CONTACT_M} - {FORWARD_SPEED_MAX_M_S} * {CONTROL_DT_S} must be "
                         f">= {DRONE_HALF_SPAN_M} + {CHORD_BOW_M}: a full-speed ending step would put a rotor tip inside the circle")


def is_footprint_obstacle(asset: AssetEntry) -> bool:
    """An imported, measured model (its radius profile exists): its footprint circle is the band's reach, and physical
    contact with it can wedge the actor. Engine primitives (cylinder, cube, crate) have no profile and stay physical."""
    return asset.radius_profile_m is not None


def footprint_obstacles(instances: Iterable[Instance], registry: AssetRegistry) -> tuple[Instance, ...]:
    """The layout instances the rule covers. An asset the registry does not know is refused: the level could not have
    been built from it, and silently treating it as a primitive would let a body reach a mesh. A ValueError, so the
    trainer's checks before it claims a run root report it as a refusal rather than a traceback."""
    out = []
    for inst in instances:
        if inst.asset not in registry.assets:
            raise ValueError(f"layout instance {inst.tag} uses asset {inst.asset!r}, which is not in the asset registry")
        if is_footprint_obstacle(registry.assets[inst.asset]):
            out.append(inst)
    return tuple(out)


def task_contact_rules(layout: Layout, registry: AssetRegistry | None = None) -> dict | None:
    """The task's contact rule a scene is scored under, for gate and training records: None where physical contact is
    the only rule (every pillar and box scene), else the footprint rule and how many obstacles it covers."""
    covered = footprint_obstacles(layout.instances, registry if registry is not None else load_registry())
    if not covered:
        return None
    return {"footprint": {"contact_m": FOOTPRINT_CONTACT_M, "obstacles": len(covered)}}


class FootprintContactRule:
    """One episode's footprint obstacles and where a step among them ends the episode."""

    def __init__(self, obstacles: Sequence[Instance], *, contact_m: float = FOOTPRINT_CONTACT_M,
                 training_margin_m: float = 0.0) -> None:
        if not (math.isfinite(contact_m) and contact_m > 0):
            raise ValueError(f"a footprint contact distance must be a positive number of metres, got {contact_m}")
        if not (math.isfinite(training_margin_m) and training_margin_m >= 0):
            raise ValueError(f"a footprint training margin must be a non-negative number of metres, got {training_margin_m}")
        self.contact_m = float(contact_m)  # the task's rule: what the gate scores
        self.training_margin_m = float(training_margin_m)
        self.termination_m = self.contact_m + self.training_margin_m  # where this env's episodes end
        # The geometry is a static pillar boundary's: the circle's surface distance from a point or a swept segment.
        self._boundary = StaticContactBoundary([(i.tag, i.x, i.y, i.radius_m) for i in obstacles], self.termination_m)

    @classmethod
    def for_episode(cls, obstacles: Iterable[Instance], movers: Iterable[str], **kw) -> "FootprintContactRule":
        """The scene's footprint obstacles less the episode's movers (by tag), whose homes are vacated once they move
        and whom the mover rule covers wherever they are."""
        moving = set(movers)
        return cls([inst for inst in obstacles if inst.tag not in moving], **kw)

    @property
    def tags(self) -> tuple[str, ...]:
        return self._boundary.tags

    def nearest(self, a_xy, b_xy) -> tuple[str, float] | None:
        """(tag, gap from its circle) of the obstacle the swept step a_xy -> b_xy came closest to, however far (None
        without obstacles). The gap is negative once the centre is inside the circle."""
        if not self.tags:
            return None
        gaps = self._boundary.gaps(a_xy, b_xy)
        i = int(np.argmin(gaps))
        return self.tags[i], float(gaps[i])

    def contact(self, a_xy, b_xy) -> tuple[str, float] | None:
        """`nearest`, if within termination_m: where the rule ends the step."""
        hit = self.nearest(a_xy, b_xy)
        return hit if hit is not None and hit[1] <= self.termination_m else None

    def source(self, gap_m: float) -> str:
        """Which rule ended a step at this gap: the task's ("footprint") or the run's margin ("footprint_margin")."""
        return "footprint" if gap_m <= self.contact_m else "footprint_margin"

    def nearest_gap(self, xy) -> float:
        return self._boundary.nearest_gap(xy)

    def to_json(self) -> dict:
        return {"contact_m": self.contact_m, "training_margin_m": self.training_margin_m,
                "termination_m": self.termination_m, "obstacles": len(self.tags)}
