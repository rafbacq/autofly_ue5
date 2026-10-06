"""Training only: where a static pillar ends an episode (2026-10-06, run 6).

Every s01d expert flew to the boundary it trained on. Against a moving pillar that was the contact rule, and training
on a rule 0.3 m wider cut run 5's gate mover collisions from 35 to 7 (`--mover-contact-margin`). Against a static pillar
the boundary was physical contact: run 5's ten static gate collisions ended with the drone's centre 0.52-0.63 m from the
pillar's surface, and its successful replays passed pillars at 0.51-0.91 m (runs/viz/s01d_r5_final). So a training run
may end a step as a collision when the drone's path passes within `boundary_m` of a static pillar's surface. The expert
observes, and the gate scores, physical contact only.

The episode's movers are left out: their homes are vacated once they move, and the mover rule covers them wherever
they are. Every other pillar of the layout stands at home for the whole episode (AutoFlyEnv.reset sends the last
episode's movers home before the first step).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from autofly_ue5.scenes.model import Instance


class StaticContactBoundary:
    """The static pillars of one episode, and the surface distance within which one ends a training step."""

    def __init__(self, pillars: Sequence[tuple[str, float, float, float]], boundary_m: float) -> None:
        """`pillars`: (tag, x, y, radius_m) of every pillar that stays home."""
        if not (math.isfinite(boundary_m) and boundary_m > 0):
            raise ValueError(f"a static contact boundary must be a positive number of metres, got {boundary_m}")
        self.boundary_m = float(boundary_m)
        self.tags = tuple(tag for tag, _x, _y, _r in pillars)
        self._xy = np.array([(x, y) for _tag, x, y, _r in pillars], dtype=np.float64).reshape(-1, 2)
        self._radius = np.array([r for _tag, _x, _y, r in pillars], dtype=np.float64)

    @classmethod
    def for_episode(cls, instances: Iterable[Instance], movers: Iterable[str],
                    boundary_m: float) -> "StaticContactBoundary":
        """The layout's pillars less the episode's movers (by tag)."""
        moving = set(movers)
        return cls([(i.tag, i.x, i.y, i.radius_m) for i in instances if i.tag not in moving], boundary_m)

    def gaps(self, a_xy, b_xy=None) -> np.ndarray:
        """Each pillar's surface distance from the point a_xy, or from the drone's swept segment a_xy -> b_xy (a step
        at full speed covers 0.4 m, as for movers: scenes/motion.surface_gaps)."""
        a = np.asarray(a_xy, dtype=np.float64)
        b = a if b_xy is None else np.asarray(b_xy, dtype=np.float64)
        v = b - a
        length2 = float(v @ v)
        t = np.zeros(len(self._xy)) if length2 == 0.0 else np.clip((self._xy - a) @ v / length2, 0.0, 1.0)
        offset = self._xy - (a + t[:, None] * v)
        return np.hypot(offset[:, 0], offset[:, 1]) - self._radius

    def contact(self, a_xy, b_xy) -> tuple[str, float] | None:
        """(tag, surface gap) of the pillar the swept step a_xy -> b_xy came closest to, if within the boundary."""
        if not self.tags:
            return None
        gaps = self.gaps(a_xy, b_xy)
        i = int(np.argmin(gaps))
        return (self.tags[i], float(gaps[i])) if gaps[i] <= self.boundary_m else None

    def nearest_gap(self, xy) -> float:
        return float(np.min(self.gaps(xy))) if self.tags else math.inf
