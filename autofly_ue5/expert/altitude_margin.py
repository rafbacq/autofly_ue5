"""A training-only cost for flying near the altitude band's edges (2026-10-05).

The task ends an episode the instant the drone leaves the 1-3 m band, and nothing in its reward cares where the drone
is inside it. So an expert learns vertical control only from those exits, and its deterministic action can carry a
small vertical bias that the training noise hides. Every s01d expert lost episodes that way:
- s01d_r1's best_model had 10 `altitude_high` exits in its gate, and its final model learned to dive (58 of 62
  out-of-bounds exits were `altitude_low`);
- s01d_r4's 350k checkpoint climbed out of 9 of its 20 evaluation episodes, at steps 28-158, far from any target.

This charges k per step at the band's floor or ceiling, falling to 0 at `margin_m` inside it:

    penalty = k * clip(1 - min(altitude - floor, ceiling - altitude) / margin_m, 0, 1)

so holding altitude away from the edges is worth something before an exit is. Like the mover penalties it is a
training choice only: the task, its success rule and the gate's reward are unchanged, and a run that pays it records it
in its identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AltitudeMarginPenalty:
    k: float          # per step, at (or beyond) the band's floor or ceiling
    margin_m: float   # distance inside the band at which the cost starts

    def __post_init__(self) -> None:
        if not (math.isfinite(self.k) and self.k > 0):
            raise ValueError(f"altitude margin penalty: k must be a positive number, got {self.k}")
        if not (math.isfinite(self.margin_m) and self.margin_m > 0):
            raise ValueError(f"altitude margin penalty: margin must be a positive number of metres, "
                             f"got {self.margin_m}")

    def check_band(self, band_m: tuple[float, float]) -> "AltitudeMarginPenalty":
        """Refuse a margin that would leave no altitude in the band free of the cost."""
        floor, ceiling = band_m
        if self.margin_m > (ceiling - floor) / 2:
            raise ValueError(f"altitude margin penalty: a margin of {self.margin_m} m leaves nothing of the "
                             f"{floor}-{ceiling} m band free of it")
        return self

    def __call__(self, altitude_m: float, *, band_m: tuple[float, float]) -> float:
        """The cost of one step that ends at `altitude_m`."""
        floor, ceiling = band_m
        inside = min(altitude_m - floor, ceiling - altitude_m)
        return self.k * min(1.0, max(0.0, 1.0 - inside / self.margin_m))

    def to_json(self) -> dict:
        return {"k": self.k, "margin_m": self.margin_m}
