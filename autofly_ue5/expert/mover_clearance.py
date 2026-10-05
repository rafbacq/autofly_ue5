"""A training-only cost for flying close to a moving pillar (2026-10-04).

A mover's contact rule fires 1.0 m from its surface (`contact_m`); a static pillar needs physical contact (~0.48 m).
s01d's experts kept failing on that difference. Every mover collision of s01d_r1's best_model was a near miss, the
drone 0.84-1.00 m from the surface (`docs/gates/m2d_r1_contact_probe_*.json`), and run 3, which sees the movers
(`obs.encode_movers`), still made the same misses at 150k steps: gaps 0.88-1.00 m, mostly beside a mover that had
stopped to yield. The task's reward only says so at the contact step (-10), a cliff that a smooth critic blurs.

This adds a ramp before the cliff, k per step at the contact boundary falling to 0 at `margin_m` outside it, for every
mover inside the margin:

    penalty = k * sum over movers of clip(1 - (surface_gap - contact_m) / margin_m, 0, 1)

so the expert is taught to keep a buffer, not just to stay outside the boundary. It is not potential-based on purpose:
the bias toward wider berths is the point, since a gated expert must hold its clearance through physics that do not
replay exactly. It is a training choice only. The task, its success rule and the gate's reward are unchanged, and a
run that uses it records it in its identity, so it cannot be resumed without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MoverClearancePenalty:
    k: float          # per step, for one mover at (or inside) the contact boundary
    margin_m: float   # clearance outside the contact boundary at which the cost starts

    def __post_init__(self) -> None:
        if not (math.isfinite(self.k) and self.k > 0):
            raise ValueError(f"mover clearance penalty: k must be a positive number, got {self.k}")
        if not (math.isfinite(self.margin_m) and self.margin_m > 0):
            raise ValueError(f"mover clearance penalty: margin must be a positive number of metres, "
                             f"got {self.margin_m}")

    def __call__(self, surface_gaps_m: Iterable[float], *, contact_m: float) -> float:
        """The cost of one step that ends with the movers' surfaces `surface_gaps_m` from the drone."""
        return self.k * sum(min(1.0, max(0.0, 1.0 - (gap - contact_m) / self.margin_m)) for gap in surface_gaps_m)

    def to_json(self) -> dict:
        return {"k": self.k, "margin_m": self.margin_m}
