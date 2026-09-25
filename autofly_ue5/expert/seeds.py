"""Episode seed ranges for training workers and evaluation, kept disjoint by construction.

`AutoFlyEnv.reset(seed=None)` draws episode `seed_base + k` for its k-th reset; an explicit `reset(seed=s)` draws
episode `s` directly. So every consumer of episodes owns a range of integer seeds, and two consumers that share a
range fly the same episodes.
"""

from __future__ import annotations

# Width of one training worker's range. Hazard #1 (Task 5-6 review, pinned by a live SubprocVecEnv measurement): two
# AutoFlyEnvs built with the same seed_base fly byte-identical episode streams.
WORKER_SEED_STRIDE = 1_000_000
EVAL_SEED_BASE = 100_000_000  # the M2 gate's episodes: comfortably above worker_seed_base(63) + WORKER_SEED_STRIDE


def worker_seed_base(rank: int) -> int:
    """Every vectorised worker must get a distinct range."""
    return rank * WORKER_SEED_STRIDE
