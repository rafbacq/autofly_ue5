"""Episode seed ranges for training workers and evaluation, kept disjoint by construction.

`AutoFlyEnv.reset(seed=None)` draws episode `seed_base + k` for its k-th reset; an explicit `reset(seed=s)` draws
episode `s` directly. So every consumer of episodes owns a range of integer seeds, and two consumers that share a
range fly the same episodes. The ranges:

    [0, 1e6)                                  explicit seeds SB3 gives a vec env's first reset in a session:
                                              seed + session * SESSION_SEED_STRIDE + rank, with seed < 99,936
    [(rank+1)*1e6, (rank+2)*1e6)              training worker `rank`, split into MAX_SESSIONS session slices
    [EVAL_SEED_BASE, +1e6)                    the M2 gate's evaluation episodes
    [EVAL_CALLBACK_SEED_BASE, +1e6)           training-time evaluation (model selection)
    [PROBE_SEED_BASE, +1e6)                   live probes (scripts/probe_crash_reset.py)
    [COLLECTION_SEED_BASE, +1e8)              dataset collection (M3/M5): never a training, gate or probe episode,
                                              in 100 slices of 1e6, one per (scene, collection run):
                                              collection_seed_base(k). Slice 0 is s01's pilot; slice 99 is for
                                              smokes and diagnostics, never a dataset
"""

from __future__ import annotations

# Width of one training worker's range. Hazard #1 (Task 5-6 review, pinned by a live SubprocVecEnv measurement): two
# AutoFlyEnvs built with the same seed_base fly byte-identical episode streams.
WORKER_SEED_STRIDE = 1_000_000
# A resumed training session starts its workers this far into their range, so it never replays an earlier
# session's episodes. 100,000 episodes per session is ~20x a 12-hour run's count.
SESSION_SEED_STRIDE = 100_000
MAX_SESSIONS = WORKER_SEED_STRIDE // SESSION_SEED_STRIDE
EVAL_SEED_BASE = 100_000_000  # the M2 gate's episodes: comfortably above worker_seed_base(63) + WORKER_SEED_STRIDE
# Training-time evaluation (best_model.zip selection). Disjoint from EVAL_SEED_BASE so the gate's episodes stay
# held out for every checkpoint -- in the 2026-09-17 run they were not (both drew from EVAL_SEED_BASE).
EVAL_CALLBACK_SEED_BASE = 200_000_000
PROBE_SEED_BASE = 300_000_000
# Dataset collection (spec §9-§10): its episodes must be none the experts trained or were gated on. 100 slices of 1e6,
# one per (scene, collection run) as M5 needs them.
COLLECTION_SEED_BASE = 400_000_000
COLLECTION_SLICE_STRIDE = 1_000_000
MAX_COLLECTION_SLICES = 100
DIAGNOSTIC_COLLECTION_SLICE = MAX_COLLECTION_SLICES - 1


def collection_seed_base(slice_index: int) -> int:
    """The first seed of collection slice `slice_index`. Each collection run takes its own slice: re-runs, and s01
    against s01d (same layout, same draw order), would otherwise fly identical setups (the M3 review, 2026-10-03)."""
    if not 0 <= slice_index < MAX_COLLECTION_SLICES:
        raise ValueError(f"collection slice must be in [0, {MAX_COLLECTION_SLICES}), got {slice_index}")
    return COLLECTION_SEED_BASE + slice_index * COLLECTION_SLICE_STRIDE


def worker_seed_base(rank: int) -> int:
    """Every vectorised worker gets a distinct range. It starts at (rank + 1) * stride, not rank * stride: SB3's
    explicit first-reset seed for worker `rank` is `seed + rank` (0 + rank here), and rank * stride made worker 0
    draw default_rng(0) twice -- once explicitly, once as its own first counter episode."""
    return (rank + 1) * WORKER_SEED_STRIDE


def session_seed_base(rank: int, session: int) -> int:
    """Worker `rank`'s seed base in training session `session` (0 = the first; resumes count up)."""
    if not 0 <= session < MAX_SESSIONS:
        raise ValueError(
            f"training session {session} is outside 0..{MAX_SESSIONS - 1}: its seeds would overflow into the next "
            f"worker's range -- start a fresh run (new --run-root) instead of resuming this one again"
        )
    return worker_seed_base(rank) + session * SESSION_SEED_STRIDE
