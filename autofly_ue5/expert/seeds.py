"""Episode seed ranges for training workers and evaluation, kept disjoint by construction.

`AutoFlyEnv.reset(seed=None)` draws episode `seed_base + k` for its k-th reset; an explicit `reset(seed=s)` draws
episode `s` directly. So every consumer of episodes owns a range of integer seeds, and two consumers that share a
range fly the same episodes. The ranges:

    [0, 1e6)                                  explicit seeds SB3 gives a vec env's first reset in a session:
                                              seed + session * SESSION_SEED_STRIDE + rank, with seed < 99,936
    [(rank+1)*1e6, (rank+2)*1e6)              training worker `rank`, split into MAX_SESSIONS session slices
    [EVAL_SEED_BASE, +1e6)                    the M2 gate's evaluation episodes
    [EVAL_CALLBACK_SEED_BASE, +1e6)           training-time evaluation (model selection)
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
