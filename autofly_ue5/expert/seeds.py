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
    [SELECTION_SEED_BASE, +1e6)               checkpoint selection after training (validation; scripts/select_checkpoint.py)
    [COLLECTION_SEED_BASE, +1e8)              dataset collection (M3/M5): never a training, gate or probe episode,
                                              in 100 slices of 1e6, one per (scene, collection run):
                                              collection_seed_base(k). Slice 0 is s01's pilot; slice 99 is for
                                              smokes and diagnostics, never a dataset; a campaign's batches take
                                              campaign_seed_slice(scene, batch), slices 1-90
    [EVAL_HARNESS_SEED_BASE, +1e7)            the closed-loop evaluation harness's frozen episode files (M6, plan 6;
                                              autofly_ue5/evaluation/episodes.py): one 1e6 slice per condition, one
                                              1e4 sub-slice per scene within it: eval_harness_seed_base(c, s)
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
# Checkpoint selection after a run (2026-10-03): many checkpoints scored on held-out validation episodes, so the gate's
# EVAL_SEED_BASE episodes stay untouched until the chosen one is gated. Training-time evaluation (20 episodes from
# EVAL_CALLBACK_SEED_BASE) picked s01d_r1's best_model at 0.85; the gate then measured 0.775.
SELECTION_SEED_BASE = 500_000_000
# Dataset collection (spec §9-§10): its episodes must be none the experts trained or were gated on. 100 slices of 1e6,
# one per (scene, collection run) as M5 needs them.
COLLECTION_SEED_BASE = 400_000_000
COLLECTION_SLICE_STRIDE = 1_000_000
MAX_COLLECTION_SLICES = 100
DIAGNOSTIC_COLLECTION_SLICE = MAX_COLLECTION_SLICES - 1
# A collection campaign's batches (plan 6 step 2e, scripts/collect_campaign.py): every (scene, batch) owns one slice,
# scene-major in this fixed order, slice = 1 + CAMPAIGN_BATCHES_PER_SCENE * (the scene's index) + batch:
#     s01 1-10, s01d 11-20, s02 21-30, s03 31-40, s04 41-50, s05 51-60, s06 61-70, s07 71-80, s09 81-90.
# So no two batches of any campaign fly the same episodes (s01 and s01d share a layout and a draw order), a store's
# seeds say which scene and batch it is, and neither the M3 pilot's slice 0 nor the smokes' 99 is ever reused; 91-98
# are unassigned. A scene appended to the order takes the next ten slices; reordering would rename every batch.
CAMPAIGN_SCENE_ORDER = ("s01", "s01d", "s02", "s03", "s04", "s05", "s06", "s07", "s09")
CAMPAIGN_BATCHES_PER_SCENE = 10
# The evaluation harness (plan 6 item 6, revision 1 item 5): a policy is scored on frozen files of (scene, seed) per
# condition, and none of those episodes may be one an expert trained, was gated or selected on, or that a dataset holds,
# or a student's score would measure memory. A slice per condition and a sub-slice per scene, so no two files and no two
# scenes share a seed: s01 and s01d share a layout and a draw order, and one seed would fly the same setup in both.
EVAL_HARNESS_SEED_BASE = 600_000_000
EVAL_HARNESS_CONDITION_STRIDE = 1_000_000
EVAL_HARNESS_SCENE_STRIDE = 10_000
MAX_EVAL_HARNESS_CONDITIONS = 10
MAX_EVAL_HARNESS_SCENES = EVAL_HARNESS_CONDITION_STRIDE // EVAL_HARNESS_SCENE_STRIDE
EVAL_HARNESS_SPAN = MAX_EVAL_HARNESS_CONDITIONS * EVAL_HARNESS_CONDITION_STRIDE


def eval_harness_seed_base(condition_slot: int, scene_slot: int) -> int:
    """The first seed of scene slot `scene_slot` in condition slot `condition_slot` of the harness's range (the slots
    are evaluation/episodes.py's CONDITION_SLOTS and SCENE_SLOTS, append-only)."""
    if not 0 <= condition_slot < MAX_EVAL_HARNESS_CONDITIONS:
        raise ValueError(f"evaluation condition slot must be in [0, {MAX_EVAL_HARNESS_CONDITIONS}), got {condition_slot}")
    if not 0 <= scene_slot < MAX_EVAL_HARNESS_SCENES:
        raise ValueError(f"evaluation scene slot must be in [0, {MAX_EVAL_HARNESS_SCENES}), got {scene_slot}")
    return EVAL_HARNESS_SEED_BASE + condition_slot * EVAL_HARNESS_CONDITION_STRIDE + scene_slot * EVAL_HARNESS_SCENE_STRIDE


def collection_seed_base(slice_index: int) -> int:
    """The first seed of collection slice `slice_index`. Each collection run takes its own slice: re-runs, and s01
    against s01d (same layout, same draw order), would otherwise fly identical setups (the M3 review, 2026-10-03)."""
    if not 0 <= slice_index < MAX_COLLECTION_SLICES:
        raise ValueError(f"collection slice must be in [0, {MAX_COLLECTION_SLICES}), got {slice_index}")
    return COLLECTION_SEED_BASE + slice_index * COLLECTION_SLICE_STRIDE


def campaign_seed_slice(scene_id: str, batch: int) -> int:
    """The collection slice of batch `batch` of scene `scene_id` in a collection campaign (see CAMPAIGN_SCENE_ORDER)."""
    if scene_id not in CAMPAIGN_SCENE_ORDER:
        raise ValueError(f"scene {scene_id!r} has no campaign slices; the campaign's scenes are {CAMPAIGN_SCENE_ORDER}")
    if not 0 <= batch < CAMPAIGN_BATCHES_PER_SCENE:
        raise ValueError(f"campaign batch must be in [0, {CAMPAIGN_BATCHES_PER_SCENE}), got {batch}")
    return 1 + CAMPAIGN_BATCHES_PER_SCENE * CAMPAIGN_SCENE_ORDER.index(scene_id) + batch


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
