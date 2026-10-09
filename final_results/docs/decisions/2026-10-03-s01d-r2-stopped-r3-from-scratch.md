# 2026-10-03 — Run 2's warm start collapsed; run 3 trains from scratch

## What happened to run 2

Run 2 (`runs/expert/s01d_r2`, plan `2026-10-03-s01d-r2-plan.md`) warm-started from run 1's best_model. During the
5,000-step policy warm-up, r1's copied policy flew at 0.75–0.85 training success with no out-of-bounds exits, so the
copy was right. Within about 500 gradient updates, training success fell to ~0.45 and out-of-bounds rose to 0.24–0.46.

A probe of r2's 20k checkpoint against its source settled it (`runs/m2d_diag/r2_probe_{r1,20k}.json`): the same 20
episodes, deterministic, probe seeds 300.95e6.

| | Success | Collisions | Out of bounds |
|---|---|---|---|
| r1 best_model | 0.85 | 3 | 0 |
| r2 at 20k | 0.25 | 10 | 5 (4 `altitude_low`) |

Paired by seed, 12 of r1's 17 successes became failures. Run 2 was stopped at 37,680 steps (15:54). Its four
simulators were stopped through their pid records. No training record was written; the run directory and the job log
remain.

## Why: fine-tuning on a fresh buffer, not the new input or the copy

An offline replay of the first 3,000 updates ran on run 2's real buffer: its first 5,000 transitions, r1's warm-up
flights. Every variant started from the same network, and drift was measured on 2,000 held-out observations against
r1's deterministic actions (`runs/m2d_diag/finetune_drift.{py,json}`):

| Variant | Mean drift after 1,000 updates (yaw rate / vertical speed, both ±1) | Actions moved > 0.25 |
|---|---|---|
| As run 2 (mover branch, lr 3e-4) | 0.38 / 0.71 | 88% |
| Without the mover branch (r1's own network) | 0.38 / 0.70 | 88% |
| Learning rate 1e-4 | 0.26 / 0.38 (0.69 at 2,000) | 71% (84% at 2,000) |
| Actor frozen | 0 | 0% |

- **The mover branch is not the cause.** r1's own network, fine-tuned on this buffer, drifts just as much.
- **The copy is not the cause either.** It is exact at step 0.
- **A lower learning rate barely helps.**
- **The cause is fine-tuning on a fresh, narrow buffer.** r1's knowledge came from a 150k-transition buffer full of
  varied failures, while its warm-up flights contain no out-of-bounds exit at all. The critic's errors in states the
  buffer no longer covers pull the actor there, vertical speed first, which is why the probe saw dives.
- **A critic warm-up would only delay this** (a08883f added one; it stays available but unused). The critic would
  forget during the warm-up just the same.

## Run 3

`runs/expert/s01d_r3`, started 2026-10-03 16:00, 12 h, from scratch.

Kept from run 2's plan:
- the mover input (ea15141);
- s01d's r_bounds = 10 (b506a5f; REWARD_VERSION 3);
- validation selection (2299b3a, b9d2ec1).

Changed:
- **No warm start.** It is the lever the experiment ruled out.
- **Replay buffer 250k instead of 150k** (21.2 GB of 62.5 GB available RAM). Run 1's evaluations swung 0.85 → 0.10 →
  0.75, the same forgetting dynamic. A larger buffer keeps older failures in view longer; SB3's own default is 1M.
- **No in-training evaluation.** `scripts/eval_watch.py` scores each 25k checkpoint on the 20 training-time
  evaluation seeds with a simulator of its own, so training never pauses and no idle evaluator renders all run
  (`2026-10-03-training-throughput.md`).

Record: `docs/gates/m2d_r3_train.json`. Afterwards come selection, the gate (`m2d_r3_gate.json`) and the audit,
following runbook-m2d §6b with r3's paths.
