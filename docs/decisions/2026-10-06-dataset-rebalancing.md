# 2026-10-06 — Dataset rebalancing (spec §10.3): how the phases are labelled, and the first measurement

Built on 2026-10-06 while run 6's gate held slots 0–4, as the next piece of the paper that needs no simulator, no GPU
and no pending decision: AutoFly's rebalancing (App. A.2.4) is what the paper does with its trajectories right after
collecting them, the M3 pilot (`data/s01_pilot`, 100 episodes) is already there to run it on, and spec §10.3 had fixed
the method. M4's scenes wait on the asset-licence choice (`2026-10-02-m4-asset-survey.md`) and on editor work that
would replace the binary live simulators run from.

Code: `autofly_ue5/dataset/rebalance.py`, `scripts/detect_targets.py`, `scripts/rebalance_dataset.py`;
tests: `tests/test_rebalance.py`; procedure: `docs/runbook-rebalance.md`.

## Decisions

1. **The transition is one-way, at the first confident detection.** The paper says both "transitioning phases upon
   confident detection" and "φ = 2 if the target is detected, otherwise 1". Spec §10.3 chose the first reading, and
   the code follows it: a record is target seeking from the first record whose score exceeds the threshold. A
   per-record label would flip phases whenever the detector flickers, and a resampling weight cannot mean that. How
   often the detector does flicker after the transition is reported as `detection_persistence`.
2. **The query is the target's name, not the whole instruction.** The paper queries Grounding DINO "with the
   instruction"; spec §10.3 says "the instruction's target". Grounding DINO scores each phrase of its prompt
   separately, and a prompt holding "white pillars", "obstacles" and "orange cylinder" would detect pillars in every
   frame. The prompt is the target name in Grounding DINO's convention: lower case, closed by a period
   (`"orange cylinder."`).
3. **The score is the model's own box confidence**: max over its 900 queries of max over text tokens of
   sigmoid(logit), which is what Grounding DINO's inference utility thresholds (`box_threshold`) and what the
   transformers processor returns from `post_process_grounded_object_detection`. Checked equal on pilot frames
   (0.28757 both ways). Batched and single-frame scores agree to 1e-7.
4. **Scores are cached per frame, the threshold is applied later.** Detection is the expensive stage (hours on the
   CPU); the threshold is a one-line parameter of the cheap stage, so 0.7 can be questioned against the data without
   scoring anything again. The cache (`detections.json`) is refused for another store or another detector (model,
   revision, transformers version, input size), and is resumed after every episode: this host freezes (MEMORY.md).
5. **CPU by default.** A torch process on the shared GPU is a foreign job to `gpu.check_gpu_for_launch`: with more
   than 2 GB held and none of our simulators up, every simulator launch is refused. The detector therefore hides
   CUDA unless `--device` says otherwise, and the runbook says when the GPU may be used.
6. **Two privileged checks ride along, because the simulator knows where the target is.** At each transition the
   result records the distance to the target (state[0]) and whether the detected box covered the target's bearing
   (state[1], through the 90° pinhole). A detector that fires on a distractor or a pillar is thereby caught rather
   than trusted. These are diagnostics in `rebalance.json`; the weights do not use them.
7. **The paper's KL figure is not reproduced, by design.** Eq. 10 on the paper's own P0 = (0.73, 0.27) gives
   0.110 nats; the paper prints "approximately 0.36 nats". No base or convention of that equation gives 0.36 for
   those numbers (0.36 nats would need P0 ≈ (0.90, 0.10)). The code follows the equation and the test pins both
   figures. The weights, which the paper prints consistently, come out as its 0.68 / 1.85.
8. **`rebalance.json` is written exclusively and whole**, never over an existing one (synced in a private file and
   linked into place, so a freeze leaves no file rather than a truncated one). A partial result (a probe over some
   episodes) needs `--allow-partial` and an `--out` outside the store, and a degenerate one (a whole phase empty) is
   refused at the store's own path: neither can pose as the store's weights.
9. **An independent review (Opus, 2026-10-06) confirmed the formulas and the split and tightened the guards**: the
   partial rule had accepted an `--out` inside the store, the writer was not atomic, a degenerate result or a
   threshold typed as 70 went to the store's path with exit 0, the detections cache had no lock against two scorers,
   and an out-of-view "detection" was filed as undecidable rather than as the false positive it is. All fixed in
   52ab313 with tests; its scale note (the cache is rewritten whole per episode, quadratic at 13K episodes) is left
   for M5.

## The 2026-10-06 probe: ten pilot episodes on the CPU

Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny` @ a2bb814d, transformers 5.19.0, the processor's default
800 px input) on the first ten episodes of `data/s01_pilot`, 8 CPU threads at nice 19 beside five live simulators.

Job `rebalance_detect_probe`; outputs `runs/rebalance/s01_pilot_probe/{detections,rebalance}.json` (PARTIAL: 10 of
100 episodes; the store's own `rebalance.json` is not written).

| | |
|---|---|
| Scored | 1,801 records of 10 episodes in 5,317 s (0.34 frames/s). The five live simulators kept 188–200 % CPU each |
| At 0.7, one-way split | P0 = (0.643, 0.357), KL 0.041 nats, weights (0.778, 1.400), resample sizes (8, 14) of (10, 10) sub-trajectories. The weights balance the records exactly: 0.778 × 1,158 = 1.400 × 643 = 900.5 |
| At 0.7, per record | 503 of 1,801 records above 0.7: P0(2) = 0.279. The paper's 0.27 |
| Never detected | 0 of 10. Every episode's maximum lies in 0.868–0.893 |
| On target | 10 of 10 first detections cover the target's bearing. Of all 503 records above 0.7 with the target in view, 0 boxes sit elsewhere: at 0.7 the detector never fires on a pillar or a distractor |
| Transition | median 24.6 m from the target (11.4–53.8 m); records 31–184 of episodes of 158–234 records |
| Persistence | 0.78: between about 50 m and 20 m the score hovers around 0.6–0.7, and it settles above 0.8 within about 18 m |

Threshold sweep on the same scores (one-way split; weights for a uniform target; the per-record column is the paper's
other wording):

| threshold | P0 | weights | per-record P0(2) | persistence | first detections on target | median distance |
|---|---|---|---|---|---|---|
| 0.50 | 0.062 / 0.938 | 8.04 / 0.533 | 0.535 | 0.57 | 3 of 10 | 64 m |
| 0.60 | 0.203 / 0.797 | 2.47 / 0.627 | 0.385 | 0.48 | 3 of 10 | 59 m |
| 0.65 | 0.453 / 0.547 | 1.105 / 0.913 | 0.339 | 0.62 | 9 of 10 | 37 m |
| **0.70** | 0.643 / 0.357 | 0.778 / 1.400 | 0.279 | 0.78 | 10 of 10 | 25 m |
| 0.75 | 0.747 / 0.253 | 0.670 / 1.975 | 0.222 | 0.88 | 10 of 10 | 21 m |
| 0.80 | 0.820 / 0.180 | 0.610 / 2.771 | 0.148 | 0.82 | 10 of 10 | 17 m |
| 0.85 | 0.889 / 0.111 | 0.562 / 4.503 | 0.087 | 0.78 | 10 of 10 | 12 m |
| 0.90 | 1.000 / 0.000 | 0.5 / none | 0 | none | never detected | none |

What it says:

- **The detector works on this scene.** At 0.7 and above, every confident detection is the target; below 0.65 the
  first "detections", at about 60 m, are boxes elsewhere in the image. 0.7 is the lowest threshold that is clean here.
- **Our split is less skewed than the paper's.** The paper measured (0.73, 0.27); the pilot's one-way split at 0.7 is
  (0.64, 0.36), because the score first crosses 0.7 far out (median 25 m, once at 54 m) and then hovers below it for a
  while (persistence 0.78). The placeholder target is a 2 m orange cylinder alone in its colour against a grid floor
  and the sky, with a0 pointing the camera at it from the first record, so it is visible from much farther than the
  paper's varied objects among trees and buildings. The per-record reading gives 0.28, the paper's number, and 0.75
  gives (0.75, 0.25) with a firmer transition (persistence 0.88).
- **Nothing is tuned.** The default stays spec §10.3's 0.7 and the one-way split. The cache makes any other choice a
  one-line rerun. M5's choice, once the real target pool exists: 0.7 as is, a higher threshold, or a persistence
  requirement (the first record from which the score stays above the threshold for k records), each measured on the
  real targets' `first_detection.off_target` and `detection_persistence`.

## What is left

- Finish the pilot (90 episodes): about 12 h on the CPU, or on the GPU (far faster; unmeasured) when no simulator
  needs to launch (runbook §2). Then `rebalance_dataset.py --raw data/s01_pilot` writes the store's own `rebalance.json`.
- M5 decides the threshold against real target pools: this probe's target is the placeholder orange cylinder, alone
  in its colour. With 60 targets and 3–5 distractors per scene, `first_detection.off_target` is the number to watch.
- Applying the weights is the training pipeline's job (a later LeRobot/OpenVLA export): `stratified_resample` draws
  the paper's rebalanced set from the sub-trajectories, with replacement where a phase is upsampled.
