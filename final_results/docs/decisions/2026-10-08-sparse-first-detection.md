# 2026-10-08: the rebalancing scores only around each episode's first detection (stride 8, alert 0.5)

**Why.** The phase split needs one number per episode: the first record whose target confidence exceeds 0.7. Scoring
every record costs about 177 detector calls per episode at 0.36 frames/s on this CPU (the 100-episode pilot took
14.5 h). The prototype dataset will hold about 1,800 episodes with 60 different targets.

**What was measured** (`runs/rebalance/sparse_study_2026-10-08/`, `scripts/sparse_search_study.py`, on the pilot's
dense scores, 100 episodes):

- A plain coarse scan (stride S) followed by a fine scan of the window before its first hit misses the first detection
  in 24 to 44 of 100 episodes for every S above 1: in 48 episodes the first record above 0.7 is a one-record flicker
  among records just below it.
- "Alert windows" keep it: probes at 0, S, 2S, ... and the last record; the records between two probes are scored when
  either probe scored above an alert level; the first record above the threshold is the transition.
- Stride 8 with alert 0.5 scores 59.4 records per episode (max 120), three times fewer than dense, and changes no
  transition, no phase label, P0 or weight on the pilot. Over every offset of the probe grid it would change 0.38 of
  100 transitions; its only misses are one- and two-record detections between probes that scored 0.39-0.49.
- Live, on the CPU: three pilot episodes scored sparsely gave first detections at records 46, 147 and 131, equal to the
  dense run, visiting 210 frames instead of 535.

**Decision.** The prototype dataset's rebalancing uses stride 8, alert 0.5, `--batch 1` (see the correction in
`2026-10-06-dataset-rebalancing.md`). Alert 0.55 would cost 43.6 calls but its 1.00 per 100 grid-offset changes is
exactly the plan's allowance, with no margin for targets other than the orange cylinder. The 0.2 alert margin is
validated on the orange cylinder only: the first pool stores get a few dense episodes per target (`--check-sparse`).
GPU throughput is unmeasured; the GPU is used only while no simulator runs.
