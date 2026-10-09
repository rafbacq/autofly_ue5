# 2026-10-08: every pool episode holds after its spawns, every session warms its targets first; the thin slice measures, the campaign collects

**What the thin slice showed.** Each pool episode records whether its first frame shows its target
(`first_frame_visibility`). Among the first ~130 records: held episodes (a target new to the session, 5 hover steps)
showed it in 33 of 36 judgeable first frames, warm ones (no hold) in 1 of 4. Looked at frame by frame:

- s05 seed 451000001 (held): the bronze whale statue's pixels equal the snow's from record 0 to 16 and turn dark by
  record 20, about 21-25 steps after its first spawn, with three other new objects spawned at once and eight simulators
  loading the host. A fixed 5-step hold cannot bound a first use.
- s01 seed 401000016 (warm): a 4 m garbage truck at 63 m is absent at records 0-1 with the camera level, and present
  near the horizon by record 7.
- The target probe run beside seven collectors (`docs/gates/m4_targets_probe_run2.json`): 59 of 60 targets, primitives
  included, invisible one step after their spawn (5 of 60 in the afternoon's lighter-load run), all 60 visible 5 steps
  later. So under load a spawned object is not drawn in the frame right after its spawn, warm or not.
- Other flags were the check's own: a nearer rock or tree inside its window. The rule now calls those
  `occluded_unmodeled` and keeps `missing` for a window whose pixels read the background behind the target.

**What changed** (`expert/env.py`, `validate/target_probe.py`, `validate/visibility.py`):

- At the first reset of every simulator session (and after every relaunch), with a pool, every (asset, material) pair
  the session could draw is spawned 6-10 m ahead of a clear view and held until the depth camera shows it (at most 60
  steps), then destroyed. First use stays out of the episodes. Live (`runs/collect/smoke_warmup_s05.json`): 48 of 48
  pairs in 18.8 s on a quiet host; under the campaign's load 48-50 pairs took 50-95 hover steps and 20-63 s.
- Every pool episode holds 5 hover steps after its spawns (`prewarm_steps` 5, `first_use` says whether a pair was still
  cold): the render lag after any spawn. Pool-free resets are unchanged (the golden s01 test).

**What the thin slice is for.** Its nine stores (`data/<scene>_c0`, 450 episodes) were collected before the change: warm
episodes may show a newly spawned object appear within their first records. They measure the pilots' kept-rates
(`docs/gates/plan6_pilots.json`) and tested the export, the figures and the loader on real pool data, and they stay as
they are, but the final dataset is collected afresh with the new reset: four 50-kept stores per scene
(`runs/collect/campaign_final.json`, batches 1-4), launched 2026-10-08 22:02 on slots 0-6, longest jobs first.
