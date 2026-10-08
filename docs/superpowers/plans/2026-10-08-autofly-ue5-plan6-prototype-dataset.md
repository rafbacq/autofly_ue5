# Plan 6: finish the prototype: diverse, VLA-ready data collection

> **Status:** written 2026-10-08 evening on the user's direction after a meeting with their lab mentor: *diversity of
> objects and scenes and a working prototype matter more than expert quality or many scenes; the paper need not be
> replicated exactly; data collection is the priority; attaching a VLA is the next phase, not part of this plan.*
> This plan supersedes the remaining items of plans 5 (M4) and the M5/M6 milestones of the design (spec §12) where
> they conflict. Each step is test-first and recorded as before; the evidence rules of CLAUDE.md are unchanged.

## What "complete" means

The project is complete for this phase when all of these hold:

1. **Scenes.** The nine training scenes are built and live-gated: s01 white pillars, s01d moving pillars, s02 sparse
   trees, s03 scattered rocks, s04 stone field, s05 standing stones (snow), s06 tree clusters, s07 stacked boxes, s09
   coloured poles; and the two test_seen layouts s05r and s06r. Done except s06r's gate (a probe-tooling gap).
2. **Targets.** All 60 pool targets are drawable (35 now; the 25 Kenney stand-ins added for category diversity) and
   verified live, including that a target is visible in the first recorded frame of an episode.
3. **A pilot per training scene** that collects at a kept-rate of at least 0.5 (stochastic, a0-aligned, under the
   task's rules): the best existing expert where one transfers, a short dedicated training where none does.
4. **The dataset.** About 400 kept episodes per training scene (about 3,600 episodes, about 0.6 M records), in
   AutoFly's format (RGB 256x256, instruction naming the drawn target, action[3], state[9]), one store per scene and
   batch, every store passing the validator, exported to RLDS and read back exactly by TFDS, with full provenance
   (scene, layout, target and distractors, pilot checkpoint and its sha256).
5. **Rebalancing** per store (spec §10.3, Grounding DINO at 0.7) and a combined summary.
6. **VLA-ready.** A dataset card (counts by scene, target and category; instruction templates; splits), per-dimension
   normalisation statistics for actions and states (q01/q99, mean/std, as OpenVLA's dataset_statistics), a fixed
   train/validation split, and a closed-loop evaluation harness: a policy interface `(rgb, instruction, state) ->
   action`, test episodes on s05r/s06r and the seen scenes, success/collision/path-efficiency metrics, proven end to
   end with a scripted baseline and an expert. No VLA is attached in this plan.
7. **`final_results/`**: one folder with the README, every important figure, flight and dataset videos for every
   scene, the records and the dataset card.

## Deliberately out of scope now

- Paper-grade experts (0.95 gates) for every scene: s02 run 2, further s01d runs. A gate that failed stays failed and
  is reported as such; a pilot only has to collect.
- s08 ruins and s10 walls/containers (no assets yet), s11/s12 unseen building scenes and photoreal vehicles and people
  (Fab, with the user at the browser). They can be added later as extra scene files without changing anything here.
- Attaching or training a VLA.

## Steps

**0. Finish what is in flight (tonight).**
- Apply the outline contact rule (reviewed twice; 1202 tests in its worktree): the task's rule in tree and rock
  scenes, which the collector rejects by.
- Fly s02 run 1's pick under the outline rule on the gate's 200 episodes, as the pre-registered separately named
  record (`docs/gates/m5_s02_gate_outline.json`). It also measures that pilot under the rule collection will use.

**1. Pilot screening (GPU, about 2 h).** Each candidate expert flies 40 stochastic episodes per scene on the probe seed
range (300e6, offset per scene), under the task's rules; the kept-rate decides. Candidates: s01's best_model (pillars,
0.98), the accepted s01d mean (pillars and movers, 0.935), s02 run 1's pick (trees). Scenes: s02, s03, s04, s05, s06,
s07, s09 (s01 and s01d fly their own experts). The best candidate per scene with a kept-rate of at least 0.5 becomes
that scene's pilot (`docs/gates/m5_pilot_screening.json`). Started 2026-10-08 18:00 with s07 and s09.

**2. Collection readiness (offline, agents, in parallel with step 1).**
- a. *First-frame visibility:* when an episode spawns a pool asset for the first time in a simulator session, the reset
  holds about five more hover steps before the first recorded frame (measured: invisible on step 1, visible by step 5).
  Pool-free episodes are untouched (golden s01).
- b. *Draw the Kenney stand-ins* (cars, trucks, people, animals: the vehicle and human categories, 22 % and 17 % of the
  paper's mix, have no other source until a Fab session). Taken under the mentor's guidance on diversity; reversible by
  one apply step.
- c. *No target that looks like the scene's own obstacles:* cylinders and cones are not drawn in the pillar and pole
  scenes (s01, s01d, s09), cubes not in the box scene (s07), so the instruction's target is never a pillar.
- d. *Re-probe all 60 targets* with settle steps (`docs/gates/m4_targets_probe_run2.json`).
- e. *A collection campaign runner:* per-scene collection seed slices (`expert/seeds.py`), N collector processes on N
  slots into per-scene stores, a failure alarm, and a tool that merges the stores' manifests into one dataset card.
- f. s06r's gate: give the 6 m depth probe fallback distances, as the 12 m one has.

**3. Short dedicated trainings, only where needed.** A scene with no pilot at 0.5 gets the M5 recipe for 6-8 h (good
enough, not 0.95), or, if three or more scenes need one, a single run over several of them. Decided from step 1's
matrix and recorded before any run starts.

**4. Collection campaign (GPU, about 10-14 h).** 400 kept episodes per training scene, four parallel slots, s01
re-collected with the real pool (the M3 pilot keeps its placeholder target and stays as it is). Validator per store,
RLDS export, TFDS read-back. Storage: about 3.4 GB per 100 episodes in both forms, about 125 GB in all (370 GB free).

**5. Rebalancing (GPU while no simulator runs, about 2-4 h).** Grounding DINO on the GPU with a sparse search for each
episode's first confident detection (score every 8th record, then the ones around the first crossing), which is all
the one-way phase split needs; per-store `rebalance.json` and a combined summary.

**6. VLA readiness (offline, agents, in parallel with step 4).** The dataset card, the normalisation statistics, the
split, and the evaluation harness (M6, minimal), proven end to end on s05r/s06r with a scripted baseline and an expert.

**7. `final_results/`.** Everything a reader needs, with flight videos of every scene's pilot and dataset playbacks.

## Timeline (estimate)

Steps 0-2 tonight; step 3 none to two short runs (0-16 h); step 4 about 12 h; step 5 about 3 h; step 6 alongside; step 7
about 2 h. About two days of mostly unattended compute.

## Risks

- A pilot transfers poorly to the rock scenes: then step 3 trains, which adds 6-8 h per run.
- Lower kept-rates slow collection proportionally; the collector's `max_attempts` covers it.
- Grounding DINO on the GPU is a foreign job to a simulator launch: it runs only while no simulator runs.
- The host can freeze (MEMORY.md): every store and record is written durably and a campaign resumes per store.
