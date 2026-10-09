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

## Revision 1 (2026-10-08, after an independent review of this plan): what changed and why

The review's points were checked against the code and adopted, except where noted.

1. **One merged dataset, VLA-tested.** `rlds.export_rlds` writes one TFDS dataset per store, split "train" only, with
   `episode_metadata` holding only the file path: 9-18 per-scene datasets would each need their own OpenVLA
   registration and normalisation key, and none would have a validation split. So the raw stores stay per scene and
   batch, but the export is **one** dataset (`autofly_ue5_proto`, 1.0.0): train and val splits (val about 5 % of
   episodes, stratified by scene x target, episode ids frozen in a file), episode metadata extended with episode id,
   scene, target name, category, seen, status and seed, and `dataset_statistics.json` in OpenVLA's schema computed on
   train only. Done means the merged dataset loads through an OpenVLA-style RLDS pipeline (dlimp, CPU, in its own venv)
   as well as through TFDS; a LeRobot conversion of a few episodes is optional. Exporting only the merged dataset also
   saves the second copy of every frame per store.
2. **A thin slice first (replaces step 1's role as a kept-rate measure).** The screening runs through the gate: no a0,
   the orange cylinder as the only target, no settle steps, and 40 episodes give about +-0.15. It ranks the candidate
   pilots, nothing more. The kept-rate is measured by a thin slice: about 50 kept episodes per scene under true
   collection conditions (the pool, a0, stochastic, settle steps), reported with a Wilson lower bound, per target too
   (to catch a target the pilot fails on more often), with a check that kept flights cross the field (detour and PER),
   and with an automated first-frame visibility check (which replaces the separate re-probe of 2d). The merged export,
   the statistics, the loader check and the harness are run on the thin slice before the campaign tops it up.
3. **No dedicated trainings; the bar is a kept-rate of 0.25.** Collection runs at the same saturated 14.9 steps/s as
   training, so 200 kept episodes cost about 1 h per scene at a kept-rate of 1.0 and 3-4 h at 0.25-0.3: a 6-8 h
   training to lift a scene from 0.3 to 0.6 never pays for itself, and it would share the GPU with collection. Step 3 is
   dropped; only a scene below 0.25 with every candidate would reopen it. **Done is 200 kept episodes per training
   scene; 400 is a stretch goal** if time allows.
4. **Privileged state.** state[0] and state[1] are the distance and the bearing to the target from its true position
   (AutoFly's own definition, kept as the record's). A VLA that reads the state could fly from them alone. The card
   marks them as privileged, and the harness zeroes them by default (an option includes them); baselines are reported
   both ways.
5. **Unseen targets are evaluated.** `TargetPool.for_split` draws unseen targets only for test_unseen scenes, which are
   out of scope. The harness gets a split override: the test_seen layouts (s05r; s06r once gated) and seen scenes with
   unseen targets, on a frozen evaluation episode file under a new seed base registered in `expert/seeds.py`, at the
   task's rules, in lock-step (a policy's latency never changes the dynamics).
6. **Rebalancing is a quality check, not a blocker.** OpenVLA and LeRobot do not use per-step weights. Its value here is
   finding targets the detector never finds (small or low-poly ones). It runs last, while no simulator runs, and its
   sparse search is first validated against the dense scores the s01 pilot already has (the same transition on at least
   99 of 100 episodes).
7. **Target exclusion by shape class, in data.** Not only the primitive matching the obstacles: barrels read as pillars
   and poles (s01, s01d, s09), the cabinet and the tool chest as boxes (s07), spheres as painted stones (s04). A table
   per scene, with a test; and the placeholder status goes into episode metadata so the Kenney stand-ins can be
   filtered out later.
8. **Language.** Two verbatim templates and one obstacle phrase per scene make every instruction a function of scene and
   target; the card says so. Paraphrases are not added in this plan (they would leave AutoFly's record convention).
9. **Cuts.** s06r's gate is deferred (the harness starts on s05r); `final_results` holds one auto-generated video per
   scene rather than hand-picked sets.

### Done checks (one per item)

| Item | Check |
|---|---|
| Scenes | The gate records pass: s01-s07, s09, s01d's mover probe, s05r |
| Targets | The thin slice's first-frame visibility check passes for every drawn target, and every target was spawned live |
| Pilots | `pilots.json`: each scene's checkpoint sha256 and its kept-rate under collection conditions, Wilson lower bound >= 0.25, with the crossing check |
| Dataset | The validator passes on every store; the merged dataset holds >= 200 episodes per training scene and every drawn seen target; TFDS reads it back |
| VLA-ready | The OpenVLA-style loader iterates the merged dataset; `dataset_statistics.json` matches the schema; the expert flown through the harness lands within the CI of its kept-rate and the scripted baseline below it |
| Rebalancing | The sparse search matches the dense scores on the s01 pilot; `rebalance.json` for the merged dataset |
| final_results | Every link in its README resolves |
