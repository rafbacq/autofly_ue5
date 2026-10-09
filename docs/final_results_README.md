# autofly_ue5: final results

**A diverse, VLA-ready UAV navigation dataset collected in Unreal Engine 5, with the closed-loop harness a VLA plugs
into.** This project recreates the simulation side of AutoFly (arXiv 2602.09657) on Unreal Engine 5.7 and Project
AirSim. Expert pilots trained here fly procedurally built obstacle scenes toward a named object; every successful,
collision-free flight is stored in AutoFly's own format: at 5 Hz, a 256x256 RGB frame, the language instruction,
action[3] and state[9].

The direction for this phase (2026-10-08) put diversity of objects and scenes and a working prototype that a VLA can be
attached to ahead of paper-grade experts. All of it is done. What falls short, and why, is listed under
[Known limits](#known-limits).

![pipeline](figures/pipeline.png)

## Headline numbers

| What | Result |
|---|---|
| Dataset | **1,800 episodes, 339,007 records**, one TFDS/RLDS dataset `autofly_ue5_proto` 1.0.0 (train 1,710 episodes / 322,125 records, val 90 / 16,882) |
| Diversity | 9 training scenes, 200 episodes each; **50 targets** in 6 categories (people 386 episodes, shapes 374, furniture 355, vehicles 324, equipment 321, statue 40); 10 more targets held out as unseen |
| VLA-ready | TFDS reads every episode back exactly; an OpenVLA-style RLDS loader (dlimp, CPU) iterates both splits; `dataset_statistics.json` in OpenVLA's schema |
| Collection | 2,709 flights, 1,800 kept (66 %); per-scene kept-rates 0.38-0.98 |
| Closed loop | each scene's pilot reproduces its kept-rate in the evaluation harness (all 9 scenes); a privileged straight-line baseline: 0.06 |
| Rebalancing | phase split 0.69 / 0.31 (the paper: 0.73 / 0.27), weights 0.72 / 1.62; every target detected in some episode |

Videos to start with: [`videos/dataset/montage_9_scenes.mp4`](videos/dataset/montage_9_scenes.mp4) (one kept flight per
scene, side by side, with its instruction) and the per-scene dataset playbacks in [`videos/dataset/`](videos/dataset/).

## 1. Scenes and objects

Nine training scenes and two held-out layouts (s05r, s06r), each a procedurally laid-out obstacle field built and
packaged in UE 5.7. Every one passed its live checks: the map matches its layout, depth matches the geometry within
centimetres, commands are tracked, collisions register (`records/gates/m4_*_gate.json`).

| Scene | Obstacles | Pilot | Kept / flown |
|---|---|---|---|
| s01 white pillars | 80 pillars | s01 expert | 200 / 212 (0.94) |
| s01d moving pillars | s01's pillars, 8-12 of them moving | s01d expert | 200 / 228 (0.88) |
| s02 sparse trees | 27 trees | s02 tree expert | 200 / 254 (0.79) |
| s03 scattered rocks | 29 boulders | s02 tree expert | 200 / 312 (0.64) |
| s04 stone field | 33 stones, some painted | s02 tree expert | 200 / 374 (0.53) |
| s05 standing stones, snow | 24 standing stones | s02 tree expert | 200 / 382 (0.52) |
| s06 tree clusters | 40 trees and bushes | s02 tree expert | 200 / 533 (0.38) |
| s07 stacked boxes | 60 crates in stacks | s01d expert | 200 / 204 (0.98) |
| s09 coloured poles | 70 coloured poles | s01d expert | 200 / 210 (0.95) |

![layouts](figures/scenes/layouts.png)
![layouts of the rock and tree scenes](figures/scenes/layouts_c4.png)

The drone's own cameras (RGB and the depth the experts fly on), one sheet per scene, are in
[`figures/scenes/snapshots/`](figures/scenes/snapshots/), for example the snow scene:

![s05 snapshots](figures/scenes/snapshots/s05_snapshots.png)

**The target pool** holds 60 objects: Poly Haven furniture, equipment, a covered car and a statue, coloured primitives,
and 25 Kenney low-poly stand-ins for vehicles, people and animals. Vehicles and people are 22 % and 17 % of AutoFly's
own mix, and photoreal ones need a manual Fab session. A live probe spawned every object in front of the drone and found
each in the spawn table, at its registry size and in the depth frame (`records/gates/m4_targets_probe_run2.json`). An
object that looks like a scene's own obstacles is never drawn there: cylinders and barrels among pillars and poles,
cubes and cabinets among boxes, spheres among stones (`dataset/target_exclusions.json`).

![targets](figures/dataset/targets_closeup.png)

## 2. The expert pilots

The pilots are SAC policies flying on privileged inputs: depth, the target's true position and, on s01d, the movers.
Each expert's 200-episode deterministic gate is kept as recorded, failures included:

| Expert | Gate (bar 0.95) | Used for |
|---|---|---|
| s01 (`s01_r2` best_model) | **0.98** pass (`records/gates/m2_gate.json`) | s01 |
| s01d (mean of run 6's 250k-550k checkpoints) | **0.935** fail; accepted for collection on 2026-10-07 | s01d, s07, s09 |
| s02 (mean of run 1's 350k-end checkpoints) | **0.580** fail under the old circle rule; **0.895** under the outline rule collection uses | s02, s03, s04, s05, s06 |

No expert was trained for s03-s07 or s09: a screening of the existing experts on each new scene chose the best one
(`records/gates/m5_pilot_screening.json`), and the thin slice measured its kept-rate under real collection conditions
(`records/gates/plan6_pilots.json`, every scene above the bar of a 0.25 lower bound).

![gates](figures/experts/gates.png)
![pilot screening](figures/pilots/pilot_screening.png)

**Why mesh scenes end at the outline.** A crash into a tree or rock can wedge the simulator's drone beyond any reset,
so on tree and rock scenes an episode ends 1.05 m from the object's real outline (the drone's measured collision reach
0.63 m plus a full-speed step), before contact. The first rule used a circle around each model, which ended flights a
median 2.7 m from any branch; that is why s02's expert gated 0.580 under it and 0.895 under the outline.

![footprint rule](figures/contact_rule/s02_footprint_rule.png)

Expert flights with RGB, depth and a top-down map: [`videos/experts/`](videos/experts/) (s01, s01d, s02) and
[`videos/pilots/`](videos/pilots/) (each pilot on s03-s07, s09).

## 3. The dataset

1,800 episodes, 200 per scene, collected stochastically by each scene's pilot from an aligned start (a0), with the
target and 3-5 distractors drawn from the pool and named in the instruction. Only successful, collision-free flights are
kept; rejected ones are recorded, never stored as episodes.

![composition](figures/dataset/composition.png)
![kept-rates](figures/dataset/kept_rates.png)

Kept flights cross the obstacle field (84-94 % of them enter its core in every scene); a trajectory map per scene is in
[`figures/dataset/`](figures/dataset/):

![s06 trajectories](figures/dataset/trajectories_s06.png)

![lengths and path efficiency](figures/dataset/lengths_per.png)
![actions and states](figures/dataset/actions_states.png)

Episodes are 171-195 records long (median by scene), 66-71 m flown. The experts fly close to the shortest path (path
efficiency 0.90-0.99 by scene): the paper's learned models score 0.73-0.78, so the data shows near-optimal flight for a
VLA to imitate. What the drone sees first, per scene: `figures/dataset/first_frames_<scene>.png`.

## 4. VLA readiness

**Format.** `autofly_ue5_proto` 1.0.0 is a TFDS dataset in AutoFly's released RLDS layout: each step holds
`observation/image_0` (256x256x3 PNG), `observation/state` (9 floats), `action` (3 floats) and `language_instruction`.
Each episode's metadata adds the episode id, store, scene, target name, category, seen flag, stand-in flag, seed and
the pilot's checkpoint hash. The validation split is about 5 % of episodes, stratified by scene and target, and frozen
in `dataset/autofly_ue5_proto_split.json`. Details: [`dataset/dataset_card.md`](dataset/dataset_card.md).

| Field | Meaning |
|---|---|
| action[0..2] | forward speed (m/s, 0 to 2), yaw rate (rad/s, -1 to 1), vertical speed (m/s, -1 to 1) |
| state[0] | horizontal distance to the target's centre, m: **privileged** (see below) |
| state[1] | bearing of the target from the drone's heading, rad, positive right: **privileged** |
| state[2] | altitude above ground minus 1.47 m |
| state[3], state[4], state[5] | speed (m/s), vertical velocity (m/s, up positive), yaw rate (rad/s) |
| state[6], state[7], state[8] | north and east offset from the episode's start (m), altitude above ground (m) |

**Loading it.** The dataset lives at `data/autofly_ue5_proto/1.0.0/` on the GPU host (32.3 GB in 113 shards):

```python
import tensorflow_datasets as tfds
ds = tfds.builder_from_directory("data/autofly_ue5_proto/1.0.0").as_dataset(split="train")
```

For OpenVLA, `dataset_statistics.json` holds the q01/q99 bounds per action and state dimension (train only). OpenVLA's
`make_oxe_dataset_kwargs` assumes 7-dim end-effector actions: pass `action_normalization_mask=[True] * 3` and no
absolute-action mask. The loader check (`dataset/vla_loader_check.json`) ran OpenVLA's own RLDS pipeline steps
(dlimp, BOUNDS_Q99 normalisation, 224x224 resize) over both splits.

**The privileged state.** state[0] and state[1] come from the target's true position. A policy that reads them can fly
to the target without looking; the evaluation harness zeroes them by default.

## 5. Closed-loop evaluation

`autofly_ue5/evaluation/` is where a VLA plugs in: a `Policy` takes `(rgb, instruction, state)` and returns the action,
and the harness flies it in lock-step on frozen episodes under the task's rules, reporting success rate (SR), collision
rate (CR) and path efficiency (PER). Three conditions: the nine training scenes with seen targets, the two held-out
layouts, and the training scenes with unseen targets.

![harness results](figures/harness_results.png)

| Condition | Pilots (expert) | Scripted straight line (privileged) | Hover |
|---|---|---|---|
| seen, 9 scenes x 20 | **0.71** [0.64, 0.77] | 0.06 [0.03, 0.11] | 0.00 (6 episodes) |
| held-out layouts s05r, s06r | 0.23 [0.12, 0.38] | 0.08 [0.03, 0.20] | |
| unseen targets, 9 scenes x 20 | 0.71 [0.64, 0.77] | | |

Each scene's expert lands within its kept-rate's interval, and the scripted baseline falls below it on every scene: the
harness measures what the dataset was collected with (`records/harness_results.md`). The experts do not look for the
target (they fly to its true position), so unseen targets cost them nothing; for a VLA that condition tests whether it
recognises objects it never trained on. The held-out layouts are harder for the s02 tree expert (0.15 and 0.30): it
generalises only moderately to new arrangements.

## 6. Rebalancing

As in AutoFly's appendix, every record before the first frame where Grounding DINO (tiny) gives the instruction's
target more than 0.7 confidence is *obstacle avoidance*, every record from it on *target seeking*, and each phase gets a
resampling weight towards an even mix. A sparse search found each episode's first detection with 44.5 detector
calls on average instead of scoring all of its ~188 records, after keeping every transition of the dense run on the pilot dataset
(`docs/decisions/2026-10-08-sparse-first-detection.md`).

![rebalancing](figures/rebalance_phases.png)

- **Phase split** 0.691 / 0.309 over 339,007 records (the paper: 0.73 / 0.27); target seeking ranges from 0.24 (s09) to
  0.37 (s04) by scene. **Weights** 0.723 / 1.621 (the paper: 0.68 / 1.85).
- The first confident detection comes a median 20.9 m from the target. 147 episodes (8 %) never reach 0.7, and every
  target is detected in some episode.
- **24 % of first detections fall on another object** (397 of 1,653), usually a distractor of a similar kind or colour;
  on the single-target pilot it was 1 %. With a diverse pool the phase boundary is noisier than the paper's.
- OpenVLA and LeRobot do not apply per-phase weights themselves: they are a sampling option, kept in
  `records/rebalance/rebalance.json` with every episode's transition.

![first detections per target](figures/rebalance_targets.png)

## Known limits

- **Expert quality.** Two of the three experts fail their 0.95 gates (s01d 0.935; s02 0.895 under the outline). Kept-rates
  fall to 0.38 on s06, and the kept flights are slightly easier than the flown ones: their straight start-to-target
  lines pass 5-15 % fewer obstacles within 2 m than the rejected flights' (s05: 1.75 against 2.03;
  `figures/dataset/crossing.json`).
- **Stochastic actions.** The pilots fly with sampled actions, as the paper's data collection does. The forward command
  sits near 2 m/s and the yaw command saturates at +-1 rad/s in turns, but the vertical command is spread almost
  uniformly over [-1, 1] m/s while the altitude holds near 2 m: SAC keeps its exploration noise in the vertical
  channel (`figures/dataset/actions_states.png`). A VLA imitating it should expect that channel to be mostly noise
  around a steady altitude.
- **Stand-ins.** 682 episodes target a Kenney low-poly model (vehicles and people); `target_stand_in` in the
  metadata filters them.
- **Language.** Two instruction templates, kept verbatim from AutoFly (including its "avioding"), and one obstacle phrase
  per scene, so every instruction is a function of scene and target.
- **First frames.** A target 60-80 m away spans a few pixels (1-3 for a chair, about 10 for a truck). Each simulator session first shows every target to the
  camera up close until the depth sees it, and every episode waits 5 frames after its spawns. An automatic check then
  found the target in 801 judgeable first frames and missed it in 52 (6 %): 41 of those rest on 1-2 pixels, most of the
  rest are open-frame objects (tables, chairs) whose projected centre shows the background.
- **Not in this phase:** s08 ruins and s10-s12 (no assets yet), photoreal vehicles and people, paraphrased
  instructions, a VLA itself. The thin slice (`figures/thin_slice/`, 450 episodes) measured the pilots but is not in
  the dataset: it was collected before the reset's warm-up existed.

## What is where

| Folder | Holds |
|---|---|
| `figures/` | everything above, plus `experts/` (training curves), `contact_rule/`, `targets/`, `thin_slice/`, `m3_pilot/` |
| `videos/` | `dataset/` (montage and per-scene playbacks), `experts/`, `pilots/`, `m3_pilot/` |
| `records/` | `gates/` (every gate and probe), `harness/`, `collect/` (each store's record and the campaign plans), `screening/`, `rebalance/`, `scenes/` |
| `dataset/` | the card, statistics, merge record, TFDS metadata, read-back checks, the target pool and exclusions, the split |
| `docs/` | the design spec, plan 6, every decision record, the runbooks |

On the GPU host: the dataset in `data/autofly_ue5_proto/1.0.0/`, its 36 raw stores in `data/<scene>_c1..c4/` (rejects
in `data/rejects/`), the code in `autofly_ue5/`, entry points in `scripts/`.

## Next: attaching a VLA

1. Fine-tune on `autofly_ue5_proto` (train split), with actions normalised by `dataset_statistics.json`; decide whether
   the policy reads state[0..1] (the harness can include or zero them).
2. Wrap it as a `Policy` (`autofly_ue5/evaluation/policy.py` says how) and run
   `scripts/eval_harness.py run --policy ... --condition seen|test_seen|unseen_targets`.
3. Compare with the table in section 5.
