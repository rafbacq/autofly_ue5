# AutoFly-style dataset collection on UE5 + Project AirSim: design

Date: 2026-09-15. Status: **design approved section by section in chat; awaiting user review of this document.**
Project root: `/home/jk_edge/research_uav/autofly_ue5/` (ROOT). Everything in this project lives under ROOT.

## 1. Goal and scope

Recreate the simulation side of AutoFly (Sun et al., *AutoFly: Vision-Language-Action Model for UAV Autonomous
Navigation in the Wild*, arXiv 2602.09657, ICLR 2026) on Unreal Engine 5 with Project AirSim, and use it to generate
our own AutoFly-format dataset. The released AutoFly dataset is no longer on this machine and AutoFly's 12 scenes,
expert agents and collection code were never released. What follows is a **recreation from the paper's description,
not a reproduction** of their exact layouts, assets or numbers.

In scope, in order: the platform install, 12 script-built scenes, one SAC expert per scene, the episode collector,
the dataset writer with provenance, rebalancing weights, validation, and finally the test splits with an evaluation
harness.

Out of scope for this design: training π0 or any VLA on the data (a later LeRobot export is a separate project step),
the pseudo-depth encoder, real-world flights, human-piloted demonstrations (AutoFly mixed some in; we have none), and
the paper's three challenge scenes (dense cylinders, dense forest, moving obstacles), which can be added later as
extra scene files.

## 2. Decisions taken with the user

| Topic | Decision |
|---|---|
| Simulator | Unreal Engine 5 prebuilt Linux editor + **Project AirSim** ("use the project airsim for now") |
| Purpose | Data generation first, evaluation harness after (both are in scope) |
| Expert pilot | **SAC reinforcement-learning agents, one per scene**, as in the paper |
| Record contents | **Exactly AutoFly's fields**: front RGB, instruction, action[3], state[9]; nothing added to the training record |
| Build approach | Scenes written as JSON files, built into UE levels by an editor Python script; targets and distractors spawned per episode at runtime |
| Location | One folder, ROOT, including the unzipped engine and the Project AirSim checkout (both git-ignored) |

## 3. Evidence this design relies on

### 3.1 From the paper

- 12 custom 70 m × 70 m AirSim scenes: 10 seen (training data) and 2 unseen with entirely different constituent
  elements. Testing uses the 2 unseen scenes plus 2 seen scenes with altered layouts and identical elements (App. A.2.1,
  Fig. 8).
- Obstacles: coloured pillars, tree clusters, stacked boxes (App. A.2.1); trees, walls, rocks, buildings (Sec. 3.3).
- 60 target object instances placed at scene boundaries, split 50 seen / 10 unseen; each scene has 3–5 distractor
  objects (Sec. 3.3, Fig. 9). Category mix (Fig. 3c): vehicles 22 %, geometry 20 %, furniture 18 %, equipment 18 %,
  human 17 %, others 5 %.
- Starts are random positions at environment edges (Sec. 3.3).
- Coarse positional or directional guidance is encoded as an initial action a0 (Sec. 3.1).
- Expert: per-scene SAC agents, CNN with downsampling layers and MLP heads, depth-only inputs, stochastic policies,
  velocity-command outputs, trained to 95 % evaluation success; twin target Q-networks; target entropy −dim(A)
  (App. A.2.3). The agents are "limited to point-to-point navigation without autonomous recognition".
- Success: within 5 m of the target and angular deviation ≤ 15° (App. A.2.1). Collision: distance to an obstacle
  ≤ d_col (value not given). Path efficiency: L_opt / max(L, L_opt) over successful trials (Sec. 4.1).
- Scale: 13K+ training episodes, 2.5M+ image-language-action triplets; evaluation 7,200 episodes (App. A.2.2).
- Rebalancing: Grounding DINO with the instruction as query splits each trajectory into obstacle-avoidance and
  target-seeking phases (threshold e.g. 0.7); P0 ≈ 0.73 / 0.27; uniform target gives weights ≈ 0.68 / 1.85 with
  stratified resampling (App. A.2.4).
- Real camera: 640×480, 90° HFOV (App. A.3.1). Released data images are 256×256.

### 3.2 From two real released AutoFly episodes

`/home/jk_edge/research_uav/results/qwen_autofly/data_smoke_v1/{train,val}.jsonl` (+ `images/`) hold one training and
one validation episode exported from the release before it was deleted (sources `UAV_VLA_Scene_1_zip/Blue_Hatchback`,
96 steps, and `UAV_VLA_Scene_7_zip/Blue_Cone`, 78 steps). The exported rows are **not in time order**; sorting by state
0 recovers an approximately smooth trajectory (median step 0.37–0.41 m, a few jumps up to 2.5 m). Measured on those two episodes:

| Field | Finding | Confidence |
|---|---|---|
| image | RGB 256×256 PNG | confirmed |
| instruction | one sentence per episode, e.g. `go through and avoid the white pillars or other obstacles to reach the blue hatchback`, `advance to blue cone while avioding stone field and other obstacles` | confirmed (2 templates seen) |
| action[0] | forward speed, mostly 1.97–1.99, max ≈ 2.0 | strong |
| action[1] | yaw rate, range ≈ ±1 | strong |
| action[2] | vertical speed, range ≈ ±1 | strong |
| state[0] | horizontal distance to target (corr 1.000 with distance to the trajectory end; ends ≈ 7.2 m) | strong |
| state[3] | speed, median 1.87–1.96 | strong |
| state[6..8] | position x, y, z; z is up, 1.2–2.6 m | strong |
| state[1] | correlates −0.88 / −0.77 with heading change | undecoded |
| state[2], state[4], state[5] | ranges within ±1.2 | undecoded |
| step period | ≈ 0.2 s (0.37 m per step at ≈ 1.9 m/s), i.e. about 5 Hz | strong |
| first action | near-zero forward, yaw rate ≈ 0.95–0.97 (a turn in place) | consistent with a0, 2 episodes only |
| scene numbering | `Scene_1` is the white-pillar scene of Fig. 8 | strong |

## 4. Platform

- **Unreal Engine 5, Epic's prebuilt Linux zip**, unzipped to `ROOT/engine/`. The user downloads it with their Epic
  account; nothing in this project signs in on their behalf.
- **Project AirSim** (`https://github.com/iamaisim/ProjectAirSim`, releases v1.0.0–v1.0.2 in 2026-09), checked out to
  `ROOT/platform/` at a pinned tag.
- **Version rule (M0 gate):** the UE5 version is the one the pinned Project AirSim tag documents for Linux, and it must
  be available as a prebuilt Linux zip. If the tag needs a source-built engine, an OS other than Ubuntu 24.04 that
  cannot be worked around, or lacks one of the capabilities in §4.1 with no workaround, M0 stops and reports; the
  fallback is Colosseum or Cosys-AirSim on the version their Linux docs name, decided with the user.
- Machine: Ubuntu 24.04.4, glibc 2.39, 32 cores, 61 GB RAM, RTX 4090 (driver 595.84), 717 GB free, X11 on `:1`.
  Known engine risks to check at M0: UE 5.7 SDL3/Wayland editor input issues (use X11), a reported Vulkan Xid 31 crash
  with UE 5.7.4 on 595.71, packaging on Ubuntu 24.04.

### 4.1 Capabilities the platform must provide

| Need | Used by |
|---|---|
| Build levels headless from Python in the editor, package a Linux binary with several maps | scenes |
| Spawn and destroy static meshes at runtime with pose and scale; set a material or colour | targets, distractors |
| Multirotor velocity command: forward (body), vertical, yaw rate | expert, collector |
| Teleport/reset the vehicle to a pose, zero velocity | every episode |
| Front RGB camera 256×256, 90° HFOV; front depth camera | collector (RGB), expert (depth) |
| Collision events with timestamps | expert reward, rejects |
| Pause / fixed step / simulator clock; ideally faster than real time | expert training, exact 5 Hz records |
| Several simulator instances on one GPU | expert training throughput |

## 5. Folder layout

```
ROOT/
├── README.md
├── docs/superpowers/{specs,plans}/
├── engine/                 prebuilt UE5 (git-ignored)
├── platform/               Project AirSim checkout + build (git-ignored)
├── ue_project/             our UE5 project: asset library, generated levels, packaging config
├── scenes/                 scene JSON files + generator
├── assets/registry.json    logical asset name -> UE asset path, scale, pivot, measured bounds, category, seen/unseen
├── configs/                Project AirSim robot, camera and scene configs
├── autofly_ue5/            Python package
│   ├── sim/                the only code importing the Project AirSim client; also a fake simulator for tests
│   ├── scenes/             scene schema, generator, reachability check, level-build driver
│   ├── expert/             SAC environment, network, training, evaluation
│   ├── collect/            episode protocol and runner
│   ├── dataset/            AutoFly-format writer, provenance, rebalancing
│   └── validate/           dataset and live-simulator checks
├── scripts/                build_scenes, package_sim, launch_sim, train_expert, collect, validate
├── tests/
├── .venv/  runs/  checkpoints/  data/     (git-ignored)
```

## 6. Scenes

### 6.1 Scene file

One JSON file per scene in `scenes/`, validated by a schema in `autofly_ue5/scenes/`:

- `id`, `split` (`train`, `test_seen`, `test_unseen`), `seed`, `bounds` (70 × 70 m, centred on the origin).
- `ground`: logical material name.
- `obstacle_groups`: list of `{asset, count, scale_range, palette, placement}`; `placement` is one of `jittered_grid`
  (pillars, poles), `poisson` (rocks, stones, ruins), `clusters` (tree clusters: cluster count, trees per cluster,
  radius), `stacks` (boxes: stack count, height range).
- `start_band` and `target_band`: distance range from the boundary (start 2–6 m inside, targets 0–3 m inside).
- `altitude_band`: 1.0–3.0 m above ground, from §3.2.
- `instruction_obstacle`: the phrase used in instructions (`white pillars`, `stone field`, …).

### 6.2 Generator and reachability

The generator expands a scene file into concrete obstacle instances with the file's seed. A 2D occupancy check then
rejects layouts where any start cell cannot reach any target cell with at least 1.0 m clearance to inflated obstacle
footprints. The check only guarantees solvable layouts; it never produces actions. `test_seen` scenes reuse a train
scene's file with a different seed.

### 6.3 The 12 scenes (from Fig. 8)

| id | split | ground | obstacles | instruction phrase |
|---|---|---|---|---|
| s01 | train | plain grid floor | white pillars | white pillars |
| s02 | train | grass | sparse young trees | trees |
| s03 | train | grass | scattered rocks | rocks |
| s04 | train | gravel | rocks and coloured stones | stone field |
| s05 | train | snow / grey stone | large standing stones, boulders | standing stones |
| s06 | train | grass | bushy tree clusters | tree clusters |
| s07 | train | sand tiles | stacked wooden crates and boxes | stacked boxes |
| s08 | train | sand | ruined wall pieces | ruins |
| s09 | train | paving stones | tall coloured poles | coloured poles |
| s10 | train | snowy yard | walls, containers, tanks, stalls | walls and containers |
| s11 | test_unseen | light ground | city blocks, towers, statue | buildings |
| s12 | test_unseen | snow / desert | mud-brick village, scaffolding | village buildings |
| s06r | test_seen | as s06 | as s06, new seed | tree clusters |
| s05r | test_seen | as s05 | as s05, new seed | standing stones |

Instruction phrases for s02–s12 are our wording except `white pillars` and `stone field`, which appear in the real
episodes (`Scene_7` = stone field is assigned to s04 here; its Fig. 8 position is not certain).

### 6.4 Assets and targets

`assets/registry.json` lists every logical asset with its UE path, category, `seen`/`unseen`, scale, pivot and bounds
measured live at M1/M4. Target pool: 60 instances, 50 seen and 10 unseen, following Fig. 9 (vehicles, sofas, chairs,
TV, barrels, suitcases, coloured primitive shapes; unseen includes crane, tiger, elephant, slide, traffic cone, ladder)
and the Fig. 3c category mix. Scene s01 uses engine primitives only, so M1 needs no downloaded assets. Asset sources
for M4 (free UE sample content, Fab free assets) are recorded per asset with licence.

## 7. Simulator interface (`autofly_ue5/sim/`)

One module wraps Project AirSim so nothing else imports it:

- `launch(map, instance) / close()` of a packaged simulator process owned by this project (PID recorded, only that
  process stopped).
- `reset(pose)`, `spawn(name, asset, pose, scale, material) / destroy(name)`.
- `command_velocity(v_forward, yaw_rate, v_z)` held for one step.
- `step(dt=0.2)`: advance the simulator clock exactly one control period (pause between steps).
- `observe()`: front RGB 256×256, front depth, pose, velocity, simulator time, collision flag.
- A fake implementation with the same interface drives all offline tests.

## 8. SAC expert (`autofly_ue5/expert/`)

- One agent per **train** scene (s01–s10); accepted when its evaluation success is ≥ 95 % over ≥ 200 fresh episodes.
  Test scenes need no expert: they are only used to evaluate students (M6), and their optimal path length L_opt comes
  from a shortest-path search on the scene's occupancy grid.
- Observation: front depth image downsampled to 84×84 and clipped to 30 m; target relative position (horizontal
  distance, bearing in the body frame, height difference); body velocity. The target position is privileged
  information for the expert only; it never enters the dataset record except where AutoFly's state[9] already carries
  it (state[0]).
- Action: `[v_forward ∈ [0, 2] m/s, yaw_rate ∈ [−1, 1] rad/s, v_z ∈ [−1, 1] m/s]`, one command per 0.2 s step.
- Network: CNN (strided convolutions) on depth, MLP on the vector, shared by actor and twin critics; SAC with automatic
  entropy tuning and target entropy −3.
- Reward per step: `+k_p · Δ(distance to target)`, `+k_h · alignment bonus when within 10 m`, `−k_t` time penalty,
  `+R_s` on success (≤ 5 m and ≤ 15°), `−R_c` on collision (episode ends), `−R_b` for leaving the altitude band or the
  bounds (episode ends). Step limit 300 (60 s simulated). Coefficients start at `k_p = 1, k_h = 0.1, k_t = 0.01,
  R_s = 10, R_c = 10, R_b = 5` and are tuned on s01 only.
- Implementation: stable-baselines3 SAC with a custom feature extractor and vectorised environments, one simulator
  instance per environment.
- Throughput gate (M2): measure environment steps per second per instance, the number of instances the RTX 4090 holds,
  and time to 95 % on s01; project the cost of all agents before M5.

## 9. Episode protocol (`autofly_ue5/collect/`)

Dataset collection runs on the 10 `train` scenes with seen targets. M6 reuses this protocol for evaluation episodes on
`test_seen` and `test_unseen` scenes, where unseen targets are also drawn.

1. Choose the scene, then a target from the pool allowed by the split (seen for `train` and `test_seen`, seen or unseen
   for `test_unseen`), then 3–5 distractors from the remaining pool; place all on the target band with spacing ≥ 4 m.
2. Choose a start pose on the start band at a random altitude in the band, with a random yaw.
3. Instruction: one template filled with `{target}` and `{obstacle}`. Templates are copied verbatim from the real
   episodes, including the original spelling: `go through and avoid the {obstacle} or other obstacles to reach the
   {target}` and `advance to {target} while avioding {obstacle} and other obstacles`.
4. Step 0 records the coarse-direction action a0: zero forward speed, yaw rate toward the target's bearing quantised
   to 8 sectors, clipped to ±1 rad/s, zero vertical speed (matches the turn-in-place first actions in §3.2).
5. The SAC expert flies with stochastic actions until success, collision, bounds exit or 300 steps.
6. Only successful, collision-free episodes enter the dataset; every other episode goes to `data/rejects/` with its
   reason.

## 10. Dataset (`autofly_ue5/dataset/`)

### 10.1 Training record (AutoFly fields only)

Per step, in time order: `observation.image_0` RGB 256×256 PNG, `observation.state` float[9], `action` float[3],
`language_instruction` string. Written as TFDS/RLDS TFRecord shards in `data/<dataset_name>/1.0.0/` with
`features.json` and `dataset_info.json`, mirroring the released `uavvlasplit_*_dataset/1.0.0` layout.

`state[9]`: `[0]` horizontal distance to the target, `[3]` speed, `[6..8]` position x, y, z (z up). Fields
`[1], [2], [4], [5]` are decoded at M3 by testing hypotheses (body-frame velocity components, roll, pitch, yaw, yaw
rate, bearing to target and its sine/cosine) against the two real episodes; a hypothesis is adopted only with
|correlation| ≥ 0.95 on both. Any field left undecoded gets a documented definition and is marked
`not_autofly_verified` in the dataset card.

### 10.2 Provenance (separate from the record)

`data/<dataset_name>/provenance/<episode_id>.json`: scene file and its SHA-256, generator seed, target and distractor
names and poses, start pose, simulator time per step, expert checkpoint and its SHA-256, platform tag, engine version,
package build hash, termination reason. `manifest.json` lists episodes, splits and counts.

### 10.3 Rebalancing

A post-processing step runs Grounding DINO on each step's image with the instruction's target as the query and a 0.7
threshold. It labels steps before the first detection as obstacle avoidance and the rest as target seeking, and writes
per-phase resampling weights to `data/<dataset_name>/rebalance.json`. The raw shards are never rewritten.

### 10.4 Scale

M3 pilot: 100 episodes on s01. M5: toward the paper's scale (13K+ episodes, 2.5M+ steps) across the 10 train scenes,
with the exact count chosen from the M2 throughput numbers.

## 11. Validation and testing

Offline, against the fake simulator: scene schema and generator determinism, reachability rejection, instruction
templates, record schema and time order, the 0.2 s step count per record, state[9] computation, provenance
completeness.

Live, on the packaged simulator (M0/M1): the RGB image changes when the pose changes; depth at known obstacles matches
scene geometry; a deliberate crash raises a collision; velocity commands track (forward, vertical, yaw rate) with
measured error; exactly one simulator step per record.

Dataset validator (M3 onward): action and state ranges, speed and altitude distributions compared against §3.2; no
NaN; images decode; episode lengths; per-scene and per-target counts.

## 12. Milestones and exit gates

| # | Milestone | Exit gate |
|---|---|---|
| M0 | UE5 + Project AirSim installed under ROOT; sample environment runs | Python smoke test: connect, spawn an object, fly a velocity command, get RGB and depth, register a collision, step the clock |
| M1 | `sim/` module; scene s01 built from its JSON file into a packaged map | live checks of §11 pass on s01 |
| M2 | SAC on s01 | ≥ 95 % success over 200 episodes; throughput numbers recorded |
| M3 | collector, dataset writer, validator; state[9] decoding | 100-episode pilot passes the validator |
| M4 | asset library; scenes s02–s12 and s05r/s06r | every scene builds, passes reachability and live checks |
| M5 | SAC experts for s02–s10; full collection on the 10 train scenes; rebalancing | per-scene expert gate; dataset validator passes |
| M6 | test splits and evaluation harness (SR, CR, PER) | harness scores a scripted baseline end to end |

Each milestone stops for the user's go-ahead before the next one starts.

## 13. Risks and open questions

- **Project AirSim fit:** the exact UE5 version, Linux packaging, runtime spawning and clock stepping are confirmed
  only at M0 (§4 version rule and fallback).
- **SAC cost:** simulator throughput decides whether 10 agents are affordable; M2 measures it before M5 commits.
- **Assets:** free assets may not match Fig. 9 exactly; the registry records substitutions.
- **Undecoded state fields:** four of nine state fields may stay undecoded (§10.1).
- **Success distance:** real episodes end ≈ 7.2 m from the state-0 reference while the paper says 5 m; M3 checks
  whether distance is measured to the object's surface or centre.
- **d_col** for the collision metric is not given in the paper; M6 picks a value and records it.
