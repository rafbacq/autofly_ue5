# AutoFly-style dataset collection on UE5 + Project AirSim: design

Date: 2026-09-15. Status: **approved by the user on 2026-09-15** ("the goal design is correct"); §8.1 added on the user's request as an optional source.
Project root: `/home/nvidiasims/research_uav/autofly_ue5/` (ROOT). Everything in this project lives under ROOT.

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

`/home/nvidiasims/research_uav/results/qwen_autofly/data_smoke_v1/{train,val}.jsonl` (+ `images/`) hold one training and
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
- **Project AirSim** (`https://github.com/iamaisim/ProjectAirSim`, MIT), checked out to `ROOT/platform/`.
- **Pinned versions** (checked 2026-09-15 against the repo, its release assets and PyPI):
  - Engine: **`Linux_Unreal_Engine_5.7.4.zip`**, the last 5.7 hotfix. Project AirSim's README supports Unreal Engine
    "5.2 or 5.7" and its Linux guide recommends an installed build, which is what Epic's prebuilt zip is. UE 5.8 is not
    supported on Project AirSim `main` yet (an open pull request).
  - Plugin: the prebuilt release asset **`ProjectAirSim-Plugin-Linux-UE5_7-1.0.1.zip`** (669,317,542 bytes, tag `v1.0.1`
    = `0975545`). Its SimLibs are Release-only, so project builds use Development or Shipping configurations.
  - Source checkout: `main` at `4d878bf` ("Prepare Project AirSim 1.0.2 release"; no `v1.0.2` tag exists). Its C++ code
    is reported identical to `v1.0.1`; M0 verifies this with `git diff 0975545 4d878bf -- unreal core_sim`.
  - Python client: **`projectairsim==1.0.2`** from PyPI (uploaded 2026-09-11, pure-Python wheel, Python ≥ 3.7).
  - Sample project: `platform/unreal/Blocks`, copied to `ROOT/ue_project/` with the prebuilt plugin in `Plugins/`.
- **Ubuntu 24.04 is not Project AirSim's supported OS** (22.04 is). Mitigations: use the engine's bundled toolchain
  (`v26_clang-20.1.8-rockylinux8`, installed by `Engine/Build/BatchFiles/Linux/SetupToolchain.sh` if missing), use the
  prebuilt plugin instead of building SimLibs, and skip `setup_linux_dev_tools.sh` (it adds an LLVM apt repository that
  returns 404 on 24.04) and `setup_linux_unreal_prereqs.sh`. Run the editor and simulator with `SDL_VIDEODRIVER=x11` and
  every Python process with `env -u PYTHONPATH` (the host's ROS Jazzy path is also Python 3.12).
- **M0 fallback trigger:** switch platforms only if, after about a day of fixing, one of these still holds: the
  `BlocksEditor` build with the prebuilt plugin fails even with `SetupToolchain.sh` and an Ubuntu 22.04 container;
  `get_images` after `world.step()` deadlocks or returns stale frames; `step()` reports no collisions for the drone; or
  NVIDIA Xid 31 / `VK_ERROR_DEVICE_LOST` crashes keep happening on driver 595.84. The fallback is Cosys-AirSim
  `5.8-v3.4.1` on `Linux_Unreal_Engine_5.8.2.zip`, decided with the user (the 5.7.4 engine cannot be reused by it).
- Machine: Ubuntu 24.04.4, glibc 2.39, 32 cores, 61 GB RAM, RTX 4090 (proprietary driver 595.84), 717 GB free, X11 on
  `:1`. Reported engine risk to watch at M0: Vulkan Xid 31 crashes with UE 5.7.4 on nvidia-open 595.71 (RTX 5070 Ti).

### 4.1 Capabilities the platform must provide

| Need | Used by | Project AirSim (per its docs and source; verified live at M0) |
|---|---|---|
| Build levels headless from Python in the editor, package a Linux binary with several maps | scenes | not a Project AirSim feature: UE's PythonScriptPlugin plus `RunUAT.sh BuildCookRun` |
| Spawn and destroy static meshes at runtime with pose and scale; set a material or colour | targets, distractors | `world.spawn_object`, `destroy_object`, `set_object_pose`, `set_object_scale`; colour only via pre-authored material instances and `set_object_material`; spawnable meshes must be cooked (`+DirectoriesToAlwaysCook`) |
| Multirotor velocity command: forward (body), vertical, yaw rate | expert, collector | `move_by_velocity_body_frame_async(v_forward, v_right, v_down, duration, yaw_is_rate=True, yaw=rad/s)`, NED (+down); a new command replaces the previous one |
| Teleport/reset the vehicle to a pose, zero velocity | every episode | `drone.set_pose(pose, reset_kinematics=True)`; flight-controller stability after teleport unverified (fallback: reload the scene) |
| Front RGB camera 256×256, 90° HFOV; front depth camera | collector (RGB), expert (depth) | robot config `capture-settings`: image types 0 (RGB) and 1 (DepthPlanar, float16 metres), `fov-degrees`; `get_images` |
| Collision events with timestamps | expert reward, rejects | `world.step()` returns `collision` events with `sim_time_ns`; `collision_info` topic |
| Pause / fixed step / simulator clock; ideally faster than real time | expert training, exact 5 Hz records | steppable clock with `pause-on-start`; `world.step(dt_ns)`; camera capture is tied to rendering, so throughput is GPU-bound |
| Several simulator instances on one GPU | expert training throughput | separate processes with `-topicsport` / `-servicesport`; not documented as tested, VRAM per instance unmeasured |

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
footprints. The check additionally runs inside a corridor around the start-target line, not the whole scene, so a
route must stay near the obstacle field rather than detour through open space far outside it. The
corridor's half-width is `p_extent + 2 * inflate_m`, clipped to the scene bounds, where `p_extent` is the line's
perpendicular extent -- the farthest any obstacle's surface reaches from the centreline -- and `inflate_m` is the
same `drone_radius_m + clearance_m` margin the occupancy grid already inflates obstacles by. The extra
`2 * inflate_m` exists because a corridor sized to the line alone can be severed by a single obstacle sitting at its
edge that a real flight would simply fly around, which made the original rule reject layouts that are trivially
flyable (see `autofly_ue5/scenes/reachability.py`). The check only guarantees solvable layouts; it never produces
actions. What it proves is opposite-edge *solvability*, not a route through the field (2026-09-24 review): every
obstacle's inflated footprint ends by `p_extent + inflate_m`, so the corridor always keeps a free lane `inflate_m`
wide beside the whole field, and even a fully sealed field (one solid block) passes when that lane is free. See
§13 for the M4 item. `test_seen` scenes reuse a train scene's file with a different seed.

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

### 7.1 Measured simulator contract (M0, 2026-09-16)

Facts measured on this build during M0. They bind every later milestone; where they contradict earlier text in this
spec, they win. Evidence: `docs/gates/m0_smoke_inst0.json`, `tests/fixtures/pas/`, and the Plan 1 ledger's rulings
R5–R9.

- **No-hit depth is `0.0`, not `+inf`.** DepthPlanar returns 0.0 for pixels with no geometry (sky). Measured: 20,145
  zero pixels, all in the frame's upper half, `inf_count` 0, finite range 1.484–254.5 m.
  `autofly_ue5.sim.decode.decode_depth` maps exact 0.0 to `+inf`, which is the project's canonical no-hit value, so
  every consumer — above all the SAC expert of §8, whose only sensor is depth — sees "nothing there" rather than an
  obstacle at the lens. Never feed raw DepthPlanar bytes to a policy or a reward.
- **Discard the first frame of every session.** The first `record()` after a scene loads returns a depth frame captured
  before the buffer stabilises: 65,522 of 65,536 pixels read closer than 1 m, against a stable ≈24,690 no-hit pixels
  and a 1.4 m minimum on every later frame. The M0 smoke tool refuses a zero warm-up; every episode runner must do the
  same (warm-up ≥ 1 record, or discard record 0).
- **Reset by the recovery sequence, not by teleporting through geometry.** A single `set_pose` that sweeps the drone
  through a solid object does fire a collision event, and its camera-versus-state agreement is not reproducible
  (0.000127 m in one run, 6.019 m in another). The up–across–down recovery sequence is reproducible to 1e-7–1e-6 m in
  every run and is the supported episode reset.
- **Backend hazards are typed, counted and recovered, never scored (M2).** A live run hits simulator-side faults that
  a fresh reset (or, after repeated failure, a relaunch of that one instance) recovers from: `CameraPoseError` (the
  camera left behind by a `set_pose` sweep -- mostly during reset), `StepTimingError`, `StaleStateError`,
  `CommandTimeoutError`, and raw pynng transport timeouts from the client. Added after the 2026-09-24 review: an
  episode must also start where it was asked -- `reset()` verifies the settled pose (`ResetPoseError`, 0.3 m / 0.1 rad),
  the env refuses a first observation that already reports a collision (`StartCollisionError`), and `step()` refuses a
  horizontal move beyond 10 m/s × dt (`KinematicsJumpError`), on collision steps too. The recorded M2 gate had 15
  one-step "collisions" whose drone was ≥ 4 m from its start, all on a reset right after a collision episode (15 of 73
  such resets, 0 of 723 others); the camera check could not see them because it is skipped on collision steps. A
  faulted step ends its episode on the last real observation; training drops that transition and evaluation replays
  the episode (`autofly_ue5/expert/resilient.py`, `evaluate.py`).
- **Lock-step holds exactly at 5 Hz.** 55/55 records had simulator time, image timestamps and kinematics timestamps
  equal to the step target; camera-to-state pose error stayed at 2.4e-7 m. The velocity command's duration is
  `dt − 2·step-ns = 0.19 s`, sent before the step, with the reply awaited after it.
- **Geometry and optics agree with prediction.** A spawned 1×3×12 m pillar at 7.1 m measured 7.0977 m of depth (2.3 mm
  error) and 54 px wide against 54.10 px predicted from the 90° FOV — so scene geometry can be checked against the
  camera arithmetic rather than by eye.
- **Built levels carry a fixed manual exposure, and frames are still not bit-identical.** A level built by
  `autofly_ue5/scenes/ue/build_level.py` contains an unbound `PostProcessVolume` with `AutoExposureMethod = Manual`,
  `AutoExposureApplyPhysicalCameraExposure = false` and a bias carried in the level spec (s01: −11.0 EV, calibrated to
  mean brightness 109 of 255). Without it the physically-based SunSky saturates every pixel to pure white — the M1 gate's
  first run measured 255.0 flat, which is why an image check belongs in every scene's gate and bounding-box checks are
  not enough. Auto-exposure is deliberately NOT used: eye adaptation carries view history, so the same pose would render
  differently depending on where the camera looked before, which an (image → action) dataset and an RL agent cannot
  tolerate. **Exposure is now history-free, but rendering is not bit-exact**: two captures of the identical pose differ by
  about 2.4 grey levels (temporal anti-aliasing). Treat frames as reproducible to a few grey levels, never as identical,
  and never gate anything on exact pixel equality. That 2.4-level figure is for timestamp-exact frames reached by the
  same reset sequence (two `reset(PILLAR_VIEW)` calls within one run); two committed measurements at the identical pose
  reached differently — `docs/gates/exposure_calibration.json`'s calibration probe (direct `set_pose` plus steps,
  decoding the first RGB frame received) versus `docs/gates/m1_gate.json`'s gate check (the timestamp-filtered
  `reset()` path) — differ by 10.7 grey levels (109.33 vs. 119.99) at the same −11.0 EV. Nothing should assume the
  rendered image is a function of pose alone.
- **Runtime colour needs a base `UMaterial`.** `set_object_material` accepts `/Game/Geometry/Materials/M_Orange`
  (patch change 55–70) and rejects the `MaterialInstanceConstant` `M_Blue`, as the server filters on `UMaterial`.
  Scene-authored colour (M1 §6.4) uses material instances created in the editor instead.
  **Binding rule for every runtime spawn (added M2, after this cost three blocked live runs):** a runtime
  `spawn()` may pass **only a base `UMaterial` package path**, or `None`. Editor-time level building and runtime
  spawning are different code paths with different class requirements — `build_level.py` applies a
  `color_instance` happily, and the runtime server cannot, ever. `spawn(material=None)` skips the call and keeps
  the mesh's own material, which is the correct choice whenever no proven base material applies. The two
  identifiers `spawn()` takes are also *not* our registry keys: `asset` is the short Unreal asset name (the last
  path segment, e.g. `Cylinder`, `1M_Cube`) and `material` is a full package path. Resolve both from the registry
  before calling.
  **The lesson is about enforcement, not documentation.** This failure was already written down twice — here and
  in `sim/smoke_m0.py:44-45` — and still reached live hardware, because `FakeSimulator` accepted any string and
  every offline test passed. A contract that only exists in prose is not a contract: the test double now rejects
  what the real server rejects, and that is what keeps this fixed.
- **Throughput and footprint:** 7.3–7.4 steps/s with `real-time-update-rate` 3 ms, 1,734 MiB of VRAM for one instance
  with cameras capturing. §8's SAC budget and §12's M2 gate use the M0 gate's measured numbers, not these single-run
  figures.

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
  **Progress stops at the success radius** (`REWARD_VERSION = "2-no-progress-inside-success-radius"`, 2026-09-24
  review): the progress term is `k_p · (max(d_prev, 5 m) − max(d, 5 m))`. Paid all the way in, closing from 5 m to 2 m
  misaligned and turning at the end out-earned an aligned success at 5 m (discounted ≈ 13.1 vs 10.5), and the first
  trained expert learned exactly that: its final checkpoint's successes ended a median 2.35 m from the target, and
  7 of its 16 real gate failures left the bounds within 5 m of a target that sits 0–3 m from the edge. Inside the
  radius only the alignment bonus and the time penalty remain, so turning onto the target at once is optimal. The
  form is still potential-based, so crossing the radius back and forth earns nothing.
- Implementation: stable-baselines3 SAC with a custom feature extractor and vectorised environments, one simulator
  instance per environment.
- Throughput gate (M2): measure environment steps per second per instance, the number of instances the RTX 4090 holds,
  and time to 95 % on s01; project the cost of all agents before M5.

### 8.1 Additional pilot: AutoFly checkpoint (optional)

User request: "use the autofly checkpoints to load it and generate the dataset with that, but add this as an additional".
The SAC expert (§8) stays the primary pilot. This pilot is an optional second data source. It stays **inactive** until the
activation condition below holds.

- **Availability (checked 2026-09-15): no public checkpoint.** `github.com/xiaolousun/AutoFly-VLA` is still at `08b5039`,
  with no tags, releases or licence, and `train/`, `val/`, `deploy/` contain only empty READMEs. On issue #1 (2026-06-19) the author said code,
  pretrained models and data are under "internal review and compliance approval", with no timeline. The project page's
  model link `huggingface.co/xlsun/AutoFly` returns 401, and user `xlsun` has 0 public models. ModelScope has 0 models for
  `jacksun001`, `xiaolousun` or `xlsun`; it only has the TFDS dataset. The paper says "the model, data and code are publicly available", which is currently false for the model and the code.
- **Model:** Prismatic `prism-dinosiglip+7b` (LLaMA-2 7B), plus a Siamese depth projector and a frozen Depth Anything V2. The
  Depth Anything size is not stated: Small is Apache-2.0, while Base, Large and Giant are CC-BY-NC-4.0. Stock OpenVLA code cannot load the depth branch, so the
  authors' model code must also be released. The model runs as a separate server process in `.venv`, with the released library versions pinned.
- **Inputs:** only the current front RGB frame (256×256, resized to 224 by the processor, no history) and the instruction.
  **a0 is not a model input.** It is a setup phase: zero linear velocity and a proportional yaw-rate turn toward the
  target bearing until roughly aligned, recorded as the first step(s) (reported from the authors' OpenReview rebuttal to reviewer nXwK, Q3; the rebuttal
  could not be re-fetched on 2026-09-15, so this is unverified). This differs from §9 step 4; M3 settles §9 step 4 against
  the two real episodes.
- **Action decoding:** each action uses 256 bins over [−1, 1], mapped to the last 256 LLaMA-2 tokens. Decode as OpenVLA does (255 bin centres,
  `a = 0.5·(n+1)·(q99−q01) + q01`) with the checkpoint's `unnorm_key` statistics, and check the decoded ranges against the
  released `dataset_statistics_*.json`. Those statistics set a forward-speed floor of 0.37–1.09 m/s. The policy cannot hover or stop, so
  the §9 step 5 end conditions end episodes. Per step: `predict_action(do_sample=False)` → `command_velocity(v_fwd,
  yaw_rate, v_z)` → `step(0.2)`. The signs of yaw and v_z are checked against the §3.2 episodes.
- **VRAM (estimate):** about 15.1 GB of bf16 weights, 0.05–0.7 GB for the depth model and 0.5–1 GB of activations, about 16–17 GB in total.
  That leaves about 7 GB of the 24.5 GB for one UE5 instance, whose use is unmeasured, so measure it first. SAC training cannot use the GPU
  at the same time. Estimated output is about 400 kept episodes per day (unconfirmed).
- **Separation:** provenance (§10.2) gains `"pilot": "sac_expert" | "autofly_checkpoint"`. For this pilot it also records the
  checkpoint SHA-256, the statistics key and the sampling mode. Its episodes go to their own `<dataset_name>` with their own manifest. Sources are
  only combined in an explicit step that lists per-pilot counts in the dataset card, never silently. The record stays
  AutoFly-only (§10.1). The §9 step 6 filter applies: an episode is kept only if it ends within 5 m and ≤ 15° of the target with no
  collision; all others go to `data/rejects/`. This pilot flies `train` scenes only, and `test_*` episodes never enter training data.
- **Limits:** AutoFly reports 47.9 % SR (55.4 % seen, 42.3 % unseen) and 21.9 % collisions in its own AirSim scenes. Our
  UE5 scenes and assets are outside its training distribution, so SR is likely lower. Kept episodes may still take detours (PER 77 %) and carry
  its biases (no stop, a speed floor, no memory), which weakens any later comparison between a student model and AutoFly.
- **Activation:** weights **and** inference code (including the depth branch) are released under a licence that permits our
  use. The pilot then slots in after M3 as optional M3b, because it needs the collector, writer and validator, and it starts only with the user's go-ahead.
  Signs to watch for: `git ls-remote` of AutoFly-VLA moving from `08b5039`, new comments on issue #1, and `huggingface.co/api/models/xlsun/AutoFly` changing from 401 to 200.

## 9. Episode protocol (`autofly_ue5/collect/`)

Dataset collection runs on the 10 `train` scenes with seen targets. M6 reuses this protocol for evaluation episodes on
`test_seen` and `test_unseen` scenes, where unseen targets are also drawn.

1. Choose the scene, then a target from the pool allowed by the split (seen for `train` and `test_seen`, seen or unseen
   for `test_unseen`), then 3–5 distractors from the remaining pool; place all on the target band with spacing ≥ 4 m.
2. Choose a start pose on the start band at a random altitude in the band, with a random yaw.
3. Instruction: one template filled with `{target}` and `{obstacle}`. Templates are copied verbatim from the real
   episodes, including the original spelling: `go through and avoid the {obstacle} or other obstacles to reach the
   {target}` and `advance to {target} while avioding {obstacle} and other obstacles`.
4. (Settled at M3 against the real episodes and §8.1's reported a0 description.) Step 0 records the coarse-direction action a0: zero forward speed, yaw rate toward the target's bearing quantised
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
| M3b | optional AutoFly-checkpoint pilot (only if the §8.1 activation condition holds) | decoded action ranges match the statistics files; an s01 pilot passes the validator with "pilot": "autofly_checkpoint" in every provenance file |
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
- **Through-field routes (M4):** the reachability rule (§6.2) proves a crossing is solvable but not that it passes
  through the obstacle field; a sealed field passes if the free lane beside it is open. Enforcing through-field
  routes needs a difficulty/detour metric (e.g. shortest-path length through vs. around the field), designed when
  M4 replaces the circle-only occupancy grid for the richer asset library. s01's jittered pillar grid is not
  affected: its gaps are what episodes cross.
- **AutoFly checkpoint:** not public as of 2026-09-15. If released, its ~48 % SR, out-of-distribution scenes, forward-speed floor and unknown licence limit it to a separately tagged optional source (§8.1).
