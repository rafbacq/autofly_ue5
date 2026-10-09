# 2026-10-02 — Moving pillars: scene s01d and milestone M2d

The user asked for some of the existing pillars to move along random routes, so the UAV learns to dodge them. This
record sets out the rules (spec §6.5), the evidence behind each one, and what stays fixed. The paper has a
moving-obstacle challenge scene of its own (spec §1, previously deferred "as extra scene files later"), and s01d is
our first such scene.

## Scope

- **A new scene id, `s01d`, on the existing S01 map.** It reuses s01's layout, built level and packaged map, and
  moves a subset of the baked pillars at runtime. Static s01 does not change: its episodes, observation space and
  simulator calls are pinned byte for byte by `tests/test_static_s01_golden.py`, taken from M2's code. M2's gate,
  its audit and every M2 checkpoint therefore stay valid.
- **A new milestone M2d** (spec §12): the mover probe passes, then an s01d expert reaches ≥ 95 % deterministic
  success over 200 held-out episodes. It runs before M3's s01d pilot; M3's s01 pilot does not wait for it.

## Verified facts the design rests on

From the Project AirSim source at `platform/` (4d878bf) and the installed client (`projectairsim` 1.0.2):

| Fact | Where |
|---|---|
| `SetObjectPose` runs synchronously on the game thread. With `teleport=true` it calls `SetActorLocationAndRotation(…, bSweep=false, TeleportPhysics)`, so the actor moves without a sweep. With `teleport=false` it sweeps and stops at the first blocking hit, which could be the drone. **Movers always teleport.** | `WorldSimApi.cpp:745-791` |
| It throws `"SetObjectPose failed. No objects of name …"` or `"… Unable to move object …, check if object state is movable!"`. The client turns either into `RuntimeError("ERROR code: …, message: …")`. | `WorldSimApi.cpp:772-787`, `client.py:387-404` |
| Lookup checks spawned objects by exact name first. It then returns the first actor whose `GetName()` *contains* the string (case-insensitive) or whose tag equals it. **Names are therefore checked exactly against an allow-list before any request.** | `UnrealHelpers.h:56-96` |
| There is no batch setter. A batch getter, `GetObjectPoses`, exists, and a missing object reads as NaN rather than raising. | `WorldSimApi.cpp:41-45, 699-728` |
| A request's receive timeout (300 s) makes `request()` **disconnect the client**, then raise a bare `RuntimeError("Fatal Timeout …")`. Every later call on that client fails, so it can be recovered only by a relaunch, never by a reset. A send timeout (1 s) raises a raw `pynng.Timeout`. | `client.py:255-282`, `client.py:52-61` (socket timeouts) |
| The S01 pillars were built `MOVABLE`, `BlockAll`, tagged `obs_NNNN`. M1 found them by tag to the micrometre. | `scenes/ue/build_level.py:spawn_mesh`, `docs/gates/m1_check_map.json` |
| M0 moved a spawned cube to within 5e-7 m. **Moving a baked actor is unmeasured**, hence the probe below. | `docs/gates/m0_smoke_inst0.json` |
| The rotor tips reach 0.472 m from the body centre (props at ±0.253 m, radius 0.1143 m). | `configs/robot_autofly_quadrotor.jsonc` |
| The drone only flies forward into its own camera view (`v_right = 0`), plus vertically. | `sim/airsim_backend.py:_step_impl` |
| In SB3 2.9, a float16 `Box` is not an image space (only uint8 is), so it is neither transposed nor divided by 255. `preprocess_obs` casts it to float32, and `DictReplayBuffer` stores the space's dtype. | `stable_baselines3/common/preprocessing.py`, `buffers.py` (verified by test) |
| s01's pillars sit at \|x\|, \|y\| ≤ 24.9 m. The closest centres are 4.42 m apart, and the median nearest neighbour is 5.19 m. | `runs/levels/s01.layout.json` |

## The rules (spec §6.5)

**Movers.** Each episode, k ∈ [8, 12] of s01's 80 baked pillars move, and the rest stay home. Of the k, 2–4 are
drawn from pillars whose centre lies within 4 m of the straight start → target segment. The rest are drawn uniformly.
Without that bias, a uniform draw puts only about one mover near the flight line, and many episodes would have none.

**Routes.** Each mover's pose is an analytic function of its own route clock τ:

- `pingpong`: back and forth at constant speed along a random heading through home, with half-length 1.0–2.5 m.
  At τ = 0 the mover sits at a random phase of its period.
- `orbit`: a circle around home with radius 1.0–2.0 m, in a random direction, starting at a random angle.
- Speed 0.4–1.2 m/s. The draws come from the episode's own generator, *after* every existing draw of
  `sample_setup`. The draw order is load-bearing: the gate, its audit and the renders replay exact episodes.

**Yielding: movers never ram the drone.** A mover's clock advances by dt only if its next pose either keeps its
surface at least `contact_m + yield_margin_m` (1.0 + 1.5 = 2.5 m) from the drone's current position, or is no
closer than its current pose. Otherwise it waits.

- The drone flies only forward, into its own field of view. So every mover contact is the policy's own doing: it
  flew into a pillar it could see, or turned into one.
- That makes the training signal fair and keeps a 0.95 gate attainable.
- It also keeps motion deterministic given the actions.

**Constraints on every mover's full sweep**, checked when the episode is sampled:

- the sweep stays inside the bounds;
- it keeps a surface gap of at least `min_gap_m` = 1.0 m to every pillar that may stand still this episode, and to
  every other mover's sweep. Pillars that may stand still are the unchosen ones, plus chosen movers not yet placed,
  which revert home if they cannot be placed. A mover's own home is excluded;
- it stays ≥ 6 m from the start and ≥ 4 m from the target and every distractor.

A mover with no valid route among 20 draws stays home.

**Path-length guard.** Plain reachability is vacuous here: s01 leaves a free ring of at least 8 m around its field, so
start → target is always reachable around it. Instead:

- The guard compares 4-connected BFS path lengths on the reachability occupancy grid (0.25 m cells, obstacles
  inflated by `DRONE_RADIUS_M + CLEARANCE_M` = 1.4 m). The path with every mover's sweep as an obstacle must be at most
  1.20 × the path through static s01.
- Seeds: on the target side, the free cells within the 5 m success radius. On the start side, the free cells within
  0.5 m of the start, widened to 1.0 m if there are none (the start sits on the inflation boundary by construction).
- Repair is greedy: drop the mover whose removal shortens the path most, and repeat. Dropping from the end of the
  list instead could strip every mover.
- **The sampler never raises.** An explicit-seed evaluation replays the same seed on any reset fault, so a raise
  would loop. If even static s01 has no path (never seen), the episode keeps no movers.

**Measured on 500 training-range seeds** (worker 0's first 500 episodes, never the gate's;
`scripts/build_scenes.py scenes/s01d_moving_pillars.json` writes `runs/levels/s01d.dynamic_report.json`):

| Measure | Result |
|---|---|
| Movers placed | mean 10.04 (min 6, max 12) |
| Movers within 4 m of the flight line | mean 3.68; 8 of 500 episodes have none |
| Chosen movers left home | 56 of 5,074: 51 with no valid route (all for the start keep-out), 5 dropped by the guard |
| Guard repairs | 5 episodes, one mover each |
| Path ratio | median 1.00, p95 1.14, max 1.19 |
| Sampling cost per reset | median 28 ms, p95 39 ms |

`tests/test_motion.py` re-checks every constraint by brute force on 150 seeds, independently of `motion.py`.

**Contact (the paper's d_col, spec §3.1).**

- A mover contact is scored when the drone's swept segment D_k → D_{k+1} passes within `contact_m` = 1.0 m of the
  mover's surface at P_{k+1}. In Unreal the mover is teleported before the step and stays put for the whole step,
  so P_{k+1} is the right pose. Static pillars are still scored physically by the simulator.
- `info["collision_source"]` is `"sim"`, `"mover"` or `"mover_inferred"`.
- On a mover collision, `info["mover_in_view"]` records whether that mover was within ±45° of the heading in any of
  the last 3 frames.

**Invariant, checked when the scene loads:** `contact_m − max_speed · dt > DRONE_HALF_SPAN_M` (0.48 m), here
1.0 − 1.2 · 0.2 = 0.76 > 0.48. Any mover the drone is still flying beside is more than `contact_m` from it, so one
teleport (at most `max_speed · dt`) can never put a mover inside the drone's rotor span. That holds even without
the yield rule.

**Defensive inference.**

- The rule: if `sim.step` raises `CameraPoseError` or `KinematicsJumpError` while the drone was within
  `contact_m + (v_forward_max + max_speed) · dt` (1.0 + 3.2 · 0.2 = 1.64 m) of a mover's surface before the step,
  the episode ends as a COLLISION on its last real observation, with `collision_source = "mover_inferred"`. That
  radius is the farthest at which the contact rule above could have fired this step. Otherwise the error re-raises
  as before.
- Why: a deterministic mover-caused fault would otherwise be dropped from training forever, and would make the
  gate replay the same seed until it gives up.
- Inferred collisions are counted separately in the train and gate records.

**Expert observation.** Depth is stacked over 3 frames (0.4 s) and stored as float16, with the newest frame last. That
is the minimum a depth-only policy needs to see velocity.

- Default: 3 frames when the scene has a `dynamic` block, else 1. `--depth-frames` overrides it.
- Static s01 keeps (1, 84, 84) float32, so every M2 checkpoint still loads.
- Replay memory, since SB3's `DictReplayBuffer` keeps both obs and next_obs, at a buffer of 150k: one float32 frame
  (today) needs 8.5 GB; three float16 frames 12.7 GB; three float32 frames 25 GB.

**Reset order** (`expert/env.py`; it keeps the drone and the movers apart at every moment):

1. Destroy the last episode's spawned objects.
2. Park every displaced pillar 50 m under its own home. The drone may have ended on a vacated home, and the reset's
   up–across–down sequence must not sweep through a pillar.
3. Sample the episode.
4. Run `sim.reset`.
5. Spawn the target and distractors.
6. Mark the union of the old and new movers displaced, before anything moves.
7. One batch: new movers to P(0), old movers that don't move this episode back home.
8. The render step.
9. The start-contact checks.
10. Set displaced to the new movers.

A relaunch starts a fresh level with every pillar home, so the env forgets its displaced set when it launches a
new simulator.

## Simulator interface

`Simulator.set_object_poses(poses: Mapping[str, Pose])`:

- The real backend checks every name against its `movable_objects` allow-list (the layout's tags) before any request.
- It then calls `set_object_pose(name, pose, True)` once per name, in order. A partial batch is possible if one fails.
- A falsy status or an `"ERROR code"` reply raises `ObjectPoseError`, which is not recoverable: it means a wrong level
  or a bug.
- A `"Fatal Timeout"` marks the connection lost and raises `SimRequestTimeoutError`, a step fault. Every later call
  then raises `SimConnectionLostError`, a launch-class fault the resilient wrapper answers with an immediate relaunch.
- `CameraPoseError` moves to `sim/types.py`, re-exported by `sim/airsim_backend.py`, so the env can catch it without
  importing the backend.

`FakeSimulator` gains named scene objects that can be moved and collided with. It rejects unknown names exactly like
the server, and applies a batch name by name, so it can fail partway.

Known residual: only the mover path maps "Fatal Timeout". The other requests the backend makes (`set_pose`,
kinematics, `world.step`) still raise it as a bare `RuntimeError`. That is the behaviour M2 ran under, with 0
occurrences in run 2. Widening the mapping is a separate change.

## Live go/no-go first: `scripts/probe_movers.py` → `docs/gates/m2d_mover_probe.json`

The probe passes only if every check holds:

| # | Check |
|---|---|
| a | Baked tags move to within 0.05 m, and the neighbouring pillars do not move. |
| b | A restore lands within 0.05 m. |
| c | Depth at the image centre, on the same step, shows a pillar teleported in front of the drone, within 0.3 m. |
| d | Flying into a displaced pillar collides, including on the first step after the move. |
| e | Flying through a vacated home spot gives no collision and no `CameraPoseError`. |
| f | Parking 50 m underground and restoring both work. |

It also records the latency of a 12-pose batch, sequential against `request_async`.

**Result (2026-10-02 16:39, `docs/gates/m2d_mover_probe.json`): PASS, every check.**

| Check | Measured |
|---|---|
| a. Move | 2.8e-6 m error; neighbours drifted 0 |
| b. Restore | 2.8e-6 m error |
| c. Depth on the same step | 4.004 m against 4.0 m expected (23.1 m before the move) |
| d1. Flying in | Collided at step 15, with the drone's centre 0.51 m from the surface |
| d2. First step after the move | Collided. The last runway step was 0.364 m; the pillar was placed 0.18 m beyond the rotor tips |
| e. Vacated home | Flew 4.63 m through the empty spot: no collision, no fault |
| f. Park and restore | Within 3e-6 m |

Latency:

- A 12-pose batch takes 15.1 ms median sent one request at a time, and 3.5 ms through `request_async`.
- A step takes 74.2 ms without moves and 87.1 ms with 10 moves, +17 %.

That is under a third of a step, so the backend keeps its sequential requests.

The physical collision in d1 registered at 0.51 m from the surface, slightly beyond the 0.48 m half-span the
invariant assumes. The invariant still holds with margin: 0.76 m against 0.51 m.

Had it failed, the fallback was a rebuilt S01D level whose candidate movers are spawned at runtime, the spawn path M0
proved. `motion.py` is unchanged by that. If latency is too high, the options are fewer movers, or
`request_async` with its disconnect-on-error wrapped. Either outcome gets its own decision record.

## Evidence safety

- Default evidence paths are per scene: s01 keeps `m2_*`, s01d gets `m2d_*`, and any other scene must pass `--out`.
- Writing over an existing `docs/gates/*` file is refused before the run starts.
- `sessions.json` locks the observation config, the scene id and sha, the base layout's sha and the motion
  parameters. A mismatched resume is refused before the run directory is claimed.
- The gate reads each checkpoint's observation space from its zip before building the env. It refuses checkpoints
  that disagree, and records the observation config.

## Levers if the M2d gate fails (the user's call, never taken silently)

`--resume` for more hours, or a warm start from s01's weights. The warm start copies `conv1` into the newest-frame
channel and zeroes the others. Beyond those: fewer or slower movers, or privileged mover state in the vector.
