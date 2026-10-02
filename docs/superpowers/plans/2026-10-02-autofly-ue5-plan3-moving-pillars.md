# M2d Moving Pillars Implementation Plan (scene s01d)

> **For agentic workers:** steps use checkbox (`- [ ]`) syntax for tracking. Offline tasks were executed test-first on
> branch `feat/moving-pillars`; each checked task names its commit. Live tasks follow `docs/runbook-m2d.md`.

**Goal:** an s01d expert that crosses s01's pillar field while 8–12 of its pillars move, with ≥ 95 % deterministic
success over 200 held-out episodes. Static s01, M2's gate and every M2 checkpoint stay byte-for-byte valid.

**Architecture:**

- A pure motion core, `scenes/motion.py`, holds routes, sampling, the guard, yielding and contact.
- A runtime controller, `expert/movers.py`, drives one episode's movers through `Simulator.set_object_poses`.
- `AutoFlyEnv` sequences resets and steps so the drone and a pillar are never in one place.
- A three-frame float16 depth stack lets the depth-only expert see motion.

Everything is testable against `FakeSimulator`. One live probe decides go/no-go before training.

**Tech stack:** unchanged from Plan 2: Python 3.12, SB3 2.9 (SAC), torch 2.14, gymnasium 1.3, numpy 1.26.4,
Project AirSim 1.0.2 on UE 5.7.4.

**Spec:** §6.5 (new), §7, §7.1, §8 and §12 (M2d), amended 2026-10-02.

**Decision record:** `docs/decisions/2026-10-02-dynamic-obstacles.md`. M2's closeout:
`docs/decisions/2026-10-02-m2-closeout.md`.

**Predecessor:** Plan 2 (M2, passed 2026-09-26).

## Global constraints

Plan 2's constraints apply unchanged: `env -u PYTHONPATH`, display `:1`, `run_job.sh`, the shared GPU, and the
evidence rules. In addition:

- **Static s01 must not change.** `tests/test_static_s01_golden.py` pins it from M2's own code, `5d0abf5`: its
  episodes, its simulator calls (never `set_object_poses`), its observations, rewards and outcomes. Its info dicts
  may only gain keys.
- **No test may change `docs/gates/`.** `tests/conftest.py` fails such a test and restores the files. A RED run once
  overwrote `m2_gate.json`, which was restored from git.
- **Movers always teleport.** A sweep would stop at the drone.
- **Names are checked exactly** against the allow-list, because the server's lookup also matches substrings.
- **The sampler never raises**, because explicit-seed evaluation replays a seed whose reset faulted.

## Decisions taken while planning

- **D1: a new scene id, not a change to s01.** s01d reuses s01's level (`level: "s01"`), so no level build and no
  packaging are needed. *Cost if wrong:* none; s01 is untouched.
- **D2: movers are baked pillars, moved at runtime.** The S01 pillars are already `MOVABLE`, tagged `obs_NNNN`.
  *Fallback:* runtime-spawned movers on a rebuilt level, the spawn path M0 proved, if the probe fails.
  `motion.py` is unchanged by that.
- **D3: movers yield to the drone** (2.5 m). Since the drone only flies forward into its view, every contact is the
  policy's doing. *Cost if wrong:* movers are easier to dodge than real traffic; the parameter is in the scene file.
- **D4: mover contact is analytic** (d_col = 1.0 m on the swept step), the paper's d_col. Static pillars stay
  physical. *Cost if wrong:* a stricter margin around movers than around static pillars, which errs on the safe side.
- **D5: a mover-adjacent `CameraPoseError` or `KinematicsJumpError` is a collision**, counted separately as
  `mover_inferred`, within the 1.64 m radius where contact was possible. *Cost if wrong:* a rare unrelated fault near
  a mover is scored as a crash. The count is on every record.
- **D6: 3 float16 depth frames** for dynamic scenes; static scenes keep 1 float32. *Cost if wrong:* a 12.7 GB buffer;
  `--depth-frames` overrides it.
- **D7: path-length guard ≤ 1.20× static**, repaired greedily. It replaces plain reachability, which is vacuous on s01.
- **D8: the first training run is fresh.** Warm starting from s01, fewer or slower movers, or privileged mover state
  are levers for after a failed gate, and they are the user's call.

## File structure

```
autofly_ue5/
  evidence.py              per-scene evidence defaults; never overwrite docs/gates (A5)
  scenes/motion.py         routes, sampling, guard, yield, contact -- pure (A3)
  scenes/resolve.py        scene id -> file, base level, layout, map, config, allow-list (A2)
  scenes/model.py, scene.schema.json   level + dynamic block, invariant (A2)
  expert/movers.py         one episode's movers in a simulator (A4)
  expert/env.py, obs.py, episode.py    reset/step integration, depth stack, movers in EpisodeSetup (A4)
  sim/protocol.py, types.py, fake.py, airsim_backend.py   set_object_poses + typed errors (A1)
scenes/s01d_moving_pillars.json (A6)
scripts/probe_movers.py (A1b), build_scenes.py (A6), watch_training.py (A7)
scripts/m2_gate.py, render_episodes.py, measure_instances.py (A5)
docs/runbook-m2d.md (A7)
```

---

### Phase 0: record-keeping and pinning

- [x] **0.1** Branch `feat/moving-pillars` from `viz/render-episodes` at `5d0abf5`, already equal to `origin`.
  Baseline: the full suite passes, and ruff finds only the 5 known hits.
- [x] **0.2** Golden record of static s01 (`d66a3cf`): 28 seeds' episodes, the observation space, the real depth
  fixture's encoding, and four scripted flights (success, out of bounds, two collisions) with full call traces.
- [x] **0.3** Decision records and spec amendments (`086b8af`).

### Phase A: offline (all against FakeSimulator)

- [x] **A1. Simulator interface** (`c70c087`):
  - `set_object_poses` with an exact allow-list, one teleporting request per name;
  - `ObjectPoseError`, plus `SimRequestTimeoutError` → `SimConnectionLostError`, which relaunches;
  - `CameraPoseError` moved to `sim/types.py`;
  - FakeSimulator scene objects;
  - the projectairsim fakes reproduce the server's replies;
  - `RecordingSimulator` records moves.
- [x] **A1b. Go/no-go probe** (`f6738a2`): checks a–f and latency, with fakes proving it passes a good simulator and
  fails three bad ones.
- [x] **A2. Schema and resolver** (`476fac3`): ids `s01d`/`s06r`; `level` (no chains, matching static fields);
  layout sha checked; one resolver replacing every id → path guess; `build_scene` refuses a reused level; episodes
  labelled by scene.
- [x] **A3. `scenes/motion.py`** (`300f1f9`): routes, sampling with constraints, guard and greedy repair, yield,
  contact. A brute-force re-check runs on 150 seeds.
- [x] **A4. Env integration** (`a31d206`): park → sample → reset → spawn → mark → batch → render → check; yield →
  move → step → contact → inference. Also the depth stack, and SAC on float16 (build, train, save, reload).
- [x] **A5. Plumbing and evidence safety** (`06651c3`):
  - per-scene evidence and refusal to overwrite;
  - run identity locked in `sessions.json`, and a host preflight;
  - `--depth-frames`, `--sim-root`;
  - collision sources in every record;
  - the gate reads checkpoint observation spaces from their zips;
  - the renderer draws movers;
  - `measure_instances --scene`.
- [x] **A5b. Evidence guard for the test suite** (`b40cec7`).
- [x] **A6. `scenes/s01d_moving_pillars.json`** (`45114ee`) and its dynamic report (`bb9afe7`). Over 500 training
  seeds: 10.04 movers, 5 repairs, path ratio ≤ 1.19, 28 ms per reset.
- [x] **A7a. Training watcher** (`c9cd870`) and `docs/runbook-m2d.md`.

### Phase A: live (GPU host, display `:1`; `docs/runbook-m2d.md`)

- [ ] **A7.1** The probe → `docs/gates/m2d_mover_probe.json`. **Go/no-go.**
- [ ] **A7.2** `measure_instances --scene s01d` → `docs/gates/m2d_instances.json`; set N.
- [ ] **A7.3** The s01 regression: 10 gate seeds with `s01_r2` best_model, same outcomes as the M2 gate.
- [ ] **A7.4** The s01d smoke: 20 minutes of training, a 10-episode gate and 2 renders.
- [ ] **A7.5** 12 h training, with the watcher, the progress plot and TensorBoard → `docs/gates/m2d_train.json`.
- [ ] **A7.6** The 200-episode gate and its audit → `docs/gates/m2d_gate.json`, `m2d_gate_audit.json`.
- [ ] **A7.7** The zero-shot baseline of the s01 expert on s01d → `docs/gates/m2d_baseline_s01_expert.json`.
- [ ] **A7.8** Render 6–10 s01d gate episodes.
- [ ] **A7.9** Close M2d: commit the evidence, update CLAUDE.md, write the closeout decision record.

## After M2d

- **Phase B: M3**, the collector, writer, validator and state[9] decoding (spec §9–§11). It gets its own plan, Plan 4.
  Its s01 pilot can start without waiting for M2d. Its s01d pilot follows a passed M2d.
- **Phase C: M4**, real-object scenes. It starts with an asset survey and a licence decision; see
  `docs/decisions/2026-10-02-m4-asset-survey.md`.
