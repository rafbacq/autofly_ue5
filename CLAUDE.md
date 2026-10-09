# autofly_ue5: instructions for agents and people working here

This project recreates the simulation side of AutoFly (arXiv 2602.09657) on Unreal Engine 5.7.4 + Project AirSim 1.0.1.
It generates an AutoFly-format UAV navigation dataset: SAC "expert" pilots fly procedurally built obstacle scenes, and
their episodes become the dataset. This file holds standing instructions. `MEMORY.md` holds what past sessions learned
the hard way; read both before changing anything.

@MEMORY.md

## Where things are

| What | Where |
|---|---|
| Design (binding) | `docs/superpowers/specs/2026-09-15-autofly-ue5-dataset-design.md`: milestones §12, risks §13 |
| Plans | `docs/superpowers/plans/`: plan1 = M0/M1, plan2 = M2 (SAC expert), plan3 = M2d (moving pillars), plan4 = M3 (collector and dataset), plan5 = M4 (scenes and assets; draft, decisions U1-U4 open) |
| Evidence | `docs/gates/*.json`; superseded runs in `docs/gates/archive/` |
| Decisions | `docs/decisions/`: rulings, the 2026-09-25 code-review findings, the 2026-10-02 M2 closeout, moving obstacles and the M4 asset survey, the 2026-10-03 M3 a0 decision, the 2026-10-06 dataset rebalancing |
| Results | `final_results/` (README, figures, videos, records, dataset card; built by `scripts/build_final_results.py` from `docs/final_results_manifest.json`); the running status report `results/2026-10-07-project-status/` |
| Live procedures | `docs/runbook-m2.md` (M0/M1/M2, done); `docs/runbook-m2d.md` (M2d: probe, throughput, smoke, 12 h run with watchers, gate); `docs/runbook-m3.md` (M3: TFDS venv, smoke, pilot); `docs/runbook-rebalance.md` (spec §10.3: detector venv, scoring, weights) |

Code (`autofly_ue5/`):

| Package | Contents |
|---|---|
| `sim/` | The `Simulator` protocol (`protocol.py`) and the only code that imports `projectairsim` (`airsim_backend.py`). Also the in-memory `FakeSimulator`, process ownership (`process.py`) and typed errors (`types.py`) |
| `scenes/` | Scene JSON → layout (`generate.py`: jittered_grid, poisson, clusters, stacks) → reachability → level spec; grid paths (`paths.py`: crossing detours for spec §13, L_opt for the paper's PER); glTF geometry and flight-band footprints (`gltf.py`); the scene resolver (`resolve.py`: id → file, base level, layout, map, config, allow-list); moving-obstacle rules (`motion.py`, pure) |
| `expert/` | `AutoFlyEnv` (`env.py`), reward, observation and depth stacking (`obs.py`), episode sampling, movers at runtime (`movers.py`), the resilient wrapper (`resilient.py`), fault classes (`faults.py`), seed ranges (`seeds.py`), vec envs (`vec.py`), the SAC trainer (`train.py`) and the evaluation harness (`evaluate.py`) |
| `collect/` | The episode collector (`collector.py`): the expert flies with a0's aligned start, successes are stored, everything else goes to the rejects, faults replay the seed |
| `dataset/` | AutoFly's state[9] (`state.py`), the canonical raw store (`raw.py`), the RLDS/TFDS exporter (`rlds.py`, with TFDS-generated metadata templates in `rlds_templates/`), the target pool (`targets.py`: `assets/targets.json`, the split's usable entries, drawn by the sampler at collection time only) and the phase rebalancing (`rebalance.py`, spec §10.3: pure; the detector runs in `scripts/detect_targets.py`) |
| `evidence.py` | Per-scene default evidence paths, and the refusal to write over `docs/gates/` |
| `validate/` | The live checks for any built scene (`live_m1.py`: M1's gate for s01, M4's per-scene gates), the geometry of every obstacle kind for them (`obstacles.py`: cylinders, yawed boxes, glTF meshes ray-cast in their own frame), the engine-fault audit (`engine_check.py`) and the dataset validator (`dataset.py`, spec §11) |
| `evaluation/` | Closed-loop evaluation of any policy (spec §12 M6, plan 6 item 6): the `Policy` interface a VLA implements and the reference adapters (`policy.py`: the SAC expert, a privileged scripted baseline, hover), the frozen episode files (`episodes.py`: `assets/eval_episodes/`), `run_condition` (`harness.py`) and SR/CR/PER with Wilson intervals (`metrics.py`) |
| `report/` | Reading raw stores, rejects and collector records for figures (`stores.py`) and the shared figure style (`style.py`), used by `scripts/dataset_figures.py`, `dataset_montage_video.py`, `harness_figures.py` and `rebalance_figures.py` |

Elsewhere in the repo:
- `scripts/`: shell and Python entry points (run_job, setup, launch/stop sim, throughput, the gate for any scene, audits,
  the mover probe, renders, `watch_training.py` (the live training dashboard), M3's `collect_dataset.py` and
  `export_rlds.py`, the rebalancing's `detect_targets.py` (Grounding DINO, in `runs/tools/gdino_venv`) and
  `rebalance_dataset.py`, `dataset_stats.py` (a store's path lengths and PER), M4's `fetch_assets.py` (the CC0 list,
  into `downloads/`, recorded in `assets/sources.json`) and `measure_assets.py` (`assets/measurements.json`), the
  evaluation harness's `eval_harness.py` (`write-episodes`, `run --policy hover|scripted|expert:<zip>`), and the
  status report's `results_figures.py` and `dataset_episode_video.py` (`results/<date>/`)).
- `configs/`: Project AirSim scene configs; `*_fast` means a 1 ms real-time update rate.
- `scenes/`: scene JSONs. `s01d_moving_pillars.json` flies s01's level (`"level": "s01"`) with 8–12 moving pillars;
  `s09_coloured_poles`, `s07_stacked_boxes`, `s02_sparse_trees` are M4's first scenes.
- `assets/`: the CC0 model and material sources (`sources.json`), their measured geometry (`measurements.json`), the
  registry every scene draws from (`registry.json`) and the frozen evaluation episodes (`eval_episodes/`, written once
  by `scripts/eval_harness.py write-episodes`, never over). The editor-side import and material scripts are in
  `autofly_ue5/scenes/ue/` (`import_assets.py`, `ue_materials.py`, `fix_foliage_materials.py`, run through
  `scripts/import_assets.py` or `scripts/ue_editor_run.py`).
- `tests/`: offline tests (the FakeSimulator plus fakes of the projectairsim API).

Git-ignored but present on the GPU host:
- `engine/` (UE 5.7.4 prebuilt);
- `platform/` (Project AirSim @4d878bf);
- `ue_project/` (Blocks + plugin; packaged binary under `Packaged/`);
- `.venv/`;
- `runs/` (every run's output: `sim/`, `jobs/`, `levels/`, `expert/`, …; `runs/tools/tfds_venv` reads RLDS exports back,
  `runs/tools/gdino_venv` + `runs/tools/hf_home` run the rebalancing's detector);
- `data/` (collected datasets and their rejects).

## Status (update when it changes)

| Milestone | State |
|---|---|
| M0 | Passed 2026-09-15 (`docs/gates/m0_gate.json`) |
| M1 | Passed 2026-09-16 (`docs/gates/m1_gate.json`) |
| M2 | Passed 2026-09-26 (`docs/gates/m2_gate.json`, run 2): best_model 0.98 deterministic / 0.99 stochastic, final 0.96 / 0.95, over 200 held-out episodes. Run 1's failed gate is archived in `docs/gates/archive/2026-09-17-m2-run1/`. Closeout: `docs/decisions/2026-10-02-m2-closeout.md` (best_model flies M3, stochastically) |
| M2d | **Gate failed** 2026-10-03 (`docs/gates/m2d_gate.json`): the s01d expert (`runs/expert/s01d_r1`, 412,828 steps) scored 0.775 (best_model) and 0.535 (final) deterministic against 0.95; the s01 expert scored 0.36 zero-shot. Half of best_model's mover collisions came from out of view, and final learned to dive below the altitude floor. Closeout and levers: `docs/decisions/2026-10-03-m2d-closeout.md`. The user left the levers to us (2026-10-03). Run 2 (warm start) collapsed and was stopped. **Run 3** (`runs/expert/s01d_r3`: from scratch, mover input, r_bounds 10, 250k buffer) survived a host freeze (`docs/decisions/2026-10-04-r3-host-freeze-and-resume.md`). It tracked run 1 through 200k, scoring 0.65 on 20 evaluation episodes with every failure a mover near miss, and was paused there on 2026-10-04 (resumable). **Run 4** (`runs/expert/s01d_r4`: run 3 plus a per-step mover clearance penalty) was stopped cleanly at 360k. Its best scored 0.78 in selection (step_320000, 100 validation episodes), level with run 1; its gate is deferred. **Run 5** (`runs/expert/s01d_r5`: run 4 plus a 0.3 m training contact margin and an altitude margin) **gated at 0.885** (`docs/gates/m2d_r5_gate.json`, final, 177/200; audit 0 flagged). **The gate still fails** (bar 0.95). On the same 200 episodes as run 1's 0.775 it is significantly better (McNemar p = 0.0046), and mover collisions fell 35 → 7. Remaining: 10 static collisions, 7 mover collisions, 5 dives, 1 lateral exit (`docs/decisions/2026-10-05-s01d-r5-margins.md`). **Run 6** (`runs/expert/s01d_r6`: run 5 plus a 1.1 m static training boundary, a 0.5 m mover margin and a V-shaped altitude cost; the user asked for it on 2026-10-06) **gated at 0.920** (`m2d_r6_gate.json`, step_240000, pre-registered), with no static collision (run 5: 10). The mean of its 31 checkpoints from 250k to 550k (`scripts/average_checkpoints.py`) **gated at 0.935** (`m2d_r6_swa_gate.json`), the best so far. **Both fail** (bar 0.95: 190/200). Against run 1's 0.775 on the same episodes the mean is significantly better (p = 9.1e-06); against run 5 not significantly (p = 0.12). Left: mover near misses at full speed (8 of 13) (`docs/decisions/2026-10-06-s01d-r6-plan.md`). Run 6 then trained 12 h more (session 1, to 1,061,800 steps). The pre-registered pick, the mean of its 250k–1,060k checkpoints, gated 0.920 and flew 0.930 on 200 unflown episodes: 370/400 = 0.925 (CI 0.895–0.947), so the expert flies at about 0.93. Mover near misses (about 4.75%) resisted every lever tried. **Accepted by the user on 2026-10-07 at 0.935** (`docs/decisions/2026-10-07-s01d-accepted-at-0.935.md`): the gate stays failed, no run 7; M5 collects s01d with `runs/expert/s01d_r6/averaged/avg_250k_550k.zip` (sha256 55e666a5…), stochastically, a0 aligned. The untried levers (closing penalty, lower entropy target, lateral margin) are recorded there |
| M4 | **Scenes done for plan 6** (plan 5 + plan 6). Built, packaged and gated live: s01, s02 sparse trees, s03 scattered rocks, s04 stone field, s05 standing stones (snow, -12 EV), s06 tree clusters, s07 stacked boxes, s09 coloured poles, and the test_seen s05r and s06r (`docs/gates/m4_*_gate.json`; s06r passed 2026-10-08 once the 6 m depth probe fell back to 5 m and 4 m as the 12 m one does). Target pool: all 60 entries drawable since 2026-10-08 (25 Kenney low-poly stand-ins marked `stand_in`, `docs/decisions/2026-10-08-target-pool-diversity.md`), minus per-scene look-alikes (`assets/target_exclusions.json`); a reset holds 5 hover steps when it spawns an object new to the session, and every pool episode records its first-frame visibility (`validate/visibility.py`); the live probe passed all 60 after the hold (`m4_targets_probe_run2.json`). **Mesh scenes' contact rule** (2026-10-08, f17c55f): an episode ends 1.05 m from the model's real outline (precomputed distance fields, `assets/footprints/`), not from a pivot circle: the circle bulged up to 3 m past leaning trees and s02's endings were a median 2.7 m from any geometry; 1.05 = the drone's measured 0.63 m collision reach + a full-speed ending step. The circle (0.9 m) stays only to evaluate runs trained under it; tools read a run's shape from its identity (`expert/run_shape.py`). Out of scope for now: s08, s10-s12 |
| M5 | **Plan 6 done 2026-10-09** (`docs/superpowers/plans/2026-10-08-autofly-ue5-plan6-prototype-dataset.md`, the user's direction 2026-10-08: diverse data collection, a prototype a VLA can be attached to). The dataset: `data/autofly_ue5_proto/1.0.0` (TFDS/RLDS, 1,800 episodes, 339,007 records, train 1,710 / val 90 frozen in `data/autofly_ue5_proto_split.json`), merged from 36 raw stores `data/<scene>_c1..c4` (200 per scene, 9 scenes, 50 seen targets); TFDS reads it back exactly, an OpenVLA-style loader iterates it, `dataset_statistics.json` in OpenVLA's schema. Pilots (`docs/gates/plan6_pilots.json`): s01 expert; s01d mean for s01d/s07/s09; s02 run 1's pick for s02-s06 (its gate: 0.580 circle, 0.895 outline, both under 0.95). The closed-loop harness reproduces every pilot's kept-rate; a privileged straight-line baseline scores 0.06 (`docs/gates/plan6_harness/`). The thin slice (`data/<scene>_c0`, 450 episodes) measured the pilots before the reset's warm-up existed and is not in the dataset (`docs/decisions/2026-10-08-reset-hold-and-warmup.md`). Everything for readers: `final_results/` |
| M3 | **Passed** 2026-10-03 (`docs/gates/m3_gate.json`): the 100-episode s01 pilot kept 100 of 101 (stochastic s01_r2 best_model, a0 = sector8), the validator passed over 17,738 records, and TFDS read every episode back exactly from `data/s01_pilot/1.0.0/`. a0 costs the expert nothing (`docs/gates/m3_a0_probe.json`: 50/50 with, 49/50 without). Decisions: `docs/decisions/2026-10-03-m3-a0-and-collection.md`, review: `2026-10-03-m3-review-findings.md` |

Each milestone stops for the user's go-ahead before the next starts. The user asked for M2d and M3 on 2026-10-02.
Ahead of M5, the rebalancing stage of spec §10.3 is built and run on the whole pilot: `data/s01_pilot/rebalance.json`
holds P0 = (0.652, 0.348) and weights (0.766, 1.438) at the paper's 0.7, every first detection but one on the target
(`docs/decisions/2026-10-06-dataset-rebalancing.md`). M4's offline groundwork is done (plan 5, C0-C1: every placement
type, the detour metric); its editor and asset work waits for decisions U1-U4 and a window with no live simulator.

## Hard rules on this host

- **Every Python process runs with `env -u PYTHONPATH`.** The host exports a ROS Jazzy path that is also Python 3.12.
- **Every Unreal process runs with `DISPLAY=:1 SDL_VIDEODRIVER=x11`** (`sim.process.sim_environment` does it for you).
  `:1` comes from the user's desktop login and vanishes on reboot. If `/tmp/.X11-unix/X1` is missing, ask the user;
  don't start X yourself.
- **Long jobs go through `bash scripts/run_job.sh start|wait|stop <name>`.** That covers anything over ~5 minutes:
  launches, smoke tests, gates, training, builds. Give the Bash tool `timeout: 600000` for `wait` calls, and use one
  `wait` per call. Logs are in `runs/jobs/<name>.log`.
- **The GPU is shared with other users** (e.g. `ameya`'s CARLA). `gpu.check_gpu_for_launch` refuses a launch while a
  foreign job holds more than 2 GB or fewer than 6 GB are free. Never kill or signal a process this project did not
  launch, and never bypass the guard without the user saying so.
- **Simulators are stopped only through their `runs/sim/inst<N>/pid.json` record:**
  - `scripts/stop_sim.py`;
  - `sim.process.stop_instance` / `stop_instances` for a run's own slots;
  - `sweep_orphaned_instances` for a dead run's.

  Never `pkill -f`. Never sweep every instance: that is how the train and eval simulators once killed each other
  (C1).
- **Ports.** Instance N uses topics `8989 + 12N` and services `8990 + 12N`. They listen on all interfaces without
  authentication.
- **Other projects.** Never modify other projects under `~/research_uav/`, or `~/Documents/AirSim`.

## Evidence rules

- **Never edit a file in `docs/gates/`.** Re-verification writes new files, e.g. `m1_gate_reverify.json`. A superseded
  run is moved unedited to `docs/gates/archive/<date>-<run>/` and explained in `docs/decisions/`.
- **A failed gate stays failed.** Never lower a threshold, re-run hunting for a lucky seed, or report a flattering
  number. Record the real one, and let the user choose the next lever.
- **Smoke runs must not overwrite evidence.** `expert/train.py`, `scripts/m2_gate.py` and `scripts/measure_instances.py`
  write into `docs/gates/<milestone>_*.json` by default (s01: `m2_`, s01d: `m2d_`; any other scene must pass `--out`),
  and refuse to write over an existing file there. A smoke run still passes `--out runs/...`, and training passes a
  fresh `--run-root runs/expert/<scene>_<tag>`. `train.py` refuses a `--run-root` that already holds a run, and a resume
  under another scene, observation or motion setting; `runs/expert/s01` holds run 1, `s01_r2` run 2.
- **No test may change `docs/gates/`.** `tests/conftest.py` points every test's evidence directory
  (`evidence.GATES_DIR`) at a scratch directory, and fails a test during which the real `docs/gates/` changed. It never
  deletes or rewrites anything, because a live run may be writing its record. A test of a refusal path must not be able
  to reach the real path even before the refusal exists (its RED run): point it at a scratch directory and stub the work.
- **Change the reward, bump the version.** Any change to `expert/reward.py`'s reward bumps `REWARD_VERSION`. Resuming
  across versions is refused, and the gate records the version.

## Commands

```bash
bash scripts/setup_venv.sh                                    # venv from requirements.lock (--relock / --recreate)
env -u PYTHONPATH .venv/bin/python -m pytest                  # ~500 tests, ~75 s, all offline; engine test skips without engine/
uvx ruff check --select F autofly_ue5 scripts tests           # unused/undefined names (5 pre-existing test-file hits)
for f in scripts/*.sh; do bash -n "$f" || echo "syntax error: $f"; done   # one file per call: `bash -n a b` checks only a
```

Live work: see `docs/runbook-m2d.md` (and `docs/runbook-m2.md` for M0–M2). Every step lists its exact command and how
to read the result. Watch a training run with `scripts/watch_training.py --run-root <run> --job <job>` and TensorBoard
on `<run>/tensorboard` (bind 127.0.0.1: the host is shared).

## How to change code here

- **Test first, always.** Write a test that fails on the current code, watch it fail for the right reason, then fix.
  For simulator behaviour, extend the fakes in `tests/test_airsim_backend.py` (the projectairsim API),
  `sim/fake.py` (the protocol) or `tests/test_expert_train.py`'s `_FlakyFakeSimulator` (fault injection).
- **Fakes must reject what the real server rejects.** A contract that lives only in prose reached live hardware three
  times (see MEMORY.md).
- **Tests never touch the real `runs/sim/`.** Pass `sim_root=tmp_path` (`make_vec_env`, `ResilientAutoFlyEnv`,
  `m2_gate.run`, `measure_n`) and `run_root=tmp_path` (`process.*`). A live training run's simulators live there.
  A test that calls `train.main()` stubs `ProjectAirSimSimulator.launch` so it can never start a real simulator.
- **New backend hazards get a typed exception.** Put it in `sim/types.py` or `airsim_backend.py`, add it to
  `expert/faults.py` if recoverable, and document it in `sim/protocol.py` and spec §7.1. A bare `RuntimeError` ends a
  12-hour run.
- **Static s01 must not change.** `tests/test_static_s01_golden.py` pins its episodes, simulator calls, observations,
  rewards and infos from M2's own code. A new feature draws its random numbers after every existing draw and makes no
  new simulator call on a static scene.
- **Seeds come from `expert/seeds.py`'s ranges.** Workers use `(rank+1)·1e6 + session·1e5`, the gate 100e6, training
  evaluation 200e6, probes 300e6, collection 400e6, selection 500e6 and the evaluation harness's frozen files 600e6.
  Don't invent new bases without adding them there and to its disjointness test.
- **Docstrings explain *why*, citing the measurement or incident that forced a choice.** Match the surrounding
  density. Keep new ones proportionate.
- **Commits.** One concern per commit on a feature branch, with a message saying what was wrong and what changed.
  End each with the Co-Authored-By trailer your harness specifies. Never push or merge without asking.
