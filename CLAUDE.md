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
| Plans | `docs/superpowers/plans/`: plan1 = M0/M1, plan2 = M2 (SAC expert) |
| Evidence | `docs/gates/*.json`; superseded runs in `docs/gates/archive/` |
| Decisions | `docs/decisions/`: rulings, and the 2026-09-25 code-review findings |
| Live procedures | `docs/runbook-m2.md`: re-verify M0/M1, then close M2 |

Code (`autofly_ue5/`):

| Package | Contents |
|---|---|
| `sim/` | The `Simulator` protocol (`protocol.py`) and the only code that imports `projectairsim` (`airsim_backend.py`). Also the in-memory `FakeSimulator`, process ownership (`process.py`) and typed errors (`types.py`) |
| `scenes/` | Scene JSON → layout → reachability → level spec |
| `expert/` | `AutoFlyEnv` (`env.py`), reward, observation, episode sampling, the resilient wrapper (`resilient.py`), fault classes (`faults.py`), seed ranges (`seeds.py`), vec envs (`vec.py`), the SAC trainer (`train.py`) and the evaluation harness (`evaluate.py`) |
| `validate/` | M0/M1 gate checks and the engine-fault audit (`engine_check.py`) |

Elsewhere in the repo:
- `scripts/`: shell and Python entry points (run_job, setup, launch/stop sim, throughput, M2 gate, audits).
- `configs/`: Project AirSim scene configs; `*_fast` means a 1 ms real-time update rate.
- `scenes/`: scene JSONs.
- `tests/`: offline tests (the FakeSimulator plus fakes of the projectairsim API).

Git-ignored but present on the GPU host:
- `engine/` (UE 5.7.4 prebuilt);
- `platform/` (Project AirSim @4d878bf);
- `ue_project/` (Blocks + plugin; packaged binary under `Packaged/`);
- `.venv/`;
- `runs/` (every run's output: `sim/`, `jobs/`, `levels/`, `expert/`, …).

## Status (update when it changes)

| Milestone | State |
|---|---|
| M0 | Passed 2026-09-15 (`docs/gates/m0_gate.json`) |
| M1 | Passed 2026-09-16 (`docs/gates/m1_gate.json`) |
| M2 | **Open.** The first run failed its gate: deterministic success 0.83 (best_model) and 0.90 (final) against 0.95. It is archived in `docs/gates/archive/2026-09-17-m2-run1/`. The C1–C9 fixes landed on branch `fix/code-review-findings` on 2026-09-25. Next: `docs/runbook-m2.md` steps 1–8 |

The M0/M1 re-verification after those fixes has not been run yet: it needs display `:1` and a GPU free of foreign
jobs. Each milestone stops for the user's go-ahead before the next starts.

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
  write into `docs/gates/` by default, so a smoke run passes `--out runs/...`, and training passes a fresh
  `--run-root runs/expert/<scene>_<tag>`. `train.py` refuses a `--run-root` that already holds a run; `runs/expert/s01`
  holds run 1.
- **Change the reward, bump the version.** Any change to `expert/reward.py`'s reward bumps `REWARD_VERSION`. Resuming
  across versions is refused, and the gate records the version.

## Commands

```bash
bash scripts/setup_venv.sh                                    # venv from requirements.lock (--relock / --recreate)
env -u PYTHONPATH .venv/bin/python -m pytest                  # ~384 tests, ~36 s, all offline; engine test skips without engine/
uvx ruff check --select F autofly_ue5 scripts tests           # unused/undefined names (5 pre-existing test-file hits)
bash -n scripts/*.sh
```

Live work: see `docs/runbook-m2.md`. Every step there lists its exact command and how to read the result.

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
- **Seeds come from `expert/seeds.py`'s ranges.** Workers use `(rank+1)·1e6 + session·1e5`, the gate 100e6, training
  evaluation 200e6, and probes 300e6. Don't invent new bases without adding them there and to its disjointness test.
- **Docstrings explain *why*, citing the measurement or incident that forced a choice.** Match the surrounding
  density. Keep new ones proportionate.
- **Commits.** One concern per commit on a feature branch, with a message saying what was wrong and what changed.
  End each with the Co-Authored-By trailer your harness specifies. Never push or merge without asking.
