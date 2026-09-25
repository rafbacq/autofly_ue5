# 2026-09-25 — Code-review findings C1–C9: evidence, fixes, and corrections to the recorded M2 run

A full review of M0–M2 (2026-09-24) raised nine findings. Each was re-verified against the code, the run logs and
offline reproductions before any fix, and each fix landed test-first on branch `fix/code-review-findings`. The
recorded evidence of the first M2 run (2026-09-16/17) is archived unedited in
`docs/gates/archive/2026-09-17-m2-run1/`, together with the C9 audit of its gate. The corrections below are to
how that evidence reads, not to its numbers.

Several modules carried a "closed for editing" note from earlier plans (`expert/env.py`, `expert/reward.py`,
`sim/process.py`, `sim/airsim_backend.py`). This review reopened them. Their notes predate the defects found here.

## The findings

| # | Finding | Evidence | Fix (commit) |
|---|---|---|---|
| C1 | A relaunch or startup sweep stopped **every** simulator on the host | The training log shows the train (inst0) and eval (inst1) simulators killing each other at every evaluation after the first genuine relaunch. All 77 `Timeout` faults and 14 of the 23 relaunches were self-inflicted. Reproduced offline with real processes: a relaunch kills a sibling's simulator, and a `close()` that returns late kills the successor in its slot | Run-owner token in `pid.json`; `stop_instance`/`stop_instances`/`sweep_orphaned_instances`; per-slot lock; `stop(expected_pid=…)`; detach-before-close; worker processes exit on unrecoverable errors (`d6cae95`) |
| C2 | A mid-step fault stored a fabricated transition | The wrapper reset internally and returned the next episode's first observation. SAC stored it as a truncation and bootstrapped from it. Reproduced: 3 resets instead of 2 and one cross-episode replay row. Faulted episodes also reached Monitor and `rollout/success_rate` | Truncate on the last real observation (`is_success=None`); `FaultFilteringDictReplayBuffer`; `Resilient(Monitor(AutoFly))`; `FaultAwareEvalCallback` (`faf529e`) |
| C3 | The recorded throughput could not be trusted | `chosen_n=1` was a crash: n=2 died on an unwrapped CameraPoseError hang (bare AutoFlyEnvs). M0 had measured 2 instances at 11.3 steps/s against 7.4 for one, and a 1 ms clock at 18.0 steps/s. `gradient_steps=1` did one update per n transitions | Measurement uses `make_vec_env`; `scene_autofly_s01_fast.jsonc`; `--scene-config` everywhere; `gradient_steps=-1` (`0afbc38`) |
| C4 | Seed ranges overlapped | `SAC(seed=0)` flew episode 0 twice (seeds `[0, 0, 1, 2]`). A resume replayed session 1's episodes. The gate's EVAL_SEED_BASE episodes were also the training-time evaluation's | Worker ranges start at `(rank+1)·1e6`; `EVAL_CALLBACK_SEED_BASE = 200e6`; `sessions.json` with session offsets, re-seeding and a reward-version guard; `--run-root`, which refuses a directory already holding a run (`f6f085c`) |
| C5 | The reachability docs overstated the guarantee | A solid 54 × 54 m block passes the crossing rule, because the corridor always keeps a free lane beside the field | Docstrings, spec §6.2 and §13 corrected; a test pins the limitation (`493dc9c`) |
| C6 | A fresh clone could not be set up or tested | `requirements.lock` lacked 39 of 84 packages, including torch, SB3, gymnasium and tensorboard. `setup_venv.sh` then passed its import check with no torch and rewrote the lock. Scripts hard-coded one home directory, `jq` and `/usr/bin/python3.12`. The package imported `scripts/`, and tests pinned this machine | Portable scripts; a lock reproduced by a clean install; an explicit 3.12 interpreter; package moves; `conftest.py` builds the s01 layout (`f3b9496`, `6b7e71d`) |
| C7 | The GPU-fault audit missed logs and failed open | Only each slot's final `sim.log` was scanned, so 24 rotated in-run logs were missed. `xid_count` reads 0 from a journal the user cannot see. `live_m1` and the M0 `faults` step scanned every log ever written. `stdout.log` was truncated by every relaunch | `instance_logs_since` (mtime), `audit_engine_faults`, `kernel_journal_readable` required by every `faults_ok`, and `stdout.log` appended (`ead89e5`) |
| C8 | The reward paid progress inside the success radius | A misaligned approach to 2 m out-earned an aligned success at 5 m (discounted ≈ 13.1 vs 10.5). `final.zip`'s successes ended a median 2.35 m out (best_model: 4.89 m). 7 of its 16 real failures were out-of-bounds within 5 m of the target | Progress clamped at 5 m; `REWARD_VERSION = "2-no-progress-inside-success-radius"`; `oob_kind`, pose and bearing in every step's info and gate record (`f5edfa4`) |
| C9 | Episodes started somewhere other than their start | 15 of the 800 gate episodes were one-step "collisions" whose drone was ≥ 4 m from its start. Every one followed a collision episode: 15 of 73 such resets, against 0 of 723 others. The camera check is skipped on collision steps, so nothing caught them | `ResetPoseError`, `KinematicsJumpError` (collision steps included), `StartCollisionError`, `SetPoseError`, all recoverable; `scripts/audit_m2_gate.py`; the live `scripts/probe_crash_reset.py` (`0532587`) |

## Corrections to the recorded M2 evidence (archived, not edited)

- **`m2_train.json` backend faults (C1):**
  - All 77 `Timeout` faults and 14 of the 23 relaunches (7 per instance) came from the two simulators stopping each other. Only 9 relaunches followed genuine faults: 8 in training and 1 in eval, all CameraPoseError.
  - The 7 `"Timeout"` entries in `outcome_histogram` are those episodes. Each is also one fabricated replay-buffer row (C2): 7 of 206,639.
  - Wall-clock cost was small: every evaluation took 720–950 s, with or without the loop.
- **`m2_gate.json` note on seeds (C4):** the note said gate seeds +0…+19 were the training evaluation's; that is wrong.
  - SB3's EvalCallback drew a new ~21-seed slice at every evaluation, walking seeds 100,000,000+0 … ≈+177, which overlaps most of the gate's 200.
  - `best_model.zip` was saved at the 125,000-step evaluation (mean return 73.27, 20/20 successes). By the fault log, that evaluation used ≈ +94…+113.
  - `final.zip` was never selected on any episode, so its gate result is held out.
- **`m2_gate.json` outcomes (C9):** 15 of its failures are the impossible episodes in `m2_gate_audit.json`. Excluding them:

  | Checkpoint | Condition | Successes | Rate |
  |---|---|---|---|
  | best_model | deterministic | 166/194 | 0.856 |
  | final | deterministic | 180/196 | 0.918 |
  | best_model | stochastic | 169/196 | 0.862 |
  | final | stochastic | 182/199 | 0.915 |

  The verdict is unchanged, because every rate is below 0.95. The worst legitimate episode used 93% of the distance it could have flown.
- **`m2_instances.json` (C3):** `chosen_n = 1` records n=2 crashing, not n=1 winning.
- **Engine-fault records (C7):** the 24 unscanned rotated logs from the training window were checked by hand, along with every other `runs/sim/inst*/sim*.log`. None contains `VK_ERROR_DEVICE_LOST` or a fatal line. The kernel journal is readable on this host (the user is in `adm`), so the recorded Xid deltas are valid. The recorded `faults_ok: true` values stand.

## Also changed

- **Tests are hermetic.** The suite no longer writes `runs/sim/inst*/client.log` or runs a global sweep, which would have killed a live run's simulators.
- **Typed backend errors.** A refused `set_pose` is `SetPoseError`, not a bare `RuntimeError` that would have ended a run.
- **Explicit gate records.** The gate records `reward_version`, `scene_config` and, when the throughput measurement is missing, why.

## Still live-only

These run on the GPU host, with display `:1` and no foreign GPU job:
- M0 and M1 re-verification;
- `probe_crash_reset.py`;
- M1 on `scene_autofly_s01_fast.jsonc`;
- `measure_instances.py`;
- the s01 retrain under reward version 2;
- the M2 gate.

The commands are in `docs/runbook-m2.md`. M2 closes when a new `m2_gate.json` has `pass: true`, and `m2_env_manifest.json`, `m2_instances.json`, `m2_train.json` and `m2_gate.json` all describe that one run (PLAN2's milestone exit).
