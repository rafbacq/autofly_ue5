# MEMORY.md: what this project has learned

Session-to-session learnings: what worked, what didn't, and traps not to fall into twice. Newest first within each
section; each entry is dated and names its evidence. When an entry turns out wrong, fix or delete it. Don't keep a
stale "fact". Standing rules live in `CLAUDE.md`; this file is the reasoning and history behind them.

## Traps: read these before touching the simulator path

- **A global sweep kills other runs' simulators (2026-09-24, C1).** Stopping "every simulator we own" from a relaunch
  made the training (inst0) and eval (inst1) simulators kill each other at every evaluation. That accounted for all
  77 Timeout faults and 14 of the 23 relaunches in run 1. Stop only your own slots. Ownership is the *main* run
  process's (pid, start-ticks) token, even for simulators launched by SubprocVecEnv workers (`sim/process.py`).
- **A `close()` abandoned by a bounded wait comes back later (2026-09-24, C1).** When it returns it must neither detach
  (Python) nor stop (process) the simulator relaunched in its place. Detach before closing, and use
  `stop(expected_pid=...)`.
- **An uncaught exception in a SubprocVecEnv worker hangs forever (Task 7, 2026-09-16).** projectairsim leaves a
  non-daemon thread that blocks interpreter shutdown, so the worker never exits, and SB3's unbounded `recv()` hangs
  the trainer with it. Workers run the wrapper in `worker_mode` (`os._exit(1)` on anything unrecoverable), and every
  cross-process wait is bounded (`expert/vec.py`). The same thread blocks the main process's exit, which is why
  `train.py`, `m2_gate.py` and `probe_crash_reset.py` end with `os._exit`.
- **pynng `Timeout` comes straight out of the projectairsim client, not through our exceptions (Task 8).** A killed
  simulator surfaced as `Timeout`, and cheaply: run 1's evaluations took the same 720–950 s with or without its
  Timeout storm (projectairsim's service socket has a 1 s send / 300 s receive timeout). `pynng.exceptions.Closed` in the logs is
  `__del__` noise, not a fault.
- **Episodes can start somewhere other than their start (2026-09-24, C9).** In run 1's gate, 15 of 73 resets right
  after a collision episode left the drone ≥ 4 m from its start; 0 of 723 other resets did. They scored as one-step
  "collisions". The camera-pose check is skipped on collision steps, so it never saw them. `reset()` now verifies the
  pose, `step()` refuses teleports (collision steps included), and `AutoFlyEnv.reset` refuses a start in contact.
  13 of the 15 fit "the drone stayed at the crash site". `scripts/probe_crash_reset.py` measures it live; its result
  is not recorded yet.
- **CameraPoseError happens almost only during reset (run 1: 97 of 97 in training).** That's the set_pose sweep of the
  up–across–down recovery sequence. Five in a row with a near-identical error (e.g. 17.916 m three times) means a
  stuck simulator: relaunch.
- **A runtime `spawn()` accepts only a base `UMaterial` package path, or `None` (M2, three blocked live runs).**
  `asset` is the short Unreal asset name (`Cylinder`, `1M_Cube`), never a registry key. The fake now rejects what the
  server rejects: a contract written only in prose reached live hardware three times.
- **The first frame of every session is corrupt** (spec §7.1): 65,522 of 65,536 depth pixels read under 1 m. Only
  `reset()`'s own steps may consume it. `step()` before `reset()` raises `SessionNotResetError`.
- **UE rotates `sim.log` to `sim-backup-<UTC time>.log` at every launch, keeping its mtime.** Find a run's logs by
  mtime, never by comparing the UTC names with local time (this host is EDT, −4). Run 1's records scanned only the
  last `sim.log`, missing 24 in-run logs; they were checked by hand afterwards and are clean.
- **`journalctl` shows nothing to a user without `adm`/`systemd-journal`,** so an Xid count of 0 before and after
  "passes" without having looked. Fault records now require `kernel_journal_readable()`. This user (nvidiasims) is in
  `adm`.
- **Display `:1` is the user's desktop session, not a daemon.** It disappeared after the 2026-09-24 14:38 reboot.
  Xvfb/VNC are not installed.
- **The GPU is shared (2026-09-24).** User `ameya` ran CARLA plus a detector, ~4.4 GB at 82% utilisation, for hours.
  The launch guard then refuses every simulator launch; wait or coordinate.

## RL and evaluation learnings

- **Progress credit inside the success radius taught a dive (2026-09-24, C8).** `final.zip` closed in misaligned and
  turned at the end: its successes ended a median 2.35 m out against 4.89 m for `best_model`. 7 of its 16 real
  failures left the bounds within 5 m of a target sitting 0–3 m from the edge. The reward now clamps progress at
  5 m (`REWARD_VERSION` 2). Watch `oob_kind` in the next gate.
- **A fault must never become a transition (2026-09-24, C2).** The old wrapper reset internally and returned the next
  episode's first observation, and SAC stored and bootstrapped from it. Faults now truncate on the last real
  observation with `is_success=None`. The replay buffer drops them, Monitor never sees them, and evaluation replays
  the seed.
- **SB3's EvalCallback doesn't use fixed episodes.** It resets with `seed=None`, so each evaluation draws the next
  ~21 counter seeds. In run 1 it walked through ~178 of the gate's 200 episodes. `FaultAwareEvalCallback` plays the
  same seeds (from 200e6) every time.
- **`SAC(seed=0)` seeds the first reset explicitly with 0** (plus rank). With worker ranges starting at 0, episode 0
  flew twice. `SAC.load` re-seeds the same way, so resumed sessions replayed session 1. Ranges now start at
  `(rank+1)·1e6`, and resumes re-seed (`expert/seeds.py`, `train.reseed_resumed_model`).
- **`gradient_steps=1` with n workers does one update per n transitions.** Use `-1`, which is also forced on resume
  because old checkpoints restore 1.
- **Run 1 in numbers:** 206,639 steps in 12 h (4.78 steps/s at n=1, 3 ms clock). Evaluations took 720–950 s each.
  `best_model` was picked at the 125k evaluation. Gate: best 0.83, final 0.90. Excluding the 15 impossible episodes:
  0.856 and 0.918. Remaining real failures are mostly pillar-field collisions, so more training steps are the main
  lever.

- **SB3 runs callbacks before storing the step's transition.** A checkpoint taken at step N holds N−1 replay rows,
  so off-by-one expectations in tests are SB3's ordering, not a bug (2026-09-25, resume round-trip test).
- **SB3's `env_method()` is all-or-nothing.** One dead worker lost every worker's fault summary, and the record read
  "0 faults". Collect per worker, bounded (`vec.collect_fault_summaries`), and record which slots were missing.
- **Claim a run directory only after every offline check** (scene, layout, config file), and don't record a resumed
  session until its checkpoint and buffer are confirmed. Otherwise the likeliest early failures on this shared host
  (a foreign GPU job, a failed launch) leave a directory you can neither restart nor resume. Found by the 2026-09-25
  final review.

## Throughput learnings

- **The 1 ms real-time update rate gave 18.0 steps/s against 7.4 at 3 ms** (M0 smoke, lock-step intact, only 50
  steps). Two concurrent instances gave 5.67 + 5.65 steps/s. `configs/scene_autofly_s01_fast.jsonc` exists; use it
  only after M1 passes on it.
- **Run 1's `chosen_n=1` was a crash, not a measurement.** The n=2 attempt died on an unwrapped CameraPoseError hang.
  The measurement now uses training's own resilient workers.
- **Packaged launches take ~3.4 s.** The ready timeout is 300 s; the old 900 s made a hung launch cost 15 min per
  attempt.

## Environment and tooling learnings

- **`requirements.lock` drifted (2026-09-24).** It lacked torch/SB3/gymnasium/tensorboard (39 of 84 packages), and
  `setup_venv.sh` rewrote it on every run. The lock is now regenerated only by `--relock`. A clean install of the
  current lock reproduces the M2 venv exactly (torch 2.14.0+cu130). PyPI and an 18 GB pip cache are available.
- **The old test suite wrote `runs/sim/inst*/client.log` and ran a real global sweep.** Running tests during a live
  run would have killed its simulators. Tests now use `sim_root=tmp_path`; keep it that way. A test that drives
  `train.main()` must also stub `ProjectAirSimSimulator.launch`: one RED run of such a test got as far as a real
  launch attempt, and only the GPU guard (a foreign job was up) stopped it (2026-09-25).
- **`uvx ruff check --select F` works without touching the venv.** 5 pre-existing unused-import hits in older test
  files remain.
- **Full suite: ~384 tests in ~36 s on this host.** A fresh clone without `runs/` passes too (`tests/conftest.py`
  builds the s01 layout, identical to the stored one).

## Open questions

- Why does a reset right after a crash sometimes leave the drone at the crash site? Run
  `scripts/probe_crash_reset.py` (runbook step 3). If a second reset always fixes it, the wrapper's retry is enough;
  if not, look at Project AirSim's collision state on `set_pose`.
- Can M2 reach 0.95 with the fixes plus 2–3× throughput? Collisions in the pillar field were the largest real
  failure class. If the retrain falls short, PLAN2's D6 levers (curriculum, `k_p`, step limit) are the user's call.
- Spec §9 step 5 says the expert collects data stochastically, while PLAN2 gates the deterministic policy. Settle this
  at M3.
