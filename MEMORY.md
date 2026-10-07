# MEMORY.md: what this project has learned

Session-to-session learnings: what worked, what didn't, and traps not to fall into twice. Newest first within each
section; each entry is dated and names its evidence. When an entry turns out wrong, fix or delete it. Don't keep a
stale "fact". Standing rules live in `CLAUDE.md`; this file is the reasoning and history behind them.

## Traps: read these before touching the simulator path

- **A step's camera frames can fail to arrive (2026-10-05).** 33 minutes into run 5, instance 0 completed a step's clock
  and command, but no rgb/depth frame came within 5 s (`FrameTimeoutError`, `sim/sync.py`). It had never happened in
  any earlier run. It was not a known fault, so the worker exited and the run ended at 25k steps. It is now a step
  fault, recovered like a stuck camera (fa7ef7b). Any newly typed backend error that is not in
  `expert/faults.py` ends a run the first time it fires: check new ones against that list.

- **The Wi-Fi driver freezes this host, and a freeze can leave a 0-byte record (2026-10-04).** Four boots between
  2026-09-25 and 2026-10-03 ended in `mt7921e … driver own failed` with no shutdown sequence. The 2026-10-03 one froze
  run 3 at 76k steps for 25 h. It also came mid-relaunch: `runs/sim/inst1/pid.json` came back empty, and any launch or
  stop in that slot would have raised `JSONDecodeError` and ended the resume. Records are now written durably, and an
  unreadable one is set aside (b7d02b3).
  - After a freeze, check that display `:1` exists, then the resume point's contents
    (`runs/m2d_diag/verify_resume_point.py`), before `--resume`.
  - The user chose to leave the host unchanged (`docs/decisions/2026-10-04-r3-host-freeze-and-resume.md`).

- **A `run_job.sh` job's recorded pid is its bash wrapper (2026-10-03).** A stall check that read that pid's CPU time saw
  0 for every job and raised a false alarm. Sum CPU over the job's process group (`pgid` in `<job>.pid.json`) instead.
  The gate and the pilot print nothing per episode, so silence alone never shows progress.

- **Arm a failure alarm for the whole of every long run (2026-10-03).** M2d's session 0 crashed at 22:51 and went
  unnoticed for 2 h 21 min: the 2 h background alarm had expired at 20:23 and was not re-armed. Re-arm the alarm and
  the log Monitor every time either expires (`docs/decisions/2026-10-03-m2d-session0-crash.md`).
- **An `--instances N` training run owns slot N too (2026-10-03).** Its evaluation simulator comes up in
  `runs/sim/inst<N>/` at every evaluation, with the trainer as `owner_pid`. A side job (smoke, probe, pilot) takes slot
  N+1 or higher. Seen live: a smoke planned for slot 4 beside a 4-worker run would have landed on the evaluator.
- **Live poses are float32; the fake's are exact (2026-10-03).** The first live collector record sat 1 µm from the
  start, and a validator check written against the fake at 1e-6 m failed it. One step later the drone is 5–11 mm
  away, so the check is now 1 mm. Size tolerances on live numbers, not the fake's.
- **Unreal can refuse a pillar move (2026-10-02 22:51, 5.5 h into M2d's run).** `SetObjectPose failed. Unable to move
  object obs_0047` ended the run, because `ObjectPoseError` was not recoverable. Now the pose is read back, the move is
  retried once, and then `ObjectMoveRefusedError` truncates the episode as a fault (202a314). The cause is unknown:
  moves at float32 precision did not reproduce it. Count `MOVE-REFUSED` in the log. Session 1 (6.5 h, 2026-10-03)
  saw 2 refusals, and both pillars read back 1 µm from the requested pose and were accepted. So Unreal seems to refuse
  a move that changes nothing, rather than one that is blocked.
- **A dead worker's stale pipe reply cost the run its record (2026-10-02).** After the crash, collecting the fault
  summaries read a queued tuple and `combine_fault_summaries` crashed, so no `m2d_train.json` was written. Replies are
  now drained and validated, and the record is assembled under guards (075e3bb).
- **A RED test of a refusal path ran the real path and wrote over committed evidence (2026-10-02).** A new test of
  `m2_gate.main()` refusing to overwrite `docs/gates/` ran, before the refusal existed, with the gate's old default
  `--out docs/gates/m2_gate.json`, and wrote a failed record over it (restored from git; sha256 3f9438f2…). The same
  run truncated `runs/sim/inst0/client.log` (`route_client_log` opens with mode "w"). `tests/conftest.py` now gives every
  test a scratch evidence directory and fails a test during which the real `docs/gates/` changed. Its first version also
  restored a snapshot, which the review caught: that would have deleted a live run's record written mid-suite. Point
  such tests at a scratch directory and stub anything that launches, so the RED run is harmless.
- **projectairsim's "Fatal Timeout" disconnects the client before it raises (2026-10-02, `client.py:255-282`).** A reset
  cannot recover; only a relaunch can. The mover path maps it to `SimRequestTimeoutError` → `SimConnectionLostError`
  (launch-class). The other backend requests (`set_pose`, kinematics, `world.step`) still raise it bare: 0 seen in run 2.
- **The server's object lookup matches substrings (2026-10-02, `UnrealHelpers.h:56-96`).** `FindActor` returns a
  spawned object of that exact name first, then the first actor whose name *contains* the string or whose tag equals
  it. `set_object_poses` therefore checks names exactly against an allow-list (the layout's tags) before any request.
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
  13 of the 15 fit "the drone stayed at the crash site". `scripts/probe_crash_reset.py` measured it live on
  2026-09-25: 0 of 30 bad first resets after a crash (`docs/gates/m1_crash_reset_probe.json`).
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

- **Average a run's checkpoints; the mean flies steadier than any one of them (2026-10-06, run 6).**
  - Neighbouring checkpoints differed sharply in deterministic exits (stage 1: 1 against 15 in 40 episodes).
  - The mean of 31 checkpoints (250k-550k, `scripts/average_checkpoints.py`) flew 100 fresh validation episodes with 0
    exits, at 0.94 against 0.83 for the stage-2 winner (p = 0.035). Narrower means did less: 6 at 0.88, 16 at 0.91.
  - It gated 0.935, the best M2d score, though still under the bar. The gain was in exits; mover contacts did not fall.
- **Late checkpoints of a long run fly worse; a wide mean still helps (2026-10-07, run 6 session 1).**
  - From 550k to 1,060k the individual checkpoints scored 0.60–0.95 on the eval watch, below the 250k–550k ones.
  - Means of late windows alone were the worst (0.88–0.89 on 100 fresh episodes). The widest mean (250k–1,060k)
    was the best at 0.95, yet gated 0.920 and confirmed 0.930.
  - Selecting the best of four on 100 episodes inflates it, as with stage 1. Pooled over 400 independent episodes the
    expert flies 0.925 (CI 0.895–0.947).
- **Mover contacts stay near 4.5% whatever the lever (2026-10-07).**
  - Unchanged by a wider training boundary, a clearance penalty, weight averaging, and 12 more hours.
  - The replayed contacts happen in saturated turns near a mover (`2026-10-06-s01d-r6-plan.md`).
  - Untried: a closing penalty, a lower entropy target.
- **Stage 1's best is inflated; trust stage 2 (2026-10-06).** Run 6's stage-1 leader (0.975 on 40 episodes) scored
  0.85 on stage 2's 100. The top of 21 noisy scores regresses. Taking five candidates into stage 2 costs nothing with
  five free slots.
- **A static training boundary removes static collisions (2026-10-06).** Run 6's 1.1 m boundary took gate static
  collisions from 10 (run 5) to 0. A wider mover boundary did not do the same: 7 at 1.3 m, then 5 and 8 at 1.5 m. Run 6's
  mover contacts were near misses at full speed past a mover that had stopped to yield, so speed is the next lever.

- **Never decide a run's fate on 20-episode scores (2026-10-05, run 4).**
  - Run 4 was stopped at 360k on an eval-watch slide: 0.85, 0.70, 0.65, 0.40 at 200k-350k.
  - On 40 validation episodes, its late checkpoints were its best (0.80-0.825). The 200k checkpoint that read 0.85
    scored 0.55.
  - Stage 2 (100 episodes) put the top three at 0.75-0.78.
  - Use the eval watch to see a run is alive. Use the selection stages to compare checkpoints or runs
    (`docs/decisions/2026-10-05-s01d-r5-margins.md`).
- **s01d experts fly to the boundary they train on, so move the boundary (2026-10-05).**
  - Mover-contact gaps cluster just inside the 1.0 m rule in every run: medians 0.94 (r1), 0.96 (r3), 0.93-0.98
    (r4).
  - Run 4's per-step clearance penalty (k 0.5) did not move the cluster, nor the ~20% of episodes it fails.
  - Run 5 trained on a 1.3 m boundary (`--mover-contact-margin 0.3`) and observed the task's rule. Gate mover
    collisions fell 35 → 7 on the same 200 episodes; success went 0.775 → 0.885 (McNemar p = 0.0046,
    `m2d_r5_gate.json`).
  - Static collisions (10) are now the largest failure, and the same pattern: a static pillar's only boundary was
    physical contact. Run 5's static collisions ended 0.52-0.63 m from the surface, measured drone centre to pillar
    surface (more than the 0.37-0.47 m rotor reach).
  - Replayed flights passed static pillars at 0.51-0.91 m and movers at 1.28-1.50 m: right at each boundary
    (`runs/m2d_diag/clearances.py` on a render's summary.json).
  - The deterministic tail reaches about 0.3 m inside the training boundary (movers: 1.3 m trained, 0.97 m at the
    gate). Run 6 trains on 0.5 m against each failure line (`2026-10-06-s01d-r6-plan.md`).
- **A training margin costs little training success (2026-10-06).** Run 5's 0.3 m mover margin trained at 0.63-0.70
  late, against run 4's 0.61-0.68 without one. Run 5's own training success was flat from 320k to 561k, so more hours
  of the same settings would not have reached 0.95.
- **The deterministic policy drifts out of the altitude band; the stochastic one rarely does (2026-10-05).**
  - Run 5's training out-of-bounds was 0.04-0.14, yet deterministic checkpoints lost up to 10 of 40 episodes to slow
    climbs and dives (r4's 160k: 38 of 40).
  - Likely cause: SAC parks its entropy in the vertical channel, which nothing in the band shapes.
  - Selection filters such checkpoints, but the drift varies sharply between neighbours. A V-shaped altitude cost
    (`--altitude-margin-penalty 0.2 1.0`) shapes the whole band; run 5's 0.1 / 0.5 m shaped only its edges.

- **Never warm-start SAC with a fresh replay buffer (2026-10-03, run 2).** r1's best_model copied into run 2 flew at
  ~0.8 during the policy warm-up, then collapsed within ~500 updates (its 20k checkpoint: 0.25 against r1's 0.85 on the
  same 20 seeds). An offline replay on the real buffer showed r1's own network drifting the same way, so the mover
  branch and the copy were not the cause: 88% of its deterministic actions moved by more than 0.25 after 1,000
  updates. A third of the learning rate barely helped. A narrow new buffer makes the critic forget what the old one
  taught, and the actor follows its errors. Train from scratch, or keep the old buffer
  (`docs/decisions/2026-10-03-s01d-r2-stopped-r3-from-scratch.md`).
- **On s01d, every mover "collision" of r1 was a near miss (2026-10-03, `docs/gates/m2d_r1_contact_probe_*.json`).**
  All 24 contacts were 0.84–1.00 m from the surface: inside the 1 m contact rule, never physical. Mostly they were
  beside the drone, past a mover that stood still while yielding. Depth alone cannot tell such a mover from a static
  pillar, hence the privileged mover input.

- **The s01d expert reached 0.775, not 0.95, in 12 h (2026-10-03, `docs/decisions/2026-10-03-m2d-closeout.md`).**
  Training on moving pillars took success from 0.36 (the s01 expert, zero-shot) to 0.775 (`best_model`, 225k).
  - 19 of its 35 mover collisions came with the mover out of the forward camera's view.
  - Its late policy (`final`, 0.535) learned to dive below the 1 m floor. This showed in training as out-of-bounds
    rising 0.09 → 0.43 while collisions fell, an hour before any evaluation reflected it.
  - Watch the outcome mix, not just the success rate.
- **One s01d gate episode does not replay (2026-10-03).** 4 of 8 rendered episodes changed outcome (s01: 1 of 12), and
  successes drifted by up to 10 steps. Resets are history-free, so this is non-bit-exact physics amplified by movers
  that yield to the drone. Judge the 200-episode rate.
  - Run 5's final (2026-10-06): 8 of 15 replays changed outcome, with the same config, checkpoint hash and
    observation as its gate (`runs/viz/s01d_r5_final/summary.json`). None of its 3 static collisions happened again.
  - An expert that flies close to obstacles makes outcomes chance. Don't diagnose one episode from one run.

- **Run 2's stochastic training success plateaued near 0.8 while its deterministic evaluations reached 1.0
  (2026-09-26 data, read 2026-10-02 with `scripts/watch_training.py`).** Training-time evaluation is 20 episodes and
  noisy: 1.00 at 200k, 0.15 at 225k, 1.00 at 275k, 0.65 at 450k. Judge a run by the 200-episode gate, never by one
  evaluation point.
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

- **Dataset size and pace (M3 pilot, 2026-10-03).** 100 episodes = 17,738 records = 1.77 GB of PNG frames, stored twice:
  the raw store and the RLDS shards each hold the same PNGs. The paper's 13K episodes would need about 460 GB for
  both (683 GB were free). Collection ran at 101 episodes in 45 min on one slot, beside two other simulators.

- **TFDS for the RLDS read-back lives in its own venv (2026-10-03).** `.venv` cannot take TensorFlow (numpy 1.26.4).
  `runs/tools/tfds_venv` holds tensorflow-cpu 2.18 + tensorflow-datasets 4.9, and needs `tensorflow-metadata==1.16.1`:
  the newest is built for protobuf 6, which TF 2.18 refuses (`VersionError` at import). Recipe: `docs/runbook-m3.md`.

- **`requirements.lock` drifted (2026-09-24).** It lacked torch/SB3/gymnasium/tensorboard (39 of 84 packages), and
  `setup_venv.sh` rewrote it on every run. The lock is now regenerated only by `--relock`. A clean install of the
  current lock reproduces the M2 venv exactly (torch 2.14.0+cu130). PyPI and an 18 GB pip cache are available.
- **The old test suite wrote `runs/sim/inst*/client.log` and ran a real global sweep.** Running tests during a live
  run would have killed its simulators. Tests now use `sim_root=tmp_path`; keep it that way. A test that drives
  `train.main()` must also stub `ProjectAirSimSimulator.launch`: one RED run of such a test got as far as a real
  launch attempt, and only the GPU guard (a foreign job was up) stopped it (2026-09-25).
- **`uvx ruff check --select F` works without touching the venv.** 5 pre-existing unused-import hits in older test
  files remain.
- **Full suite: ~500 tests in ~75 s on this host (2026-10-02; ~384 in ~36 s before moving pillars).** A fresh clone without `runs/` passes too (`tests/conftest.py`
  builds the s01 layout, identical to the stored one).

## Open questions

- What gets s01d from 0.885 to 0.95? Run 5 (mover input, r_bounds 10, a 0.3 m training contact margin, an altitude
  margin) gated 0.885. Its 23 failures were 10 static, 7 mover, 5 altitude and 1 lateral. Run 6 (2026-10-06) trains
  each boundary 0.5 m outside the task's failure line. It gated 0.920, and the mean of its checkpoints gated 0.935. 12 h
  more and a wider mean flew 0.925 over 400 episodes, so the expert is at about 0.93. Mover near misses (about 4.5%)
  resisted every lever. Untried: a closing penalty, a lower entropy target, a lateral margin. The user chooses
  (`docs/decisions/2026-10-06-s01d-r6-plan.md`).

Settled:
- a0's aligned start costs the s01 expert nothing (2026-10-03, `docs/gates/m3_a0_probe.json`): 50/50 with a0, 49/50
  without, over the same 50 seeds. The first smoke's 3 of 5 was chance.
- Moving a baked pillar works live (2026-10-02, `docs/gates/m2d_mover_probe.json`): moves land within 3e-6 m, show in
  the same step's depth, and collide at once. 10 moves add 13 ms to a 74 ms step (+17 %), and at N = 4 nothing
  (13.49 against 13.37 steps/s). A hovering drone covers only centimetres in its first step (M0: 2.33 m in 10 steps
  at 2 m/s), and the rotor tips reach 0.367 m straight ahead (0.472 m on the diagonal). The probe's first version
  missed both and would have failed live.
- The crash-then-reset probe (2026-09-25, `docs/gates/m1_crash_reset_probe.json`): 0 of 30 first resets after a crash
  missed their pose (worst 1.9 mm). Run 2 still raised 22 `ResetPoseError`s in 12 h, all recovered by the retry.
- M2 reached 0.98 with the C1–C9 fixes and N = 4 (run 2).
- Stochastic or deterministic collection: stochastic, flown by best_model (`docs/decisions/2026-10-02-m2-closeout.md`).
  The deterministic gate stays the acceptance bar.
