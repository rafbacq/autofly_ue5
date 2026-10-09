# 2026-10-03 — M2d's 12 h run: session 0 crashed on a refused pillar move, and went unnoticed for 2 h 21 min

Evidence: `runs/jobs/m2d_train.log` (session 0), `runs/expert/s01d_r1/sessions.json`, this session's transcript.

## What happened

- **22:51:45 EDT, 2026-10-02**, 5 h 28 min into the 12 h run `runs/expert/s01d_r1`, at 192,772 steps. Worker 2's
  `set_object_pose('obs_0047')` failed with Unreal's own refusal: `SetObjectPose failed. Unable to move object
  obs_0047, check if object state is movable!` (log line 10053).
- `ObjectPoseError` was not a recoverable fault, so the worker exited by design (`worker_mode`). The trainer saw
  `EOFError` and stopped: `training failed: EOFError`.
- **No run record was written.** Collecting the fault summaries afterwards hit a stale reply still queued in a worker
  pipe (a tuple, not a summary dict), and `combine_fault_summaries` crashed on it (`AttributeError: 'tuple' object has
  no attribute 'get'`, line 10118).

Before the crash, the run was healthy. It had recovered 115 `CameraPoseError`s and 11 `ResetPoseError`s, with 11
relaunches. Its evaluations (20 fixed seeds, deterministic) were 0.00 at 25k and 50k, 0.05 at 75k, 0.70 at 100k, 0.60
at 125k, 0.65 at 150k and 0.85 at 175k.

## Why Unreal refused the move: not confirmed

The packaged pillars are MOVABLE and BlockAll, and every other move in 5.5 h succeeded. In Project AirSim's
`SetObjectPose`, "Unable to move" means the actor was found but the move itself returned false. Live moves as small
as float32 precision allows (one ULP is about 1 µm near 11.5 m) did not reproduce it. The cause is open.

## What changed

- **202a314: a refused move is recoverable.** The backend reads the pose back. If the pillar is within 1 mm of the
  requested pose, the move counts as done. Otherwise it retries once, and then raises `ObjectMoveRefusedError`, a step
  fault: the episode is truncated on its last real observation and never becomes a transition, as for every other
  fault (C2). Each refusal prints `MOVE-REFUSED` to the log, and `scripts/watch_training.py` counts them.
- **075e3bb: a dead worker can no longer cost the record.**
  - Stale pipe replies are drained, and only a reply that is a fault summary is accepted.
  - Summaries are validated before they are combined.
  - The record is assembled under guards that note what was missing instead of crashing.
  - A resumed session writes its own record, `docs/gates/m2d_train_session<k>.json`.
- **Resume.** Session 1 resumed from the 190,000-step checkpoint and its replay buffer at 02:20 EDT, 2026-10-03, with
  `--hours 6.5`, the run's remaining budget. The run's total training time stays 12 h. The 2,772 steps after the last
  checkpoint were lost.

## The monitoring failure

The crash went unnoticed for 2 h 21 min. It was found at 01:13 EDT, only because the user asked whether anything was
still running. Training then stood idle until the resume at 02:20, 3 h 29 min after the crash.

The cause was mine. A background alarm watched for the job's exit file and for failure lines in the log. At 20:23 EDT
it reached its 2 h limit, the longest a background check may run. I chose not to re-arm it, relying on a one-off check
instead. Nothing was armed when the run crashed.

**Rule from now on.** For as long as a run lasts, a failure alarm is always armed. Every time one expires, it is
re-armed at once. A Monitor on the log covers the same failure lines plus evaluations and refused moves, and it is
re-armed when it expires too. The user's own view, `scripts/watch_training.py`, is independent of both.
