# 2026-10-04 — Run 3 lost to a host freeze; resumed from its 70k checkpoint

## What happened

Run 3 (`runs/expert/s01d_r3`, plan `2026-10-03-s01d-r2-stopped-r3-from-scratch.md`) started at 15:59 on 2026-10-03.
At 17:38:49 the whole host froze, 1 h 40 min into the 12 h budget:

- Session 0's last SB3 table read 76,068 steps. Its newest checkpoint was 70,000, written at 17:30:29, with its replay
  buffer at 17:30:36.
- The kernel's last lines were the Wi-Fi driver failing to wake its card, ten times in 18 seconds, and then nothing:
  `mt7921e 0000:0b:00.0: driver own failed` / `Timeout for driver own`. The training log, the 5-minute dashboard and
  the journal all stop there.
- Nobody could reach the host (its only network link is that card; `eno1` has no link), and it stayed frozen until
  it was power-cycled at 18:39 on 2026-10-04: 25 h.

It was not the first time. Every boot since 2026-09-25 ended the same way, with no shutdown sequence:

| Boot ended | Kernel's last lines | Down |
|---|---|---|
| 2026-09-25 16:53 | `mt7921e … Message 00020001 timeout`, after ~4 min of `driver own failed` | 1 min |
| 2026-09-28 10:26 | ~4 min of `driver own failed` / `Timeout for driver own` | 1 min |
| 2026-10-01 08:09 | `chip reset failed`, then `BUG: unable to handle page fault` (a kernel oops) | 30 h |
| 2026-10-03 17:38 | 18 s of `driver own failed` / `Timeout for driver own` | 25 h |

The card runs with PCIe ASPM L1 on (`/sys/module/mt7921e/parameters/disable_aspm` = N) and NetworkManager's Wi-Fi
power saving on (`wifi.powersave = 3`). "driver own" is the driver reclaiming the chip from a low-power state. The
usual remedies are a wired link, `disable_aspm=1`, or power saving off. All need root on a shared host, and **the user
decided on 2026-10-04 to leave the host unchanged for now.** A freeze costs at most the steps since the last checkpoint
(10k steps, about 13 minutes), but the downtime lasts until someone power-cycles the machine.

## What it cost, and what was checked

- **Steps 70,000–76,068** (8 minutes). The model and buffer resume from 70,000.
- **Session 0's training record.** `docs/gates/m2d_r3_train.json` is written only when a session ends, so it does not
  exist. Session 0's numbers live in `runs/jobs/s01d_r3_train.log`, its monitor CSVs and TensorBoard.
- **Display `:1`.** The reboot brought it down; the user logged in again at 19:17 (the MEMORY.md recipe).
- **A 0-byte `runs/sim/inst1/pid.json`.** The freeze landed mid-relaunch of instance 1. `os.replace()` had put the new
  record in place, but ext4 had not yet written its data. Any launch or stop in that slot would have raised
  `JSONDecodeError`, ending the resume in its first minute, after it had claimed its session. It was set aside by hand
  as `pid.json.empty-after-2026-10-03-host-freeze`. b7d02b3 removes the trap:
  - the slot record and `sessions.json` are now written durably (data synced before the rename, the directory after);
  - an unreadable record is set aside by the next launch, and `stop()` / `stop_instance()` return `unreadable_record`
    instead of raising.
- **The resume point, read in full** (`runs/m2d_diag/verify_resume_point.py rl_model_70000_steps.zip`; output in
  `runs/jobs/r3_resume_verify.log`):
  - `rl_model_70000_steps.zip`: CRCs match, sha256 `18c484e0…`, `num_timesteps` 70,000.
  - Its buffer: a `FaultFilteringDictReplayBuffer` of 62,500 rows × 4 envs at `pos` 17,499.
  - Every stored transition is finite, and none is all-zero. A real depth frame never is, so zero-filled blocks from
    the crash would have shown. Every unwritten row is still zero.
  - Rewards lie in [−10.3, 10.4], `dones` are 0 or 1 (576 episode ends), actions lie within the box, and 0 fault rows
    were dropped.
- **The code.** Only documentation changed between session 0's start and the resume (`git diff 83ab9c2 32563d7`), and
  the offline suite passed.

## The resume

Session 1 started at 19:19:34 on 2026-10-04:

```bash
$J start s01d_r3_train_s1 -- $PY -m autofly_ue5.expert.train --scene s01d --instances 4 --hours 10.5 \
    --scene-config $CFG --run-root runs/expert/s01d_r3 --out docs/gates/m2d_r3_train_session1.json \
    --eval-freq 1000000000 --buffer-size 250000 --resume
```

- **`--hours 10.5`.** The 70k checkpoint held 1.52 h of session 0's training, and 12 − 1.52 ≈ 10.5, so the run keeps
  its 12 h budget.
- **`--out` is explicit.** Without it, session 1's record would default to `m2d_train_session1.json`, which is run 1's
  committed record, and the trainer would refuse to start.
- **It resumed exactly.** It loaded 70,000 steps and continued `n_updates` and the entropy coefficient (0.00984) from
  the checkpoint.
- **New job names, same jobs.** The dashboard, eval watch (slot 4, every 25k) and TensorBoard restarted as
  `s01d_r3_{watch,evalwatch,tensorboard}_s1`.

At equal steps run 3 is ahead of run 1, from the monitor CSVs (rolling 200 training episodes, stochastic) and the same
20 fixed evaluation seeds:

| Steps | r1 training success | r3 training success | r1 evaluation | r3 evaluation |
|---|---|---|---|---|
| 50k | 0.04 | 0.08 | 0.00 | 0.30 |
| 75k | 0.16 | 0.26 | 0.05 | — |

Run 3 also leaves the bounds far less often: 0.21 of training episodes at 75k, against r1's 0.43. Its main failure is
now mover collisions. Run 1 plateaued near 0.55 training success, with evaluations swinging between 0.10 and 0.85.
Whether run 3 clears that plateau is the question for the rest of the run.
