# 2026-10-04 — Run 3 paused at ~200k; run 4 adds a mover clearance penalty

## Why run 3 was paused

Run 3 (`runs/expert/s01d_r3`) added three levers to run 1: the mover input, r_bounds = 10, and a 250k buffer
(`2026-10-03-s01d-r2-stopped-r3-from-scratch.md`). Through 200k steps its training curve tracked run 1's almost
exactly. Run 1 plateaued from there and gated at 0.775 (`m2d_gate.json`).

Rolling 200 training episodes, stochastic, from the monitor CSVs:

| Steps | Success (r1 / r3) | Mover collisions (r1 / r3) | Out of bounds (r1 / r3) |
|---|---|---|---|
| 100k | 0.35 / 0.40 | 0.33 / 0.34 | 0.27 / 0.14 |
| 125k | 0.52 / 0.47 | 0.30 / 0.32 | 0.15 / 0.12 |
| 150k | 0.56 / 0.54 | 0.28 / 0.28 | 0.13 / 0.14 |
| 200k | 0.55 / 0.60 | 0.17 / 0.26 | 0.24 / 0.10 |

Deterministic, on the 20 training-time evaluation seeds:

| Steps | r1 | r3 (eval watch) |
|---|---|---|
| 50k | 0.00 | 0.30 |
| 100k | 0.70 | 0.25 |
| 150k | 0.65 | 0.65 |
| 200k | 0.50 | 0.65 |

**The mover input did not move the mover-collision rate.** Run 3's failures are the ones run 1's contact probe found:

- **Every mover collision is a near miss.** All 30 across the four evaluations had the drone 0.80–1.00 m from the
  mover's surface (median 0.96), inside the 1.0 m contact rule.
- **Mostly beside a mover that had stopped to yield.** In 26 of 30 the mover was standing still, and 27 of 30 were 30°
  or more off the heading.
- **17 of 30 had the mover in view.** The drone flies past movers it can see.
- **At 200k they were the only failure left:** 7 of 7, with no static collision and no out-of-bounds exit.
- **The out-of-bounds escapes are gone** from the evaluations (0 of 60 at 100k–200k against 5 of 20 at 50k), so
  r_bounds did its job.

So the policy now knows where the movers are, and still passes them as if they were static pillars (~0.48 m of physical
clearance). Nothing in the reward asks for more until the contact step: a −10 cliff exactly at the boundary, which a
smooth critic blurs.

**Paused, not abandoned.** Run 3 was stopped at 22:18 on 2026-10-04, at 208,176 steps, just before its 210k save. Its
200k checkpoint and replay buffer had reached the disk 12 minutes earlier. Its trainer predates the clean stop request
(below), so session 1 wrote no record. Its checkpoints, buffer, monitor CSVs and eval-watch scores remain, and its
identity is unchanged. It can resume from 200k for the 7.7 h left of its 12 h (session 0 banked 1.52 h, session 1
2.78 h):

```bash
$J start s01d_r3_train_s2 -- ... the session 1 command ... --resume --hours 7.7 \
    --out docs/gates/m2d_r3_train_session2.json
```

## Run 4: one change

`runs/expert/s01d_r4` is run 3's command plus `--mover-clearance-penalty 0.5 1.0` (38ac0bc). Every step, each mover
whose surface is closer than 2.0 m costs

    0.5 * clip(1 - (gap - 1.0) / 1.0, 0, 1)

That is 0 at 2.0 m from the surface and 0.5 at the 1.0 m contact boundary, so the cliff becomes a ramp and the expert
learns a buffer. The scale was chosen against the reward:

- **At the boundary** the penalty (0.5 per step) slightly exceeds what a full-speed step earns in progress (0.4).
- **A 1 m wider berth past a mover** costs about 0.2 of progress, against roughly 1–2 of penalty for a close pass.
- **A mover inside the margin is always standing still.** It yields 2.5 m from its surface, so it cannot move into the
  zone on its own. The penalty therefore prices only the drone's own choice to fly close.

It is training only. The evaluation env, the eval watch, selection and the gate score the task's own reward and
success rule. `reward.py` and `REWARD_VERSION` are unchanged. A run that pays the penalty records it in its identity,
so it cannot resume without it. Everything else is identical to run 3, the seeds included, so the same episodes fly in
the same order and the two runs compare directly.

Two operational additions:
- **`touch <run root>/STOP`** ends a session cleanly, with final.zip and its record (5466891).
- **`outcomes/mover_clearance_penalty`** in the log and TensorBoard shows what the penalty costs per episode, so it
  can be seen falling as the expert learns to keep its distance (06d45be).

## What would show it worked

- **In training:** fewer mover collisions than runs 1 and 3 at equal steps, without more timeouts, and the per-episode
  penalty falling.
- **In the eval watch:** 20-episode scores holding above 0.85.
- **At the gate:** ≥ 0.95.
- **If it falls short:** the per-contact details (gap, bearing, moving or yielding) will again say why.
