# 2026-10-03 — M2d closeout: the s01d expert fails its gate; the next lever is the user's

Evidence:
- `docs/gates/m2d_gate.json` and `m2d_gate_audit.json` (0 flagged);
- `docs/gates/m2d_baseline_s01_expert.json`;
- `docs/gates/m2d_train_session1.json` (session 0's record was lost: `2026-10-03-m2d-session0-crash.md`);
- `runs/viz/s01d_r1_best_model/` (8 rendered gate episodes).

## Result

The 12 h s01d run (`runs/expert/s01d_r1`, 412,828 steps, 3-frame float16 depth, 1 ms config) was gated on 200 held-out
episodes per checkpoint and condition. The bar is 0.95, deterministic.

| Checkpoint | Deterministic | Stochastic | Main failure |
|---|---|---|---|
| `best_model` (225k) | **0.775** | 0.725 | mover collisions: 35 deterministic, 19 with the mover out of view |
| `final` (412k) | **0.535** | 0.440 | dives below the 1 m floor: 58 / 70 `altitude_low` exits, at a median of 79 steps |
| *s01 expert, zero-shot* | *0.36* | — | *122 mover collisions* |

**The gate fails, and stays failed.** Training on moving pillars took success from 0.36 to 0.775, but not to 0.95.
The run itself was sound: status ok, a clean engine audit, every backend fault recovered, and no flagged episode.

## What the failures say

- **Out of view.** Of `best_model`'s 35 mover collisions, 19 had the mover outside the camera's view. The runbook
  reads that as the observation's limit, not the learning's: a forward depth camera cannot see a pillar moving in from
  the side.
- **In view.** The other 16 had the mover in view, which is still learning to do.
- **The late policy learned to dive.** In training, out-of-bounds episodes rose from 0.09 to 0.43 between 225k and
  250k as collisions fell from 0.36 to 0.15. `final` leaves the altitude band low, early in the episode. Model
  selection (`best_model`) kept the earlier policy, as it should.
- **Training-time evaluations did not trend up after 225k:** 0.85, 0.10, 0.45, 0.60, 0.45, 0.40, 0.70, 0.75. SAC's
  own signals stayed healthy (entropy coefficient 0.0064–0.0076, critic loss 0.5–1.8), so this is not a broken run.
- **One s01d episode does not replay.** Of 8 rendered gate episodes, the 3 successes replayed as successes. 4 of the
  5 failures succeeded on replay, and successes drifted by up to 10 steps (s01: 11 of 12 outcomes and 0–3 steps).
  Resets are history-free: pillars are parked and then sent home, and the depth stack is refilled each episode. So
  this is the physics' known non-bit-exactness, amplified by movers that yield to the drone. The 200-episode rate is
  the measure; single gate episodes are not reproducible on s01d.
- **Refused pillar moves are harmless now.** Session 1 had 2 refusals, and both pillars read back 1 µm from their
  target.

## Levers (the user's choice)

1. **Privileged mover state in the expert's vector observation**, e.g. the nearest movers' relative positions and
   velocities. It addresses the out-of-view half of the collisions directly. The expert is only the data source, and
   the dataset records RGB and state[9] whatever the expert sees, so this costs the dataset nothing. Needs a new
   observation space, so a new run.
2. **Warm start from s01's weights.** The s01 expert already flies the field (0.36 zero-shot, 0.98 on static s01). Its
   observation is one float32 frame against s01d's three float16 frames, so the first layer needs adapting.
3. **Fewer or slower movers.** This lowers the difficulty directly, but changes what s01d is.
4. **More hours (`--resume`).** The evaluations since 225k and the late dive suggest more of the same will not reach
   0.95 alone.
5. **Discourage the dive**, e.g. a penalty for approaching the altitude floor. That is a reward change, so
   `REWARD_VERSION` must be bumped. Model selection already avoids it.

My recommendation is 1, with 2 if the run should be shorter. The data say the expert cannot see half of what hits it.

## What does not wait for M2d

M3 passed on static s01 (`docs/gates/m3_gate.json`). An s01d collection pilot waits for a passed M2d gate.
