# Evaluation: scripted on test_seen

- Policy: `{"aligned_rad": 0.3, "altitude_m": 2.0, "climb_gain": 1.0, "cruise_m_s": 2.0, "kind": "privileged_straight_line", "name": "scripted", "requires_privileged_state": true, "turn_gain": 2.0}`
- Condition file: /home/nvidiasims/research_uav/autofly_ue5/assets/eval_episodes/test_seen.json (sha256 9c188a81112d), 20 episodes per scene
- a0: sector8; step limit 300; mesh scenes' contact rule: the outline
- State: state[0] and state[1] (the target's true distance and bearing) given
- Target pool: /home/nvidiasims/research_uav/autofly_ue5/assets/targets.json (sha256 0a8bfa72f11d)
- Status: ok; 2026-10-09 02:01:51 to 2026-10-09 02:13:53; engine faults ok: True; attempts replayed after backend faults (never scored): 0

| Scene | Status | Episodes | SR [95% CI] | CR [95% CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s05r | ok | 20 | 0.100 [0.028, 0.301] | 0.900 [0.699, 0.972] | 0.000 | 0.000 | 0.999-1.000 (n=2) |
| s06r | ok | 20 | 0.050 [0.009, 0.236] | 0.950 [0.764, 0.991] | 0.000 | 0.000 | 0.999-1.000 (n=1) |
| all | ok | 40 | 0.075 [0.026, 0.199] | 0.925 [0.801, 0.974] | 0.000 | 0.000 | 0.999-1.000 (n=3) |

PER is L_opt / max(L, L_opt) over successful episodes, L the flown horizontal path. L_opt lies between the straight line to the success disc (the lower PER) and the shortest path on the 0.5 m grid inflated by 1.4 m (the upper).
