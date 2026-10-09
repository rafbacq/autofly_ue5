# Evaluation: scripted on seen

- Policy: `{"aligned_rad": 0.3, "altitude_m": 2.0, "climb_gain": 1.0, "cruise_m_s": 2.0, "kind": "privileged_straight_line", "name": "scripted", "requires_privileged_state": true, "turn_gain": 2.0}`
- Condition file: /home/nvidiasims/research_uav/autofly_ue5/assets/eval_episodes/seen.json (sha256 adaa9f7c582c), 20 episodes per scene
- a0: sector8; step limit 300; mesh scenes' contact rule: the outline
- State: state[0] and state[1] (the target's true distance and bearing) given
- Target pool: /home/nvidiasims/research_uav/autofly_ue5/assets/targets.json (sha256 0a8bfa72f11d)
- Status: ok; 2026-10-09 00:32:12 to 2026-10-09 01:33:37; engine faults ok: True; attempts replayed after backend faults (never scored): 0

| Scene | Status | Episodes | SR [95% CI] | CR [95% CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s01 | ok | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s01d | ok | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s02 | ok | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s03 | ok | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s04 | ok | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s05 | ok | 20 | 0.050 [0.009, 0.236] | 0.950 [0.764, 0.991] | 0.000 | 0.000 | 0.998-1.000 (n=1) |
| s06 | ok | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s07 | ok | 20 | 0.400 [0.219, 0.613] | 0.600 [0.387, 0.781] | 0.000 | 0.000 | 0.996-1.000 (n=8) |
| s09 | ok | 20 | 0.100 [0.028, 0.301] | 0.900 [0.699, 0.972] | 0.000 | 0.000 | 0.998-1.000 (n=2) |
| all | ok | 180 | 0.061 [0.034, 0.106] | 0.939 [0.894, 0.966] | 0.000 | 0.000 | 0.996-1.000 (n=11) |

PER is L_opt / max(L, L_opt) over successful episodes, L the flown horizontal path. L_opt lies between the straight line to the success disc (the lower PER) and the shortest path on the 0.5 m grid inflated by 1.4 m (the upper).
