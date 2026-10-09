# Evaluation: hover on seen

- Policy: `{"action": [0.0, 0.0, 0.0], "kind": "floor", "name": "hover"}`
- Condition file: /home/nvidiasims/research_uav/autofly_ue5/assets/eval_episodes/seen.json (sha256 adaa9f7c582c), 20 episodes per scene, at most 3 flown
- a0: sector8; step limit 300; mesh scenes' contact rule: the outline
- State: state[0] and state[1] (the target's true distance and bearing) zeroed
- Target pool: /home/nvidiasims/research_uav/autofly_ue5/assets/targets.json (sha256 0a8bfa72f11d)
- Status: ok; 2026-10-09 02:45:59 to 2026-10-09 02:53:29; engine faults ok: True; attempts replayed after backend faults (never scored): 0

| Scene | Status | Episodes | SR [95% CI] | CR [95% CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s01 | ok | 3 | 0.000 [0.000, 0.561] | 0.000 [0.000, 0.561] | 0.000 | 1.000 |  |
| s05 | ok | 3 | 0.000 [0.000, 0.561] | 0.000 [0.000, 0.561] | 0.000 | 1.000 |  |
| all | ok | 6 | 0.000 [0.000, 0.390] | 0.000 [0.000, 0.390] | 0.000 | 1.000 |  |

PER is L_opt / max(L, L_opt) over successful episodes, L the flown horizontal path. L_opt lies between the straight line to the success disc (the lower PER) and the shortest path on the 0.5 m grid inflated by 1.4 m (the upper).
