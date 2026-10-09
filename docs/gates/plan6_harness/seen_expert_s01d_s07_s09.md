# Evaluation: expert on seen

- Policy: `{"checkpoint": "runs/expert/s01d_r6/averaged/avg_250k_550k.zip", "deterministic": false, "kind": "sac_expert", "name": "expert", "obs_config": {"depth_dtype": "float16", "depth_frames": 3, "mover_slots": 4}, "privileged_observation": true, "sha256": "55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd", "trained_footprint_shape": null}`
- Condition file: /home/nvidiasims/research_uav/autofly_ue5/assets/eval_episodes/seen.json (sha256 adaa9f7c582c), 20 episodes per scene
- a0: sector8; step limit 300; mesh scenes' contact rule: the outline
- State: state[0] and state[1] (the target's true distance and bearing) zeroed
- Target pool: /home/nvidiasims/research_uav/autofly_ue5/assets/targets.json (sha256 0a8bfa72f11d)
- Status: ok; 2026-10-08 22:25:39 to 2026-10-08 23:14:39; engine faults ok: True; attempts replayed after backend faults (never scored): 0

| Scene | Status | Episodes | SR [95% CI] | CR [95% CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s01d | ok | 20 | 0.950 [0.764, 0.991] | 0.050 [0.009, 0.236] | 0.000 | 0.000 | 0.949-0.989 (n=19) |
| s07 | ok | 20 | 0.950 [0.764, 0.991] | 0.050 [0.009, 0.236] | 0.000 | 0.000 | 0.982-0.999 (n=19) |
| s09 | ok | 20 | 0.950 [0.764, 0.991] | 0.050 [0.009, 0.236] | 0.000 | 0.000 | 0.966-0.998 (n=19) |
| all | ok | 60 | 0.950 [0.863, 0.983] | 0.050 [0.017, 0.137] | 0.000 | 0.000 | 0.966-0.996 (n=57) |

PER is L_opt / max(L, L_opt) over successful episodes, L the flown horizontal path. L_opt lies between the straight line to the success disc (the lower PER) and the shortest path on the 0.5 m grid inflated by 1.4 m (the upper).
