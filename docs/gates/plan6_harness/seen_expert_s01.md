# Evaluation: expert on seen

- Policy: `{"checkpoint": "runs/expert/s01_r2/best/best_model.zip", "deterministic": false, "kind": "sac_expert", "name": "expert", "obs_config": {"depth_dtype": "float32", "depth_frames": 1}, "privileged_observation": true, "sha256": "8c2c7500041907a98d5d696b9007e30578731ff5136f63c384270c0f4fcbf8dc", "trained_footprint_shape": null}`
- Condition file: /home/nvidiasims/research_uav/autofly_ue5/assets/eval_episodes/seen.json (sha256 adaa9f7c582c), 20 episodes per scene
- a0: sector8; step limit 300; mesh scenes' contact rule: the outline
- State: state[0] and state[1] (the target's true distance and bearing) zeroed
- Target pool: /home/nvidiasims/research_uav/autofly_ue5/assets/targets.json (sha256 0a8bfa72f11d)
- Status: ok; 2026-10-08 22:07:17 to 2026-10-08 22:25:06; engine faults ok: True; attempts replayed after backend faults (never scored): 0

| Scene | Status | Episodes | SR [95% CI] | CR [95% CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s01 | ok | 20 | 1.000 [0.839, 1.000] | 0.000 [0.000, 0.161] | 0.000 | 0.000 | 0.972-0.997 (n=20) |
| all | ok | 20 | 1.000 [0.839, 1.000] | 0.000 [0.000, 0.161] | 0.000 | 0.000 | 0.972-0.997 (n=20) |

PER is L_opt / max(L, L_opt) over successful episodes, L the flown horizontal path. L_opt lies between the straight line to the success disc (the lower PER) and the shortest path on the 0.5 m grid inflated by 1.4 m (the upper).
