# Evaluation: expert on test_seen

- Policy: `{"checkpoint": "runs/expert/s02_r1/averaged/avg_350k_end.zip", "deterministic": false, "kind": "sac_expert", "name": "expert", "obs_config": {"depth_dtype": "float32", "depth_frames": 1}, "privileged_observation": true, "sha256": "0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73", "trained_footprint_shape": "circle"}`
- Condition file: /home/nvidiasims/research_uav/autofly_ue5/assets/eval_episodes/test_seen.json (sha256 9c188a81112d), 20 episodes per scene
- a0: sector8; step limit 300; mesh scenes' contact rule: the outline
- State: state[0] and state[1] (the target's true distance and bearing) zeroed
- Target pool: /home/nvidiasims/research_uav/autofly_ue5/assets/targets.json (sha256 0a8bfa72f11d)
- Status: ok; 2026-10-09 01:34:10 to 2026-10-09 02:01:20; engine faults ok: True; attempts replayed after backend faults (never scored): 0

| Scene | Status | Episodes | SR [95% CI] | CR [95% CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s05r | ok | 20 | 0.150 [0.052, 0.360] | 0.400 [0.219, 0.613] | 0.450 | 0.000 | 0.904-0.955 (n=3) |
| s06r | ok | 20 | 0.300 [0.145, 0.519] | 0.550 [0.342, 0.742] | 0.150 | 0.000 | 0.939-0.997 (n=6) |
| all | ok | 40 | 0.225 [0.123, 0.375] | 0.475 [0.329, 0.625] | 0.300 | 0.000 | 0.928-0.983 (n=9) |

PER is L_opt / max(L, L_opt) over successful episodes, L the flown horizontal path. L_opt lies between the straight line to the success disc (the lower PER) and the shortest path on the 0.5 m grid inflated by 1.4 m (the upper).
