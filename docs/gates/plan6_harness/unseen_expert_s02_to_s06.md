# Evaluation: expert on unseen_targets

- Policy: `{"checkpoint": "runs/expert/s02_r1/averaged/avg_350k_end.zip", "deterministic": false, "kind": "sac_expert", "name": "expert", "obs_config": {"depth_dtype": "float32", "depth_frames": 1}, "privileged_observation": true, "sha256": "0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73", "trained_footprint_shape": "circle"}`
- Condition file: /home/nvidiasims/research_uav/autofly_ue5/assets/eval_episodes/unseen_targets.json (sha256 7b6b0fcb8e99), 20 episodes per scene
- a0: sector8; step limit 300; mesh scenes' contact rule: the outline
- State: state[0] and state[1] (the target's true distance and bearing) zeroed
- Target pool: /home/nvidiasims/research_uav/autofly_ue5/assets/targets.json (sha256 0a8bfa72f11d)
- Status: ok; 2026-10-09 02:45:02 to 2026-10-09 03:23:48; engine faults ok: True; attempts replayed after backend faults (never scored): 0

| Scene | Status | Episodes | SR [95% CI] | CR [95% CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s02 | ok | 20 | 0.850 [0.640, 0.948] | 0.100 [0.028, 0.301] | 0.050 | 0.000 | 0.954-0.997 (n=17) |
| s03 | ok | 20 | 0.500 [0.299, 0.701] | 0.300 [0.145, 0.519] | 0.150 | 0.050 | 0.933-0.999 (n=10) |
| s04 | ok | 20 | 0.500 [0.299, 0.701] | 0.200 [0.081, 0.416] | 0.300 | 0.000 | 0.953-0.997 (n=10) |
| s05 | ok | 20 | 0.400 [0.219, 0.613] | 0.300 [0.145, 0.519] | 0.300 | 0.000 | 0.925-0.990 (n=8) |
| s06 | ok | 20 | 0.300 [0.145, 0.519] | 0.500 [0.299, 0.701] | 0.200 | 0.000 | 0.955-1.000 (n=6) |
| all | ok | 100 | 0.510 [0.413, 0.606] | 0.280 [0.201, 0.375] | 0.200 | 0.010 | 0.945-0.997 (n=51) |

PER is L_opt / max(L, L_opt) over successful episodes, L the flown horizontal path. L_opt lies between the straight line to the success disc (the lower PER) and the shortest path on the 0.5 m grid inflated by 1.4 m (the upper).
