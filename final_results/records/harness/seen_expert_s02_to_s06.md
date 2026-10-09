# Evaluation: expert on seen

- Policy: `{"checkpoint": "runs/expert/s02_r1/averaged/avg_350k_end.zip", "deterministic": false, "kind": "sac_expert", "name": "expert", "obs_config": {"depth_dtype": "float32", "depth_frames": 1}, "privileged_observation": true, "sha256": "0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73", "trained_footprint_shape": "circle"}`
- Condition file: /home/nvidiasims/research_uav/autofly_ue5/assets/eval_episodes/seen.json (sha256 adaa9f7c582c), 20 episodes per scene
- a0: sector8; step limit 300; mesh scenes' contact rule: the outline
- State: state[0] and state[1] (the target's true distance and bearing) zeroed
- Target pool: /home/nvidiasims/research_uav/autofly_ue5/assets/targets.json (sha256 0a8bfa72f11d)
- Status: ok; 2026-10-08 23:15:12 to 2026-10-09 00:31:41; engine faults ok: True; attempts replayed after backend faults (never scored): 0

| Scene | Status | Episodes | SR [95% CI] | CR [95% CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s02 | ok | 20 | 0.650 [0.433, 0.819] | 0.100 [0.028, 0.301] | 0.150 | 0.100 | 0.966-1.000 (n=13) |
| s03 | ok | 20 | 0.450 [0.258, 0.658] | 0.300 [0.145, 0.519] | 0.200 | 0.050 | 0.915-0.996 (n=9) |
| s04 | ok | 20 | 0.550 [0.342, 0.742] | 0.350 [0.181, 0.567] | 0.100 | 0.000 | 0.936-0.989 (n=11) |
| s05 | ok | 20 | 0.500 [0.299, 0.701] | 0.400 [0.219, 0.613] | 0.100 | 0.000 | 0.925-0.980 (n=10) |
| s06 | ok | 20 | 0.350 [0.181, 0.567] | 0.500 [0.299, 0.701] | 0.150 | 0.000 | 0.957-1.000 (n=7) |
| all | ok | 100 | 0.500 [0.404, 0.596] | 0.330 [0.246, 0.427] | 0.140 | 0.030 | 0.941-0.993 (n=50) |

PER is L_opt / max(L, L_opt) over successful episodes, L the flown horizontal path. L_opt lies between the straight line to the success disc (the lower PER) and the shortest path on the 0.5 m grid inflated by 1.4 m (the upper).
