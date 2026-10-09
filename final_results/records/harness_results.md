# Closed-loop evaluation

From 10 harness records by `scripts/harness_figures.py` (2026-10-09 03:30:36); every number is in harness_results.json. Success, collision, out-of-bounds and timeout rates are shares of the scene's episodes with Wilson 95 % intervals; PER = L_opt / max(L, L_opt) over successful episodes, between the straight-line and the grid-optimum bounds.

## seen: the nine training scenes, seen targets

Condition file `assets/eval_episodes/seen.json` (sha256 adaa9f7c582c), 20 episodes per scene; a0 sector8, step limit 300, mesh scenes' contact rule: the outline.

| Scene | Policy | Episodes | Success [95 % CI] | Collision [95 % CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s01 white pillars | expert | 20 | 1.000 [0.839, 1.000] | 0.000 [0.000, 0.161] | 0.000 | 0.000 | 0.972-0.997 (n=20) |
|  | scripted (privileged state) | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
|  | hover | 3 of 20 (ok) | 0.000 [0.000, 0.561] | 0.000 [0.000, 0.561] | 0.000 | 1.000 |  |
| s01d moving pillars | expert | 20 | 0.950 [0.764, 0.991] | 0.050 [0.009, 0.236] | 0.000 | 0.000 | 0.949-0.989 (n=19) |
|  | scripted (privileged state) | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s02 sparse trees | expert | 20 | 0.650 [0.433, 0.819] | 0.100 [0.028, 0.301] | 0.150 | 0.100 | 0.966-1.000 (n=13) |
|  | scripted (privileged state) | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s03 scattered rocks | expert | 20 | 0.450 [0.258, 0.658] | 0.300 [0.145, 0.519] | 0.200 | 0.050 | 0.915-0.996 (n=9) |
|  | scripted (privileged state) | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s04 stone field | expert | 20 | 0.550 [0.342, 0.742] | 0.350 [0.181, 0.567] | 0.100 | 0.000 | 0.936-0.989 (n=11) |
|  | scripted (privileged state) | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s05 standing stones | expert | 20 | 0.500 [0.299, 0.701] | 0.400 [0.219, 0.613] | 0.100 | 0.000 | 0.925-0.980 (n=10) |
|  | scripted (privileged state) | 20 | 0.050 [0.009, 0.236] | 0.950 [0.764, 0.991] | 0.000 | 0.000 | 0.998-1.000 (n=1) |
|  | hover | 3 of 20 (ok) | 0.000 [0.000, 0.561] | 0.000 [0.000, 0.561] | 0.000 | 1.000 |  |
| s06 tree clusters | expert | 20 | 0.350 [0.181, 0.567] | 0.500 [0.299, 0.701] | 0.150 | 0.000 | 0.957-1.000 (n=7) |
|  | scripted (privileged state) | 20 | 0.000 [0.000, 0.161] | 1.000 [0.839, 1.000] | 0.000 | 0.000 |  |
| s07 stacked boxes | expert | 20 | 0.950 [0.764, 0.991] | 0.050 [0.009, 0.236] | 0.000 | 0.000 | 0.982-0.999 (n=19) |
|  | scripted (privileged state) | 20 | 0.400 [0.219, 0.613] | 0.600 [0.387, 0.781] | 0.000 | 0.000 | 0.996-1.000 (n=8) |
| s09 coloured poles | expert | 20 | 0.950 [0.764, 0.991] | 0.050 [0.009, 0.236] | 0.000 | 0.000 | 0.966-0.998 (n=19) |
|  | scripted (privileged state) | 20 | 0.100 [0.028, 0.301] | 0.900 [0.699, 0.972] | 0.000 | 0.000 | 0.998-1.000 (n=2) |
| **all** | expert | 180 | 0.706 [0.635, 0.767] | 0.200 [0.148, 0.264] | 0.078 | 0.017 | 0.957-0.995 (n=127) |
|  | scripted (privileged state) | 180 | 0.061 [0.035, 0.106] | 0.939 [0.894, 0.966] | 0.000 | 0.000 | 0.996-1.000 (n=11) |
|  | hover | 6 | 0.000 [0.000, 0.390] | 0.000 [0.000, 0.390] | 0.000 | 1.000 |  |

## test_seen: the training scenes' elements in new layouts

Condition file `assets/eval_episodes/test_seen.json` (sha256 9c188a81112d), 20 episodes per scene; a0 sector8, step limit 300, mesh scenes' contact rule: the outline.

| Scene | Policy | Episodes | Success [95 % CI] | Collision [95 % CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s05r standing stones | expert | 20 | 0.150 [0.052, 0.360] | 0.400 [0.219, 0.613] | 0.450 | 0.000 | 0.904-0.955 (n=3) |
|  | scripted (privileged state) | 20 | 0.100 [0.028, 0.301] | 0.900 [0.699, 0.972] | 0.000 | 0.000 | 0.999-1.000 (n=2) |
| s06r tree clusters | expert | 20 | 0.300 [0.145, 0.519] | 0.550 [0.342, 0.742] | 0.150 | 0.000 | 0.939-0.997 (n=6) |
|  | scripted (privileged state) | 20 | 0.050 [0.009, 0.236] | 0.950 [0.764, 0.991] | 0.000 | 0.000 | 0.999-1.000 (n=1) |
| **all** | expert | 40 | 0.225 [0.123, 0.375] | 0.475 [0.329, 0.625] | 0.300 | 0.000 | 0.928-0.983 (n=9) |
|  | scripted (privileged state) | 40 | 0.075 [0.026, 0.199] | 0.925 [0.801, 0.974] | 0.000 | 0.000 | 0.999-1.000 (n=3) |

## unseen targets: the training scenes, targets from the pool's unseen entries

Condition file `assets/eval_episodes/unseen_targets.json` (sha256 7b6b0fcb8e99), 20 episodes per scene; a0 sector8, step limit 300, mesh scenes' contact rule: the outline.

| Scene | Policy | Episodes | Success [95 % CI] | Collision [95 % CI] | Out of bounds | Timeout | PER over successes |
|---|---|---|---|---|---|---|---|
| s01 white pillars | expert | 20 | 0.950 [0.764, 0.991] | 0.000 [0.000, 0.161] | 0.050 | 0.000 | 0.978-1.000 (n=19) |
| s01d moving pillars | expert | 20 | 0.850 [0.640, 0.948] | 0.100 [0.028, 0.301] | 0.050 | 0.000 | 0.920-0.971 (n=17) |
| s02 sparse trees | expert | 20 | 0.850 [0.640, 0.948] | 0.100 [0.028, 0.301] | 0.050 | 0.000 | 0.954-0.997 (n=17) |
| s03 scattered rocks | expert | 20 | 0.500 [0.299, 0.701] | 0.300 [0.145, 0.519] | 0.150 | 0.050 | 0.933-0.999 (n=10) |
| s04 stone field | expert | 20 | 0.500 [0.299, 0.701] | 0.200 [0.081, 0.416] | 0.300 | 0.000 | 0.953-0.997 (n=10) |
| s05 standing stones | expert | 20 | 0.400 [0.219, 0.613] | 0.300 [0.145, 0.519] | 0.300 | 0.000 | 0.925-0.990 (n=8) |
| s06 tree clusters | expert | 20 | 0.300 [0.145, 0.519] | 0.500 [0.299, 0.701] | 0.200 | 0.000 | 0.955-1.000 (n=6) |
| s07 stacked boxes | expert | 20 | 1.000 [0.839, 1.000] | 0.000 [0.000, 0.161] | 0.000 | 0.000 | 0.984-1.000 (n=20) |
| s09 coloured poles | expert | 20 | 1.000 [0.839, 1.000] | 0.000 [0.000, 0.161] | 0.000 | 0.000 | 0.962-0.996 (n=20) |
| **all** | expert | 180 | 0.706 [0.635, 0.767] | 0.167 [0.119, 0.228] | 0.122 | 0.006 | 0.956-0.994 (n=127) |

## Plan 6's done check: the expert against its pilot's kept-rate

The seen condition's expert (expert) against each scene's pilot in `docs/gates/plan6_pilots.json` (sha256 d41fa851eacd): plan 6 revision 1: the expert flown through the harness lands within the CI of its kept-rate and the scripted baseline below it. Here: the expert's Wilson 95 % success interval overlaps the pilot's kept-rate interval (thin slice), and the scripted baseline's success rate is below the expert's ('separated' when its whole interval is).

| Scene | Expert success [95 % CI] | Pilot kept-rate [95 % CI] | Intervals overlap | Same checkpoint | Scripted success [95 % CI] | Scripted below |
|---|---|---|---|---|---|---|
| s01 white pillars | 1.000 [0.839, 1.000] | 0.962 [0.870, 0.989] | yes | yes | 0.000 [0.000, 0.161] | yes (separated) |
| s01d moving pillars | 0.950 [0.764, 0.991] | 0.909 [0.804, 0.961] | yes | yes | 0.000 [0.000, 0.161] | yes (separated) |
| s02 sparse trees | 0.650 [0.433, 0.819] | 0.735 [0.620, 0.826] | yes | yes | 0.000 [0.000, 0.161] | yes (separated) |
| s03 scattered rocks | 0.450 [0.258, 0.658] | 0.602 [0.495, 0.701] | yes | yes | 0.000 [0.000, 0.161] | yes (separated) |
| s04 stone field | 0.550 [0.342, 0.742] | 0.562 [0.458, 0.660] | yes | yes | 0.000 [0.000, 0.161] | yes (separated) |
| s05 standing stones | 0.500 [0.299, 0.701] | 0.495 [0.400, 0.591] | yes | yes | 0.050 [0.009, 0.236] | yes (separated) |
| s06 tree clusters | 0.350 [0.181, 0.567] | 0.394 [0.313, 0.481] | yes | yes | 0.000 [0.000, 0.161] | yes (separated) |
| s07 stacked boxes | 0.950 [0.764, 0.991] | 0.980 [0.897, 0.997] | yes | yes | 0.400 [0.219, 0.613] | yes (separated) |
| s09 coloured poles | 0.950 [0.764, 0.991] | 1.000 [0.929, 1.000] | yes | yes | 0.100 [0.028, 0.301] | yes (separated) |

- Every scene's expert interval overlaps its pilot's: yes
- The scripted baseline below the expert on every scene: yes
- The expert flew each scene's pilot checkpoint: yes

## Notes

- seen: each policy's whole is over the scenes it flew (expert 9, scripted (privileged state) 9, hover 2)

## Records

- `runs/eval/final/seen_expert_s01d_s07_s09.json` (sha256 fac18d6f1ddd): expert on seen, status ok
- `runs/eval/final/seen_expert_s01.json` (sha256 dff00173e8f3): expert on seen, status ok
- `runs/eval/final/seen_expert_s02_to_s06.json` (sha256 68e8c32d03a5): expert on seen, status ok
- `runs/eval/final/seen_hover_s01_s05.json` (sha256 6505e82af0e1): hover on seen, status ok
- `runs/eval/final/seen_scripted_privileged.json` (sha256 f3c77166f292): scripted (privileged state) on seen, status ok
- `runs/eval/final/test_seen_expert_s02.json` (sha256 414d10cbcc94): expert on test_seen, status ok
- `runs/eval/final/test_seen_scripted.json` (sha256 1a5a9c802909): scripted (privileged state) on test_seen, status ok
- `runs/eval/final/unseen_expert_s01d_s07_s09.json` (sha256 20c594ae856a): expert on unseen_targets, status ok
- `runs/eval/final/unseen_expert_s01.json` (sha256 adcf455c9099): expert on unseen_targets, status ok
- `runs/eval/final/unseen_expert_s02_to_s06.json` (sha256 40c9897193d9): expert on unseen_targets, status ok
