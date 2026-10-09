# Rebalancing (spec §10.3; AutoFly App. A.2.4)

`runs/rebalance/final/rebalance.json` (sha256 ac95fec3024b, union of stores: s01_c1, s01_c2, s01_c3, s01_c4, s01d_c1, s01d_c2, s01d_c3, s01d_c4, s02_c1, s02_c2, s02_c3, s02_c4, s03_c1, s03_c2, s03_c3, s03_c4, s04_c1, s04_c2, s04_c3, s04_c4, s05_c1, s05_c2, s05_c3, s05_c4, s06_c1, s06_c2, s06_c3, s06_c4, s07_c1, s07_c2, s07_c3, s07_c4, s09_c1, s09_c2, s09_c3, s09_c4), drawn by `scripts/rebalance_figures.py` (2026-10-09 04:48:11). A record is obstacle avoidance until the episode's first detection of its target above 0.7, target seeking from it on; the weights even the two phases out (alpha 0.0).

| | this dataset | the paper |
|---|---|---|
| phase split P0 (avoidance, seeking) | 0.691 / 0.309 | 0.73 / 0.27 |
| resampling weights | 0.723 / 1.621 | 0.68 / 1.85 |
| share of records with a confident detection | n/a (sparse) | 0.27 |
| episodes never detected | 147 of 1800 | |
| detection persistence after the first detection | n/a (sparse) | |

## Per scene

| Scene | Episodes | Records | Obstacle avoidance | Target seeking | Never detected | Median distance at first detection (m) |
|---|---|---|---|---|---|---|
| all | 1800 | 339007 | 0.691 | 0.309 | 147 | 20.9 |
| s01 white pillars | 200 | 38868 | 0.686 | 0.314 | 9 | 18.9 |
| s01d moving pillars | 200 | 39572 | 0.704 | 0.296 | 11 | 21.5 |
| s02 sparse trees | 200 | 36612 | 0.708 | 0.292 | 22 | 20.0 |
| s03 scattered rocks | 200 | 39137 | 0.723 | 0.277 | 16 | 19.3 |
| s04 stone field | 200 | 37813 | 0.627 | 0.373 | 15 | 23.6 |
| s05 standing stones | 200 | 38136 | 0.666 | 0.334 | 16 | 21.4 |
| s06 tree clusters | 200 | 37066 | 0.709 | 0.291 | 26 | 21.7 |
| s07 stacked boxes | 200 | 35249 | 0.643 | 0.357 | 16 | 23.4 |
| s09 coloured poles | 200 | 36554 | 0.756 | 0.244 | 16 | 19.3 |

## Per target

| Target | Episodes | Never detected | Median distance at first detection (m) | Median share of the episode before it | On target / off |
|---|---|---|---|---|---|
| yellow suv | 28 | 0 | 42.1 | 0.43 | 21 / 7 |
| white pickup truck | 28 | 0 | 41.8 | 0.45 | 21 / 7 |
| green garbage truck | 31 | 0 | 41.7 | 0.46 | 21 / 10 |
| red sedan | 42 | 0 | 41.2 | 0.45 | 28 / 14 |
| yellow taxi | 23 | 0 | 40.5 | 0.45 | 14 / 9 |
| orange cylinder | 37 | 0 | 40.3 | 0.50 | 22 / 15 |
| blue cylinder | 31 | 0 | 39.6 | 0.49 | 24 / 7 |
| green hatchback | 32 | 0 | 38.3 | 0.45 | 20 / 12 |
| police car | 32 | 0 | 33.8 | 0.54 | 20 / 12 |
| blue van | 32 | 0 | 33.6 | 0.55 | 27 / 5 |
| wooden barrel | 28 | 0 | 30.6 | 0.58 | 17 / 11 |
| blue cone | 34 | 0 | 30.2 | 0.67 | 20 / 14 |
| green suitcase | 34 | 14 | 29.9 | 0.72 | 8 / 12 |
| red cylinder | 27 | 0 | 29.0 | 0.64 | 21 / 6 |
| red cube | 41 | 0 | 27.9 | 0.63 | 29 / 12 |
| orange cone | 47 | 0 | 27.3 | 0.67 | 37 / 10 |
| covered car | 28 | 0 | 27.0 | 0.64 | 19 / 9 |
| red barrel | 25 | 0 | 26.3 | 0.68 | 16 / 9 |
| bronze whale statue | 40 | 0 | 25.8 | 0.69 | 23 / 17 |
| yellow cylinder | 24 | 1 | 24.8 | 0.68 | 19 / 4 |
| green cube | 23 | 0 | 24.7 | 0.68 | 19 / 4 |
| red fire truck | 24 | 0 | 21.6 | 0.70 | 19 / 5 |
| red wooden bench | 37 | 1 | 21.6 | 0.71 | 23 / 13 |
| man in blue jacket | 39 | 0 | 21.5 | 0.75 | 32 / 7 |
| person in purple shirt | 44 | 3 | 20.9 | 0.76 | 34 / 7 |
| yellow cube | 44 | 0 | 20.5 | 0.75 | 36 / 8 |
| blue barrel | 29 | 0 | 20.3 | 0.77 | 19 / 10 |
| bald man in green shirt | 47 | 2 | 20.2 | 0.76 | 42 / 3 |
| man in red shirt | 40 | 6 | 19.9 | 0.78 | 29 / 5 |
| beige sofa | 38 | 1 | 19.6 | 0.78 | 26 / 11 |
| doctor in white coat | 45 | 0 | 19.1 | 0.80 | 40 / 5 |
| yellow sphere | 23 | 0 | 18.6 | 0.80 | 22 / 1 |
| man in brown jacket | 39 | 10 | 18.5 | 0.80 | 28 / 1 |
| red tool chest | 33 | 0 | 16.7 | 0.72 | 18 / 15 |
| police officer | 37 | 15 | 16.3 | 0.84 | 16 / 6 |
| man in black suit | 51 | 18 | 16.2 | 0.84 | 29 / 4 |
| grey armchair | 36 | 0 | 15.2 | 0.85 | 32 / 4 |
| green armchair | 44 | 0 | 15.1 | 0.81 | 28 / 16 |
| black leather sofa | 34 | 1 | 14.8 | 0.86 | 27 / 6 |
| girl in yellow shirt | 44 | 0 | 14.4 | 0.86 | 41 / 3 |
| blue school chair | 40 | 0 | 14.1 | 0.84 | 24 / 16 |
| white plastic chair | 39 | 0 | 13.6 | 0.86 | 29 / 10 |
| yellow generator | 33 | 0 | 12.3 | 0.89 | 24 / 9 |
| round wooden table | 43 | 0 | 11.8 | 0.88 | 39 / 4 |
| white cabinet | 44 | 2 | 11.3 | 0.90 | 39 / 3 |
| ambulance | 24 | 16 | 10.9 | 0.89 | 8 / 0 |
| green sphere | 43 | 1 | 10.9 | 0.91 | 42 / 0 |
| red fire extinguisher | 41 | 1 | 9.6 | 0.93 | 38 / 2 |
| yellow wet floor sign | 53 | 28 | 7.7 | 0.95 | 16 / 9 |
| old television | 45 | 27 | 7.4 | 0.90 | 10 / 8 |
