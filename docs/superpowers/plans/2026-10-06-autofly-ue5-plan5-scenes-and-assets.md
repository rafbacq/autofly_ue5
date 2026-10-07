# M4 Scenes and Assets Plan (draft for the user's go-ahead)

> **Status:** drafted 2026-10-06 while run 6's follow-up held the simulator slots; **go-ahead 2026-10-07**: the user
> delegated U1-U4 ("do what you recommend and what ... would be most aligned with what they would do in the paper or in
> convention") and allowed any milestone to start once run 6's work had finished. The decisions below are therefore
> taken as recommended. Steps use checkbox (`- [ ]`) syntax; each task is test-first, like Plans 2-4.

**Goal (spec §12, M4):** the asset library and scenes s02-s12, s05r and s06r (spec §6.3); every scene builds, passes
reachability and the live checks of §11. Then M5 trains one expert per train scene and collects.

**Spec:** §6 (scenes, generator, assets, dynamic obstacles), §7.1 (measured contract: exposure, runtime spawn rules),
§13 (through-field routes). Sources and licences: `docs/decisions/2026-10-02-m4-asset-survey.md`.

## Facts found while planning (2026-10-06)

1. **The generator places every type spec §6.1 names** (C0, 75a9222): `poisson`, `clusters` and `stacks` beside s01's
   `jittered_grid`, box footprints (the circumscribed circle), several groups per scene without overlap. s01's layout is
   pinned to M1's build (`S01_LAYOUT_SHA256`); it did not move.
2. **A detour metric answers spec §13** (C1, 64e87ce, hardened after review): `scenes/paths.py:crossing_detours`
   flies evenly spaced start-band cells straight across to the opposite target band on the inflated grid and reports
   the detour over the straight line *and* whether a crossing exists inside the field's own lateral extent, where
   rounding the field is not allowed. s01: max detour 1.013, nothing unreachable, every start inside the field crosses
   through it. A wall with one gap: 1.0-1.6. A wall to the bounds: unreachable (max = inf). A compact 20 x 20 m block
   of pillars: detour 1.14 but `through_unreachable` > 0, which is what catches it. The bound is U4.
3. **Two scenes need no download.** s07 (stacked boxes) is the engine cube stacked by `stacks`; s09 (tall coloured
   poles) is the engine cylinder with a colour palette (colour instances are built in the editor, as MI_White was).
   Both need only a ground material and an exposure calibration, so they can be built first, while assets for the
   others are chosen.
4. **Every other scene needs assets**: trees (s02, s06), rocks and stones (s03, s04, s05), ruins (s08), walls and
   containers (s10), buildings (s11, s12), plus the 60-target pool (§6.4). Poly Haven and ambientCG are CC0 and
   scriptable; photoreal vehicles (22 % of the paper's targets) exist only on Fab under its Standard License, through
   the user's account, excluding NoAI listings (the survey).
5. **Footprints must be measured at flight altitude, not taken from the mesh bounds.** Depth sees the visual mesh,
   collisions use the collision mesh, and a tree's footprint at 1-3 m is its trunk plus low canopy. The registry's
   `measured_extent_cm_at_unit_scale` is filled by the editor build (as for s01's cylinder), and the live M1-style check
   compares depth against the layout's circles.
6. **Exposure is per scene** (-11 EV calibrated s01's grid floor to mean brightness 109; `docs/gates/exposure_calibration.json`).
   A ground that is brighter or darker needs its own value, found the same way, so a placeholder ground would mean
   calibrating twice.
7. **The pilot's expert flies near-optimally on s01**: PER in [0.956, 0.994] over the 100 kept episodes
   (`scripts/dataset_stats.py`, `runs/rebalance/s01_pilot_path_efficiency.json`; the grid bounds it from above, the
   straight line from below). Harder scenes should lower it; the paper's models score 73-78 %.
8. **Decided now, because changing it later would move every pinned layout:** scattered and clustered obstacles
   (poisson, clusters) face a random way; grid pillars keep yaw 0 (s01). Boxes share one yaw per stack.
10. **Measured from the glTF geometry (`scripts/measure_assets.py`, `assets/measurements.json`, 2026-10-07):**
   natural-size boulders are below the flight band (`boulder_01` 1.0 m tall, the Namaqualand boulders 0.5–1.9 m; only
   `_03` and `_04` reach into 1–3 m), so s03–s05 must scale them to 2–4 m standing stones or a drone at 1–3 m meets no
   rock. The young trees are wide at drone height: flight-band radii 3.3 m (`island_tree_02`), 2.8 m (`tree_small_02`),
   2.9 / 2.0 / 1.1 m (`searsia_burchellii` large / medium / small), 2.2 m (`island_tree_03`), 0.3–1.2 m (`searsia_lucida`
   a–e; f and g are under 0.8 m), 0.7 m (`quiver_tree_01`), with canopy centres 0.1–1.0 m off the pivot. Variant sets
   import as one static mesh per node (`searsia_burchellii` 3, `searsia_lucida` 7). Triangle counts run to 2 M per
   tree: import with Nanite, and give each mesh a convex collision that wraps the canopy, which is what the circle
   footprint models and what the live check must confirm.
9. **The live checks assume circles.** `sim/check_map.py` expects a (2r, 2r, h) bounding box and `validate/geometry.py`
   models standing columns, so a box instance's circumscribed `radius_m` would fail them (1.0 m wide against 1.41
   expected). C6 gives boxes their real extents before C8 runs on s07.

## Decisions (taken 2026-10-07 as recommended; the user delegated them)

- **U1: assets.** (a) CC0 only for now: Poly Haven trees and rocks, ambientCG grounds, Kenney/Quaternius low-poly
  props and vehicles as targets (a visible cartoon gap next to photoreal trees); (b) CC0 for obstacles and grounds,
  Fab (user's account, Standard License, no NoAI listings, terms confirmed in a browser) for vehicles and buildings.
  **Decided: (b).** The CC0 list is `scripts/fetch_assets.py`'s (6 young trees 2.3-4.7 m tall and up to 8.5 m wide, 7
  boulders and rock groups of 0.5-1.9 m, 6 grounds; sizes from the APIs on 2026-10-07; 662 MB), recorded with sha256 and authors in `assets/sources.json`. Vehicles and buildings
  wait for a Fab session with the user at the browser (its licence page refuses automated fetches; NoAI listings
  excluded).
- **U2: grounds.** Textured CC0 materials imported once, or flat-colour placeholders first? **Decided: textures**, so each
  scene is calibrated once (fact 6): ambientCG 2K JPG sets, one material per ground built in the editor (C3).
- **U3: order.** s09 and s07 first (fact 3), then s02 and s06 (trees), s03-s05 (rocks), s08, s10, s11-s12, and the
  re-seeded s05r/s06r last. **Decided: as listed.**
- **U4: the acceptance rule** for a generated layout (fact 2). **Decided:** at the sampler's 1.4 m inflation, 12
  evenly spaced starts per edge: `max` <= 1.25, `unreachable` = 0 and `through_unreachable` = 0, recorded in each
  scene's layout file beside the reachability result (C4 adds it to `scenes/build.py`).

## Tasks

- [x] **C0. Placements** (75a9222): `poisson`, `clusters`, `stacks`, box footprints, several groups per scene; s01 pinned.
- [x] **C1. Detour metric** (64e87ce): `crossing_detours` and `optimal_path_m` on the inflated grid; PER for a store.
- [x] **C2. Fetch CC0 assets by script** (`scripts/fetch_assets.py`, 2026-10-07): the curated list (6 young trees, 7
  boulders and rock groups, 6 grounds) with source URL, licence, authors, dimensions and sha256 into `downloads/`
  (git-ignored), recorded in `assets/sources.json`. Nothing is imported here.
- [x] **C3a. Measure the models offline** (`autofly_ue5/scenes/gltf.py`, `scripts/measure_assets.py`, 2026-10-07):
  height, extents and the 1–3 m flight-band footprint of every glTF node, from the buffers (fact 10).
- [ ] **C3b. Import in the editor** (`scenes/ue/import_assets.py`): import each glTF node as its own static mesh with
  Nanite and a convex collision, build one material per ambientCG set, read each mesh's bounds back against
  `measurements.json`, and write the registry entries (`ue_path`, `base_size_m`, `pivot` = base, `footprint` circle of
  `band_radius_m`, `category`, `role`, `seen`). Add `+DirectoriesToAlwaysCook` for spawnable targets.
- [ ] **C4. Scene files** `scenes/s02_*.json` ... `s12_*.json`, `s05r`, `s06r`: groups, placements, bands, ground,
  `instruction_obstacle`; each passes schema, generation, reachability and the detour bound offline before any build.
- [ ] **C5. Target pool** (`assets/targets.json`, spec §6.4): 60 instances, 50 seen / 10 unseen, the Fig. 3c category
  mix, a name for instructions each. The sampler draws the target and 3-5 distractors from the split's pool *after*
  every existing draw, and the instruction names the target; s01's pool stays the orange cylinder so its golden setups
  do not move. The collector's `--target-name` placeholder goes.
- [ ] **C6. Footprints and the runtime spawn table**: footprint circles from measured extents at flight altitude;
  box extents (base x scale, yaw) for `check_map.py`, `validate/geometry.py` and `motion.py`'s `CircleFootprint`
  (fact 9); `_spawn_asset_name` checked against `world.list_assets()` for every target (spec §7.1: short names, base
  `UMaterial` only).
- [ ] **C7. Exposure per scene**: the s01 procedure (`docs/gates/exposure_calibration.json`) on each built level, the
  bias recorded in the level spec.
- [ ] **C8. Build, package, live checks** per scene (`scripts/build_level.sh`, `package_sim.sh`, `validate/live_m1.py`):
  `docs/gates/m4_<scene>_gate.json` each (image changes with pose, depth matches geometry at known obstacles, a crash
  collides, commands track, one step per record). Packaging only while no simulator of ours is running.
- [ ] **C9. Status**: CLAUDE.md's table, spec §6.3 wording where a scene's phrase changed, MEMORY.md's learnings.

## Not in M4

Experts for s02-s10 and the full collection (M5); the test splits' evaluation harness (M6; `scenes/paths.py` already
gives its L_opt); dynamic variants of the new scenes (an `s0Xd` file each, when wanted).
