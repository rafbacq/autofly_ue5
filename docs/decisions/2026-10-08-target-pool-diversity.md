# 2026-10-08 — The target pool: the Kenney stand-ins are drawn, and no scene draws a target that looks like its obstacles

## The decision

Plan 6 (step 2b, revision 1 item 7), taken under the user's lab mentor's guidance that diversity of objects matters
more than replicating the paper exactly:

1. **The 25 Kenney models are drawn.** Their status in `assets/targets.json` goes from `placeholder` to `imported`, and
   each keeps `"stand_in": true`. The pool now draws 60 entries, 50 seen and 10 unseen (35 before: 31 and 4).
2. **No scene draws a target that looks like its own obstacles.** `assets/target_exclusions.json`, beside the pool and
   loaded with it, names per scene the registry assets never drawn there, target or distractor, with a reason each.

## Why the stand-ins

The paper's mix (Fig. 3c) is 22 % vehicles and 17 % humans. Without the Kenney models the pool had one vehicle (the
covered car) and no human at all, and nothing else covers those categories until a Fab session with the user at the
browser (photoreal vehicles and people). The Kenney models are CC0, imported and cooked in the C6 window, and the live
probe spawned every one of them (`docs/gates/m4_targets_probe.json`; its first-frame misses were first-use loading,
plan 6 step 2a). They are low-poly: 12 vehicles (10 seen), 10 people (9 seen), the traffic cone and two animals
(unseen).

## Reversible

- **The mark travels with the data.** Every entry of `assets/targets.json` states `stand_in`, true for the 25 Kenney
  models and false for the other 35; the loader refuses an entry without it, so the mark is never inferred.
  `TargetEntry.to_json()` carries it, so every pool episode's provenance says, as a JSON boolean, for its target
  (`target.stand_in`, which the merged export reads into its episode metadata) and each distractor, whether it was a
  stand-in. A dataset can drop or keep those episodes later without re-collecting anything else.
- **Undoing it is one data edit:** set the 25 entries' status back to `placeholder`. They keep `stand_in: true`, and
  `scripts/import_assets.py apply --targets --use-placeholders` writes the same mark when it promotes a placeholder, so
  a re-import cannot draw a stand-in without it.

## The per-scene exclusions

Keyed by registry asset, not entry name: the shape is what reads as an obstacle, so an asset takes every colour of a
primitive with it (and a colour added later). A scene without an entry excludes nothing; s01d draws by its own entry.

| Scenes | Excluded assets | Why |
|---|---|---|
| s01, s01d (white pillars), s09 (coloured poles) | `cylinder` (4 colours), `tgt_barrel_01`, `tgt_barrel_02`, `tgt_wine_barrel_01` | A cylinder is one more pillar or pole (s09's poles wear the same palette); an upright 0.88 m barrel is a short cylinder |
| s07 (stacked boxes) | `cube` (3 colours), `tgt_painted_wooden_cabinet`, `tgt_metal_tool_chest` | A cube is one more box; the white cabinet and the red tool chest are closed boxes |
| s03, s04, s05, s05r (rocks, stones) | `sphere` (2 colours) | A sphere on the ground reads as a rounded stone; s04 paints its stones in the spheres' palette |

Seen entries each scene draws after its exclusions: 43 in s01, s01d and s09; 45 in s07; 48 in s03, s04, s05 and s05r;
50 in s02, s06 and s06r. No unseen entry is excluded anywhere, so a test_unseen draw adds 10.

The cones stay in every scene. Plan 6's step 2c named "cylinders and cones" for the pillar scenes; revision 1 item 7
names the barrels instead, and a 1.5 m cone tapering to a point does not read as a pillar or a pole. If a live look
says otherwise, `cone` (and the traffic cone, `tgt_cone`) is one line in the table.

## How it is enforced

`TargetPool.for_split(split, scene_id=...)` drops the scene's excluded assets after the status and split filter, in
file order, so the sampler's pinned draws (`expert/episode.py`) keep their form over the scene's list, and the six-entry
floor (a target and up to five distractors) holds after the exclusions. The env, the sampler and
`scripts/collect_dataset.py` pass the scene's id. A table naming a scene no scene file claims, or an asset no entry
uses, is refused when the pool loads. The pool's record in every collection names what its scene's entry excluded and
the table's sha256.

## Verification

`tests/test_targets_pool.py` (the real pool and table: 60 drawn, 50 seen, the per-scene counts above, every Kenney entry
a marked stand-in), `tests/test_dataset_targets.py` (the mechanism on a test pool), `tests/test_expert_episode.py` and
`tests/test_expert_env.py` (the draws under exclusions), `tests/test_import_targets.py` (the apply step's mark),
`tests/test_collect.py` (the record); `tests/test_static_s01_golden.py` unchanged and green (no pool: nothing changes).
