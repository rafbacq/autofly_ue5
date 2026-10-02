# 2026-10-02 — M4 asset survey: where real trees and vehicles can come from (for the user's licence decision)

M4 builds the remaining scenes (spec §6.3: trees, rocks, crates, ruins, poles, walls, containers, buildings) and the
60-target pool (§6.4, Fig. 9: vehicles, furniture, equipment, people, primitives). s01 needed no downloaded asset;
everything after it does. This record lists the sources checked on 2026-10-02 and what each licence allows, so the
user can choose. **Nothing has been downloaded.** M4 starts after M3, but downloads, licence checks and the import
pipeline have lead time, so the choice is worth making now.

## What the dataset needs from a licence

The project distributes **rendered images** (the dataset), never the assets themselves. A student model, a VLA, is
trained on those images. Every asset's source, licence and SHA-256 go into `assets/registry.json` (spec §6.4).

## Sources checked

| Source | Licence (checked 2026-10-02) | What it has for us | Download | Fit |
|---|---|---|---|---|
| **Poly Haven** (polyhaven.com) | CC0: "use our assets for any purpose, including commercial work", redistribution allowed, no attribution needed | 521 models. Trees (category "trees", 20): `fir_tree_01`, `pine_tree_01`, `island_tree_01..03`, `jacaranda_tree`, `quiver_tree_01..02`, `tree_small_02`, saplings, stumps, trunks. Rocks (37): `boulder_01`, `namaqualand_boulder_02..06`, `coast_rocks_*`. Props: `barrel_03`, `wooden_barrels_01`, `cardboard_box_01`, `street_lamp_01..02`, benches, `covered_car` (its only vehicle) | Scriptable through the public API (`api.polyhaven.com`), glTF/FBX with textures | **Best for s02/s03/s04/s05/s06:** photoscanned and photoreal. High-poly, so it needs Nanite, and collision checked at flight altitude |
| **Kenney** (kenney.nl) | CC0: "public domain licensed (CC0). You're free to use them, even in commercial projects" | Low-poly kits: cars, city, nature, furniture | Zip per kit | Cartoon style: a visible domain gap next to photoreal trees. Usable for targets and primitives |
| **Quaternius** (quaternius.com) | CC0: "can be used for free without the need for attribution in commercial, educational, and personal projects" | Low-poly cars, nature, buildings, animals | Zip per pack | As Kenney |
| **ambientCG** | CC0 (textures and materials) | Ground materials: grass, gravel, sand, snow, paving | Scriptable | Ground for s02–s12 |
| **Fab** (fab.com, Epic) | Fab Standard License. Assets may be used and "commercially distribute[d]" inside a project, and rendered images and video may be distributed. No standalone redistribution of the asset. **Listings may carry a "NoAI" tag that contractually forbids use for generative-AI data collection** | Photoreal vehicles, buildings, city kits; Megascans | Needs the user's Epic account on the host | The only realistic source of **photoreal vehicles**. Any NoAI-tagged listing must be excluded, and each listing's terms recorded |
| **Sketchfab** (now under Fab) | Per model: CC0, CC-BY or Standard; NoAI possible | Many vehicles | Per model | Check model by model |

The Fab licence page itself returned 403 to an automated fetch. Its terms above are from the search results quoting
it and Epic's NoAI announcement. They need confirming in a browser before any Fab asset is used.

## Recommendation (the decision is the user's)

1. **CC0 first.** Poly Haven trees and rocks for the forest and rock scenes (s02, s06, s03, s04, s05), ambientCG
   grounds, Kenney or Quaternius props and primitive targets. All can be downloaded by script, need no account, and
   carry no AI restriction. Record source, licence and sha256 per asset.
2. **Vehicles: the real choice.** CC0 vehicles are low-poly cartoons. Photoreal ones need Fab, through the user's
   account, under its Standard License, excluding NoAI listings. Vehicles make up 22 % of AutoFly's targets (Fig. 3c),
   so this matters for the target pool more than for obstacles.
3. **Scene order stays as planned:** s02 (sparse trees), then s06 (tree clusters), then urban scenes with parked
   vehicles (s10/s11). A scene that uses vehicles as obstacles excludes vehicle targets, so instructions stay
   unambiguous.

## What M4 must handle whichever source is chosen

- **Collision must match the visual mesh in the flight band (1–3 m).** Depth sees the visual mesh; collisions use the
  collision mesh. Photoscanned trees ship with simple or no collision, so check the convex hulls or use complex
  collision; otherwise depth and collisions disagree. Measure live, as M1 did for the pillars.
- **Footprints in the flight band.** A tree's obstacle footprint at 1–3 m is its trunk plus any low canopy, not the
  trunk alone. A vehicle's is an oriented box. `motion.py`'s `CircleFootprint` is the hook for both.
- **Exposure calibration per scene.** s01 needed −11 EV (`docs/gates/exposure_calibration.json`).
- **Packaging.** New maps and cooked assets must go into the packaged binary (`+DirectoriesToAlwaysCook`). Runtime
  spawn takes the short asset name and a base `UMaterial` only (spec §7.1).

Sources: [Poly Haven licence](https://polyhaven.com/license), [Kenney support](https://kenney.nl/support),
[Quaternius FAQ](https://quaternius.com/faq.html), [Fab Standard License](https://www.fab.com/eula?lang=en),
[Fab NoAI tags](https://support.fab.com/s/article/Introducing-NoAI-meta-tags-and-Created-with-AI-self-declaration?language=en_US),
[Poly Haven API](https://api.polyhaven.com/assets?type=models).
