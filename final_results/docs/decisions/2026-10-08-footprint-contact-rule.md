# 2026-10-08 — The footprint contact rule: in mesh scenes the body never reaches wood

## Why

A crash into a tree can wedge the simulator's swept actor beyond any teleport, and a wedged actor cannot fly
(`2026-10-07-m4-live-checks-and-foliage.md`; `docs/gates/m4_s02_wood_crash_reset_probe.json`: 2 of 4 crashes into
wood stayed wedged after eight escape attempts). Convex solids never wedge. So before any expert trains on s02, the
task's contact rule in a scene with mesh obstacles had to stop being physical contact. The user asked for exactly this
in the 2026-10-07 handover: a geometric rule at the footprint circle, evaluated per step as s01d's mover rule is, with
the physical event as a backstop, built test-first, with s01's golden test untouched, and a training margin on top.

## The rule (`autofly_ue5/expert/footprint_contact.py`)

- **Which obstacles.** Every layout instance whose registry asset has a radius profile (an imported, measured model:
  trees, rocks). Engine primitives (cylinder, cube, crate) stay physical. The registry decides, not the scene file, so
  s06 and the rock scenes get the rule without a field anyone could forget; a layout naming an asset the registry lacks
  is refused.
- **Where.** An episode ends as a collision when the drone's swept step passes within **0.9 m** of an obstacle's
  footprint circle (the layout's `radius_m`). Precedence on a step: mover contact, then the simulator's own event
  ("sim"), then this rule ("footprint"), then a training run's static boundary.
- **Why 0.9 m.** The rule is checked after the step is flown, so the ending step overshoots the boundary by up to a
  full-speed step (2 m/s × 0.2 s = 0.4 m); the rotor tips reach 0.48 m from the centre; a full-rate turning step bows
  its chord by 0.01 m. 0.48 + 0.4 + 0.01 = 0.89, rounded up, and asserted at import as spec §6.5's mover invariant is
  at load. The first version used 0.5 m, reasoning only about steps the rule does not end: on 40 scripted fake flights
  straight at s02's trees every ending step put a rotor tip inside the circle (gaps 0.11–0.48 m). The figure that was
  meant to illustrate the rule caught it (`results/2026-10-07-project-status/figures/s02_footprint_rule.png`, made by
  `scripts/footprint_rule_figure.py`); the tests had not, because the fake's only wood was a 0.3 m trunk deep inside
  every circle. At 0.9 m the same 40 flights end at 0.51–0.88 m with no rotor tip inside any circle.
- **The circle is the model's whole reach, not the band's.** Until today `radius_m` was the model's widest slice inside
  the 1–3 m band. The body is not a point at the band's altitude, and after an episode ends beside a tree the reset
  climbs straight up through whatever hangs above the band: tree_small_02 reaches 0.175 m farther above 3 m than inside
  the band, searsia_burchellii_small 0.17 m below 1 m. With the rotor tips 0.02 m outside the circle at the closest
  ending, that overhang is a contact. So `scenes/generate.py` now gives a measured model the widest slice of its whole
  profile (what already spaced instances on the ground), keeping the flight-band check as the refusal of a model that
  is below the drone. For rocks that is the base, 0–0.5 m wider than their band slices at the 2–3× scales s03–s05 need.
  Regenerating the layouts: s09 and s07 byte-identical; s02's level spec identical (the packaged level is unchanged),
  its layout changed only in two tree_small_02 instances' radii (obs_0010 2.954 → 3.141 m, obs_0012 2.850 → 3.125 m)
  and in the third decimal of the crossing report (max 1.113, accepted). The s02 gate was re-run on the new layout:
  `docs/gates/m4_s02_gate_run4.json` **passed** (2026-10-08 00:26, slot 5, 1,690 MiB): 54 actors as built, the three
  depth probes on trunk interiors, the crash on obs_0002's wood at 8.95 m against 8.68 m predicted, the reset freed by
  an 8 m retrace on its fourth attempt (run 3: a 4 m retrace on its second; freeing a body that reached wood stays a
  matter of attempts, which is the rule's whole point). Run 3 stays as the record of the earlier layout file.
- **A mesh is a column over the whole band.** The rule is two-dimensional: a quiver tree 2.6 m tall or a 2 m boulder
  blocks the drone at 2.9 m too, although the depth camera shows air above it. Any per-altitude rule would let the
  body under a crown, which is where the actor wedges, so the column stays. Rock scenes must be designed with that in
  mind: the drone goes around rocks, never over them.
- **The backstop.** The simulator's collision event still ends the episode, scored "sim". On a mesh scene a "sim"
  ending means a target or distractor, wood outside its circle, or a rule miss; the per-episode record carries the
  nearest circle's tag and gap so the three can be told apart. Watch `outcomes/collision_sim` on every tree run.
- **Training margin.** `--footprint-contact-margin M` ends training episodes at 0.9 + M m, recorded in the run
  identity (`footprint_contact`) and the record; the gate and the collector apply the task's 0.9 m. Starts keep 1.4 m
  from every circle and a reset may land 0.3 m off, so the trainer refuses a boundary beyond 1.1 m: **M ≤ 0.2 m.** That
  is half a full-speed step, so about half the ending steps of a full-speed approach cross both lines at once and are
  the task rule's (`collision_footprint`); `collision_footprint_margin` counts only endings the margin alone caused.

## What is recorded

- A gate record on any scene now carries `contact_rules` (None on a pillar or box scene; the footprint rule and its
  obstacle count on a mesh scene), and each episode ending by the rule carries `footprint_contact` (tag, asset, gap,
  which rule, bearing, forward speed), as mover contacts do since 2026-10-03.
- A training record carries the rule in its identity, so a resume under another distance is refused; pillar scenes'
  identities are unchanged, so run 6 and every earlier run still resume.
- Log lines `FOOTPRINT instance N: <tag> (<asset>) at <gap> m from its circle (<rule>), bearing, forward` count the
  endings; `scripts/watch_training.py` shows them.

## Known limits, and the lever if M5's s02 expert needs it

1. The 0.2 m margin ceiling comes from the sampler's 1.4 m clearance, which is the reachability inflation the layouts
   are accepted with. Every s01d expert's deterministic tail reached about 0.3 m inside its training boundary, so a
   tree expert trained at 1.1 m may still end gate flights at the 0.9 m rule. If s02's gate shows that, the lever is
   a scene-level start/target clearance (a `spawn_clearance_m` field used by the sampler only, leaving the layout's
   acceptance at 1.4 m): starts farther from circles allow a wider margin without making any layout less flyable. It
   changes s02's scene file, so its layout file would be regenerated (instances unchanged) and its gate re-run.

   **The field now exists (2026-10-08).** A scene file may set `spawn_clearance_m` (≥ 1.4 m, the schema's floor). The
   sampler keeps starts, targets and distractors that far from every circle (`episode.scene_spawn_clearance_m`); the
   layout's acceptance, the movers' path-length guard and the built level stay at 1.4 m. The trainer's ceilings follow
   it: with 1.7 m, `--footprint-contact-margin` may reach 0.5 m (0.9 + 0.5 = 1.7 − 0.3) and `--static-contact-m`
   1.4 m. A scene that sets it names it in the run identity; one that does not is unchanged, so every earlier run
   resumes. It is one of the static fields a scene reusing a level must match. No scene sets it yet.
2. The reset after a footprint ending still climbs straight up from 0.5–0.9 m outside the circle; the circle now holds
   every slice, so the climb clears the model, but a spawned target beside a tree is not in any circle.
3. The rule covers movers' homes like any other circle only on static scenes: a scene with both a dynamic block and
   mesh obstacles is refused until moving a mesh (and its separate cards actor) is designed.

## Verification

- `tests/test_footprint_contact.py` (the rule, the env, the trainer, the gate record, the watcher), on a tree scene
  generated from the real registry against the fake; `tests/test_static_s01_golden.py` unchanged and green.
- `scripts/footprint_rule_figure.py` on s02's real layout: 40 seeds × {task rule, 0.2 m margin}; summary in
  `results/2026-10-07-project-status/figures/s02_footprint_rule.json`.
- Independent review (Opus, 2026-10-08) of the diff: static s01 and s01d unchanged in calls, draws, rewards, infos and
  observations; the refusal arithmetic sound; its should-fix items (the registry read in the unguarded record, the
  missing per-episode detail, the band overhang) are what the paragraphs above resolve.

## 2026-10-08 (afternoon): the rule measures to the outline, at 1.05 m

### Why

The circle about the pivot, with the model's widest reach as radius, holds every branch. On a leaning tree it also
reaches far past anything on the side away from the lean. `scripts/footprint_endings_figure.py` measured s02_r1's
eval-watch endings against the trees' real glTF geometry
(`results/2026-10-07-project-status/figures/s02_footprint_endings.json`):

- 82 of 220 deterministic flights ended at a circle.
- At those endings the drone was a median 2.70 m from any geometry, wood or leaf cards, and never closer than 0.94 m.
- On the three leaning species (island_tree_02, island_tree_03, tree_small_02) the median was 3.0 m.
- The tree was inside the forward camera's view at 8 of the 82 endings, and 49 of the 62 endings on leaning species
  were on the side away from the lean.

The expert sees only depth, and it was being stopped by an invisible bulge of the circle. A wider training margin
would have made that worse, not better.

### What changed

- **The distance is to the model's outline.** That is the horizontal projection of all its geometry above the ground,
  wood and leaf cards, at every height, at the instance's position, yaw and horizontal scale. It is still a column, as
  the circle was: after an ending the reset climbs straight up, so nothing overhead may be within reach. A crown
  2–3.5 m up counts for a drone at 1.5 m.
- **At 1.05 m, not 0.9 m** (`OUTLINE_CONTACT_M`; the circle keeps `CIRCLE_CONTACT_M` = 0.9 m, s02_r1's rule).
  - The 0.9 m invariant took the drone's reach to be its rotor tips, `DRONE_HALF_SPAN_M` = 0.48 m. A physical collision
    registers farther out: the mover probe's scripted flight collided with the drone's centre 0.51 m from a pillar's
    surface (`docs/gates/m2d_mover_probe.json`, check d1; `2026-10-02-dynamic-obstacles.md`), and run 5's gate static
    collisions ended 0.52–0.63 m from the surface, drone centre to pillar surface (`2026-10-06-s01d-r6-plan.md`,
    measuring the gate of `2026-10-05-s01d-r5-margins.md`). The UE collision of the drone's meshes reaches past the
    rotor discs. `scenes/model.py` names it `DRONE_COLLISION_REACH_M` = 0.63 m; `DRONE_HALF_SPAN_M` stays the mover
    invariant's.
  - The circle hid the difference in its slack: it reaches past the model everywhere but at the model's widest point.
    The outline has no slack on a trunk-side approach.
  - So the outline's distance keeps the measured reach outside it after a full-speed ending step (0.4 m) and a
    full-rate turn's chord bow (0.01 m), asserted at import: 1.05 − 0.4 − 0.01 = 0.64 ≥ 0.63. Neither the ending step
    nor the reset's straight climb from where it ended can reach the model.
  - Everything that reports or records the distance uses the shape's: the identity's `footprint_contact.contact_m`, a
    record's `contact_rules`, the step info, the trainer's ceilings and its help.
  - The gate's starts still never meet the task's rule: they keep 1.4 m from every circle (so from every outline inside
    it), and a reset may land 0.3 m off, which leaves 1.1 m ≥ 1.05 m.
- **Distance fields per asset** (`autofly_ue5/scenes/footprint_field.py`).
  - Files: `assets/footprints/<key>.npz` and `manifest.json`, built by `scripts/footprint_fields.py`; `--check` holds
    the manifest to the files, `assets/sources.json` and the registry.
  - Size: all 21 mesh obstacles in 2.58 MB, built in 10 s (1–2 s for a two-million-triangle tree).
  - Build: the triangles come from the loader the live depth checks use (MeshLibrary), in `Mesh._to_gltf`'s frame.
    They are rasterised exactly (each edge's every cell, each cell whose centre a triangle holds) at 6.25 mm, four
    times finer than the stored 2.5 cm cell. cv2's exact distance transform follows, less half a fine cell diagonal;
    float16 storage rounds down.
  - Why not cv2's fill and line drawing: `cv2.fillPoly` with several polygons XORs where they overlap, and its
    sub-pixel lines start half a pixel off the true line, so they were not used.
  - Accuracy: the field never overstates the clearance. Around one instance of each s02 species, on 11,470 random
    points 0.05–2.5 m from the outline, field minus exact distance lies in [−0.0132, −0.0007] m (median −0.006 m),
    against a brute force over every projected triangle. Over all 21 models (about 370 points each, at unit scale) it
    lies in [−12.6, −0.6] mm.
  - The tests hold that. `tests/test_footprint_field.py` holds the real models to a maximum of 0 (it allowed +5 mm),
    and a synthetic square whose sides lie on fine-cell boundaries to the same. A layout's lowest corner always falls
    on a fine-cell centre, which is why the review could remove the half-diagonal subtraction with every earlier
    distance test still passing. Without it, exactly those two tests fail: the square's sides read 3.1 mm too far, and
    island_tree_02 2.9 mm.
  - A damaged or missing field or manifest is refused as a ValueError naming the file, so the trainer reports it as a
    refusal. A non-finite drone position reads as +inf, no contact, instead of raising from the step's sample count;
    non-finite kinematics are the backend's to refuse.
- **Geometry below the ground is cut away.** Poly Haven's rocks sit up to 10 cm into the ground. The buried part of
  boulder_01, namaqualand_boulder_02 and sand_rocks_small_01 reached 6–17 mm past the widest slice above the ground,
  and that slice is the circle every layout was generated with.
  - Cut at the ground, every outline reaches as far from the pivot as its circle. The manifest's exact reach and the
    registry's widest profile slice agree to 0.05 mm (the registry rounds to 0.1 mm). `--check` holds every field
    within its `REACH_TOLERANCE_M` of 2 mm.
  - The cut assumes the instance stands on the ground. Under the outline an instance whose `z_center` is not 0 (within
    1e-6 m) is refused. Every stored layout's meshes stand at 0.
- **How a step is checked.**
  - Screen: a circle about each pivot whose radius is the field's own reach (`FootprintField.pivot_reach_m`). It is a
    bound read from the field itself, so the screen holds whatever radius an older layout gave the instance. At unit
    scale it can exceed the true reach by about a stored cell's diagonal (3.5 cm) plus the field's understatement, and
    in the world by that times the scale. On the stored layouts it exceeds the layout's `radius_m` by 2.2–4.1 cm on s02
    and s06, 3.5–10.0 cm on s03, 3.0–10.2 cm on s04 and 3.1–6.0 cm on s05.
  - A step more than termination + `PREFILTER_SLACK_M` (0.1 m) from its screen is skipped. The slack covers the field's
    understatement of the outline distance: at most 1.3 cm at unit scale (12.6 mm measured), so 3.2 cm in the world at
    the largest layout scale (namaqualand_boulder_06 at 3.17 on s04; the review measured 2.7 cm at scale 3).
  - Every other step's swept segment is read from the fields of the trees it passes near, sampled every 3.1 mm at unit
    scale, which is 3.1 mm times the instance's scale in the world (1 cm at 3.17).
  - Cost: on 60 scripted s02 flights at 1.05 m plus a 0.2 m margin, 0.034 ms per step on average (0.22 ms at most),
    against 0.020 ms for the circle (0.03 and 0.21 ms at 0.9 m). A live step takes about 70 ms.
- **The boundary.** On s02's 27 instances, the area within 1.05 m of the outline against the area within 0.9 m of the
  circle: median −6 %, from −49 % (tree_small_02) to 0 %. The leaning species lose 40–49 %, the upright ones 1–6 %. At
  0.9 m against 0.9 m it was median −21 %, max −55 %.
- **Records name the shape.**
  - The step info's `footprint_contact` and `footprint_nearest` gain `shape` and `circle_gap_m` beside `gap_m`, which
    is the shape's gap.
  - The `FOOTPRINT` log line says "from its outline"; the watcher's counts are unchanged.
  - The run identity's `footprint_contact` gains `shape`, and its `contact_m` is the shape's. For the outline it also
    gains `fields`: each covered asset's field content sha256, not the manifest's, so fields built later for other
    assets do not refuse a resume.
  - A record's `contact_rules` carries the shape, the shape's distance, the fields and the manifest's sha256. A model
    scored against the shape its run trained under also carries `trained_shape`.
  - A training record's config names `footprint_shape` on a mesh scene only.
  - `scripts/footprint_endings_figure.py` checks each recorded gap against the gap of the shape its record was scored
    under, and counts the outline rule's endings at 1.05 m.
- **Which shape scores a model is read, not defaulted** (`autofly_ue5/expert/run_shape.py`).
  - Why: s02_r1's pre-registered evaluation (`2026-10-08-m5-s02-r1-evaluation-plan.md`) was written before the shape
    existed. With "outline" as every script's default, its commands would have scored its circle-trained models under
    another rule without a word.
  - A model's run is the training run whose `sessions.json` lies in the model's directory or up to two above it:
    `final.zip`, `averaged/`, `checkpoints/`, `best/`; a run root is its own. Its shape is the last session's identity's
    `footprint_contact.shape`. A `footprint_contact` without a shape is from before shapes: the circle (s02_r1, which
    also resumes under the circle on the same reading). An identity without one trained without a footprint rule.
  - `m2_gate.py` and `collect_dataset.py` score that shape. Models of runs that disagree are refused (score them
    separately). `--footprint-shape` is needed only where no run names a shape: a model outside any run, or one from
    a run without a footprint rule. A flag that contradicts a run's shape is refused, unless `--against-trained-shape`
    says that is the point; the record then names `trained_shape`.
  - `eval_watch.py` and `select_checkpoint.py` read `--run-root`'s `sessions.json` the same way and hand the shape to
    every gate. The watch waits for the trainer to write its record, rather than refusing, until a checkpoint is due.
  - `render_episodes.py` replays under the shape its gate record scored, and without a record under the checkpoint's
    run's.
  - On a scene without mesh obstacles none of this applies: the shape is never asked for and no record is read.
  - `train.py` keeps `--footprint-shape {outline,circle}`, default outline: a new run trains under the task's rule, and a
    resume under another shape than its record's is refused.
- **A training run may keep its own starts farther out** (`train.py --spawn-clearance-m M`).
  - The training workers' sampler keeps starts, targets and distractors M from every circle, M at least the scene's own
    clearance. The evaluation env keeps the scene's own, as the gate and the collector do.
  - Why not a scene-file field (known limit 1 above): a scene's `spawn_clearance_m` changes the gate's episodes for the
    same seeds, and with them every comparison of runs on the same 200 gate episodes.
  - The ceilings follow it: contact distance + margin ≤ M − 0.3 m, and `--static-contact-m` ≤ M − 0.3 m. At the scene's
    own 1.4 m the outline leaves a margin of 0.05 m, so the runbook's 0.2 m margin trains with M = 1.55 m (0.5 m needs
    1.85 m). On s02, 2,000 seeds drew every start at 1.55, 1.85 and 1.9 m, and fewer than 1 % of the scene's own starts
    lie within 1.9 m of a circle.
  - The identity names `training_spawn_clearance_m` only when the flag is set, so every earlier identity still matches;
    the record's config carries it.
  - The trainer checks a margin's ceiling before it builds the rule, so an excessive margin is refused as one, not as
    a field too short to reach it.
- **One definition of a mesh obstacle**: a measured model (a radius profile) in the obstacle role
  (`footprint_contact.is_measured_obstacle`, which `scripts/footprint_fields.py` builds fields by). The target pool's
  models are measured too, but never placed in a layout: a layout instance of a measured asset in another role is
  refused.
- **An asset without a field is refused under the outline.** The env, the trainer (before it claims the run root),
  the gate and the collector (before anything launches) all refuse it. It is never measured by its circle instead.

### What did not change

- **Layouts and levels.** The layouts, the sampler's 1.4 m clearance (and `spawn_clearance_m`), reachability, the
  crossing detour and the built levels are all drawn against circles. The circles hold the outlines, so all of these
  stay conservative. No layout is regenerated and no level is rebuilt.
- **Where the ceilings measure from.** A start drawn 1.4 m (or M) from every circle is at least that far from every
  outline. Only the contact distance a margin adds to is the shape's.
- **Pillar and box scenes.** They have no footprint rule and their identities are unchanged. Static s01's golden test
  is unchanged and green.

### The circle is kept for s02_r1

`footprint_shape="circle"` is the rule as it was, step for step, at 0.9 m: a test holds it to the static boundary's
computation over the layout's radii.

- **Resuming.** s02_r1 trained under the circle. Its identity predates shapes, and a recorded `footprint_contact`
  without a shape is read as the circle. So s02_r1 resumes under `--footprint-shape circle` and is refused under the
  outline.
- **Evaluating.** Its pre-registered comparison, gate and renders are scored under the circle, the rule at
  pre-registration: on main as it stood before this patch, and after it by every script reading s02_r1's identity,
  without a flag. The outline record of its pick, at 1.05 m on the same 200 gate episodes, is flown with
  `--footprint-shape outline --against-trained-shape` as `docs/gates/m5_s02_gate_outline.json` (the plan's addendum).

### Verification (2026-10-08, evening)

- `tests/test_footprint_contact.py`: the distances and both invariants, each shape's ending distance in the env and the
  records, the training clearance (workers sample at M, the evaluation env at the scene's own, the ceilings, the
  identity), an excessive margin refused as one, the role and ground refusals, non-finite positions, a damaged field
  refused before the run root is claimed, and the gate's shape from a run (`--against-trained-shape` included).
- `tests/test_run_shape.py` (the run record's shape), `tests/test_footprint_field.py` (the field's bounds, the
  boundary-aligned square, non-finite points, damaged files), `tests/test_footprint_fields_script.py`, and the entry
  points' own tests (`test_collect.py`, `test_eval_watch.py`, `test_select_checkpoint.py`, `test_render_episodes.py`,
  `test_footprint_endings_figure.py`).
- `tests/test_static_s01_golden.py` unchanged and green; `scripts/footprint_fields.py --check` clean.
