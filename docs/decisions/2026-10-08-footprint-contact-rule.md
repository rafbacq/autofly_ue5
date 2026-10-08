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
