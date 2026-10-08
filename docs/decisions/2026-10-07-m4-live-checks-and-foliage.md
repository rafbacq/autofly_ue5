# M4 C8: live checks for every obstacle kind, and what the trees taught (2026-10-07)

## What was done

The M1 live checks (`validate/live_m1.py`) assumed centred cylinders. They now take any layout obstacle through
`validate/obstacles.py` (cylinders, yawed engine cubes, imported glTF meshes ray-cast against their own triangles),
derive their poses from the scene's bounds (s01's constants fall out of them), check every actor's world box inside the
same simulator session (`map_matches_layout`, which `sim/check_map.py` used to do in a separate step through a new
`Simulator.object_bounds`), and save each depth probe's frames. The depth probe of a solid obstacle is M1's: a 4 x 4
window on its surface, the window medians within max(0.15 m, 2 %), the row silhouette at the pinhole width. A mesh is
foliage at most heights, so it is probed on a 16 x 16 window pixel by pixel: the camera must measure the ray-cast
depth on the *interior* of a solid patch (a trunk: solid pixels with solid neighbours; an isolated twig pixel's depth is
decided by half a pixel), never nearer than a leaf card where a card comes first, and farther where nothing is
predicted. The chooser lowers the camera into the band (1.5, 1.2 m) where trunks are solid, and lets the 12 m probe
fall back to 10 or 8 m where a trunk is too thin to fill three pixels. s01's probes are unchanged (same pillars, same
poses) and its gate's checks pass on the fake simulator, which now renders planar depth of its pillars on request.

Records: `docs/gates/m4_s09_gate.json` and `m4_s07_gate.json` **passed** (fast configs, slot 5):

| | s09 coloured poles | s07 stacked boxes |
|---|---|---|
| actors placed as built | 70 of 70, centre 2.4 µm, size 0.1 mm | 60 of 60, 2.4 µm, 0.15 µm |
| depth at 3 / 6 / 12 m | errors 0.3 / 1.1 / 1.6 mm, silhouette 22 / 12 / 6 px exact | 0.1 / 1.1 / 5.0 mm (box faces 0.157 m past the footprint circle), 39 / 22 / 12 px within 1 |
| crash | impact on the surface (0.000 m from the footprint), 6.400 m travelled against 6.405 predicted | on the yawed face, 6.441 against 6.557 (a corner meets the body first) |
| tracking, one step, spawn, faults | as M1 | as M1 |

## What the trees taught

1. **Interchange imports glTF `BLEND` materials as Translucent, and translucent surfaces write no depth.** Every Poly
   Haven tree carries its leaves and twigs as alpha-blended cards (island_tree_02: 715k card triangles against 357k of
   trunk and branches). The first s02 live check saw full canopies in RGB and bare trunks in depth: the depth-only
   expert would have flown into leaves it could not see, and the collision mesh held every card. Fixed at the import:
   each translucent instance of Interchange's `M_Default` master is overridden to **Masked** (clip 0.5, the glTF
   default) by `scenes/ue/ue_materials.py`, in `import_assets.py` for new imports and `fix_foliage_materials.py` for
   this project (8 instances; the variant sets share theirs). The host-side report check refuses a translucent slot.
   The depth camera then sees the canopies, and the three s02 depth probes agreed to 0.7-1.9 mm on trunk interiors
   with every interior pixel an inlier and 93-100 % of card pixels no nearer than their card.

2. **A crash into a canopy can wedge the simulator's actor for good.** The plugin moves the Unreal actor to the
   kinematics by a collision *sweep* every tick (`UnrealRobot.cpp MoveRobotToUnrealPose`). After a shallow contact a
   teleport straight up leaves the actor among the cards while the kinematics rise (`CameraPoseError`, 18 m then
   50 m), or stops it at the first leaf above, whereupon FastPhysics snaps the kinematics back (a reset that "settled"
   38 m away with no camera error). The backend's reset now climbs out by a ladder (`RESET_ESCAPES`: a plain climb; a
   nudge along the last collision's normal, the direction UE lets a penetrating actor leave by; retraces along the
   heading of 1.5, 4 and 8 m, which the body-forward-only action space makes free space), verifying that each hop both
   landed and kept its camera. On s02 it freed three of three shallow crashes (two by the 0.5 m nudge, one by the
   1.5 m retrace, 6-8 steps each, 0.5-1.0 s). But a *deep* contact -- the body straddling several cards whose
   minimum-translation directions disagree -- blocks every sweep at its start, and FastPhysics re-pins the kinematics
   to the actor on every tick: the instrumented hops (climb, nudge, retraces of 1.5 and 8 m, 0.3 m up and down, a
   far teleport) all returned the same pose with a collision event each. No API call moves a robot without a sweep.
   `docs/gates/m4_s02_gate.json` **failed** on exactly this: its crash check's reset could not free the actor. The
   record stays as written.

3. **One crash is one sample.** The same crash geometry (8.32 m of travel, to the centimetre) freed in one run and
   wedged in the next. Resets after foliage crashes have to be judged over many trials (`scripts/probe_canopy_reset.py`),
   not one gate.

## Decision: collision on solid geometry only

A cloud of thin cards is the wrong shape for sweep-based collision, and UE foliage convention is trunk collision
without leaf collision. The next step imports each glTF with cards as two static meshes from the same file: the opaque
primitives (trunk, branches, rock) with complex collision under the registry key, and the card primitives as a
`<key>_cards` companion the level builder spawns at the same transform with collision disabled. The depth camera sees
both; the body stops only at wood. The layout, footprints and depth prediction (which already separates opaque from
card hits) do not change; the registry, level spec, level builder, map check and the crash check's expected contact
(the nearest *opaque* geometry across the body's width) do. The s02 gate is then re-run as a new record.

Rejected: relaunching the simulator on every wedged reset (the resilient wrapper would, after repeated faults; at the
collision rates of early training that is a relaunch a minute); per-section collision flags (not reachable from the
editor's Python); simple convex collision from the whole mesh (hulls around the canopy, worse than cards).

## Also learned, operationally

- A client process that dies without `close()` hangs on projectairsim's non-daemon thread and its nng dialers
  **reconnect to the next simulator on the same ports**: two launches in a row then failed with "Timeout waiting to
  get topic info" until that process was killed. Kill the client before relaunching a slot.
- `scripts/stop_sim.py` returns before the process has exited (a minute, this time); a launch in that window connects
  to the dying server. Wait for the process to go.
- `scripts/probe_canopy_reset.py`'s run reported "no collision" for every trial after its third reset although the
  same approach collides when flown fresh; unexplained, and the probe is to be re-run after the collision split.
