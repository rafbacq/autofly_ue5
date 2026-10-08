# Runbook: the target pool's editor window (plan 5 C5/C6)

This window makes `assets/targets.json` usable for collection (`collect_dataset.py --targets`). It makes the palette's
colours as base materials the runtime can apply, imports the 48 model targets as one static mesh each, puts them in the
cook, repackages, and checks every target live. Run it in one sitting, after training has ended. The editor and the
packager need the GPU and the packaged binary to themselves.

Everything below was prepared and tested offline on 2026-10-08. The editor-side scripts have not run yet. The
"Not verified offline" list at the end says what only this window can show.

| Step | What | Where it writes | Time (estimate) |
|---|---|---|---|
| 1 | Runtime colours: base UMaterials M_Red, M_Blue, M_Yellow, M_Green, M_White | `ue_project/Content/AutoFly/Materials/Runtime/`, `runs/assets/runtime_materials.json` | 2–4 min (editor) |
| 2 | Merge their registry entries | `assets/registry.json` | seconds |
| 3 | Plan the targets: merged glTFs and the editor job | `runs/assets/targets/` | ~10 s |
| 4 | Import the targets | `ue_project/Content/AutoFly/Targets/`, `runs/assets/targets/import_report.json` | 5–15 min (editor) |
| 5 | Apply: registry entries and pool entries | `assets/registry.json`, `assets/targets.json` | seconds |
| 6 | Cook list | `ue_project/Config/DefaultGame.ini` | seconds |
| 7 | Package | `ue_project/Packaged/` | minutes (new shaders) |
| 8 | Package manifest | `runs/package/package_manifest.json` | seconds |
| 9 | Live probe of every target | `runs/m4/targets_probe/` | ~5 min |

```bash
cd ~/research_uav/autofly_ue5
J="bash scripts/run_job.sh"
PY="env -u PYTHONPATH .venv/bin/python"
```

## 0. Preconditions

```bash
git status --short                          # clean, on a feature branch for this window's commits
ls /tmp/.X11-unix/X1                        # display :1 must exist; if not, ask the user (MEMORY.md); never start X
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # no foreign job over 2 GB, 6 GB free
ls runs/sim/inst*/pid.json 2>/dev/null      # no live simulator of ours (runbook-m5.md §0 shows each slot's owner)
$PY -m pytest -q tests/test_runtime_materials.py tests/test_gltf_merge.py tests/test_import_targets.py \
    tests/test_ensure_cook_dirs.py tests/test_probe_targets.py tests/test_targets_pool.py
```

- Training must have ended: `$J wait <its job> 1` returns 0 or 1, not 124.
- Its eval watch and any other side job must have ended too.
- A simulator of ours still running blocks packaging (step 7 refuses). Stop it only through its record
  (`$PY scripts/stop_sim.py --instance <k>`), never with `pkill`.
- The editor is a foreign GPU job to any run's relaunch guard. That is why this waits for training to end.

## 1. Runtime colours (editor)

```bash
$J start runtime_materials -- $PY scripts/ue_editor_run.py autofly_ue5/scenes/ue/create_runtime_materials.py \
    --out runs/assets/runtime_materials.json --arg patch=$PWD/assets/pending/runtime_materials.registry.json
$J wait runtime_materials 600
$PY -c "import json; r=json.load(open('runs/assets/runtime_materials.json')); print(r['pass'], r['error'], r['shading'])
for m in r['materials']: print(m['key'], m['class'], m['path'], m['base_color_readback'], m['connected'], m['blend_mode'])"
grep -c "Failed to compile Material" runs/assets/runtime_materials.log    # must print 0
```

**Expect:**

- `True None`.
- Five rows, each with class `Material`, its path under `/Game/AutoFly/Materials/Runtime/`, and the palette's RGBA read
  back. `MP_BASE_COLOR` must be among the connected inputs, and the blend mode must be `BLEND_OPAQUE`.
- `shading` says where the roughness came from: BasicShapeMaterial (the palette's parent), M_Orange, or the 0.5
  default.

**On failure:**

- **`pass` False.** `error` holds the traceback. Most likely a UE Python API name differs on 5.7.4 (e.g.
  `get_material_property_input_node`, `MaterialExpressionScalarParameter`). Fix `create_runtime_materials.py` and
  run it again. A re-run deletes and re-makes each material.
- **No report.** The editor crashed or did not start: see `runs/assets/runtime_materials.log` and `.log.stdout`.
- **A "Failed to compile Material" line.** The editor only warns about it and renders the default grid in its place
  (2026-10-07: six grounds reached a packaged build that way). Read the lines after it in the log for the compiler
  error, fix the script, and run step 1 again. Step 2 refuses while the log holds such a line.

## 2. Merge the colours into the registry

```bash
$PY scripts/apply_runtime_materials.py --report runs/assets/runtime_materials.json
$PY -c "from autofly_ue5.dataset.targets import TargetPool; p = TargetPool.load('assets/targets.json')
print(sum(e.usable for e in p.entries), 'usable;', len(p.for_split('train')), 'for train')"
```

**Expect:**

- `registry updated: 5 base materials (runtime_blue, runtime_green, runtime_red, runtime_white, runtime_yellow)`.
- `12 usable; 11 for train`: every geometry target is now drawable ("green cone" is unseen).

**On failure:** the script exits 2, lists every problem and leaves the registry untouched. A compile failure, a wrong
class or a colour mismatch means step 1 again. "exists with different content" means a re-made material changed (e.g.
its roughness): pass `--replace` only if that change was meant.

## 3. Plan the targets (host)

```bash
$PY scripts/import_assets.py plan --targets
```

**Expect:** 48 lines and `48 targets planned in runs/assets/targets/import_spec.json`. Each line shows the key
(`tgt_<model>`), the pool name, the triangle count, the extent (glTF x, y up, z) and any material rewrite. There are
three kinds of rewrite:

- **Glass becomes masked.** The clock, the fire extinguisher and the generator have glass with
  `alphaMode BLEND -> MASK`.
- **Unlit becomes lit.** Seven Kenney characters change `KHR_materials_unlit -> lit`. An unlit surface would render
  near black under the scenes' -11 EV exposure.
- **Unlisted.** The suitcase file holds two suitcases, and only `vintage_suitcase_01_*` is taken. Its pivot is moved
  0.46 m onto it.

A dry run on 2026-10-08 planned all 48. Every extent matched `assets/measurements.json` to 0.1 mm, and the suitcase
matched its pool entry.

**On failure:** the plan prints `refused: ...` and exits 2. Nothing goes to the editor.

| Message | Cause | Fix |
|---|---|---|
| `... sha256 ...: not the file that was fetched` | The download changed | `scripts/fetch_assets.py` |
| `... is already the short name of ...` | A cooked asset already has the key, in any case | Rename the key's source, or ask |
| `extent` | The merge disagrees with the measurement | `scripts/measure_assets.py`, then compare |
| `% of its triangles are not opaque` | The model is foliage-like | Drop it from the pool |

On the key clash: Project AirSim's spawn table is a case-insensitive map of short names. The engine already has two
assets called `cone`, which is why every key has the `tgt_` prefix.

## 4. Import the targets (editor)

```bash
$J start import_targets -- $PY scripts/import_assets.py run --targets
$J wait import_targets 600                      # repeat while it prints 124
tail -2 runs/jobs/import_targets.log            # "editor exit code ... report written: .../runs/assets/targets/import_report.json"
grep -c "Failed to compile Material" runs/assets/targets/import_editor.log   # must print 0
```

It is the obstacles' editor script (`autofly_ue5/scenes/ue/import_assets.py`) on the targets' spec, with these
settings:

- **Path.** One static mesh per target, at `/Game/AutoFly/Targets/<key>/<key>`.
- **Nanite.** On, with the fallback keeping 100 % of the triangles (`nanite_fallback_percent` 1.0 in the spec).
  Complex collision runs on the fallback, and 10 % of a 72-triangle character is a handful of triangles.
- **Collision.** Complex-as-simple.
- **Textures.** Never streamed.
- **Materials.** Any translucent slot is made Masked. The merge already asked for masked glass, so none should be
  left.

**On failure:**

- **No report.** See `runs/assets/targets/import_editor.log` (and `.stdout`).
- **A hung editor** (the job never ends): `$J stop import_targets`, then look at the log's tail.
- **`pass` False.** Read `runs/assets/targets/import_report.json` `error`. A re-run deletes each target's folder
  before importing.
- **A "Failed to compile Material" line.** The failure is in one of Interchange's glTF material instances under
  `/Game/AutoFly/Targets/<key>/`. Find the key on that line, read the compiler error after it, and do not apply.
  Collected, that target would render as the default grid under a name that says "red ...". Either drop the entry
  for now, by setting its status to `manual` in `assets/targets.json` and planning again, or fix the cause and re-run
  step 4.

## 5. Apply (host)

```bash
$PY scripts/import_assets.py apply --targets                 # Kenney stand-ins stay "placeholder" (not drawn)
# or, if the user has decided the Kenney stand-ins are drawn until a Fab session replaces them:
# $PY scripts/import_assets.py apply --targets --use-placeholders
git diff --stat assets/registry.json assets/targets.json
```

**Expect:** `registry updated: 48 targets; 23 pool entries imported (drawn): ...`, or 48 with `--use-placeholders`.

- Every model entry in `assets/targets.json` now has `asset: "tgt_..."`.
- Poly Haven entries become `imported`.
- Kenney entries stay `placeholder` without the flag. They are still probed in step 9.

Whether the Kenney vehicles and people are drawn is **open with the user** (plan 5 C5). Without them the pool's only
vehicle is the covered car, and it has no human at all, against 22 % vehicles and 17 % humans in the paper's mix.

**On failure:** exit 2 with every problem listed. The registry and the pool are untouched.

| Problem | Meaning |
|---|---|
| `meshes imported; a target must be one static mesh` | Interchange split the merged file. Report it; this would be new. |
| `bounds centre at ...: the merge's pivot did not survive` | Interchange re-pivoted the mesh. |
| `complex collision would run on N of M triangles` | The fallback setting did not take. |
| `translucent material slots` | A masked override failed. |
| `material failed to compile` | Step 4's failure. |

## 6. Cook list

```bash
$PY scripts/ensure_cook_dirs.py
$PY scripts/ensure_cook_dirs.py --check
```

**Expect:**

- `added +DirectoriesToAlwaysCook for ['/Game/AutoFly/Targets']`. A second run says `unchanged`.
- `--check` then says `cook list covers the 49 runtime-spawn folders of the registry`: 48 target folders and
  `/Materials/Runtime`.

Nothing in the repo owns `DefaultGame.ini` (ue_project/ is git-ignored). This script is the only writer, and it only
ever adds lines. `/Game/AutoFly/Materials` was already cooked, and it covers the runtime colours.

## 7. Package (no simulator of ours running)

```bash
$J start package_targets -- bash scripts/package_sim.sh
$J wait package_targets 600                     # repeat while it prints 124
tail -4 runs/jobs/package_targets.log           # "RunUAT exit code: 0", "BUILD SUCCESSFUL", "packaged binary: ..."
```

`package_sim.sh` refuses to package in two cases, before it cooks anything:

- while a simulator of ours is running;
- while a registry target's or runtime material's folder is not in the cook list (`ensure_cook_dirs.py --check`).
  The fix is step 6.

On a cook failure, read `runs/package/package_dev.log`. Search it for `Error:` and for the target and material paths.

## 8. Package manifest

```bash
$PY scripts/make_package_manifest.py --level-spec runs/levels/s01.level.json --level-verify runs/levels/s01.verify.json \
    --out runs/package/package_manifest.json
```

The binary changed, and `validate/live_m1.py` compares its sha256 with this manifest. Every later gate needs the new
one.

## 9. Live probe of every target

```bash
$J start probe_targets -- $PY scripts/probe_targets.py --scene s01 --instance 5 --targets assets/targets.json \
    --out runs/m4/targets_probe
$J wait probe_targets 600
$PY -c "import json; s=json.load(open('runs/m4/targets_probe/summary.json')); print(s['pass'], s['error'], s['failed'])
print('not in the spawn table:', s['spawn_table']['missing'])
for r in s['targets']:
    print(f\"{r['name']:24s} {r['short_name']:32s} {'ok' if r['pass'] else r['failed']}  depth {r['depth'].get('hit_share')} \"
          f\"near {r['depth'].get('expected_near_m')}/{r['depth'].get('centre_depth_m')}  colour {(r['colour'] or {}).get('observed_mean_rgb')}\")"
```

Then **look at `runs/m4/targets_probe/contact_sheet.png`**. It shows every target as the drone's camera saw it,
cropped to its box and labelled with its instruction name. Read each label against its picture: a "red cube" must be
red, a "white cabinet" a cabinet, a person must stand upright. The per-target `NN_<key>_rgb.png` / `_depth.png` are
the full frames.

The probe flies s01 (slot 5, the 1 ms config) from one view on its open ring. For each of the 60 entries it does the
following, then destroys the object:

1. Checks the short name is in `world.list_assets()`.
2. Spawns the entry 4–10 m straight ahead, standing on the ground as collection stands it, at its pool scale and with
   its runtime material.
3. Steps once.
4. Compares the bounding box with the registry, checks the depth inside the projected box against the frame before
   the spawn, and checks the channel order of a runtime colour.

Exit codes: 0 all pass, 1 some fail, 2 the pool did not load (nothing launched).

**On failure, by check:**

| Check | What it means and what to do |
|---|---|
| `listed` for every `tgt_*` | The cook list was not applied before packaging: steps 6–8, then 9. |
| `listed` for one | Its key collides or its import is missing. Compare the registry with `ue_project/Content/AutoFly/Targets/`. |
| `spawned` with `set_object_material(...) failed` | The runtime material is not cooked, or is not a base UMaterial (step 1's report, step 6). |
| `bounds` | Size or centre off. A factor of 100 is a unit error. A wrong centre height is a pivot error: `bounds_origin_cm` against `standing_z_ned`. |
| `depth` | The camera does not see the target where it stands: translucent material, wrong pivot, or not rendered. Check the depth PNG. |
| `colour` | The runtime material did not show (the mesh's own colour). Check the material path in the cook. |
| `collided` | The target was spawned onto the drone. It should not happen at 4 m or more; report it. |

Keep the record either way: copy `runs/m4/targets_probe/summary.json` to `docs/gates/m4_targets_probe.json`, never
over an existing file. Do the same with the contact sheet (`m4_targets_probe_sheet.png`).

## 10. After the window

- **Commit** on the feature branch, one concern per commit:
  - the registry with the runtime colours;
  - the registry and pool with the targets;
  - the probe record.
- **Record a decision** (`docs/decisions/<date>-c6-targets-import.md`): what passed, what was dropped, the placeholder
  decision, and the probe's numbers.
- **Collection can use the pool:** `collect_dataset.py ... --targets assets/targets.json` (runbook-m5.md §5).
- **Still open in C6:** excluding a target that looks like the scene's own obstacles (s07 cubes, s09 poles, later
  vehicles in s10/s11).

## Not verified offline

- **The editor-side scripts have never run.** `create_runtime_materials.py` is new. `import_assets.py` has imported
  obstacles, but never a merged single-node file, a GLB-derived file, or Kenney's PNG textures.
- **Interchange's handling of the merged files is assumed:**
  - one mesh named after the glTF mesh, at the identity transform;
  - the `alphaMode MASK` glass;
  - `KHR_texture_transform` with only a texCoord;
  - vertex colours on the television.
- **That `fallback_percent_triangles` 1.0 keeps every triangle.** The report's `fallback_triangles` shows it.
- **That the runtime spawns and paints the new assets** (Project AirSim's spawn table filled from the cook) is
  exactly what step 9 is for.
