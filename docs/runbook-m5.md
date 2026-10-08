# Runbook: M5, one SAC expert per new scene, then collection

M5's gate (spec §12): every train scene has an expert that passes its own gate, and the collected dataset passes the
validator (spec §11), with the rebalancing of §10.3 on top. This runbook takes **one scene** through it. Repeat it per
scene. The order is: throughput, smoke, a 12 h run, selection and weight averaging, one gate, collection, rebalancing.

The recipe comes from M2d's six runs on s01d (`docs/runbook-m2d.md` §6b–§6e, `docs/decisions/2026-10-06-s01d-r6-plan.md`).
The contact rule for mesh scenes is in `docs/decisions/2026-10-08-footprint-contact-rule.md`. How an accepted expert
flies collection is in `docs/decisions/2026-10-07-s01d-accepted-at-0.935.md`. Collection, export and rebalancing are as
in `docs/runbook-m3.md` and `docs/runbook-rebalance.md`. Every command below was taken from s02's job records of
2026-10-08 (`runs/jobs/<job>.pid.json`) or from the scripts' `--help`, and is written for s02.

Every command runs from the repo root on the GPU host. Long steps go through `scripts/run_job.sh`:

- `start` returns at once.
- `wait <name> <seconds>` returns 0 when the job finished OK, 1 on failure, and 124 while it is still running (call it
  again). Agents: one `wait` per Bash call, with `timeout: 600000`.
- Logs are in `runs/jobs/<name>.log`.

Only s01 and s01d have default evidence paths (`autofly_ue5/evidence.py`). Every other scene's training, gate,
throughput and collection **must pass `--out`**. Evidence goes to `docs/gates/m5_<id>_*.json`, smokes to `runs/m5/`.
Every script refuses to write over an existing file in `docs/gates/`.

```bash
cd ~/research_uav/autofly_ue5
J="bash scripts/run_job.sh"
PY="env -u PYTHONPATH .venv/bin/python"
ID=s02                                  # the scene; every command below is written with it
CFG=scene_autofly_${ID}_fast.jsonc      # the 1 ms clock: training, selection, the gate and collection all fly it
RUN=runs/expert/${ID}_r1                # this scene's training run (a later run: ${ID}_r2, with m5_${ID}_r2_* records)
N=4                                     # training workers, from step 1 (s02: 4)
TFDS=runs/tools/tfds_venv/bin/python    # git-ignored; runbook-m3.md §1 builds it
```

The records, and what each one is for:

| Record | Step | What it is for |
|---|---|---|
| `docs/gates/m5_<id>_instances.json` | 1 | Throughput per N. It decides N. |
| `runs/m5/<id>_train_smoke.json`, `<id>_gate_smoke.json` | 2 | Smoke: every path once, live. Not evidence. |
| `docs/gates/m5_<id>_train.json` (`_train_session<k>.json` for a resume) | 3 | The training run: identity, contact rules, outcomes, faults. |
| `$RUN/eval_watch/` | 3 | The eval watch: 20 episodes per 50k. It shows that the run is alive, and nothing more. |
| `$RUN/selection/stage{1,2}_*.json`, `stage3_*.json` | 4 | Selection and the comparison of the means, on validation seeds. Not evidence. |
| `$RUN/averaged/<name>.zip` + `.json` | 4 | A weight-averaged expert and its sources by sha256. |
| `docs/gates/m5_<id>_gate.json`, `_gate_audit.json` | 4 | **The scene's expert gate**, and its audit for impossible episodes. |
| `docs/gates/m5_<id>_avg_sources.json` | 4 | Only if a mean is gated: a copy of its `.json`, like `m2d_r6_swa_sources.json`. |
| `docs/gates/m5_<id>_confirm.json` | 4 | Only if a second model is gated on the same 200 episodes (see 4e). |
| `docs/decisions/<date>-m5-<id>-expert.md` | 4 | The closeout: which file flies collection, and why. |
| `docs/gates/m5_<id>_collect_<k>.json` | 5 | A collection batch: kept/flown, validator, RLDS, TFDS read-back. |
| `data/<name>/` (`detections.json`, `rebalance.json`) | 5 | The dataset store, then its detections and resampling weights. |

## 0. Preconditions (every session)

```bash
git status --short                          # clean, on main or a feature branch
ls /tmp/.X11-unix/X1                        # display :1 must exist; if not, ask the user (MEMORY.md); never start X
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # no foreign job over 2 GB, 6 GB free
journalctl _TRANSPORT=kernel -b -n 1 -q --no-pager   # must print a line (the engine-fault audit reads it)
ls runs/sim/inst*/pid.json 2>/dev/null      # each file is a slot in use; see below
$PY -m pytest -q                            # everything passes, offline (safe beside a live run: tests use scratch dirs)
```

**Slots.** A live slot has a `runs/sim/inst<k>/pid.json`. To see who owns each one:

```bash
for f in runs/sim/inst*/pid.json; do $PY -c "import json, os, sys; r = json.load(open(sys.argv[1])); o = r.get('owner_pid')
print(sys.argv[1], 'sim', r['pid'], 'owner', o, 'alive' if o and os.path.exists(f'/proc/{o}') else 'GONE')" "$f"; done
```

A slot whose owner is alive belongs to that run. Never stop it unless the run is yours. A slot whose owner is gone is
swept by the next launch's startup sweep. To stop one of your own, use `$PY scripts/stop_sim.py --instance <k>`, and
never `pkill`. `stop_sim.py` returns before the process has exited (about a minute). If a client died without
`close()`, kill it before relaunching that slot, or it takes over the new launch's ports (MEMORY.md, 2026-10-07).

| Slots | Held by |
|---|---|
| 0 … N−1 | The training run's workers. |
| N | The trainer's evaluator, which launches only at an in-training evaluation. With `--eval-freq 1000000000` that never happens, so the eval watch takes slot N (s02: 4). |
| N+1 and up | Side jobs: smokes, gates, renders, selection and collection for another scene (s02's gate smoke used 5). |

**The scene must be built, packaged and gated.**

```bash
ls docs/gates/m4_${ID}_gate*.json           # the newest passing record must be on the current layout
$PY -c "import json; print(json.load(open('docs/gates/m4_s02_gate_run4.json'))['pass'])"   # s02: run 4 (2026-10-08)
$PY -c "from autofly_ue5.scenes.resolve import resolve_scene; from autofly_ue5.expert.footprint_contact import task_contact_rules
r = resolve_scene('$ID'); rule = (task_contact_rules(r.layout) or {}).get('footprint')
print(r.scene.split, len(r.layout.instances), 'obstacles;', rule and {k: v for k, v in rule.items() if 'fields' not in k})"
$PY scripts/footprint_fields.py --check     # every mesh obstacle's outline field (assets/footprints/) matches its sources
ls configs/$CFG
```

The python line names the scene's kind, and the kind decides step 3's recipe. On 2026-10-08 it printed:

| Kind | Scenes | Contact rule |
|---|---|---|
| Mesh | s02 (27 obstacles), s03 (29), s04 (33), s05 (24), s06 (40) | `{'contact_m': 1.05, 'obstacles': n, 'shape': 'outline'}` |
| Pillar or box | s07, s09 (and s01) | `None` (physical contact) |
| Mixed | none yet | A footprint count smaller than the obstacle count |

The footprint rule ends an episode 1.05 m from a mesh's outline, the horizontal projection of all its geometry
(`2026-10-08-footprint-contact-rule.md`). s02_r1 trained under the earlier rule, 0.9 m from the footprint circle: every
script that scores a model reads the shape from the model's own run (`expert/run_shape.py`), so its records keep the
circle without any flag.

Packaging another scene needs a window with **no simulator of ours running** (`package_sim.sh` refuses otherwise), so
plan builds between training runs. An editor is also a foreign GPU job to a live run's relaunch guard.

## 1. Throughput (~11 min on s02): decides N

```bash
$J start m5_${ID}_instances -- $PY scripts/measure_instances.py --scene $ID --scene-config $CFG --candidate-ns 1 2 4 \
    --out docs/gates/m5_${ID}_instances.json
$J wait m5_${ID}_instances 600
$PY -c "import json; d=json.load(open('docs/gates/m5_${ID}_instances.json')); print(d['chosen_n'], d['stop_reason'], {n: round(r['env_steps_per_s_total'], 2) for n, r in d['per_n'].items()})"
N=4   # set to chosen_n
```

s02 gave N = 4 at 11.25 / 12.40 / 14.93 steps/s for N = 1 / 2 / 4, with no stop condition hit. Static s01 gave 13.4 at
N = 4 (`m2_instances.json`). A 12 h run at about 14 steps/s is about 600k steps.

## 2. Smoke (~20 min + a few minutes): every path, live

Use the step-3 recipe for the scene's kind. This is s02's, a mesh scene. A pillar or box scene passes
`--static-contact-m 1.1` instead of `--spawn-clearance-m 1.55 --footprint-contact-margin 0.2`.

```bash
$J start ${ID}_train_smoke -- $PY -m autofly_ue5.expert.train --scene $ID --instances $N --hours 0.33 --scene-config $CFG \
    --run-root runs/expert/${ID}_smoke --eval-freq 2000 --eval-episodes 3 --out runs/m5/${ID}_train_smoke.json \
    --spawn-clearance-m 1.55 --footprint-contact-margin 0.2 --altitude-margin-penalty 0.2 1.0
$J wait ${ID}_train_smoke 600           # repeat until it finishes
$J start ${ID}_gate_smoke -- $PY -m scripts.m2_gate --scene $ID --run-root runs/expert/${ID}_smoke --scene-config $CFG \
    --episodes 3 --conditions deterministic --instance $((N + 1)) --out runs/m5/${ID}_gate_smoke.json
$J wait ${ID}_gate_smoke 600
```

Read the training smoke:

```bash
$PY -c "import json; t=json.load(open('runs/m5/${ID}_train_smoke.json')); i=t['identity']
print(t['status'], t['obs_config'], i['scene'], 'host', t['host']['ok'], 'faults_ok', t['faults_ok'])
print(t['contact_rules'], t['collision_sources'], {k: v for k, v in i.items() if 'contact' in k or 'altitude' in k or 'clearance' in k})"
```

- `status` is `ok`. `obs_config` is `{"depth_frames": 1, "depth_dtype": "float32"}`: a static scene sees one frame.
- `host.ok` and `faults_ok` are true.
- **Mesh scene:** `contact_rules` names the footprint rule, its obstacle count, its shape (`outline`) and the outline
  fields it measured with. The identity holds `footprint_contact: {contact_m: 1.05, training_margin_m: 0.2, shape:
  outline, fields: {<asset>: <field sha256>, ...}}`, `training_spawn_clearance_m: 1.55` and the altitude cost; the
  config holds `footprint_shape: outline` and `training_spawn_clearance_m: 1.55` beside the scene's own
  `spawn_clearance_m: 1.4`. `collision_sources` shows `footprint_margin`, `footprint` and perhaps a few `sim` (s02's
  circle smoke: 48 / 3 / 1).
- **Pillar or box scene:** `contact_rules` is `None`, the identity holds `static_contact_m: 1.1`, and
  `collision_sources` shows `static_margin`.

Read the gate smoke:

```bash
$PY -c "import json; g=json.load(open('runs/m5/${ID}_gate_smoke.json')); print(g['contact_rules'], 'faults_ok', g['faults_ok'])
[print(n, e['outcome'], e['collision_source'], (e.get('footprint_contact') or {}).get('gap_m')) for n, c in g['checkpoints'].items() for e in c['deterministic']['per_episode']]"
```

The gate's log says which footprint shape it scored and where it read it (`footprint shape: outline (from the run's
identity)`). On a mesh scene, every collision says `footprint` and carries `footprint_contact` with its `shape` and a
gap between about 0.65 and 1.05 m (the outline; a circle run's were 0.5–0.9 m, s02's circle smoke 0.85–0.89 m). The
gate scores the task's 1.05 m, never the training margin. Success means nothing at this length.

## 3. The 12 h run, with watchers

**The recipe.** It is run 6's (the best s01d run) with the margin moved to where each scene's failure line is:

| | Mesh scene (trees, rocks) | Pillar or box scene | Why |
|---|---|---|---|
| Contact margin | `--spawn-clearance-m 1.55 --footprint-contact-margin 0.2` | `--static-contact-m 1.1` | See below. |
| Altitude | `--altitude-margin-penalty 0.2 1.0` | the same | A V: 0 at 2.0 m, 0.2 per step at the band's edges. Run 5's flat cost let the deterministic policy drift out of the band. |
| Buffer | `--buffer-size 250000` | the same | Runs 3–6. Always from scratch: never warm-start SAC with a fresh buffer (run 2). |
| Evaluation | `--eval-freq 1000000000` | the same | No in-training evaluation, which would pause all N workers. The eval watch scores checkpoints beside training instead. |
| Movers | none | none | M5's scenes are static. A scene with both movers and meshes is refused (`train.py`). |

**Why a margin at all.** Every s01d expert flew right up to the boundary it trained on, and its deterministic tail
reached about 0.3 m inside it (MEMORY.md, 2026-10-05). So training ends episodes some way outside the line the gate
scores. The expert observes, and the gate scores, the task's own rule.

- **Mesh scenes: `--spawn-clearance-m 1.55 --footprint-contact-margin 0.2`.** The task rule ends an episode 1.05 m from
  a mesh's outline, because a crash into wood or rock can wedge the simulator's actor beyond any teleport (0.63 m of
  measured collision reach, a full-speed step and a turn's chord bow). Training adds 0.2 m on top of that, for a
  boundary of 1.25 m. A boundary may reach no farther than a start can be: starts keep the scene's 1.4 m from every
  circle and a reset may land 0.3 m off, which leaves the outline 0.05 m. So the training workers keep their own starts
  1.55 m out (`--spawn-clearance-m`, training only: the evaluation env, the gate and the collector keep the scene's own,
  so the gate's 200 episodes stay those every run is compared on). A wider margin needs a wider clearance by as much
  (0.5 m: 1.85 m); on s02, 2,000 seeds drew every start at 1.55, 1.85 and 1.9 m, and fewer than 1 % of the scene's own
  starts lie within 1.9 m of a circle. 0.2 m is half a full-speed step, so about half of the endings on a full-speed
  approach cross both lines in one step and count as `footprint` (the task rule's), not `footprint_margin`.
- **Pillar and box scenes: `--static-contact-m 1.1`.** Engine primitives never wedge, so the task rule there is
  physical contact. Run 6's 1.1 m static boundary took the gate's static collisions from 10 (run 5) to 0. 1.1 m is the
  same ceiling (1.4 m start clearance less the 0.3 m reset tolerance). The distance is measured from the layout's
  `radius_m`. For a box that is the circumscribed circle of its base, so the boundary at a face sits farther out than
  1.1 m.
- **Mixed scenes:** pass both flags. The static boundary covers only the solids, and the footprint rule covers the
  meshes (`expert/env.py`). The trainer refuses `--static-contact-m` on a scene that has only meshes, and
  `--footprint-contact-margin` on a scene with no mesh. Both ceilings follow `--spawn-clearance-m` when it is given.

```bash
$J start ${ID}_r1_train -- $PY -m autofly_ue5.expert.train --scene $ID --instances $N --hours 12 --scene-config $CFG \
    --run-root $RUN --out docs/gates/m5_${ID}_train.json --eval-freq 1000000000 --buffer-size 250000 \
    --spawn-clearance-m 1.55 --footprint-contact-margin 0.2 --altitude-margin-penalty 0.2 1.0
#   pillar or box scene: --static-contact-m 1.1 in place of --spawn-clearance-m 1.55 --footprint-contact-margin 0.2
$J start ${ID}_r1_watch -- $PY scripts/watch_training.py --run-root $RUN --job ${ID}_r1_train --interval 300 \
    --png $RUN/progress.png
$J start ${ID}_r1_evalwatch -- env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 .venv/bin/python scripts/eval_watch.py \
    --run-root $RUN --job ${ID}_r1_train --every 50000 --instance $N --scene $ID --scene-config $CFG
#   no --footprint-shape: the watch reads the run's own from $RUN/sessions.json once the trainer has written it
```

**The failure alarm: arm it before anything else, and keep it armed until training ends** (MEMORY.md, 2026-10-03: a
crash once went unseen for 2 h 21 min). The repo has no alarm script. This is the loop:

```bash
$J start ${ID}_r1_alarm -- bash -c 'job=$1; log=runs/jobs/$job.log
  until [ -f runs/jobs/$job.exit ] || grep -qE "^(Traceback|FATAL)|Killed" "$log" \
        || [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt 900 ]; do sleep 60; done
  echo "ALARM $job $(date): exit=$(cat runs/jobs/$job.exit 2>/dev/null)"; tail -n 30 "$log"' _ ${ID}_r1_train
$J wait ${ID}_r1_alarm 600              # 124: quiet; 0: it fired (read its log: exit file, failure line or silence)
```

- It fires when the job's exit file appears (a clean end needs action too: step 4), when the log shows a traceback, a
  `FATAL` line or `Killed`, or when the log has been silent for 15 minutes. SB3's progress table prints every few
  episodes, so a silent log means a stall.
- **Do not match bare `Error`.** Every recovered fault prints `FAULT instance k: caught CameraPoseError …`: run 6's two
  clean sessions logged 5 and 13 of them.
- As a `run_job` job the loop has no time limit. An agent that runs it as a time-limited background command (a 2 h
  limit) **re-arms it at every expiry, in the same turn**, and does the same with any Monitor on the log.

**Reading the dashboard** (`tail -f runs/jobs/${ID}_r1_watch.log`, every 5 min; one snapshot:
`$PY scripts/watch_training.py --run-root $RUN --job ${ID}_r1_train --interval 0`). s02 at 52 min:

```
timesteps 45,288 at 14.0 steps/s (SB3)
episodes 468 | last 200: success 0.00  return 11.5  length 121  | collision 0.69  out_of_bounds 0.26  timeout 0.06  (collisions: footprint 2, footprint_margin 133, sim 3)
faults: none   relaunches 0   mover-inferred collisions 0   footprint endings 2 (+190 at the training margin)   refused moves 0   FATAL 0   skipped evaluations 0
```

- **`last 200`** covers the last 200 *training* episodes. They are stochastic and scored under the training rules, so
  training success runs below what the gate will see. Returns include the altitude cost.
- **`(collisions: …)`**, over the same 200 episodes:
  - `footprint`: endings at the task's 1.05 m (0.9 m on a circle run such as s02_r1).
  - `footprint_margin`: endings that only the 0.2 m margin caused, at 1.05–1.25 m. Expect these high early and falling.
  - `static_margin`: the same for the 1.1 m static boundary on a pillar or box scene.
  - `sim`: physical contact.
- **`footprint endings N (+M at the training margin)`** counts the job log's `FOOTPRINT instance k: <tag> (<asset>) at
  <gap> m …` lines over the whole run, not over the window.
- **`sim` on a mesh scene is the backstop firing.** Each one is a `SIM-CONTACT` line in the training log:

  ```bash
  grep SIM-CONTACT runs/jobs/${ID}_r1_train.log    # "... nearest outline <tag> (<asset>) <gap> m away, forward <v> m/s"
  ```

  The gap is to the nearest model's outline (to its circle on a circle run), and it tells what was hit:
  - A gap inside the rule's 1.05 m, down to 0, is wood or rock the rule did not stop: a rule miss.
  - A large gap, or "beyond its field's reach" (3 m and more at unit scale), is a spawned target or distractor (placed
    1.4 m or more from every circle), or something else away from every model.
  - On a circle run: a small gap is wood or rock outside its circle, and a gap of 0 or less a body inside a circle.

  s02_r1's first three were 10.95–15.76 m from the nearest tree, so not wood. What they did hit is not diagnosed yet.
  Watch `outcomes/collision_sim` on every mesh run: a rising count of small-gap contacts would mean wedged actors.
- **`faults` / `relaunches` / `FATAL`.** Recovered faults are normal (CameraPoseError mostly at reset). Five in a row
  with a near-identical error means a stuck simulator, and the wrapper relaunches it. Any `FATAL` ends the run.
- **Eval watch:** `grep EVALWATCH runs/jobs/${ID}_r1_evalwatch.log`. It flies 20 deterministic episodes per 50k on the
  training-evaluation seeds (200e6) under the task's rules. It shows the run is alive. **Never decide anything on it**:
  run 4's checkpoint at 0.85 there scored 0.55 on 40 validation episodes.
- **TensorBoard** (optional): `$J start ${ID}_r1_tensorboard -- .venv/bin/tensorboard --logdir $RUN/tensorboard --host
  127.0.0.1 --port 6006`. It binds to localhost only, because the host is shared.

**To end a session early and keep its record**, `touch $RUN/STOP`. Training stops within 25 steps, writes `final.zip`
and the session record, and tears its simulators down. Remove the file before any resume.

**If training dies partway**, resume with the same command plus `--resume`, the hours left (`--hours <h>`) and
`--out docs/gates/m5_${ID}_train_session<k>.json`. The resume continues from the newest periodic checkpoint and its
buffer. If it died before the first checkpoint (10k steps), rerun without `--resume`. **After a host freeze**, check
`/tmp/.X11-unix/X1` and the resume point (`runs/m2d_diag/verify_resume_point.py`) first (MEMORY.md, 2026-10-04).

## 4. After training: selection, averaging, one gate

When training ends, first check that `$J wait ${ID}_r1_train 600` returns 0 and that the record says `status: ok` and
`faults_ok: true`. Then let the eval watch finish (`$J wait ${ID}_r1_evalwatch 600`; it exits after the last due
checkpoint) and stop the viewers. Slots 0–4 are then free for selection, unless another scene's run holds them.

The seeds. Validation episodes come from `SELECTION_SEED_BASE` = 500e6 (`expert/seeds.py`). The gate's own episodes
(100e6) stay untouched until one model is gated.

| Stage | Seeds | Episodes |
|---|---|---|
| Stage 1 | 500,000,000 + 0…39 | 40 |
| Stage 2 | 500,001,000 + 0…99 | 100 |
| Comparison of the means | 500,002,000 + 0…99 | 100 |
| Gate | 100,000,000 + 0…199 | 200 |
| Confirmation (only if needed) | 100,000,200 + 0…199 | 200 |

### 4a. Stage 1 (s01d: ~2 h on five slots): every 20k checkpoint, 40 episodes

```bash
$PY scripts/select_checkpoint.py plan --run-root $RUN --every 20000 --min-steps <k> --slots 0 1 2 3 4 \
    --episodes 40 --scene $ID --scene-config $CFG --launch      # jobs select_s1_<slot>, launched 30 s apart
#   on a mesh scene the plan writes the run's footprint shape (from $RUN/sessions.json) into every part's command
$J wait select_s1_<slot> 600                                     # each slot; repeat until all have finished
$PY scripts/select_checkpoint.py rank --run-root $RUN            # exits 0 only when every part is in
```

- `--min-steps`: skip the checkpoints from before the eval watch first showed the run flying. s01d's runs used
  60,000 (run 2) and 160,000 (run 6). Write down the value used before ranking.
- Candidates also include `final` (and `best_model`, if there is one). The plan is
  `$RUN/selection/stage1_plan.json`, and the ranking `stage1_ranking.json`, best first.
- The job names `select_s1_<slot>` repeat across runs. Starting them replaces an earlier selection's logs in
  `runs/jobs/`, but never its results, which live in that run's `selection/`.

### 4b. Stage 2 (s01d: ~1 h): the top five, 100 further episodes

```bash
$PY scripts/select_checkpoint.py plan --run-root $RUN --stage 2 --names <top five> --seed-offset 1000 \
    --slots 0 1 2 3 4 --episodes 100 --scene $ID --scene-config $CFG --launch
$PY scripts/select_checkpoint.py rank --run-root $RUN --stage 2
```

Stage 1's best regresses: run 6's 0.975 on 40 episodes scored 0.85 on stage 2's 100. Take five candidates into stage 2,
not three, and trust stage 2.

### 4c. Weight averaging (s01d: ~1 h): means of wide windows against the stage-2 winner, on fresh episodes

The mean of a run's checkpoints flew steadier than any one of them. Run 6's 31-checkpoint mean had 0 exits against the
winner's 9, on the same 100 episodes (0.94 against 0.83, p = 0.035). Narrower and later windows did worse, and the
widest window was best both times.

**Pre-register first.** Before any mean flies, write the candidate windows and the rule into the closeout draft. The
windows are the widest (from the step where the eval watch first plateaued to the last checkpoint) and one or two
narrower late ones. The rule: *the best of them on the fresh episodes is gated if it beats the stage-2 winner there;
otherwise the winner is gated.*

```bash
LAST=$(ls $RUN/checkpoints | sed -n 's/^rl_model_\([0-9]*\)_steps\.zip$/\1/p' | sort -n | tail -1)
FROM=<first step of the window>       # e.g. 250000 for run 6
$PY -m scripts.average_checkpoints $RUN/averaged/avg_$((FROM / 1000))k_end.zip \
    $(for s in $(seq $FROM 10000 $LAST); do echo $RUN/checkpoints/rl_model_${s}_steps.zip; done)
# writes the zip and <zip>.json (every source by sha256); refuses an existing zip
```

Every candidate and the stage-2 winner fly the same 100 fresh validation episodes, one slot each, launched 30 s apart
(launches must not race the GPU guard):

```bash
$J start ${ID}_avg_eval_<k> -- env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 .venv/bin/python -m scripts.m2_gate \
    --scene $ID --scene-config $CFG --conditions deterministic --episodes 100 --seed-base 500002000 --instance <slot> \
    --out $RUN/selection/stage3_<name>.json --model <name>=<path>
#   <name>=<path>: e.g. avg_250k_end=$RUN/averaged/avg_250k_end.zip, or step_240000=$RUN/checkpoints/rl_model_240000_steps.zip
```

- No `--footprint-shape` anywhere in step 4: the gate reads each model's run (`final.zip`, `averaged/`, `checkpoints/`,
  `best/` all sit in or under the run root) and scores the shape it trained under; models of runs that trained under
  different shapes are refused together.
- `select_checkpoint.py` sees only its own stage plans, so a later `plan` must use a `--seed-offset` clear of
  2000–2099.
- To compare two of them on the same seeds: `$PY runs/m2d_diag/paired_compare.py <stage3 A> <name A> <stage3 B> <name
  B>` (McNemar, Wilson; git-ignored, a diagnostic).

### 4d. The gate (s01d: ~1.3 h for one model): one model, once

```bash
$J start m5_${ID}_gate -- $PY -m scripts.m2_gate --scene $ID --scene-config $CFG --episodes 200 \
    --conditions deterministic --model <name>=<path> --out docs/gates/m5_${ID}_gate.json
#   add --instance $((N + 1)) or higher if another scene's run holds slot 0
$J wait m5_${ID}_gate 600            # repeat
cp $RUN/averaged/<name>.json docs/gates/m5_${ID}_avg_sources.json      # only if a mean was gated
$PY scripts/audit_m2_gate.py --gate docs/gates/m5_${ID}_gate.json --out docs/gates/m5_${ID}_gate_audit.json   # n_flagged 0
```

The 12 h run trained with `--eval-freq 1000000000`, so there is no `best_model`. Always name `--model`. Read the record:

```bash
$PY - <<EOF
import json, collections
g = json.load(open("docs/gates/m5_${ID}_gate.json"))
print("PASS" if g["pass"] else "FAIL", g["obs_config"], g["contact_rules"], "faults_ok", g["faults_ok"])
for name, c in g["checkpoints"].items():
    d = c["deterministic"]; eps = d["per_episode"]
    print(name, d["success_rate"], d["collision_sources"], collections.Counter(e["outcome"] for e in eps),
          collections.Counter(e["oob_kind"] for e in eps if e["oob_kind"]))
    print([(e["seed"] - 100_000_000, e["collision_source"], (e.get("footprint_contact") or e.get("footprint_nearest") or {}).get("gap_m"))
           for e in eps if e["outcome"] == "collision"])
EOF
```

- The bar is 0.95 deterministic over 200 episodes, as in M2 and M2d.
- On a mesh scene a `footprint` collision carries its tree or rock, the shape, the gap (and the circle's gap beside it),
  bearing and speed. A `sim` collision carries the nearest model (`footprint_nearest`); read it as in step 3.
- **Scoring a model under the other shape is deliberate.** A circle-trained run's pick is flown under the outline only
  as a separately named record on the same 200 episodes, with `--footprint-shape outline --against-trained-shape
  --out docs/gates/m5_${ID}_gate_outline.json`; its `contact_rules` names `trained_shape: circle`. Without
  `--against-trained-shape` the gate refuses a flag that contradicts the run.
- **A failed gate stays failed.** No lowered bar, no re-gating hunting for a seed, no second model on the same episodes
  without a confirmation. The levers are the user's to choose:
  - more hours (`--resume`);
  - a new run with a changed recipe;
  - for a tree expert that ends gate flights at the 1.05 m rule, a wider training margin with a wider training
    clearance (`--spawn-clearance-m`, e.g. 1.85 m for a 0.5 m margin). It is training only, so the gate's episodes stay
    the same 200; a scene file's own `spawn_clearance_m` would change them for the same seeds
    (`2026-10-08-footprint-contact-rule.md`);
  - acceptance below the bar, as s01d was accepted at 0.935 (`2026-10-07-s01d-accepted-at-0.935.md`).

### 4e. Confirmation (only when a second model is gated on the same episodes)

On s01d the stage-2 winner was gated first and the mean second, on the same 200 episodes. A pass by the second model
counts only if it also clears 0.95 on 200 gate-range episodes never flown before:

```bash
$J start m5_${ID}_confirm -- $PY -m scripts.m2_gate --scene $ID --scene-config $CFG --episodes 200 \
    --conditions deterministic --seed-base 100000200 --model <name>=<path> --out docs/gates/m5_${ID}_confirm.json
```

The gate and the confirmation may run side by side once the model is fixed. Following 4c gates only one model, so this
step is not needed.

### 4f. Watch it fly (~10 min per few episodes)

Pick 6–10 gate episodes, mixing successes and each kind of failure:

```bash
$J start ${ID}_render -- $PY -m scripts.render_episodes --scene $ID --model <name>=<path> --scene-config $CFG \
    --episodes <i> <i> ... --gate docs/gates/m5_${ID}_gate.json --out-dir runs/viz/${ID}_r1_<name> --instance $((N + 1))
```

Each episode gives an MP4 (RGB, depth, top-down map) and a map PNG. `summary.json` compares each replay with the gate's
outcome. Replays are not bit-exact, so judge the 200-episode rate, never one replay. A replay flies the footprint shape
its gate record scored (without `--gate`: the shape the checkpoint's run trained under).

### 4g. The closeout

Write `docs/decisions/<date>-m5-<id>-expert.md`, modelled on `2026-10-07-s01d-accepted-at-0.935.md`:

- the expert's file and sha256, `obs_config`, scene config and reward version;
- the gate (and confirmation), CI, failures by kind, and the audit;
- the selection, with the pre-registered windows and rule, and what was compared;
- how it flies collection: stochastic, a0 sector8, under the task's rules, with no margin or penalty.

Then update `CLAUDE.md`'s status table (an M5 row per scene) and commit the `docs/gates/m5_${ID}_*.json` records, one
concern per commit, when the user agrees.

## 5. Collection, export, read-back, rebalancing

Collection waits for an expert that passed its gate or was accepted by the user. The collector flies it
**stochastically** (the default; `--deterministic` is for comparison only), with **a0 = sector8** built in (no flag),
under the task's own rules: on a mesh scene, the footprint shape the expert's run trained under, read from its
sessions.json (the record and every episode's provenance name it). A footprint ending is a reject (collision), like any
other collision. Only
successful, collision-free episodes enter the store, and the rest go to `data/rejects/<name>/` with their reasons.

**Seed slices.** Each collection run takes its own slice of 1e6 collection seeds (`--seed-slice k` →
400e6 + k·1e6, `expert/seeds.py`). Re-runs and scenes that share a layout would otherwise fly identical setups. The
allocation so far:

| Slice | Use |
|---|---|
| 0 | `s01_pilot` (M3) |
| 99 | Smokes and diagnostics, never a dataset |
| 1–98 | Unassigned. No M5 scheme is recorded yet. Take the lowest free slice per batch, and record it here and in the batch's closeout. |

**The target pool.** `--targets assets/targets.json` (plan 5 C5) draws each episode's target and 3–5 distractors
from the split's pool, and the instruction names the target. **It cannot be used yet.** Checked offline on 2026-10-08,
the pool loader refuses `assets/targets.json` (`'blue cylinder': material 'blue' is a color_instance, not a base
UMaterial`). Until C6 imports the targets and gives them base materials, the collector refuses `--targets` at startup,
before anything launches. Without the flag the target is M3's orange cylinder, a placeholder (`--target-name`), and the
card says so. Whether to collect at scale before the pool works is the user's call: every instruction would name the
orange cylinder. `docs/runbook-targets.md` is the editor window that makes the pool usable (runtime base materials, the
48 model targets imported, the cook list, a repackage and a live probe of every target).

### 5a. Smoke (~3 min): never into evidence

```bash
$J start m5_${ID}_collect_smoke -- $PY scripts/collect_dataset.py --scene $ID --model <expert.zip> --scene-config $CFG \
    --name ${ID}_smoke<k> --episodes 3 --max-attempts 6 --seed-slice 99 --instance <slot> \
    --data-root runs/m5_smoke/data --out runs/m5_smoke/${ID}_collect_smoke<k>.json --check-python $TFDS
#   once the pool is usable: add --targets assets/targets.json
$J wait m5_${ID}_collect_smoke 420
```

Expect `kept: 3`, `validator_pass: true` and `rlds.tfds_check.pass: true`.

### 5b. A batch (s01: 101 episodes in 45 min on one slot)

```bash
$J start m5_${ID}_collect_<k> -- $PY scripts/collect_dataset.py --scene $ID --model <expert.zip> --scene-config $CFG \
    --name ${ID}_c<k> --episodes 100 --seed-slice <slice> --instance <slot> --check-python $TFDS \
    --out docs/gates/m5_${ID}_collect_<k>.json
#   once the pool is usable: add --targets assets/targets.json
grep --line-buffered COLLECT runs/jobs/m5_${ID}_collect_<k>.log      # one line per episode: outcome, kept, rejected
$J wait m5_${ID}_collect_<k> 600                                      # repeat
$PY -c "import json; g=json.load(open('docs/gates/m5_${ID}_collect_<k>.json')); c=g['collection']; v=g['validation']
print('PASS' if g['pass'] else 'FAIL', c['kept'], 'kept of', c['attempted'], c['outcomes'], 'faults_ok', g['faults_ok'])
print(v['failures'][:5], v['warnings']); print(v['distributions']); print(g['rlds'])"
```

- **Naming.** `--out` is required (the default `m3_gate.json` exists and is refused). The `m5_<id>_collect_<k>` record
  name and the `<id>_c<k>` store name are this runbook's convention, not the code's. `--name` must be new: the
  collector refuses a store that exists.
- **The first batch measures the stochastic kept rate.** That rate has no gate measurement. If it falls clearly under
  the deterministic gate rate (for s01d: below about 0.85 over 100 flown), stop and look before scaling up. Do not
  re-gate.
- **Parallel batches.** One collector holds one slot. Several batches can run on separate slots, each with its own
  `--name` and `--seed-slice`, started 30 s apart. Collection beside a training run takes slots N+1 and up.
- **Storage.** M3's 100 s01 episodes were 17,738 records and 1.77 GB of PNG frames, stored twice (the raw store and
  the RLDS shards): `data/s01_pilot` is 3.4 GB on disk. Scenes with longer episodes cost proportionally more. The
  paper's 13K episodes would be about 460 GB. Check `df -h .` before each batch (397 GB free on 2026-10-08).
- A failed batch stays failed. Record it, find the cause, and only then run a new `--name` and `--seed-slice`.

### 5c. Export and TFDS read-back (only if the collector's own failed)

`collect_dataset.py` already exports to `data/<name>/1.0.0/` and reads it back with `--check-python`. Run these only to
redo a step whose record failed:

```bash
$PY scripts/export_rlds.py --raw data/<name> --check-python $TFDS     # refuses an existing data/<name>/1.0.0/
env -u PYTHONPATH $TFDS scripts/check_rlds_with_tfds.py --rlds data/<name>/1.0.0 --raw data/<name>   # read-back only
```

The read-back's docstring says `data/<name>/rlds/<dataset>/1.0.0`, but the exporter writes `data/<name>/1.0.0/`, as
the M3 record shows.

### 5d. Rebalancing (per store; `docs/runbook-rebalance.md`)

```bash
GD="env -u PYTHONPATH HF_HOME=$PWD/runs/tools/hf_home nice -n 19 runs/tools/gdino_venv/bin/python"
$J start detect_<name> -- $GD scripts/detect_targets.py --raw data/<name> --threads 8 --batch 4    # CPU, resumable
grep --line-buffered DETECT runs/jobs/detect_<name>.log
$J wait detect_<name> 600                                            # repeat until "complete"
$PY scripts/rebalance_dataset.py --raw data/<name>                   # data/<name>/rebalance.json; refuses an existing one
```

- The detector queries each episode's own target name, so a pool store scores every target by its name.
- Read `p0`, `weights`, `coverage` and `first_detection.off_target` as in runbook-rebalance §3. A rising `off_target`
  count means the 0.7 threshold is too low for that target or scene.
- **Pace.** At 0.34–0.37 frames/s on 8 CPU threads, 100 s01 episodes took 13–14 h. The paper's scale would be on the
  order of 1,700 CPU-hours, which is not feasible here. `--device cuda --allow-gpu` is unmeasured, and a torch process
  on the GPU makes the launch guard refuse every simulator launch. Use the GPU only with the user's go-ahead, when
  nothing may need to launch.
- **Before the full collection is scored,** `detections.json` needs an appendable form. It is rewritten whole after
  every episode, which is quadratic at M5's size.

## 6. Closing M5

One scene is done when its gate record, audit and closeout are committed and its batches pass. M5 closes when every
train scene's expert has its gate and the collected stores pass the validator, with rebalancing run on them. As with
every milestone, it stops for the user's go-ahead.

**Not yet determined offline** (decide with the user, then record here):

1. The seed-slice scheme for M5's batches.
2. Episodes per scene toward the paper's 13K: spec §10.4 chooses them from throughput.
3. Whether to collect before the target pool is usable.
4. How per-store rebalancing combines into one dataset-wide weighting: `rebalance_dataset.py` works on one store at a
   time.
5. A GPU or faster route for the detector at full scale.
