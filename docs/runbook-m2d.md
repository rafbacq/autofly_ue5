# Runbook: M2d, moving pillars (scene s01d)

M2d gate (spec §12): the mover probe passes, and the s01d expert's deterministic success is ≥ 95 % over 200 held-out
episodes. Rules and evidence: `docs/decisions/2026-10-02-dynamic-obstacles.md`. Plan:
`docs/superpowers/plans/2026-10-02-autofly-ue5-plan3-moving-pillars.md`.

Every command runs from the repo root on the GPU host. Long steps go through `scripts/run_job.sh`:

- `start` returns at once.
- `wait <name> <seconds>` returns 0 when the job finished OK, 1 on failure, and 124 while it is still running (call it
  again).
- Logs are in `runs/jobs/<name>.log`.

Smoke outputs go to `runs/`. Real evidence goes to `docs/gates/m2d_*.json`, which is the default `--out` for scene
s01d. Training, the gate and the throughput script refuse to write over an existing file there.

```bash
cd ~/research_uav/autofly_ue5
J="bash scripts/run_job.sh"
PY="env -u PYTHONPATH .venv/bin/python"
CFG=scene_autofly_s01_fast.jsonc        # the 1 ms clock M1 passed on and M2 trained on
```

## 0. Preconditions (every session)

```bash
git branch --show-current                   # feat/moving-pillars (or main once merged)
ls /tmp/.X11-unix/X1                        # display :1 must exist; if not, log in to the desktop session (MEMORY.md)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # no foreign job over 2 GB
journalctl _TRANSPORT=kernel -b -n 1 -q --no-pager   # must print a line
$PY -m pytest -q                            # everything passes, offline
```

Don't run two of the steps below at once. Startup sweeps stop orphaned simulators.

## 1. The go/no-go probe (~10 min)

```bash
$J start m2d_probe -- $PY scripts/probe_movers.py --scene-config $CFG --out runs/m2d/mover_probe.json
$J wait m2d_probe 900
$PY -c "import json; r=json.load(open('runs/m2d/mover_probe.json')); print('PASS' if r['pass'] else 'FAIL', {k: v['pass'] for k, v in r['checks'].items()}); print(json.dumps(r['latency'], indent=1))"
cp runs/m2d/mover_probe.json docs/gates/m2d_mover_probe.json     # recorded whether it passes or not
```

How to read it:

| Check | What it shows |
|---|---|
| `a_move`, `b_restore` | A baked pillar moves, and returns home, to within 0.05 m; its neighbours stay put. |
| `c_depth_same_step` | The first frame after a move shows the pillar at its new place. |
| `d_collision` | Flying into a moved pillar collides. `d2`: also on the first step after the move. |
| `e_vacated_home` | Flying through a vacated home spot neither collides nor raises. |
| `f_park_and_restore` | Parking 50 m underground and restoring both work. |

The `latency` block gives the cost of moving pillars:

- `sequential_batch_ms`: 12 poses sent one request at a time, the way the backend sends them.
- `async_batch_ms`: the same 12 poses sent concurrently.
- `step_ms_without_moves` and `step_ms_with_10_moves`: one step with and without moves.

**If any check fails, stop.** The fallback (runtime-spawned movers on a rebuilt S01D level) is the user's decision and
gets its own decision record. **If moves cost more than about a third of a step**, compare the sequential and async
timings. The levers, fewer movers or `request_async` with its disconnect-on-error wrapped, are also the user's call.

## 2. Throughput with movers (~30 min): decides N

```bash
$J start m2d_instances -- $PY scripts/measure_instances.py --scene s01d --scene-config $CFG --candidate-ns 1 2 4
$J wait m2d_instances 540     # repeat until it finishes; writes docs/gates/m2d_instances.json
$PY -c "import json; d=json.load(open('docs/gates/m2d_instances.json')); print(d['chosen_n'], d['stop_reason'], {n: round(r['env_steps_per_s_total'], 2) for n, r in d['per_n'].items()})"
N=4   # set to chosen_n; training runs N simulators plus one for evaluation
```

Compare with `docs/gates/m2_instances.json`, which measured 13.4 steps/s at N = 4 on static s01.

## 3. s01 regression (~15 min): static s01 still flies as recorded

```bash
$J start m2d_s01_regression -- $PY -m scripts.m2_gate --scene s01 --scene-config $CFG --episodes 10 \
    --conditions deterministic --model best_model=runs/expert/s01_r2/best/best_model.zip \
    --out runs/m2d/s01_regression_gate.json
$J wait m2d_s01_regression 900
$PY - <<'EOF'
import json
new = json.load(open("runs/m2d/s01_regression_gate.json"))["checkpoints"]["best_model"]["deterministic"]["per_episode"]
old = {e["seed"]: e for e in json.load(open("docs/gates/m2_gate.json"))["checkpoints"]["best_model"]["deterministic"]["per_episode"]}
print([(e["seed"] - 100_000_000, e["outcome"], old[e["seed"]]["outcome"]) for e in new])
EOF
```

Expect the same outcomes as the M2 gate's first 10 episodes. Replays are not bit-exact (MEMORY.md): step counts drift
by a few, and about 1 marginal episode in 12 can flip. A systematic difference means static s01 changed: stop.

## 4. s01d smoke (~40 min): every new path, live

```bash
$J start m2d_train_smoke -- $PY -m autofly_ue5.expert.train --scene s01d --instances $N --hours 0.33 --scene-config $CFG \
    --run-root runs/expert/s01d_smoke --eval-freq 2000 --eval-episodes 3 --out runs/m2d/train_smoke.json
$J wait m2d_train_smoke 540   # repeat until it finishes
$J start m2d_gate_smoke -- $PY -m scripts.m2_gate --scene s01d --run-root runs/expert/s01d_smoke --scene-config $CFG \
    --episodes 10 --out runs/m2d/gate_smoke.json
$J wait m2d_gate_smoke 540
$J start m2d_render_smoke -- $PY -m scripts.render_episodes --scene s01d --run-root runs/expert/s01d_smoke \
    --checkpoint final --scene-config $CFG --episodes 0 1 --gate runs/m2d/gate_smoke.json --out-dir runs/viz/s01d_smoke
$J wait m2d_render_smoke 900
```

Check `runs/m2d/train_smoke.json`:

- `status` is `ok` and `obs_config` is `{"depth_frames": 3, "depth_dtype": "float16"}`;
- `identity.scene` is `s01d`, and `host.ok` is true;
- `backend_faults` shows no storm, and `faults_ok` is true;
- `collision_sources` is present.

The success rate means nothing at this length. In the gate smoke, every episode has `n_movers` ≥ 2. In
`runs/viz/s01d_smoke/`, the videos show red pillars moving, and the map leaves their homes empty.

## 5. The 12 h run, with watchers

```bash
$J start m2d_train -- $PY -m autofly_ue5.expert.train --scene s01d --instances $N --hours 12 --scene-config $CFG \
    --run-root runs/expert/s01d_r1                       # writes docs/gates/m2d_train.json at the end
$J start m2d_watch -- $PY scripts/watch_training.py --run-root runs/expert/s01d_r1 --job m2d_train --interval 300 \
    --png runs/expert/s01d_r1/progress.png
$J start m2d_tensorboard -- .venv/bin/tensorboard --logdir runs/expert/s01d_r1/tensorboard --host 127.0.0.1 --port 6006
```

To watch it:

- **Dashboard**: `tail -f runs/jobs/m2d_watch.log`, refreshed every 5 minutes. For one snapshot now:
  `$PY scripts/watch_training.py --run-root runs/expert/s01d_r1 --job m2d_train --interval 0`.
- **Progress plot**: `runs/expert/s01d_r1/progress.png`, rewritten every 5 minutes.
- **TensorBoard**: http://127.0.0.1:6006 from the desktop session, or from a laptop with
  `ssh -L 6006:127.0.0.1:6006 <host>`. It binds to localhost only, because this host is shared.
  `eval/success_rate` is the fixed-seed curve; `rollout/success_rate` and `outcomes/*` are training episodes.
- **Raw log**: `tail -f runs/jobs/m2d_train.log`.

When training ends, stop the viewers: `$J stop m2d_tensorboard`. The watch job ends by itself.

If training dies partway, resume the same run with the same command plus `--resume` and a shorter `--hours` (what is
left of the budget). A resume refuses a run recorded under another scene, observation or motion setting, and it writes
its own record, `docs/gates/m2d_train_session<k>.json`, so session 0's record is never overwritten. If it died before
its first checkpoint (10,000 steps), rerun the command without `--resume`; the run directory is reused.

**Slots while training runs.** An `--instances N` run owns slots 0 to N−1 for its workers *and slot N for its
evaluation simulator*, which comes up at every evaluation (`runs/sim/inst<N>/pid.json`, `owner_pid` = the trainer).
Anything run beside it (a smoke, a probe, a pilot) takes slot N+1 or higher. Never stop a slot whose `owner_pid` is
alive and not yours.

## 6. The gate (~6 h): M2d's evidence

```bash
$J start m2d_gate -- $PY -m scripts.m2_gate --scene s01d --run-root runs/expert/s01d_r1 --scene-config $CFG --episodes 200
$J wait m2d_gate 540          # repeat; writes docs/gates/m2d_gate.json
$PY -c "import json; g=json.load(open('docs/gates/m2d_gate.json')); print('PASS' if g['pass'] else 'FAIL', g['obs_config'], {k: (v['deterministic'].get('success_rate'), v['deterministic'].get('collision_sources')) for k, v in g['checkpoints'].items()})"
$PY scripts/audit_m2_gate.py --gate docs/gates/m2d_gate.json --out docs/gates/m2d_gate_audit.json   # expect n_flagged 0
```

A failure stays failed: no lowered threshold, and no re-runs hunting for a lucky seed. On s01d a single episode does
not replay: 4 of 8 rendered gate episodes changed outcome (2026-10-03), so judge the 200-episode rate, never one seed. The levers are the user's to
choose:

- `--resume` for more hours;
- a warm start from s01's weights;
- fewer or slower movers;
- privileged mover state in the vector.

Read `collision_sources` and `mover_in_view` to see which lever fits. Many mover collisions with the mover in view
point at learning; mover collisions out of view point at the observation.

## 6b. Run 2 (2026-10-03): warm start, mover input, validation selection

Run 1 failed its gate at 0.775 (`docs/decisions/2026-10-03-m2d-closeout.md`). Run 2's changes and their reasons:
`docs/decisions/2026-10-03-s01d-r2-plan.md`. Its records carry `m2d_r2_` names, so pass every `--out` explicitly.

```bash
# training: 12 h, warm-started from run 1's best_model; evaluation every 50k (selection does not rely on it)
$J start s01d_r2_train -- $PY -m autofly_ue5.expert.train --scene s01d --instances 4 --hours 12 --scene-config $CFG \
    --run-root runs/expert/s01d_r2 --out docs/gates/m2d_r2_train.json \
    --warm-start runs/expert/s01d_r1/best/best_model.zip --eval-freq 50000
$J start s01d_r2_watch -- $PY scripts/watch_training.py --run-root runs/expert/s01d_r2 --job s01d_r2_train --interval 300 \
    --png runs/expert/s01d_r2/progress.png
$J start s01d_r2_tensorboard -- .venv/bin/tensorboard --logdir runs/expert/s01d_r2/tensorboard --host 127.0.0.1 --port 6006
# a resume (more hours): the same command plus --resume, --hours <h>, and --out docs/gates/m2d_r2_train_session<k>.json

# selection on validation seeds (SELECTION_SEED_BASE), four slots, after training has ended
$PY scripts/select_checkpoint.py plan --run-root runs/expert/s01d_r2 --every 20000 --min-steps 60000 --slots 0 1 2 3 \
    --episodes 40 --scene s01d --scene-config $CFG --launch
$PY scripts/select_checkpoint.py rank --run-root runs/expert/s01d_r2          # exit 0 only when every part is in
$PY scripts/select_checkpoint.py plan --run-root runs/expert/s01d_r2 --stage 2 --names <top 3> --seed-offset 1000 \
    --slots 0 1 2 --episodes 100 --scene s01d --scene-config $CFG --launch
$PY scripts/select_checkpoint.py rank --run-root runs/expert/s01d_r2 --stage 2

# the gate: the stage-2 winner and final, on the gate's seeds
$J start m2d_r2_gate -- $PY -m scripts.m2_gate --scene s01d --scene-config $CFG --episodes 200 \
    --model selected=runs/expert/s01d_r2/<winner path> --model final=runs/expert/s01d_r2/final.zip \
    --out docs/gates/m2d_r2_gate.json
$PY scripts/audit_m2_gate.py --gate docs/gates/m2d_r2_gate.json --out docs/gates/m2d_r2_gate_audit.json
```

Each mover collision in a gate record now says how it happened (`mover_contact`: gap, bearing, moving or yielding).

## 6c. Run 4 (2026-10-04): run 3 plus a mover clearance penalty

Why: `docs/decisions/2026-10-04-s01d-r4-clearance-penalty.md`. Run 4 is run 3's command plus one flag, so the two
differ in nothing else (the same seeds fly the same episodes):

```bash
$J start s01d_r4_train -- $PY -m autofly_ue5.expert.train --scene s01d --instances 4 --hours 12 --scene-config $CFG \
    --run-root runs/expert/s01d_r4 --out docs/gates/m2d_r4_train.json --eval-freq 1000000000 --buffer-size 250000 \
    --mover-clearance-penalty 0.5 1.0
$J start s01d_r4_watch -- $PY scripts/watch_training.py --run-root runs/expert/s01d_r4 --job s01d_r4_train \
    --interval 300 --png runs/expert/s01d_r4/progress.png
$J start s01d_r4_evalwatch -- env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 .venv/bin/python scripts/eval_watch.py \
    --run-root runs/expert/s01d_r4 --job s01d_r4_train --every 50000 --instance 4 --scene s01d --scene-config $CFG
$J start s01d_r4_tensorboard -- .venv/bin/tensorboard --logdir runs/expert/s01d_r4/tensorboard --host 127.0.0.1 \
    --port 6006
```

- **Training returns include the penalty:** Monitor's `r`, `rollout/ep_rew_mean` and the dashboard's return. The eval
  watch, selection and the gate score the task's own reward, so their returns compare with every earlier run's.
- **To end a session early and keep its record**, `touch runs/expert/s01d_r4/STOP`. Training stops within 25 steps,
  writes final.zip and the session record, and tears its simulators down. Remove the file before a resume: a session
  refuses to start while a request is pending. A resume continues from the newest periodic checkpoint and its replay
  buffer, since final.zip carries no buffer, so it repeats up to 10k steps.
- **Afterwards**, selection and the gate as in §6b with run 4's paths: `m2d_r4_gate.json`, `m2d_r4_gate_audit.json`.

## 6d. Run 5 (2026-10-05): run 4 plus a training contact margin and an altitude margin

Why: `docs/decisions/2026-10-05-s01d-r5-margins.md`. Run 5 is run 4's command, the best so far, with two training-only
changes, each aimed at a failure run 4's best checkpoints still showed:

- **`--mover-contact-margin 0.3`.** A mover contact ends a training episode at 1.3 m from the mover's surface instead
  of the task's 1.0 m. The expert still observes, and the gate still scores, the 1.0 m rule.
- **`--altitude-margin-penalty 0.1 0.5`.** This charges 0.1 per step at the band's floor or ceiling, falling to 0 at
  0.5 m inside it.

```bash
$J start s01d_r5_train -- $PY -m autofly_ue5.expert.train --scene s01d --instances 4 --hours 12 --scene-config $CFG \
    --run-root runs/expert/s01d_r5 --out docs/gates/m2d_r5_train.json --eval-freq 1000000000 --buffer-size 250000 \
    --mover-clearance-penalty 0.5 1.0 --mover-contact-margin 0.3 --altitude-margin-penalty 0.1 0.5
$J start s01d_r5_watch -- $PY scripts/watch_training.py --run-root runs/expert/s01d_r5 --job s01d_r5_train \
    --interval 300 --png runs/expert/s01d_r5/progress.png
$J start s01d_r5_evalwatch -- env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 .venv/bin/python scripts/eval_watch.py \
    --run-root runs/expert/s01d_r5 --job s01d_r5_train --every 50000 --instance 4 --scene s01d --scene-config $CFG
```

**Training numbers read differently.**
- **Mover collisions count at 1.3 m**, so they include contacts the task would not score. Per-contact `gap_m`
  values between 1.0 and 1.3 are the margin at work.
- **Returns include both penalties.**
- **The eval watch, selection and the gate use the task's own rule and reward**, so their numbers compare directly
  with runs 1–4.

**Afterwards:** selection and the gate as in §6b, with run 5's paths: `m2d_r5_gate.json`, `m2d_r5_gate_audit.json`.

## 7. Diagnostic: the s01 expert, zero-shot, on s01d

```bash
$J start m2d_baseline -- $PY -m scripts.m2_gate --scene s01d --scene-config $CFG --episodes 200 \
    --conditions deterministic --model best_model=runs/expert/s01_r2/best/best_model.zip \
    --out docs/gates/m2d_baseline_s01_expert.json
```

The gate reads the checkpoint's one-frame observation from its zip and flies s01d with it. The gap between this and
step 6 is what training on moving pillars bought.

## 8. Watch it dodge

Pick 6–10 gate episodes, mixing successes and mover collisions, from `docs/gates/m2d_gate.json`, then:

```bash
$J start m2d_render -- $PY -m scripts.render_episodes --scene s01d --run-root runs/expert/s01d_r1 \
    --checkpoint best_model --scene-config $CFG --episodes <i> <i> ... --gate docs/gates/m2d_gate.json \
    --out-dir runs/viz/s01d_r1_best_model
```

## 9. Closing M2d

1. Commit the `docs/gates/m2d_*.json` records.
2. Update the status table in `CLAUDE.md`.
3. Record which checkpoint flies M3's s01d pilot in `docs/decisions/<date>-m2d-closeout.md`.
