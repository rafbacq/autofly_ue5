# Runbook: re-verify M0/M1 and close M2

Every command runs from the repo root on the GPU host. Long steps go through `scripts/run_job.sh`: `start` returns at
once, and `wait <name> <seconds>` returns 0 when the job finished OK, 1 on failure, and 124 while it is still running
(call it again). Logs are in `runs/jobs/<name>.log`. Re-verification outputs go to `runs/`; copy them to
`docs/gates/` only when they pass. Never edit an existing gate file.

```bash
cd ~/research_uav/autofly_ue5
J="bash scripts/run_job.sh"
PY="env -u PYTHONPATH .venv/bin/python"
```

## 0. Preconditions (every session)

```bash
git branch --show-current                   # fix/code-review-findings (or main after merging it)
ls /tmp/.X11-unix/X1                        # display :1 must exist: log in to the desktop session that provides it
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
journalctl _TRANSPORT=kernel -b -n 1 -q --no-pager   # must print a line (you need the adm group)
bash scripts/setup_venv.sh                  # verifies the venv against requirements.lock; changes nothing if current
$PY -m pytest                               # ~384 tests, ~36 s; everything passes
```

The GPU must hold no foreign job over 2 GB. The launch guard refuses a simulator while one is running: on
2026-09-24, user `ameya`'s CARLA plus detector held ~4.4 GB. Coordinate or wait; never kill someone else's process.
Don't run two of the steps below at once unless a step says so: startup sweeps stop orphaned simulators, and
`launch_sim.py` simulators count as orphans once their launcher exits.

## 1. M0 re-verification (~1 h; opens an Unreal Editor window on :1 for 2 minutes)

```bash
mkdir -p runs/archive && cp -a runs/m0 runs/archive/m0_run1        # the M0 scripts write into runs/m0/
$J start engine_check -- $PY -m autofly_ue5.validate.engine_check --out runs/m0/engine_check.json
$J wait engine_check 600
$PY -m autofly_ue5.validate.m0_gate vram baseline
$J start launch_inst0 -- $PY scripts/launch_sim.py --mode editor --map /Game/BlocksMap --instance 0 --timeout 900
$J wait launch_inst0 900
$PY -m autofly_ue5.validate.m0_gate vram idle_instance
$J start smoke_inst0 -- $PY -m autofly_ue5.sim.smoke_m0 --instance 0 --scene scene_autofly_m0.jsonc --warmup-steps 5 --out runs/m0/smoke_inst0.json
$J wait smoke_inst0 600
$J start smoke_fast -- $PY -m autofly_ue5.sim.smoke_m0 --instance 0 --scene scene_autofly_m0_fast.jsonc --phases lockstep --warmup-steps 5 --steps 50 --out runs/m0/smoke_fast.json
$J wait smoke_fast 600
$J start launch_inst1 -- $PY scripts/launch_sim.py --mode editor --map /Game/BlocksMap --instance 1 --timeout 900
$J wait launch_inst1 900
START_AT=$(( $(date +%s) + 240 ))
$J start smoke_inst0_concurrent -- $PY -m autofly_ue5.sim.smoke_m0 --instance 0 --scene scene_autofly_m0.jsonc --phases lockstep --warmup-steps 5 --start-at "$START_AT" --steps 100 --out runs/m0/smoke_inst0_concurrent.json
$J start smoke_inst1 -- $PY -m autofly_ue5.sim.smoke_m0 --instance 1 --scene scene_autofly_m0.jsonc --phases lockstep --warmup-steps 5 --start-at "$START_AT" --steps 100 --out runs/m0/smoke_inst1.json
$J wait smoke_inst0_concurrent 900; $J wait smoke_inst1 900
$PY scripts/stop_sim.py --instance 1; $PY scripts/stop_sim.py --instance 0
$PY -m autofly_ue5.validate.m0_gate faults       # scans only this run's logs now; needs a readable journal
$PY -m autofly_ue5.validate.m0_gate assemble     # prints the gate; "pass": true is the verdict
cp runs/m0/m0_gate.json docs/gates/m0_gate_reverify.json   # only if it passed
```

The build and plugin inputs (`runs/build/`, `downloads/plugin_manifest_report.json`) are reused unchanged. Rebuilding
the editor is not part of re-verification.

## 2. M1 re-verification at 3 ms (~15 min, packaged binary)

```bash
cp -a runs/m1 runs/archive/m1_run1
$J start launch_s01 -- $PY scripts/launch_sim.py --mode packaged --map /Game/AutoFly/Maps/S01 --instance 0 --timeout 600
$J wait launch_s01 600
$J start check_map_s01 -- $PY -m autofly_ue5.sim.check_map --instance 0 --layout runs/levels/s01.layout.json --out runs/m1/check_map.json
$J wait check_map_s01 600
$PY scripts/stop_sim.py --instance 0
$J start live_m1 -- $PY -m autofly_ue5.validate.live_m1 --out runs/m1/m1_gate_reverify.json
$J wait live_m1 900
cp runs/m1/m1_gate_reverify.json docs/gates/m1_gate_reverify.json   # only if "pass": true
```

If `crash_raises_collision.reset_after_crash` fails with `ResetPoseError`, that is the C9 defect showing, not a
regression. Run step 3 before anything else.

## 3. Crash-then-reset probe (~10 min)

```bash
$J start crash_reset_probe -- $PY scripts/probe_crash_reset.py --trials 30 --out runs/m1/crash_reset_probe.json
$J wait crash_reset_probe 900
$PY -c "import json; print(json.load(open('runs/m1/crash_reset_probe.json'))['summary'])"
```

How to read the summary:
- `bad_first_reset` is how many resets right after a crash missed their pose. The 2026-09-17 gate suggests about 20%.
- `fixed_by_second_reset` is how many of those a second reset repaired. Training and the gate already retry a bad
  reset. If a second reset always fixes it, retries are cheap. If it doesn't, relaunches take over, and they are slow.
- Keep the file: `cp runs/m1/crash_reset_probe.json docs/gates/m1_crash_reset_probe.json`.

## 4. M1 on the 1 ms clock (~10 min): decides the scene config for M2

```bash
$J start live_m1_fast -- $PY -m autofly_ue5.validate.live_m1 --scene-config scene_autofly_s01_fast.jsonc --out runs/m1/m1_gate_fast.json
$J wait live_m1_fast 900
```

`CFG=scene_autofly_s01_fast.jsonc` if it passes (M0 measured 2.4× the throughput). Otherwise
`CFG=scene_autofly_s01.jsonc`. Record the result either way: `cp runs/m1/m1_gate_fast.json docs/gates/`.

## 5. Throughput (~30 min): decides the number of workers

```bash
$J start m2_instances -- $PY scripts/measure_instances.py --scene-config $CFG --candidate-ns 1 2 4 --out docs/gates/m2_instances.json
$J wait m2_instances 540     # repeat until it finishes
$PY -c "import json; d=json.load(open('docs/gates/m2_instances.json')); print(d['chosen_n'], d['stop_reason'], {n: round(r['env_steps_per_s_total'], 2) for n, r in d['per_n'].items()})"
```

Set `N` to `chosen_n`. Training runs N simulators plus one for evaluation, so check VRAM headroom at N+1.

## 6. Smoke training + smoke gate (~30 min): every new code path, live

```bash
$J start m2_train_smoke -- $PY -m autofly_ue5.expert.train --scene s01 --instances $N --hours 0.33 --scene-config $CFG \
    --run-root runs/expert/s01_smoke --eval-freq 2000 --eval-episodes 3 --out runs/m2/train_smoke.json
$J wait m2_train_smoke 540   # repeat until it finishes
$J start m2_gate_smoke -- $PY -m scripts.m2_gate --run-root runs/expert/s01_smoke --scene-config $CFG --episodes 10 --out runs/m2/gate_smoke.json
$J wait m2_gate_smoke 540
```

Check in `runs/m2/train_smoke.json`:
- `status` is `ok`;
- `backend_faults` shows no Timeout storm;
- `dropped_fault_rows` is small;
- `engine_faults.kernel_journal_readable` is true;
- `faults_ok` is true;
- `eval.n_evaluations` is 1 or more.

Check in `runs/m2/gate_smoke.json`: every episode has `final_pose` and `oob_kind`, and the success rate is
meaningless at this length.

## 7. The retrain (12 h) and the gate (~6 h): M2's evidence

```bash
$J start m2_train -- $PY -m autofly_ue5.expert.train --scene s01 --instances $N --hours 12 --scene-config $CFG \
    --run-root runs/expert/s01_r2 --out docs/gates/m2_train.json
$J wait m2_train 540        # repeat, or: tail -f runs/jobs/m2_train.log
# progress: .venv/bin/tensorboard --logdir runs/expert/s01_r2/tensorboard

$J start m2_gate -- $PY -m scripts.m2_gate --run-root runs/expert/s01_r2 --scene-config $CFG --episodes 200 --out docs/gates/m2_gate.json
$J wait m2_gate 540         # repeat
$PY -c "import json; g=json.load(open('docs/gates/m2_gate.json')); print('PASS' if g['pass'] else 'FAIL', {k: v['deterministic'].get('success_rate') for k, v in g['checkpoints'].items()})"
$PY scripts/audit_m2_gate.py --gate docs/gates/m2_gate.json --out docs/gates/m2_gate_audit.json   # expect n_flagged 0
```

If training dies partway, resume the same run with the same command plus `--resume`. The resume refuses a run
recorded under another reward version. If it died before its first checkpoint (10,000 steps), there is nothing to
resume: rerun the same command without `--resume`, and the run directory is reused.

## 8. Closing M2

M2 closes when `docs/gates/m2_gate.json` has `"pass": true` and `m2_env_manifest.json`, `m2_instances.json`,
`m2_train.json` and `m2_gate.json` all come from this run.

Then:
1. Commit the gate files.
2. Choose which checkpoint (`best_model` or `final`) flies M3's collection, and record it in
   `docs/decisions/<date>-m2-closeout.md`.

If the gate fails, `pass: false` stays on the record, with no lowered threshold and no re-runs hunting for a lucky
seed. PLAN2's levers (D6) are then yours to choose: a start–target distance curriculum, raising `k_p`, or a longer
step limit. More training time via `--hours` is the cheapest first lever, because `--resume` continues the same run.
