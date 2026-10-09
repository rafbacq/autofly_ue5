# Runbook: M3, the collector and the s01 pilot

M3's gate (spec §12) is a 100-episode s01 pilot that passes the dataset validator (spec §11). Code: `collect/`
(the collector), `dataset/` (state[9], the raw store, the RLDS exporter), `validate/dataset.py`, and
`scripts/collect_dataset.py`, which runs all of it and writes `docs/gates/m3_gate.json`. Decisions:
`docs/decisions/2026-10-03-m3-a0-and-collection.md` (a0, the target name, state[9]) and
`docs/decisions/2026-10-02-m2-closeout.md` (which checkpoint flies, stochastically).

```bash
PY="env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 .venv/bin/python"
J="bash scripts/run_job.sh"
CFG=scene_autofly_s01_fast.jsonc        # the clock s01_r2 was trained and gated on (docs/gates/m2_train.json)
MODEL=runs/expert/s01_r2/best/best_model.zip
TFDS=runs/tools/tfds_venv/bin/python    # git-ignored; step 1 builds it
```

Pick a free slot. While a training run with `--instances N` is up, slots 0..N are its own (N is its evaluation
simulator); use N+1 or higher. With nothing else running, any slot works. Check `ls runs/sim/inst*/pid.json`.

## 1. The TFDS venv (once)

TensorFlow cannot go into `.venv` (numpy 1.26.4 is pinned for projectairsim), so the RLDS read-back runs in its own:

```bash
uv venv runs/tools/tfds_venv --python 3.11
env -u PYTHONPATH uv pip install --python $TFDS "tensorflow-cpu==2.18.*" "tensorflow-datasets==4.9.*" \
    "tensorflow-metadata==1.16.1" pillow     # the newest tensorflow-metadata needs protobuf 6, which TF 2.18 refuses
```

## 2. Smoke (~3 min): never into evidence

```bash
$J start m3_smoke -- $PY scripts/collect_dataset.py --scene s01 --model $MODEL --scene-config $CFG --name smoke<k> \
    --episodes 3 --max-attempts 6 --seed-slice 99 --instance <slot> --data-root runs/m3_smoke/data \
    --out runs/m3_smoke/m3_gate_smoke<k>.json --check-python $TFDS
$J wait m3_smoke 420
```

Expect `kept: 3`, `validator_pass: true`, and `rlds.tfds_check.pass: true` in the record.

## 3. The pilot (~40 min at one slot)

```bash
$J start m3_pilot -- $PY scripts/collect_dataset.py --scene s01 --model $MODEL --scene-config $CFG --name s01_pilot \
    --episodes 100 --seed-slice 0 --instance <slot> --check-python $TFDS   # data/s01_pilot/, docs/gates/m3_gate.json
grep --line-buffered COLLECT runs/jobs/m3_pilot.log              # one line per episode: outcome, kept, rejected
$J wait m3_pilot 540                                             # repeat until it finishes
```

Read the record:

```bash
$PY -c "import json; g=json.load(open('docs/gates/m3_gate.json')); c=g['collection']; v=g['validation']
print('PASS' if g['pass'] else 'FAIL', c['kept'], 'kept of', c['attempted'], c['outcomes'], 'faults_ok', g['faults_ok'])
print(v['failures'][:5], v['warnings']); print(v['distributions']); print(g['rlds'])"
```

- `pass` needs 100 kept episodes, a validator pass, a clean engine audit, a usable RLDS export
  (`data/s01_pilot/1.0.0/`, spec §10.1) and, with `--check-python`, a passing TFDS read-back.
- A run that flew any episode writes its record as evidence, even if it failed or every episode was rejected. One that
  never flew (a refused launch) goes to `runs/not_started/` and claims no dataset name, so it can simply be retried.
- `collection.outcomes` is the stochastic expert's record on fresh seeds: its success rate should be near the M2
  gate's 0.99 stochastic. Rejected episodes are in `data/rejects/s01_pilot/`, each with its reason and provenance.
- The validator's distributions sit next to the release's (spec §3.2): median speed ~1.9 m/s, forward action ~2 m/s.
- `rlds.tfds_check` is TFDS reading every episode back and comparing it with the raw store.

A failed pilot stays failed: record it, find the cause, and run a new `--name` and `--seed-slice` with a new record
only after a fix. Slot rule: `main()` refuses a slot another live run holds.

## 4. Closing M3

1. Commit `docs/gates/m3_gate.json` (the dataset itself stays out of git: `/data/` is ignored).
2. Update `CLAUDE.md`'s status table and Plan 4's checkboxes.
3. The s01d pilot follows a passed M2d, with its own checkpoint and `--scene s01d`.
