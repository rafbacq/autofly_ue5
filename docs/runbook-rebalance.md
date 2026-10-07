# Runbook: dataset rebalancing (spec §10.3, for M5)

AutoFly resamples its trajectories by phase (App. A.2.4): obstacle avoidance until Grounding DINO first detects the
target with confidence above a threshold, target seeking from then on. The result here is `data/<name>/rebalance.json`:
the phase distribution P0, its KL divergence from uniform, the per-phase weights and the resampling sizes, plus every
episode's transition. The raw store and the RLDS shards are never rewritten. Code: `autofly_ue5/dataset/rebalance.py`
(pure), `scripts/detect_targets.py` (the detector, in its own venv), `scripts/rebalance_dataset.py` (the weights).
Decisions and the first measurement: `docs/decisions/2026-10-06-dataset-rebalancing.md`.

```bash
PY="env -u PYTHONPATH .venv/bin/python"
GD="env -u PYTHONPATH HF_HOME=$PWD/runs/tools/hf_home runs/tools/gdino_venv/bin/python"   # step 1 builds it
J="bash scripts/run_job.sh"
```

## 1. The detector venv (once)

`transformers` cannot go into `.venv` (numpy 1.26.4 is pinned for projectairsim), so the detector runs in its own venv
with a CPU-only torch; the model (about 700 MB) is cached under `runs/tools/hf_home`, inside the project.

```bash
uv venv runs/tools/gdino_venv --python 3.12
env -u PYTHONPATH uv pip install --python runs/tools/gdino_venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch
env -u PYTHONPATH uv pip install --python runs/tools/gdino_venv/bin/python transformers pillow numpy
# 2026-10-06: torch 2.14.1+cpu, transformers 5.19.0, pillow 12.3.0, numpy 2.5.3; grounding-dino-tiny at a2bb814d
```

## 2. Score the frames (stage one; long)

One score per frame: the model's highest box confidence for the episode's target phrase (`"orange cylinder."`), and
that box. Written after every episode and resumable, so a host freeze costs one episode.

```bash
$J start detect_<name> -- $GD scripts/detect_targets.py --raw data/<name> --threads 8 --batch 4     # CPU, default
grep --line-buffered DETECT runs/jobs/detect_<name>.log      # one line per episode: frames/s, max score, first > 0.7
$J wait detect_<name> 600                                    # repeat until "complete"
```

- **CPU pace (2026-10-06):** 0.34-0.37 frames/s at 8 threads, so the 100-episode pilot (17,738 frames) is 13-14 h. Run it
  niced (`nice -n 19`) beside live simulators; it leaves them their CPU.
- **GPU:** `--device cuda --allow-gpu` should be far faster (unmeasured here), but a torch process on the GPU is a
  *foreign job* to `gpu.check_gpu_for_launch`: with more than 2 GB held and no simulator of ours up, every simulator
  launch is refused, and a live run's relaunch would fail. Hence the explicit flag: use the GPU only when no training,
  gate or collection is running or about to relaunch, and with the user's go-ahead. The output file does not care which
  device scored which episode (`runs` records each run's device); expect scores to agree to about 1e-4 between devices
  (CPU batched against single-frame agreed to 1e-7; CPU against GPU is unmeasured).
- `--episodes N` scores the manifest's first N episodes (a probe); a later run without it continues from there.
  Writing anywhere but `data/<name>/detections.json`: `--out`.
- A detections file for another store, or by another detector (model, revision, transformers version, input size), is
  refused: pass another `--out`. The default model is pinned to the revision scored on 2026-10-06 (`--revision` for
  another), so a cached model loads without the network.
- `<out>.lock` holds the scorer's pid while it runs; a second scorer of the same file is refused, a lock left by a
  dead process is taken over.
- The whole file is rewritten after every episode: fine for the pilot's 100 episodes, quadratic at M5's ~13K. Give the
  cache an appendable form (one line per episode) before scoring the full collection.

## 3. The weights (stage two; seconds)

```bash
$PY scripts/rebalance_dataset.py --raw data/<name>                     # data/<name>/rebalance.json; refuses an existing one
$PY scripts/rebalance_dataset.py --raw data/<name> --threshold 0.6 --out runs/rebalance/<name>_t06.json   # another threshold
```

Read it:

- `coverage`: every episode must be covered (`partial: false`). A probe over a subset needs `--allow-partial` *and* an
  `--out` away from the store, and says `PARTIAL`.
- `p0` and `kl_nats`: the paper measured (0.73, 0.27) and printed 0.36 nats (its own Eq. 10 gives 0.110 for those two
  numbers; the code follows the equation). `weights`: the paper's 0.68 / 1.85 for a uniform target (`--alpha 0`).
- `counts.records_above_threshold`: the paper's other wording ("2 if detected, otherwise 1") counted per record, for
  comparison with the one-way split in `counts.records`.
- `counts.episodes_never_detected`: episodes the detector never scored above the threshold (all obstacle avoidance).
  `degenerate: true` means a whole phase is empty: no weight can rebalance that, so the CLI refuses to write such a
  result as the store's `rebalance.json` (inspect it with `--out`). A `--threshold` outside [0, 1) is refused too.
- `detection_persistence`: of the records from each first detection on, the share still above the threshold. Near 1
  means the transition is firm; low means the threshold sits where the detector flickers.
- `first_detection`: what the simulator knew at each transition: the median distance to the target, and whether the
  detected box covered the target's bearing (`on_target`) or something else (`off_target`: a distractor, a pillar;
  `off_target_out_of_view` counts those made while the target was not even in the image). A rising `off_target` count
  means the detector is not detecting the target and the threshold is too low for this scene.
- `episodes[]`: per episode, `first_detection` (record index or null), `phase_records`, and the two checks above.

## 4. The 2026-10-06 probe

Ten episodes of `data/s01_pilot`, CPU, beside run 6's gate (slots 0-4 were live):

```bash
$J start rebalance_detect_probe -- env -u PYTHONPATH HF_HOME=$PWD/runs/tools/hf_home nice -n 19 \
    runs/tools/gdino_venv/bin/python scripts/detect_targets.py --raw data/s01_pilot \
    --out runs/rebalance/s01_pilot_probe/detections.json --episodes 10 --threads 8 --batch 4
$PY scripts/rebalance_dataset.py --raw data/s01_pilot --detections runs/rebalance/s01_pilot_probe/detections.json \
    --allow-partial --out runs/rebalance/s01_pilot_probe/rebalance.json
```

The numbers are in the decision record. The pilot was then finished the same way on 2026-10-07: the probe's file copied
to `data/s01_pilot/detections.json`, the same `detect_targets.py` command without `--episodes` and `--out` (13 h, job
`detect_s01_pilot`; the file was continued, not redone), then `rebalance_dataset.py --raw data/s01_pilot` wrote
`data/s01_pilot/rebalance.json`: P0 = (0.652, 0.348), weights (0.766, 1.438).
