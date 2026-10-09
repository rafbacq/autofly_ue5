# Runbook: dataset rebalancing (spec §10.3, for M5)

AutoFly resamples its trajectories by phase (App. A.2.4): obstacle avoidance until Grounding DINO first detects the
target with confidence above a threshold, target seeking from then on. The result here is `data/<name>/rebalance.json`
(or one document for several stores): the phase distribution P0, its KL divergence from uniform, the per-phase weights
and the resampling sizes, plus every episode's transition. The raw store and the RLDS shards are never rewritten. Code:
`autofly_ue5/dataset/rebalance.py` (pure), `autofly_ue5/dataset/sparse.py` (the sparse search),
`autofly_ue5/dataset/detections.py` (the detections cache), `scripts/detect_targets.py` (the detector, in its own venv),
`scripts/rebalance_dataset.py` (the weights), `scripts/sparse_search_study.py` (the sparse search against dense
scores). Decisions and the first measurement: `docs/decisions/2026-10-06-dataset-rebalancing.md`.

```bash
PY="env -u PYTHONPATH .venv/bin/python"
GD="env -u PYTHONPATH HF_HOME=$PWD/runs/tools/hf_home nice -n 19 runs/tools/gdino_venv/bin/python"   # step 1 builds it
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

The script imports two modules of `autofly_ue5` in that venv, the cache format and the sparse search, so that the
detector and the rebalancing stage run one piece of code. Both are standard-library only, and a test holds them so
(`tests/test_sparse_search.py`).

## 2. Score the frames (stage one; long)

One score per scored frame: the model's highest box confidence for the episode's target phrase (`"orange cylinder."`;
with a target pool, the drawn entry's name, which is the phrase the instruction uses), and that box. Two modes:

- **Dense** (the default): every record, as the pilot was scored. Any threshold can be applied later.
- **Sparse** (`--sparse-stride S`): only the records the search of `dataset/sparse.py` visits on its way to the
  episode's first record above `--sparse-threshold` (default 0.7). That is all the one-way phase split needs, but the
  file then answers for that threshold only: the rebalancing stage refuses any other.

```bash
$J start detect_<name> -- $GD scripts/detect_targets.py --raw data/<name> --threads 8 --batch 1 \
    --sparse-stride 8 --sparse-alert 0.5                     # sparse; leave out both flags for dense (--batch 4 as before)
grep --line-buffered DETECT runs/jobs/detect_<name>.log      # one line per episode: frames scored, first > threshold
$J wait detect_<name> 600                                    # repeat until "complete"
```

- **The search, and its one bet:** coarse probes at records 0, S, 2S, ... and the last; the records between two
  probes are scored in order whenever either probe scored above the alert level; the first record found above the
  threshold is the transition. It assumes that a window whose two probes both score at or below the alert level holds
  no detection. Section 4 measures it. `--sparse-alert` equal to the threshold is the plain coarse scan, which the
  study rules out.
- **CPU pace (2026-10-06):** 0.34-0.37 frames/s at 8 threads. Dense, the 100-episode pilot (17,738 frames) was 13-14 h.
  Sparse at stride 8 and alert 0.5, the same episodes need about 5,940 frames (6,190 at `--batch 4`), so about 5 h.
  At the prototype's ~2,000 episodes of similar length that is about 124,000 frames, roughly 100 CPU-hours: plan for the
  GPU or for days. Run it niced beside live simulators; it leaves them their CPU.
- **A frame's score depends on its batch-mates (measured 2026-10-08, CPU).** Pilot frames scored again in their dense
  batch reproduce the stored scores exactly, on every run. Scored alone or beside other frames, the same frame moves
  by up to 4.1e-4: record 117 of `s01_400000001` scores 0.404919 in its dense batch and 0.405330 alone. The likely
  cause is the model's top-900 proposal selection, which float noise can flip; it is not established. The 2026-10-06
  record's "batched and single-frame scores agree to 1e-7" does not hold in general. Two consequences:
  - A transition is defined only up to that noise. Two of the pilot's 100 dense transitions (`s01_400000045`,
    `s01_400000066`) lie within 4.1e-4 of 0.7.
  - A sparse search, which batches other frames together than a dense run does, can differ from a dense file there.
- **On the CPU, prefer `--batch 1` for sparse runs.** It costs nothing per frame here (2.4-2.5 s alone against 2.5-2.6 s
  per frame in batches of 3-4). It scores no look-ahead records past the stop: 59.4 frames per pilot episode against
  61.9 at batch 4. And every score then depends on its frame alone, not on which frames the search happened to batch
  with it.
- **GPU:** `--device cuda --allow-gpu`. Its throughput is **unmeasured** here. A torch process on the GPU is a
  *foreign job* to `gpu.check_gpu_for_launch`: with more than 2 GB held and no simulator of ours up, every simulator
  launch is refused, and a live run's relaunch would fail. Hence the explicit flag: use the GPU only when no training,
  gate or collection is running or about to relaunch, and with the user's go-ahead. The file does not record which
  device scored which episode beyond each run's line. Expect scores on another device to differ by at least the batch
  noise above; CPU against GPU is unmeasured. A sparse decision sits on the alert and threshold levels, so a different
  device or batch can change which records are visited. Check a few episodes against dense CPU scores (section 4)
  before trusting a GPU run.
- **The cache** is `data/<name>/detections.jsonl` (`--out` for another path), in the form of `dataset/detections.py`:
  a header line (store, detector, search), a line per run, and one line per finished episode, each appended and synced
  as it is written. A host freeze costs at most the episode being scored. A torn last line is cut off on resume, and
  the log says so; any other damaged line is refused.
- A run continues a file only for the same store, the same detector (model, revision, transformers version, input
  size) and the same search (dense, or the same stride, threshold and alert). Anything else is refused: pass another
  `--out`. The default model is pinned to the revision scored on 2026-10-06 (`--revision` for another), so a cached
  model loads without the network.
- The pilot's `data/s01_pilot/detections.json` is the older whole-document form. The rebalancing stage still reads it,
  but the detector never continues it. With no `--out`, a store that holds one is refused rather than scored a second
  time.
- `--episodes N` scores the manifest's first N episodes (a probe); a later run without it continues from there.
- An episode whose `target_name` is not a phrase of its own instruction (a registry key, say) is refused before
  anything is scored. The detector must look for the words the instruction uses.
- `<out>.lock` holds the scorer's pid while it runs; a second scorer of the same file is refused, a lock left by a
  dead process is taken over.

## 3. The weights (stage two; seconds)

```bash
$PY scripts/rebalance_dataset.py --raw data/<name>                     # data/<name>/rebalance.json; refuses an existing one
$PY scripts/rebalance_dataset.py --raw data/<name> --threshold 0.6 --out runs/rebalance/<name>_t06.json   # dense files only
$PY scripts/rebalance_dataset.py --raw data/<a> --raw data/<b> --raw data/<c> --out data/<merged>/rebalance.json
```

A store's detections are its `detections.jsonl` or its `detections.json`, whichever it holds. A store holding both is
refused: pass `--detections` (once per `--raw`, in the same order). Read it:

- `coverage`: every episode must be covered (`partial: false`). A probe over a subset needs `--allow-partial` *and* an
  `--out` away from every store, and says `PARTIAL`.
- `p0` and `kl_nats`: the paper measured (0.73, 0.27) and printed 0.36 nats (its own Eq. 10 gives 0.110 for those two
  numbers; the code follows the equation). `weights`: the paper's 0.68 / 1.85 for a uniform target (`--alpha 0`).
- `sparse`: null for dense detections, else the search they came from. A sparse file is accepted only at its own
  threshold, and only after the search, replayed on its recorded scores, needs no record it lacks and finds the first
  detection it recorded.
- `counts.records_above_threshold`: the paper's other wording ("2 if detected, otherwise 1") counted per record, for
  comparison with the one-way split in `counts.records`. None for sparse detections.
- `counts.episodes_never_detected`: episodes the detector never scored above the threshold (all obstacle avoidance).
  `degenerate: true` means a whole phase is empty: no weight can rebalance that, so the CLI refuses to write such a
  result as a store's `rebalance.json` (inspect it with `--out`). A `--threshold` outside [0, 1) is refused too.
- `detection_persistence`: of the records from each first detection on, the share still above the threshold. Near 1
  means the transition is firm; low means the threshold sits where the detector flickers. None for sparse detections.
- `first_detection`: what the simulator knew at each transition: the median distance to the target, and whether the
  detected box covered the target's bearing (`on_target`) or something else (`off_target`: a distractor, a pillar;
  `off_target_out_of_view` counts those made while the target was not even in the image). A rising `off_target` count
  means the detector is not detecting the target and the threshold is too low for this scene.
- `by_target` and `targets_never_detected`: per target name, the episodes, how many never reached the threshold, the
  median first-detection distance and position in the episode, and on/off-target counts. With a pool of 50+ targets
  this is where a target the detector cannot find shows up: its episodes are all obstacle avoidance.
- `episodes[]`: per episode, `first_detection` (record index or null), `phase_records`, and the two checks above.

**Several stores** give one document for their union (`autofly_ue5_rebalance_union/1`; `--out` required). The records
are summed, so `p0` and `weights` are the union's. `coverage.by_store` and `stores.<name>` hold each store's own counts,
P0, weights, search, first-detection summary, `by_target` and the detections file's path and sha256. `episodes[]` are
keyed by `store` and `id` and name their `target`. Stores scored by different detectors are refused. Stores scored by
different searches (dense and sparse, or two sparse searches) need `--allow-mixed-search`: `search_mixed` is then true,
and the union's records-above-threshold and persistence are None.

## 4. A sparse search against dense scores (before scoring sparsely)

Plan 6 (Revision 1 item 6) asks that the sparse search keep the dense transition on at least 99 of the s01 pilot's 100
episodes. The study replays it on the pilot's dense scores. A strategy is a stride and an alert level, and each one is
weighed exactly as the rebalancing stage weighs a sparse file holding its records:

```bash
$PY scripts/sparse_search_study.py --raw data/s01_pilot --with-states --strides 1-16      # seconds
#   runs/rebalance/sparse_study.json and .md: calls per episode, transitions changed, P0, weights, labels changed
$J start sparse_check -- $GD scripts/detect_targets.py --raw data/s01_pilot --out runs/rebalance/live_check/detections.jsonl \
    --episodes 3 --threads 8 --batch 4 --sparse-stride 8 --sparse-alert 0.5                  # live, ~10 min on the CPU
$PY scripts/sparse_search_study.py --raw data/s01_pilot --check-sparse runs/rebalance/live_check/detections.jsonl \
    --out runs/rebalance/live_check/check.json
```

Read the study: `episodes_differing` is on the detector's own probe grid; `differing_over_offsets_per_100` averages
over every position of the grid, which shows how much a row owes to where its probes happened to fall. `offset_misses`
names the episodes behind it, and `differences` gives each changed transition's dense record, score, distance and
whether it was on target. The `recommendation` is a rule's pick, with the rule written out; the choice belongs in the
decision record. The check compares a live sparse file with the dense scores of the same frames, record by record, and
compares its first detections with the dense ones and with the study's offline prediction.

**2026-10-08, on the pilot (all 100 episodes, threshold 0.7):**

- **Dense:** 177.4 calls per episode, P0 (0.652, 0.348).
- **The plain coarse scan:** changes 24 transitions already at stride 2, and 37-44 at strides 4-16. In 48 episodes
  the first record above 0.7 is a one-record flicker among records hovering just below it.
- **Alert windows at stride 8, alert 0.5:** the rule's pick. 59.4 calls per episode (max 120; 61.9 at `--batch 4`),
  2.98 times fewer than dense. No transition changed, and P0 and the weights are unchanged. Over all eight positions of
  the probe grid it changes 0.38 per 100 (`s01_400000035` at two positions, `s01_400000095` at one). Both are genuine
  on-target detections of one and two records (0.755 and 0.737), between probes scoring 0.39-0.49, found 4 and 5
  records late.
- **The cheaper alternative, alert 0.55 at stride 8:** 43.6 calls (4.07 times fewer), no transition changed on the
  detector's grid, 1.00 per 100 over grid positions.
- **The live check:** 3 episodes, `--batch 4`, CPU, 8 threads at nice 10, 210 frames in 568 s against 535 dense.
  - Every first detection equalled the dense one and the prediction: records 46, 147 and 131.
  - Visited scores differed from the dense ones by at most 4.1e-4 (mean 2.4e-5) and boxes by 1.3e-5, all from batch
    composition (above).

## 5. The 2026-10-06 probe

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
`data/s01_pilot/rebalance.json`: P0 = (0.652, 0.348), weights (0.766, 1.438). Those commands wrote the whole-document
form, which today's detector reads only through the rebalancing stage. A rerun of the probe goes to a `.jsonl` file.
