# M3 Collector and Dataset Plan (draft for the user's go-ahead)

> **Status:** draft, written 2026-10-02 once Plan 3's offline work had landed. Nothing in it has been implemented.
> Steps use checkbox (`- [ ]`) syntax; each task is test-first against `FakeSimulator`, like Plans 2 and 3.

**Goal (spec §12, M3):** a collector, a dataset writer, a validator and the state[9] decoding. The gate is a
100-episode **s01** pilot that passes the validator → `docs/gates/m3_gate.json`. An s01d pilot follows a passed M2d.

**Spec:** §9 (episode protocol), §10 (dataset, provenance), §11 (validation). The M2 closeout fixes the pilot:
`best_model` of run 2, flown **stochastically**.

## Facts found while planning (2026-10-02, from the two real episodes on this host)

The two released episodes are at `/home/nvidiasims/research_uav/results/qwen_autofly/data_smoke_v1/`:

- `train.jsonl`: one episode, 96 steps, `UAV_VLA_Scene_1_zip/Blue_Hatchback`;
- `val.jsonl`: one episode, 78 steps, `UAV_VLA_Scene_7_zip/Blue_Cone`;
- PNG frames, and an `audit.json` with the source shards' sha256.

No copy of the release's `features.json` or `dataset_info.json` exists on this host. Spec §8.1 notes that ModelScope
still hosts the TFDS dataset, so those two small metadata files can be fetched from it. A scratch analysis (numpy
only) found:

1. **The file order is one fixed permutation, shared by both episodes.** Sorting by state[0] (distance to target,
   decreasing) recovers time order: the median step is 0.35–0.41 m, against 13–18 m in file order. Sorted that way,
   file row *i* lands on nearly the same time step in both episodes (75/74, 36/35, 3/3, 42/41, 51/50, 10/10, …), so
   the true order is recoverable exactly, not just approximately.
2. **Spec §3.2's "first action is a turn in place" comes from file row 0, which is not the first step.** File row 0
   (forward 0.03 / 0.16 m/s, yaw rate 0.95 / 0.97) is time step 75 of 96 and 74 of 78, with speed state 1.86 / 1.84.
   In time order, **both episodes start flying forward at about 1.98 m/s**. So the evidence for the a0 rule of §9
   step 4 is gone, and **M3 must settle a0 again** before the collector records anything at step 0 (task B1).
3. **state[6..8] are relative to the episode's start.** Both start within 0.25 m of (0, 0) horizontally, at
   z ≈ 2.04–2.07, with z up.
4. **state[2] ≈ 0.985·z − 1.47** in both episodes (fit residual 0.03–0.04; |r| = 0.986 and 0.996). It is the only
   unknown field that already clears the spec's 0.95 bar on both. It reads as height relative to about 1.47 m, but
   needs testing against the target's height.
5. **state[1]** ~ bearing to the target relative to the direction of motion: r = 0.92 / 0.88, with the target
   trilaterated from state[0] (residual 0.13 / 0.02 m). Below the bar, but the motion direction is a noisy stand-in
   for yaw.
6. **state[4]** ~ vertical velocity (r = 0.77 / 0.60). **state[5]** ~ yaw rate (r = 0.54 / 0.76). Both are estimated
   from position differences, which smoothing makes worse. They are probably instantaneous simulator values the
   positions cannot recover.

These are preliminary, from a throwaway script. Task B1 makes them reproducible and decides them by the spec's rule.

## Decisions needed from the user before B3

- **U1: a0.** Options: (a) drop the a0 setup step: the time-ordered data shows none at the start; (b) keep §9
  step 4's turn-in-place a0; (c) record whatever B1's exact ordering shows. *Recommendation: decide after B1.*
- **U2: the real episodes in the test suite.** Copy their numeric fields (174 rows; no images) into
  `tests/fixtures/`, or keep reading them from the host path and skip the tests elsewhere? The release's licence is
  unstated (spec §8.1). *Recommendation: keep them out of git.*
- **U3: the s01 pilot's target name.** Until M4's pool exists, Plan 2 D1 named the target literally "target". The
  pilot's spawned target is an orange cylinder. *Recommendation: name it "orange cylinder" in instructions, and
  record that as a placeholder in the dataset card.*

## Tasks

- [ ] **B0. Seeds.** Add `COLLECTION_SEED_BASE` (400e6) and move the probes' 300e6 into `expert/seeds.py`, both
  under the disjointness test. Collection never reuses a gate or training seed.
- [ ] **B1. `scripts/decode_state.py`.**
  - Recover the exact step order: state[0] order, checked against the shared permutation of finding 1.
  - Test each hypothesis for state[1], [2], [4], [5]:
    - body-frame velocities;
    - roll, pitch, yaw and yaw rate;
    - bearing to the target and its sin/cos;
    - height above the start, or relative to the target's centre or surface.
  - Adopt a hypothesis only at |r| ≥ 0.95 on both episodes (spec §10.1). Anything else gets a documented definition
    and is marked `not_autofly_verified`.
  - Settle a0 (U1), and whether the ≈ 7.2 m final distance is measured to the target's surface or its centre
    (§13).
  - Output: `docs/gates/m3_state_decoding.json`.
- [ ] **B2. `dataset/state.py`.** Compute state[9] from an `Observation` and the episode setup, with B1's
  definitions, unit-tested against hand-computed poses.
- [ ] **B3. `collect/`.** The collector flies `AutoFlyEnv` (resilient) with the stochastic `best_model`.
  - Add `AutoFlyEnv.last_observation` for the RGB frame. It is never an expert input.
  - a0 per U1. With movers, a0 goes through `env.step`, so they advance during it.
  - Keep success-only episodes. Every other episode goes to `data/rejects/` with its reason.
  - Backend faults replay the seed, as the gate does.
- [ ] **B4. Raw store (`dataset/raw.py`).** Per episode: PNG frames, `steps.npz` (state, action, simulator time) and a
  provenance JSON (spec §10.2: scene and sha, seed, setup including mover routes and per-step poses, checkpoint and
  sha, platform/engine/package hashes, termination reason). Plus `manifest.json` with splits and counts. This store
  is canonical; every export derives from it.
- [ ] **B5. RLDS TFRecord exporter (`dataset/rlds.py`)**, mirroring `uavvlasplit_*_dataset/1.0.0`, without adding
  TensorFlow to the numpy-1.26.4 venv:
  - TFRecord framing with masked CRC32C; tensorboard's record writer, checked present in tensorboard 2.21 first;
  - a hand-written `tf.train.Example` protobuf encoder;
  - `features.json` and `dataset_info.json` copied in structure from the release (prerequisite above);
  - read back once with TensorFlow/TFDS in a throwaway venv, outside `.venv`.
- [ ] **B6. `validate/dataset.py` (§11).**
  - action and state ranges against §3.2;
  - speed and altitude distributions against the real episodes;
  - no NaN; images decode at 256×256;
  - time order, and exactly one 0.2 s simulator step per record;
  - episode lengths; per-scene and per-target counts; provenance complete.
- [ ] **B7. Live pilot.** 100 s01 episodes → raw store → validator → RLDS export → `docs/gates/m3_gate.json`. About
  30 minutes of flying at N = 1, plus rejected episodes and resets: 100 × ~180 steps at the 11.5 steps/s that
  `docs/gates/m2_instances.json` measured for one instance. It can run beside M2d's training only if VRAM and the
  launch guard allow; otherwise after it.

## Not in M3

Rebalancing (§10.3, M5), M4's target pool and assets (`docs/decisions/2026-10-02-m4-asset-survey.md`), and the
optional AutoFly-checkpoint pilot (§8.1; still inactive).
