# 2026-10-03 — M3: how a0 is handled, and the pilot's two other open choices

The user asked me (2026-10-03) to decide M3's a0 handling as I recommend, and to build M3. This record settles Plan 4's
three open questions, U1–U3. Evidence: `scripts/decode_state.py` on the two real released episodes
(`/home/nvidiasims/research_uav/results/qwen_autofly/data_smoke_v1/`), and the decision record's own checks below.

## U1 — a0 is an unrecorded coarse alignment of the starting heading

**Decision.** Before an episode is recorded, the drone's start yaw is set to the bearing of the target from the start,
rounded to the nearest of 8 sector centres (multiples of 45°). So the drone starts within ±22.5° of facing the target.
The record begins with the expert's own first action. No synthetic turn step is recorded.

**Evidence.**

1. **No turn in place at the start.** In time order (state[0] decreasing), both episodes' first actions fly forward
   at 1.98–1.99 m/s with yaw rates of −0.06 / −0.05 rad/s. Spec §3.2's "turn in place first" came from file row 0,
   which is time step 75 of 96 and 74 of 78. The export lists steps in a fixed order of its own (Plan 4, finding 1).
2. **The starts already face the target.** The first recorded state's state[1] is 0.374 rad (21.4°) and 0.065 rad
   (3.7°). The heading of motion over the first three steps is 25.8° and 8.4° off the bearing to the target
   (trilaterated from state[0]). Both are consistent with a coarse alignment of up to 22.5°. A random start yaw
   (spec §9.2) would put only one start in four within 45°.
3. **The paper's wording.** "Coarse positional or directional guidance is encoded as an initial action a0" (Sec. 3.1).
   Read with the data, the guidance is the heading the episode starts with: applied before the first recorded frame,
   not recorded as a step.

**Caveats.** Two episodes. state[1] is not verified as a bearing (|r| 0.92 / 0.88, under the spec's 0.95 bar). The
real starts are already moving (state[3] 0.80 and 0.68 m/s), while ours start from a hover. 8-sector rounding is a
choice; a proportional turn that stops within ~25° would fit the data as well.

**What changes.**
- Spec §9 step 4 now reads as above.
- Collection only: `AutoFlyEnv.reset(options={"a0": "sector8"})` overrides the start yaw after sampling, so a seed
  still draws the same episode (start position, target, distractors, movers). Training, the gates and static s01's
  golden record are unchanged.
- The SAC experts were trained on random start yaws, so an aligned start is inside their training distribution.

**Cost if wrong.** Our dataset's episodes start slightly better aligned than AutoFly's. A re-collection with a recorded
turn needs no change to the record format.

## U2 — the two real episodes stay out of git

Their release's licence is unstated (spec §8.1). Tests that need them read them from the host path above and skip
where it is absent (`tests/test_decode_state.py` already does).

## U3 — the s01 pilot names its target "orange cylinder"

Until M4's 60-target pool exists, Plan 2's D1 had every instruction end in "…reach the target". The pilot's target
really is an orange cylinder (the `Cylinder` mesh with `M_Orange`), and its distractors are unpainted cylinders, so the
name is accurate and unambiguous. It follows the release's "colour + category" names ("blue hatchback", "blue cone").

- Only the dataset's instruction uses it. The expert never sees instructions, and `sample_setup`'s placeholder stays,
  so the golden record holds.
- The dataset card marks it a placeholder until M4.

## State[9] as M3 writes it (spec §10.1)

Only state[2] cleared the spec's bar. Every field is written, with its status recorded in the dataset card:

| Index | Our definition | Status |
|---|---|---|
| 0 | Horizontal distance to the target's centre, m | spec §3.2 (strong) |
| 1 | Bearing to the target relative to the drone's heading, rad, wrapped to ±π | `not_autofly_verified` (r 0.92 / 0.88 against a motion-based estimate) |
| 2 | Altitude − 1.47 m | adopted, \|r\| 0.986 / 0.996; fits 0.985·z − 1.470 and 0.987·z − 1.477 |
| 3 | Speed, the norm of the velocity, m/s | spec §3.2 (strong) |
| 4 | Vertical velocity, up positive, m/s | `not_autofly_verified` (r 0.77 / 0.60) |
| 5 | Yaw rate, rad/s | `not_autofly_verified` (r 0.54 / 0.76) |
| 6, 7 | x, y relative to the episode's start, m (the release's episodes start within 0.25 m of (0, 0)) | spec §3.2 |
| 8 | Altitude, m (z up) | spec §3.2 |
