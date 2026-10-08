# 2026-10-07 — s01d: the 0.935 expert is accepted for collection; the M2d gate stays failed

## The decision

The user decided on 2026-10-07: **accept the 0.935 s01d expert** and do not start a run 7. Spec §12's M2d gate
(deterministic success ≥ 0.95 over 200 held-out episodes) **stays failed**: every record in `docs/gates/` stands as
written, no threshold moves, and nothing is re-gated. What changes is that s01d now has an expert for M5's collection,
chosen the way s01's was (`2026-10-02-m2-closeout.md`): the best gated checkpoint, flown **stochastically**.

## The accepted expert

| | |
|---|---|
| File | `runs/expert/s01d_r6/averaged/avg_250k_550k.zip` |
| sha256 | `55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd` (checked on disk 2026-10-07, equal to the gate record's) |
| What it is | The parameter mean of run 6's 31 checkpoints from 250k to 550k steps (`scripts/average_checkpoints.py`; sources and their hashes in `docs/gates/m2d_r6_swa_sources.json`) |
| Observation | 3 stacked float16 depth frames, 4 mover slots (`obs_config` in the gate record) |
| Scene config | `configs/scene_autofly_s01_fast.jsonc` (the 1 ms clock the gate flew) |
| Reward version | `3-scene-reward-overrides` |
| Gate | `docs/gates/m2d_r6_swa_gate.json`: **187/200 = 0.935** deterministic (CI 0.892–0.962), audit `m2d_r6_swa_gate_audit.json` flagged 0 |
| Its 13 failures | 8 mover contacts (every one a near miss, 0.90–1.00 m from the surface), 1 physical collision, 2 above the ceiling, 2 lateral exits |

Why this one and not the later mean. Session 1's pre-registered pick (the 250k–1,060k mean, `avg_250k_end.zip`)
gated 0.920 and confirmed 0.930 on 200 unflown episodes: 370/400 = 0.925 (CI 0.895–0.947). On the gate's own 200
episodes the two means are level (10 won, 13 lost, p = 0.68). The 250k–550k mean holds the best gated number, so it
is the one accepted; the user may swap in the wider mean later without any new evidence being needed, since both
are already gated on the same episodes.

## How it is flown

- **Stochastic actions**, as spec §9 step 5 says and as s01's `best_model` flies M3 (its stochastic gate score, 0.99,
  was at least its deterministic one, 0.98). This expert has **no stochastic gate measurement**: the swa gate ran the
  deterministic condition only. The collector keeps every reject with its reason, so the first s01d collection batch
  measures the stochastic kept-rate for free. If it falls clearly under the deterministic 0.93 (say below 0.85 over
  100 flown episodes), stop and look before scaling up; do not re-gate.
- **a0 aligned start** (the collector always applies sector8; there is no flag), as for s01 (`2026-10-03-m3-a0-and-collection.md`).
- The task's own rules: physical contact, the 1.0 m mover contact rule, the 1–3 m band. No training margin or
  penalty is applied at collection (the env applies them only when asked, and the collector never asks).
- Episodes that end in a mover contact are rejects (collision), like any other collision.

## What this means for the dataset

About 4.5 % of s01d flights end in a mover near miss and about 2 % leave the bounds or touch a pillar, so roughly
one flown episode in fourteen is rejected. That is a throughput cost (about 7 % more flights per kept episode), not a
data-quality cost: only successful, collision-free episodes enter the dataset (spec §9 step 6).

## The levers left untried, should near misses matter later

Recorded so a future run 7 starts from the evidence, not from memory (`2026-10-06-s01d-r6-plan.md` has the data):

1. **`--mover-closing-penalty K MARGIN_M`** (implemented in b40163e, never trained): charges for closing in on a
   mover rather than for being near one. The three replayed contacts of the mean were full-rate turns that closed on
   a yielding mover, which is what this prices.
2. **A lower SAC entropy target** (the default is −dim(A) = −3): the stochastic policy training optimises would fly
   more like the deterministic one the gate scores. Near movers the deterministic policy saturates its yaw command
   twice as often as elsewhere.
3. **A lateral training margin**: worth the 1–1.5 % of episodes whose straight route runs along the field's edge
   (every lateral exit replayed; the targets sit a median 1.3 m from the edge, inside the 5 m success radius, so
   none would become unreachable).

Each is a fresh 12 h run from scratch (never a warm start with a fresh buffer: run 2), plus about 4 h of selection
and gating, with checkpoint averaging kept. None has evidence yet that it removes the mover contacts, which stayed at
4–5 % through a 1.3 m then 1.5 m training boundary, a clearance penalty, averaging and 12 more hours.

## What does not change

- `docs/gates/m2d_*.json`: untouched. The gate records say "pass: false" and keep saying it.
- CLAUDE.md's status row reads "gate failed; accepted by the user at 0.935 for collection".
- s01d's observation space, scene file and layout: every run 6 checkpoint keeps loading.
