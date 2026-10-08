# 2026-10-08 — s02 run 1: the evaluation plan, fixed before any of it runs

Written at 13:20 EDT on 2026-10-08, with s02_r1 at about 560k of an expected ~640k steps (it ends near 14:50) and
before any selection, averaging or gate result exists. Its commit time is the pre-registration.

## What is known when this is written

Only the eval watch: 20 deterministic episodes per 50k checkpoint on the training-time evaluation seeds (200e6), under
the task's rules (the 0.9 m footprint rule, the 1–3 m band): 0.00, 0.35, 0.50, 0.55, 0.70, 0.45, 0.60, 0.60, 0.35,
0.45 at 50k–500k. Footprint endings are the commonest failure (74 of 200 watched flights), 56 of them on the three
leaning species. Twenty-episode points swing too much to rank checkpoints (MEMORY.md, run 4), so none of them is used
below.

## The plan (a deliberate shortening of runbook-m5 §4)

The runbook's per-checkpoint stages 1 and 2 (about 3.5 h on five slots) exist to find a single-checkpoint winner. On
s01d, run 6, the widest checkpoint mean beat that winner on fresh episodes (0.94 against 0.83, p = 0.035) and was the
best gated model; in session 1 the widest mean was again the best of four. So run 1 skips the per-checkpoint search
and compares three pre-named candidates directly:

| Candidate | What |
|---|---|
| `avg_200k_end` | the mean of every 10k checkpoint from 200k to the last (the eval watch first reached 0.55 at 200k) |
| `avg_350k_end` | the mean of every 10k checkpoint from 350k to the last (the late half) |
| `final` | `runs/expert/s02_r1/final.zip` |

1. **Comparison.** Each candidate flies the same 100 fresh validation episodes, deterministic, seed base 500,002,000
   (never flown on s02), one slot each, under the task's rules (`scripts/m2_gate.py`, outputs in
   `runs/expert/s02_r1/selection/stage3_<name>.json`).
2. **Rule.** The candidate with the highest success rate is gated; a tie goes to the wider window. No other candidate
   is added after any result is seen.
3. **Gate.** The pick flies the usual 200 gate episodes at 100e6 once: `docs/gates/m5_s02_gate.json`, then the audit
   (`m5_s02_gate_audit.json`). The bar is 0.95 deterministic. A failed gate stays failed; what follows (another run,
   a changed rule, or acceptance) is the user's choice, and the reasons go in the closeout.
4. **Renders** of 8 gate episodes mixing successes and each failure kind (`runs/viz/s02_r1_<pick>/`), for the status
   report.

If the footprint rule changes for run 2 (the clearance diagnostic of `scripts/footprint_endings_figure.py` decides
that), run 1's pick is also flown under the new rule on the same 200 episodes as a separately named record, so the
two runs are compared under one rule. That record does not replace the gate.

## Commands

```bash
J="bash scripts/run_job.sh"; PY="env -u PYTHONPATH .venv/bin/python"; RUN=runs/expert/s02_r1; CFG=scene_autofly_s02_fast.jsonc
LAST=$(ls $RUN/checkpoints | sed -n 's/^rl_model_\([0-9]*\)_steps\.zip$/\1/p' | sort -n | tail -1)
for FROM in 200000 350000; do
  $PY -m scripts.average_checkpoints $RUN/averaged/avg_$((FROM / 1000))k_end.zip \
      $(for s in $(seq $FROM 10000 $LAST); do echo $RUN/checkpoints/rl_model_${s}_steps.zip; done)
done
i=0; for M in avg_200k_end=$RUN/averaged/avg_200k_end.zip avg_350k_end=$RUN/averaged/avg_350k_end.zip final=$RUN/final.zip; do
  $J start s02_r1_cmp_$i -- env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 .venv/bin/python -m scripts.m2_gate \
      --scene s02 --scene-config $CFG --conditions deterministic --episodes 100 --seed-base 500002000 --instance $i \
      --out $RUN/selection/stage3_${M%%=*}.json --model $M
  i=$((i + 1)); sleep 30
done
# then: the pick -> $J start m5_s02_gate -- $PY -m scripts.m2_gate --scene s02 --scene-config $CFG --episodes 200 \
#   --conditions deterministic --model <pick>=<path> --out docs/gates/m5_s02_gate.json
```

Addendum (2026-10-08, before the gate ran): the circle gate is flown on main as it stood before the outline patch (pure
circle, 0.9 m); after the patch, m2_gate reads s02_r1's identity and scores it under the circle without a flag. The
outline record of the pick, at the outline's 1.05 m, is docs/gates/m5_s02_gate_outline.json.
