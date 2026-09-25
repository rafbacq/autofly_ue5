# M2 run 1 (2026-09-16/17): archived evidence

The first SAC expert run on s01, exactly as recorded (moved here unedited, from `docs/gates/` at commit 68f221a):

| File | What it records |
|---|---|
| `m2_env_manifest.json` | the Python environment the run was measured on |
| `m2_instances.json` | the throughput measurement (Task 7) |
| `m2_train.json` | the 12 h training run (Task 8) |
| `m2_gate.json` | the 200-episode gate (Task 9); `pass: false`, with deterministic success of 0.83 (best_model) and 0.90 (final) |

`m2_gate_audit.json` was added afterwards: `scripts/audit_m2_gate.py` found 15 physically impossible episodes in that
gate.

Read these together with `docs/decisions/2026-09-25-code-review-findings.md`. It explains which of these numbers the
2026-09-24 review showed to be self-inflicted (C1), mislabelled (C4) or not policy failures at all (C9). It also
explains why the next run (reward version 2) starts fresh rather than resuming this one.
