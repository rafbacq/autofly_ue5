# 2026-10-03 — M3 code review: findings and fixes

A fresh-eyes review of M3 (commits 9e55b63..91932d7) ran before the 100-episode pilot. It traced the data path and
ran experiments against FakeSimulator.

**The data path held.** Record t's frame, state[9], simulator time and pose all come from the observation the
expert acted on, and its action is the command the env flies (the same float32 clip). On a fake run, the
displacement from record t to t+1 matched action[t] · 0.2 s within 4e-6 m. The state[9] signs and units match the
a0 decision's table. a0 changes only the start yaw, so the RNG stream is unchanged and static s01's golden test still
holds. A fault-cut attempt is never stored. The protobuf field numbers and wire types, and the TFDS metadata, are
correct.

| # | Finding | Fix |
|---|---|---|
| 1 | A pilot that crashed partway lost its counts and its export, and was filed under `runs/not_started/`. So was a pilot whose every episode was rejected | 5da3e41: counts survive exceptions; any flown episode makes the record evidence |
| 2 | The dataset name was claimed before the launch, so a launch refused by the GPU guard blocked retries under that name | 5da3e41: nothing is created until the first episode or reject, then the claim is exclusive |
| 3 | Provenance lacked target and distractor names, and the placeholder target name (U3) was not marked | 58bd600: asset, material and spawned name for each object; `card.target_names` |
| 4 | Neither the gate nor the pilot checked that its slot was free | a382c2c: `process.slot_busy`, refused before anything touches the slot |
| 5 | The gate could pass without a usable RLDS export | 5da3e41: the pass rule needs the export, and the TFDS read-back when one is requested |
| 6 | A raising validator or audit left no record; a wrong-shaped episode crashed the validator | 5da3e41, 000ceb8 |
| 7 | The instruction check matched a wildcard, so Plan 2's placeholder passed; provenance was compared by length only | 000ceb8: exact template with the episode's target and the scene's obstacles; seed and times compared by value |
| 8 | The final record write was not exclusive | 5da3e41: a clash goes to a `.conflict-<time>` sibling |
| 9 | Every collection run used the same seeds, though `seeds.py` promised one slice per run | 3ac2c15: `--seed-slice`; slice 0 is s01's pilot, 99 is for smokes |
| 10 | The 1 mm start check is far tighter than reset's 0.3 m acceptance | **Kept.** Live resets land about 1e-6 m off horizontally (30 crash resets, 6 smoke episodes), and one step later the drone is already 5–11 mm away. A reset that lands millimetres off would be an anomaly the validator should report |
| — | The shards were not at spec §10.1's `data/<dataset_name>/1.0.0/` | 0386ccf |

Live re-check after the fixes (`runs/m3_smoke/m3_gate_smoke2.json`, slot 5, seed slice 99): 3 of 3 episodes kept,
the validator passed, and TFDS read every episode back exactly. A stored last frame shows the orange target in true
colour, so RGB order is right end to end.
