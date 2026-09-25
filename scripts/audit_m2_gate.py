"""Audit a recorded M2 gate run for physically impossible episodes (C9, 2026-09-24 review).

Each gate episode is reconstructed from its seed (`sample_setup` is deterministic), giving its start distance to the
target. The drone flies at most 2 m/s, 0.4 m per 0.2 s step, so no episode can end more than 0.4 m x steps (+0.5 m
slack) nearer to or farther from its target than it started; one that does began somewhere other than its start.
The 2026-09-17 gate had 15 such episodes, every one a one-step "collision" right after a collision episode.

env -u PYTHONPATH .venv/bin/python scripts/audit_m2_gate.py --gate docs/gates/m2_gate.json --out <audit.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from autofly_ue5.expert.episode import sample_setup  # noqa: E402
from autofly_ue5.expert.train import scene_and_layout  # noqa: E402

MAX_STEP_M = 0.4  # 2 m/s x 0.2 s
SLACK_M = 0.5


def _start_distance(scene, layout, seed: int) -> float:
    setup = sample_setup(scene, layout, np.random.default_rng(seed))
    return math.hypot(setup.target_xy_z[0] - setup.start.x, setup.target_xy_z[1] - setup.start.y)


def audit_gate(gate: dict, scene, layout, max_step_m: float = MAX_STEP_M, slack_m: float = SLACK_M) -> dict:
    flagged: list[dict] = []
    excluded: dict[str, dict] = {}
    after_collision: dict[str, dict] = {}
    after_other: dict[str, dict] = {}
    worst_legit_ratio = 0.0
    for name, checkpoint in gate["checkpoints"].items():
        for condition in ("deterministic", "stochastic"):
            episodes = (checkpoint.get(condition) or {}).get("per_episode") or []
            combo = f"{name}:{condition}"
            flagged_here = []
            previous = None
            for episode in episodes:
                d0 = _start_distance(scene, layout, episode["seed"])
                change = abs(episode["final_distance_m"] - d0)
                reach = max_step_m * episode["steps"]
                impossible = change > reach + slack_m
                if impossible:
                    record = {"checkpoint": name, "condition": condition, "seed": episode["seed"],
                              "outcome": episode["outcome"], "steps": episode["steps"],
                              "start_distance_m": round(d0, 3), "final_distance_m": round(episode["final_distance_m"], 3),
                              "distance_change_m": round(change, 3), "preceding_outcome": previous}
                    flagged.append(record)
                    flagged_here.append(episode["seed"])
                else:
                    worst_legit_ratio = max(worst_legit_ratio, change / reach if reach else 0.0)
                if previous is not None:
                    bucket = after_collision if previous == "collision" else after_other
                    tally = bucket.setdefault(combo, {"flagged": 0, "resets": 0})
                    tally["resets"] += 1
                    tally["flagged"] += int(impossible)
                previous = episode["outcome"]
            kept = [e for e in episodes if e["seed"] not in flagged_here]
            successes = sum(1 for e in kept if e.get("is_success"))
            excluded.setdefault(name, {})[condition] = {
                "successes": successes, "episodes": len(kept), "rate": (successes / len(kept)) if kept else None,
            }
    return {
        "rule": f"|final_distance - start_distance| > {max_step_m} m x steps + {slack_m} m",
        "flagged": flagged,
        "n_flagged": len(flagged),
        "worst_legitimate_ratio": round(worst_legit_ratio, 4),
        "success_rate_excluding_flagged": excluded,
        "after_a_collision": after_collision,
        "after_other_outcomes": after_other,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", default=None, help="default: the gate record's own scene")
    args = parser.parse_args(argv)
    gate = json.loads(args.gate.read_text())
    scene, layout = scene_and_layout(args.scene or gate["scene"])
    audit = audit_gate(gate, scene, layout)
    commit = subprocess.run(["git", "-C", str(_ROOT), "log", "-1", "--format=%H", "--", str(args.gate.resolve())],
                            capture_output=True, text=True).stdout.strip() or None
    audit = {
        "description": "C9 audit: gate episodes that could not have been flown from their start (see the rule).",
        "source": str(args.gate),
        "source_sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
        "source_last_commit": commit,
        **audit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"n_flagged": audit["n_flagged"], "success_rate_excluding_flagged": audit["success_rate_excluding_flagged"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
