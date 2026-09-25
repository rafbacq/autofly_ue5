"""scripts/audit_m2_gate.py: an episode cannot end farther from (or nearer to) its target than it could fly."""

import math

import numpy as np

from tests.test_expert_episode import scene_and_layout


def _start_distance(seed: int) -> float:
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    s = sample_setup(scene, layout, np.random.default_rng(seed))
    return math.hypot(s.target_xy_z[0] - s.start.x, s.target_xy_z[1] - s.start.y)


def _episode(seed, outcome, steps, final_distance_m):
    return {"seed": seed, "outcome": outcome, "steps": steps, "final_distance_m": final_distance_m,
            "is_success": outcome == "success"}


def test_the_audit_flags_an_episode_that_moved_farther_than_it_could_fly():
    from scripts.audit_m2_gate import audit_gate

    d = {s: _start_distance(s) for s in (10, 11, 12)}
    episodes = [
        _episode(10, "collision", 50, d[10] - 15.0),  # 15 m in 50 steps: possible (limit 20 m)
        _episode(11, "collision", 1, d[11] - 12.0),   # 12 m in one step: impossible
        _episode(12, "success", 200, 4.9),        # <= 80 m reachable in 200 steps
    ]
    gate = {"checkpoints": {"final": {"deterministic": {"per_episode": episodes}, "stochastic": {"per_episode": []}}}}
    scene, layout = scene_and_layout()
    audit = audit_gate(gate, scene, layout)

    flagged = audit["flagged"]
    assert [(f["seed"], f["preceding_outcome"]) for f in flagged] == [(11, "collision")]
    excluded = audit["success_rate_excluding_flagged"]["final"]["deterministic"]
    assert excluded == {"successes": 1, "episodes": 2, "rate": 0.5}
    assert audit["after_a_collision"] == {"final:deterministic": {"flagged": 1, "resets": 2}}
