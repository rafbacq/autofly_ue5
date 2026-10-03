"""scripts/eval_watch.py: score each new checkpoint of a live run with the gate's machinery on the 20 training-time
evaluation seeds, one simulator launched per evaluation and stopped after it. Training never pauses, and no idle
evaluation simulator renders all run long (it took a fifth of the GPU's time slices in s01d's runs)."""

from __future__ import annotations

import json

from autofly_ue5.expert.seeds import EVAL_CALLBACK_SEED_BASE


def _run(tmp_path, steps):
    root = tmp_path / "run"
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    for s in steps:
        (root / "checkpoints" / f"rl_model_{s}_steps.zip").write_bytes(b"x")
    return root


def _fake_gate(calls):
    def run_gate(argv):
        calls.append(argv)
        out = argv[argv.index("--out") + 1]
        name, path = argv[argv.index("--model") + 1].split("=", 1)
        with open(out, "w") as handle:
            json.dump({"status": "ok", "eval_seed_base": int(argv[argv.index("--seed-base") + 1]),
                       "checkpoints": {name: {"path": path, "deterministic": {
                           "status": "ok", "n_episodes": 20, "success_rate": 0.8, "mean_return": 60.0,
                           "collision_rate": 0.15, "out_of_bounds_rate": 0.05, "collision_sources": {"mover": 3}}}}},
                      handle)
        return 0
    return run_gate


def test_each_new_checkpoint_at_the_interval_is_scored_once_on_the_callback_seeds(tmp_path):
    from scripts.eval_watch import evaluate_pending

    root = _run(tmp_path, [10_000, 20_000, 30_000, 40_000, 50_000, 100_000])
    calls, lines = [], []
    done = evaluate_pending(root, every=50_000, episodes=20, instance=4, scene="s01d",
                            scene_config="scene_autofly_s01_fast.jsonc", run_gate=_fake_gate(calls), log=lines.append)
    assert done == [50_000, 100_000] and len(calls) == 2
    for argv in calls:
        assert argv[argv.index("--seed-base") + 1] == str(EVAL_CALLBACK_SEED_BASE)
        assert argv[argv.index("--instance") + 1] == "4" and "--conditions" in argv
        assert argv[argv.index("--out") + 1].startswith(str(root / "eval_watch"))
    assert any("EVALWATCH step 50000: success 0.80" in line for line in lines)
    assert evaluate_pending(root, every=50_000, episodes=20, instance=4, scene="s01d",
                            scene_config="scene_autofly_s01_fast.jsonc", run_gate=_fake_gate(calls), log=lines.append) == []
    assert len(calls) == 2, "an evaluated checkpoint is never evaluated again"


def test_a_failed_evaluation_is_reported_and_retried_next_time(tmp_path):
    from scripts.eval_watch import evaluate_pending

    root = _run(tmp_path, [50_000])
    lines = []
    assert evaluate_pending(root, every=50_000, episodes=20, instance=4, scene="s01d",
                            scene_config="c.jsonc", run_gate=lambda argv: 1, log=lines.append) == []
    assert any("EVALWATCH step 50000: FAILED" in line for line in lines)
    calls = []
    assert evaluate_pending(root, every=50_000, episodes=20, instance=4, scene="s01d", scene_config="c.jsonc",
                            run_gate=_fake_gate(calls), log=lines.append) == [50_000]
