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


def test_a_shape_handed_to_the_watch_reaches_every_gate_and_none_reaches_none(tmp_path):
    # The footprint shape a run trains under (2026-10-08): its eval watch scores the same rule. Without one nothing is
    # passed, as every earlier watch's command line had it, and the gate reads the checkpoint's run itself.
    from scripts.eval_watch import build_arg_parser, evaluate_pending
    from scripts.m2_gate import build_arg_parser as gate_parser

    root = _run(tmp_path, [50_000])
    calls = []
    evaluate_pending(root, every=50_000, episodes=20, instance=4, scene="s02", scene_config="c.jsonc",
                     run_gate=_fake_gate(calls), log=lambda line: None, footprint_shape="circle")
    assert calls[0][calls[0].index("--footprint-shape") + 1] == "circle"
    assert gate_parser().parse_args(calls[0]).footprint_shape == "circle"
    other = _run(tmp_path / "other", [50_000])
    evaluate_pending(other, every=50_000, episodes=20, instance=4, scene="s02", scene_config="c.jsonc",
                     run_gate=_fake_gate(calls), log=lambda line: None)
    assert "--footprint-shape" not in calls[1]
    argv = ["--run-root", "r", "--job", "j", "--instance", "4", "--scene", "s02", "--scene-config", "c"]
    assert build_arg_parser().parse_args(argv).footprint_shape is None
    assert build_arg_parser().parse_args([*argv, "--footprint-shape", "circle"]).footprint_shape == "circle"


def _watch(tmp_path, monkeypatch, root, *argv, layout=None):
    """eval_watch.main() for one pass (the training job's exit file already there) on the test tree scene (or
    `layout`), with the gate faked: (exit code, the gate argv lists)."""
    import scripts.eval_watch as ew
    from tests.test_footprint_contact import _tree_resolved

    jobs = tmp_path / "runs_dir" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "train.exit").write_text("0")
    monkeypatch.setattr(ew, "RUNS_DIR", tmp_path / "runs_dir")
    monkeypatch.setattr(ew, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path, layout=layout))
    calls = []
    monkeypatch.setattr(ew, "_gate_subprocess", _fake_gate(calls))
    code = ew.main(["--run-root", str(root), "--job", "train", "--instance", "4", "--scene", "s99", "--scene-config",
                    "c.jsonc", "--poll-s", "0", *argv])
    return code, calls


def test_the_watch_scores_a_runs_checkpoints_under_the_shape_the_run_trains_under(tmp_path, monkeypatch, capsys):
    # The run's sessions.json names its shape (expert/run_shape.py): the watch passes it to every gate, unasked.
    from tests.test_run_shape import LEGACY, OUTLINE, run_root

    root = run_root(tmp_path, "s02_r1", LEGACY)
    code, calls = _watch(tmp_path, monkeypatch, root)
    assert code == 0 and len(calls) == 1 and calls[0][calls[0].index("--footprint-shape") + 1] == "circle"
    other = run_root(tmp_path, "s02_r2", OUTLINE)
    code, calls = _watch(tmp_path, monkeypatch, other)
    assert code == 0 and calls[0][calls[0].index("--footprint-shape") + 1] == "outline"
    code, calls = _watch(tmp_path, monkeypatch, run_root(tmp_path, "s02_r3", LEGACY), "--footprint-shape", "outline")
    assert code == 2 and calls == [] and "--footprint-shape outline contradicts" in capsys.readouterr().err


def test_a_watch_whose_run_names_no_shape_needs_the_flag_and_a_pillar_scene_never_does(tmp_path, monkeypatch, capsys):
    from tests.test_expert_episode import scene_and_layout

    root = _run(tmp_path, [50_000])  # checkpoints, but no sessions.json
    code, calls = _watch(tmp_path, monkeypatch, root)
    assert code == 2 and calls == [] and "pass --footprint-shape outline or" in capsys.readouterr().err
    code, calls = _watch(tmp_path, monkeypatch, root, "--footprint-shape", "circle")
    assert code == 0 and calls[0][calls[0].index("--footprint-shape") + 1] == "circle"
    code, calls = _watch(tmp_path, monkeypatch, _run(tmp_path / "pillars", [50_000]), layout=scene_and_layout()[1])
    assert code == 0 and len(calls) == 1 and "--footprint-shape" not in calls[0]


def test_a_watch_started_before_its_run_has_a_record_waits_for_it_instead_of_refusing(tmp_path, monkeypatch):
    # The runbook starts the watch beside the trainer, which writes sessions.json a few seconds in; until a checkpoint
    # is due there is nothing to score and nothing to refuse.
    root = tmp_path / "run"
    root.mkdir()
    code, calls = _watch(tmp_path, monkeypatch, root)
    assert code == 0 and calls == []
