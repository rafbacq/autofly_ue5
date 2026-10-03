"""scripts/select_checkpoint.py: which checkpoints are candidates, how they are shared across slots, and how the
gate-format results are ranked (validation success first, then mean return; an unfinished one last, never dropped)."""

from __future__ import annotations

import json

import pytest

from autofly_ue5.expert.seeds import EVAL_SEED_BASE, SELECTION_SEED_BASE


def _run(tmp_path, steps=(10_000, 20_000, 30_000, 40_000, 50_000, 60_000), best=True, final=True):
    root = tmp_path / "run"
    (root / "checkpoints").mkdir(parents=True)
    for s in steps:
        (root / "checkpoints" / f"rl_model_{s}_steps.zip").write_bytes(b"x")
        (root / "checkpoints" / f"rl_model_replay_buffer_{s}_steps.pkl").write_bytes(b"x")
    if best:
        (root / "best").mkdir()
        (root / "best" / "best_model.zip").write_bytes(b"x")
    if final:
        (root / "final.zip").write_bytes(b"x")
    return root


def test_candidates_are_the_periodic_checkpoints_from_a_floor_plus_best_and_final(tmp_path):
    from scripts.select_checkpoint import candidates

    chosen = candidates(_run(tmp_path), every=20_000, min_steps=20_000)
    assert sorted(chosen) == ["best_model", "final", "step_20000", "step_40000", "step_60000"]
    assert candidates(_run(tmp_path / "b"), every=20_000, min_steps=0, names=["step_30000", "final"]) == {
        "step_30000": tmp_path / "b" / "run" / "checkpoints" / "rl_model_30000_steps.zip",
        "final": tmp_path / "b" / "run" / "final.zip"}
    with pytest.raises(ValueError):
        candidates(_run(tmp_path / "c"), every=10_000, min_steps=0, names=["step_70000"])


def test_the_plan_deals_candidates_round_robin_onto_validation_seeds_never_the_gate_s(tmp_path):
    from scripts.select_checkpoint import candidates, plan_commands

    root = _run(tmp_path)
    parts = plan_commands(root, candidates(root, every=10_000, min_steps=0), slots=[0, 1, 2, 3], episodes=40,
                          scene="s01d", scene_config="scene_autofly_s01_fast.jsonc")
    names = [n for part in parts for n in part["names"]]
    assert sorted(names) == sorted(candidates(root, every=10_000, min_steps=0)) and len(names) == len(set(names))
    assert [len(part["names"]) for part in parts] == [2, 2, 2, 2]
    for part in parts:
        cmd = part["command"]
        assert f"--seed-base {SELECTION_SEED_BASE}" in cmd and str(EVAL_SEED_BASE) not in cmd
        assert "--conditions deterministic" in cmd and f"--instance {part['slot']}" in cmd
        assert part["out"].startswith(str(root / "selection")), "into the run's own directory, never docs/gates"


def _record(rows):
    return {"checkpoints": {name: {"path": f"/{name}.zip", "sha256": "0" * 64, "deterministic": det}
                            for name, det in rows.items()}}


def test_ranking_puts_the_best_validation_success_first_then_return_and_unfinished_last():
    from scripts.select_checkpoint import rank

    ok = lambda s, r: {"status": "ok", "n_episodes": 40, "success_rate": s, "mean_return": r}  # noqa: E731
    ranking = rank([_record({"step_20000": ok(0.80, 60.0), "step_40000": ok(0.95, 50.0)}),
                    _record({"final": ok(0.95, 62.0), "best_model": {"status": "failed", "n_episodes": 12}})])
    assert [r["name"] for r in ranking] == ["final", "step_40000", "step_20000", "best_model"]
    assert ranking[0]["steps"] is None and ranking[1]["steps"] == 40_000
    assert ranking[-1]["success_rate"] is None and ranking[-1]["status"] == "failed"


def test_rank_writes_the_ranking_and_says_when_a_part_is_missing(tmp_path, capsys):
    from scripts.select_checkpoint import main

    root = _run(tmp_path)
    assert main(["plan", "--run-root", str(root), "--every", "20000", "--slots", "0", "1", "--episodes", "40",
                 "--scene", "s01d", "--scene-config", "scene_autofly_s01_fast.jsonc"]) == 0
    plan = json.loads((root / "selection" / "stage1_plan.json").read_text())
    first = plan["parts"][0]
    with open(first["out"], "w") as handle:
        json.dump(_record({n: {"status": "ok", "n_episodes": 40, "success_rate": 0.9, "mean_return": 55.0}
                           for n in first["names"]}), handle)
    assert main(["rank", "--run-root", str(root)]) == 1, "a part is missing"
    ranking = json.loads((root / "selection" / "stage1_ranking.json").read_text())
    assert ranking["missing_parts"] == [plan["parts"][1]["out"]] and len(ranking["ranking"]) == len(first["names"])


def test_a_plan_outside_the_validation_range_is_refused(tmp_path):
    from scripts.select_checkpoint import main

    assert main(["plan", "--run-root", str(_run(tmp_path)), "--slots", "0", "--episodes", "40", "--seed-offset",
                 "999990", "--scene", "s01d", "--scene-config", "scene_autofly_s01_fast.jsonc"]) == 2
