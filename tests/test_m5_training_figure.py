"""scripts/m5_training_figure.py draws a training run from its own files: it orders every worker's Monitor episodes by
the time they ended, reads outcome and collision source from their columns, skips an eval-watch record still being
written, and writes only its --out and --json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import m5_training_figure as fig


def _monitor(path: Path, t_start: float, rows: list[tuple[int, float, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'#{{"t_start": {t_start}, "env_id": "None"}}', "r,l,t,is_success,outcome,collision_source"]
    for length, t, outcome, source in rows:
        lines.append(f"0.0,{length},{t},{outcome == 'success'},{outcome},{source}")
    path.write_text("\n".join(lines) + "\n")


def _watch(path: Path, step: int, outcomes: list[tuple[str, str | None]], status: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    per_episode = [{"seed": 200_000_000 + i, "outcome": o, "collision_source": s} for i, (o, s) in enumerate(outcomes)]
    det = {"status": status, "n_episodes": len(outcomes),
           "success_rate": sum(o == "success" for o, _ in outcomes) / len(outcomes), "per_episode": per_episode}
    path.write_text(json.dumps({"checkpoints": {f"step_{step}": {"deterministic": det}}}))


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "s02_r9"
    # Worker 1 started 100 s after worker 0, so its 5 s episode ends between worker 0's two.
    _monitor(run / "monitor/session0/0.monitor.csv", 1000.0, [(10, 50.0, "success", ""), (20, 200.0, "collision", "footprint_margin")])
    _monitor(run / "monitor/session0/1.monitor.csv", 1100.0, [(30, 5.0, "out_of_bounds", "")])
    _monitor(run / "monitor/session0/eval/0.monitor.csv", 1000.0, [(999, 1.0, "success", "")])  # not training
    _watch(run / "eval_watch/step_100.json", 100, [("success", None), ("collision", "footprint"), ("out_of_bounds", None), ("success", None)])
    _watch(run / "eval_watch/step_50.json", 50, [("collision", "footprint"), ("collision", "sim"), ("timeout", None), ("success", None)])
    _watch(run / "eval_watch/step_150.partial.json", 150, [("success", None)] * 4)
    _watch(run / "eval_watch/step_200.json", 200, [("success", None)] * 4, status="error")
    return run


def test_training_episodes_are_in_time_order_across_workers(tmp_path):
    eps = fig.training_episodes(_run(tmp_path))
    assert eps["outcome"] == ["success", "out_of_bounds", "collision"]
    assert eps["collision_source"] == ["", "", "footprint_margin"]
    assert list(eps["steps"]) == [10, 40, 60]


def test_rolling_rate():
    assert np.allclose(fig.rolling_rate(np.array([1, 0, 1, 1]), 2), [0.5, 0.5, 1.0])
    assert len(fig.rolling_rate(np.array([1, 0]), 3)) == 0


def test_eval_watch_skips_partial_and_unfinished_records(tmp_path):
    points = fig.eval_watch(_run(tmp_path))
    assert [p["step"] for p in points] == [50, 100]
    assert points[0] == {"step": 50, "episodes": 4, "success_rate": 0.25, "successes": 1, "footprint": 1,
                         "other_collisions": 1, "out_of_bounds": 0, "timeout": 1}
    assert (points[1]["footprint"], points[1]["out_of_bounds"], points[1]["successes"]) == (1, 1, 2)


def test_main_writes_only_its_outputs(tmp_path):
    run = _run(tmp_path)
    before = sorted(p for p in run.rglob("*"))
    out, js = tmp_path / "out/fig.png", tmp_path / "out/fig.json"
    record = tmp_path / "train.json"
    record.write_text(json.dumps({"num_timesteps": 60, "wall_clock_s": 7200.0, "env_steps_per_s": 1.5, "instances": 2,
                                  "outcome_histogram": {"success": 1, "collision": 1, "out_of_bounds": 1},
                                  "backend_faults": {"fault_counts": {"Timeout": 0}, "relaunch_count": 0}}))
    assert fig.main(["--run-root", str(run), "--out", str(out), "--json", str(js), "--train-record", str(record),
                     "--window", "2"]) == 0
    assert out.stat().st_size > 0
    summary = json.loads(js.read_text())
    assert summary["training"]["episodes"] == 3
    assert summary["training"]["collision_sources"] == {"footprint_margin": 1}
    assert summary["train_record"].startswith("60 steps in 2.0 h")
    assert sorted(p for p in run.rglob("*")) == before
