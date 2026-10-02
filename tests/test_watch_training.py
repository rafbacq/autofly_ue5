"""scripts/watch_training.py reads a run's own files and summarises it; checked on a synthetic run directory."""

from __future__ import annotations

import json

import numpy as np


def _run(tmp_path):
    root = tmp_path / "run"
    (root / "monitor" / "session0").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "eval_logs").mkdir()
    identity = {"scene": "s01d", "obs_config": {"depth_frames": 3, "depth_dtype": "float16"}}
    (root / "sessions.json").write_text(json.dumps({"sessions": [
        {"index": 0, "started": "2026-10-02 18:00:00", "resume": False, "identity": identity}]}))
    rows = ["#{\"t_start\": 0}", "r,l,t,is_success,outcome,collision_source"]
    for i in range(10):
        outcome = ("success", "collision", "collision", "out_of_bounds")[i % 4]
        source = {"collision": "mover" if i % 8 == 1 else "sim"}.get(outcome, "")
        rows.append(f"{float(i)},{100 + i},{float(i)},{outcome == 'success'},{outcome},{source}")
    (root / "monitor" / "session0" / "0.monitor.csv").write_text("\n".join(rows) + "\n")
    for steps in (10000, 20000):
        (root / "checkpoints" / f"rl_model_{steps}_steps.zip").write_text("x")
    np.savez(root / "eval_logs" / "evaluations.npz", timesteps=[25000, 50000], results=[[1.0, 2.0], [5.0, 7.0]],
             ep_lengths=[[10, 10], [10, 10]], successes=[[False, True], [True, True]])
    log = tmp_path / "job.log"
    log.write_text("\n".join([
        "| time/              |          |", "|    fps             | 9.5      |", "|    total_timesteps | 21234    |",
        "FAULT instance 2: caught CameraPoseError during reset() (occurrence #1 ...)",
        "RELAUNCH instance 2: relaunching (this will be relaunch #1)",
        "MOVER instance 1: CameraPoseError: injected within 1.20 m of a mover: scored as a collision",
    ]) + "\n")
    return root, log


def test_the_watcher_summarises_episodes_evaluations_faults_and_progress(tmp_path):
    from scripts.watch_training import render_text, summarize

    root, log = _run(tmp_path)
    s = summarize(root, log, window=8, host=False)
    assert s["episodes_total"] == 10 and s["rolling"]["episodes"] == 8
    assert s["rolling"]["outcomes"]["success"] == 2 / 8 and s["rolling"]["collision_sources"] == {"mover": 1, "sim": 3}
    assert s["newest_checkpoint_steps"] == 20000
    assert [e["success"] for e in s["evaluations"]] == [0.5, 1.0]
    assert s["log"]["sb3"]["total_timesteps"] == 21234 and s["log"]["fault_names"] == {"CameraPoseError": 1}
    assert s["log"]["counts"]["relaunches"] == 1 and s["log"]["counts"]["mover_inferred"] == 1
    text = render_text(s)
    assert "scene s01d" in text and "3xfloat16" in text and "21,234" in text and "mover 1" in text
    assert s["log"]["finished"] is None
    log.with_suffix(".exit").write_text("0\n")  # what run_job.sh writes when the job ends
    assert summarize(root, log, host=False)["log"]["finished"] == "job job finished: exit=0"


def test_the_watcher_draws_its_progress_plot_and_tolerates_an_empty_run(tmp_path):
    from scripts.watch_training import main, plot, summarize

    root, log = _run(tmp_path)
    plot(summarize(root, log, host=False), tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 1000
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--run-root", str(empty), "--interval", "0"]) == 0
