"""scripts/collect_campaign.py (plan 6 step 2e): a collection campaign's plan (one scripts/collect_dataset.py command per
(scene, batch)), its launcher (one job per free slot via scripts/run_job.sh) and its status (each store's counts).
Offline: plans against stores and records in tmp_path (never the real data/ or runs/collect/), launches with a stubbed
job starter and harmless commands, scene layouts from the worktree's s01 build or the test tree scene."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from autofly_ue5.sim.process import proc_start_ticks


@pytest.fixture
def cli(monkeypatch):
    """The script's module, with s02 resolved to the test tree scene (renamed s02, so its base and config are s02's)
    and the job starter stubbed so nothing can reach scripts/run_job.sh by accident."""
    import scripts.collect_campaign as cli
    from autofly_ue5.scenes.resolve import resolve_scene
    from tests.test_footprint_contact import _tree_resolved

    def resolve(scene_id):
        if scene_id != "s02":
            return resolve_scene(scene_id)
        tree = _tree_resolved(Path("/nonexistent"))
        return dataclasses.replace(tree, scene=dataclasses.replace(tree.scene, id="s02"),
                                   base_scene=dataclasses.replace(tree.base_scene, id="s02"),
                                   default_scene_config="scene_autofly_s02.jsonc")

    monkeypatch.setattr(cli, "resolve_scene", resolve)
    monkeypatch.setattr(cli, "start_job", lambda job, argv, **kw: pytest.fail(f"start_job reached for {job}"))
    return cli


def _pilots(tmp_path):
    from tests.test_run_shape import LEGACY, run_root

    s01 = tmp_path / "pilots" / "s01_best.zip"  # a model outside any run: a pillar scene never asks its shape
    s01.parent.mkdir(parents=True)
    s01.write_bytes(b"s01 pilot")
    s02_r1 = run_root(tmp_path, "s02_r1", LEGACY)  # trained under the circle
    return {"s01": s01, "s02": s02_r1 / "averaged" / "avg_1k_end.zip"}


def _plan(cli, tmp_path, pilots=None, *extra):
    pilots = pilots if pilots is not None else _pilots(tmp_path)
    argv = ["plan", "--campaign", str(tmp_path / "campaign.json"), "--data-root", str(tmp_path / "data"),
            "--records-dir", str(tmp_path / "records"), "--episodes", "50", "--batches", "2", "--slots", "6", "7", "8",
            *sum((["--pilot", f"{scene}={path}"] for scene, path in pilots.items()), []), *extra]
    return cli.main(argv)


def test_plan_writes_one_collect_dataset_command_per_scene_and_batch(cli, tmp_path):
    from autofly_ue5.expert.seeds import collection_seed_base

    pilots = _pilots(tmp_path)
    assert _plan(cli, tmp_path, pilots) == 0
    campaign = json.loads((tmp_path / "campaign.json").read_text())
    jobs = campaign["jobs"]
    assert [(j["scene"], j["batch"], j["name"], j["job"], j["slot"], j["seed_slice"]) for j in jobs] == [
        ("s01", 0, "s01_c0", "collect_s01_c0", 6, 1), ("s01", 1, "s01_c1", "collect_s01_c1", 7, 2),
        ("s02", 0, "s02_c0", "collect_s02_c0", 8, 21), ("s02", 1, "s02_c1", "collect_s02_c1", 6, 22)]
    assert all(j["seed_base"] == collection_seed_base(j["seed_slice"]) and j["max_attempts"] == 200 for j in jobs)
    data, records = tmp_path / "data", tmp_path / "records"
    head = ["env", "-u", "PYTHONPATH", ".venv/bin/python", "scripts/collect_dataset.py"]
    # --no-export: plan 6 revision 1 exports one merged dataset, never a copy per store
    assert jobs[0]["argv"] == [*head, "--scene", "s01", "--model", str(pilots["s01"]), "--scene-config",
                               "scene_autofly_s01_fast.jsonc", "--name", "s01_c0", "--episodes", "50", "--max-attempts",
                               "200", "--seed-slice", "1", "--instance", "6", "--targets", "assets/targets.json", "--out",
                               str(records / "s01_c0.json"), "--no-export", "--data-root", str(data)]
    # a mesh scene collects under the task's outline rule; s02_r1's pilot trained under the circle, so on purpose
    assert jobs[2]["argv"] == [*head, "--scene", "s02", "--model", str(pilots["s02"]), "--scene-config",
                               "scene_autofly_s02_fast.jsonc", "--name", "s02_c0", "--episodes", "50", "--max-attempts",
                               "200", "--seed-slice", "21", "--instance", "8", "--targets", "assets/targets.json", "--out",
                               str(records / "s02_c0.json"), "--no-export", "--data-root", str(data),
                               "--footprint-shape", "outline", "--against-trained-shape"]
    assert jobs[2]["command"] == " ".join(jobs[2]["argv"])
    scenes = campaign["scenes"]
    assert scenes["s01"]["pilot"] == str(pilots["s01"]) and scenes["s01"]["footprint"] is None
    assert scenes["s02"]["footprint"] == {"shape": "outline", "trained_shape": "circle", "against_trained_shape": True}
    assert {s: (v["episodes_per_batch"], v["batches"], v["first_batch"], v["slots"]) for s, v in scenes.items()} == {
        "s01": (50, 2, 0, [6, 7]), "s02": (50, 2, 0, [8, 6])}
    assert len(scenes["s01"]["pilot_sha256"]) == 64 and campaign["slots"] == [6, 7, 8]
    assert not data.exists() and not records.exists(), "a plan claims nothing"


def test_s01d_flies_s01_s_config_and_a_later_batch_takes_its_own_slice(cli, tmp_path):
    pilot = tmp_path / "s01d_mean.zip"
    pilot.write_bytes(b"x")
    assert _plan(cli, tmp_path, {"s01d": pilot}, "--first-batch", "3") == 0
    jobs = json.loads((tmp_path / "campaign.json").read_text())["jobs"]
    assert [(j["name"], j["seed_slice"]) for j in jobs] == [("s01d_c3", 14), ("s01d_c4", 15)]
    argv = jobs[0]["argv"]
    assert argv[argv.index("--scene-config") + 1] == "scene_autofly_s01_fast.jsonc"
    assert "--footprint-shape" not in argv


@pytest.mark.parametrize("footprints, flags", [
    ("legacy", ["--footprint-shape", "outline", "--against-trained-shape"]),  # s02_r1: the circle
    ("outline", ["--footprint-shape", "outline"]),
    ("none", ["--footprint-shape", "outline"]),  # a pillar run (the s01d mean) flying a mesh scene: no rule of its own
])
def test_a_mesh_scene_s_flags_follow_the_shape_its_pilot_s_run_trained_under(cli, tmp_path, footprints, flags):
    from tests.test_run_shape import LEGACY, OUTLINE, run_root

    footprint = {"legacy": LEGACY, "outline": OUTLINE, "none": None}[footprints]
    root = run_root(tmp_path, "pilot_run", footprint) if footprint is not None else run_root(tmp_path, "pilot_run")
    assert _plan(cli, tmp_path, {"s02": root / "final.zip"}) == 0
    argv = json.loads((tmp_path / "campaign.json").read_text())["jobs"][0]["argv"]
    assert argv[argv.index("--data-root") + 2:] == flags


def _store(data: Path, name: str, seeds: list[int], rejects: dict[int, str] | None = None) -> None:
    (data / name).mkdir(parents=True)
    episodes = [{"id": f"x_{s}", "seed": s, "steps": 10} for s in seeds]
    (data / name / "manifest.json").write_text(json.dumps({"episodes": episodes, "counts": {
        "episodes": len(seeds), "records": 10 * len(seeds), "rejects": len(rejects or {})}}))
    for seed, reason in (rejects or {}).items():
        (data / "rejects" / name).mkdir(parents=True, exist_ok=True)
        (data / "rejects" / name / f"x_{seed}.json").write_text(json.dumps(
            {"id": f"x_{seed}", "reason": reason, "provenance": {"seed": seed}}))


@pytest.mark.parametrize("arrange, match", [
    (lambda t: _store(t / "data", "s01_c1", []), "s01_c1 exists"),  # the job's own store exists (no slice flown yet)
    (lambda t: _store(t / "data", "old", [402_000_005]), "slice 2"),  # another store flew s01 batch 1's slice
    (lambda t: _store(t / "data", "old", [], {401_000_003: "collision"}), "slice 1"),  # a store of rejects only
    (lambda t: (t / "data" / "rejects" / "s01_c0").mkdir(parents=True) or
     (t / "data" / "rejects" / "s01_c0" / "a.json").write_text("{}"), "rejects"),
    (lambda t: (t / "records").mkdir() or (t / "records" / "s01_c0.json").write_text("{}"), "s01_c0.json"),
    (lambda t: (t / "records").mkdir() or (t / "records" / "old.json").write_text(json.dumps(
        {"seed_base": 401_000_000})), "slice 1"),  # a collection record of the slice
    (lambda t: (t / "campaign.json").write_text("{}"), "campaign.json"),
])
def test_plan_refuses_what_would_reuse_a_name_or_a_slice_and_writes_nothing(cli, tmp_path, capsys, arrange, match):
    arrange(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert _plan(cli, tmp_path, {"s01": _pilots(tmp_path)["s01"]}) == 2
    assert match in capsys.readouterr().err
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if "pilots" not in str(p) and "runs" not in str(p))
    assert after == [p for p in before if "pilots" not in str(p) and "runs" not in str(p)]


@pytest.mark.parametrize("pilots, extra, match", [
    ({"s05r": "s01"}, [], "s05r"),  # not a campaign scene (expert/seeds.py CAMPAIGN_SCENE_ORDER)
    ({"s01": None}, [], "no pilot"),
    ({"s01": "s01"}, ["--first-batch", "9"], "batch"),  # batches 9 and 10: there are ten per scene
    ({"s01": "s01"}, ["--max-attempts", "10"], "max-attempts"),
])
def test_plan_refuses_a_bad_campaign_before_writing_it(cli, tmp_path, capsys, pilots, extra, match):
    models = _pilots(tmp_path)
    resolved = {scene: (models[src] if src is not None else tmp_path / "missing.zip") for scene, src in pilots.items()}
    assert _plan(cli, tmp_path, resolved, *extra) == 2
    assert match in capsys.readouterr().err and not (tmp_path / "campaign.json").exists()


def test_plan_refuses_a_pilot_named_twice_and_bad_slots(cli, tmp_path, capsys):
    model = _pilots(tmp_path)["s01"]
    common = ["plan", "--campaign", str(tmp_path / "campaign.json"), "--data-root", str(tmp_path / "data"),
              "--records-dir", str(tmp_path / "records"), "--episodes", "5", "--batches", "1"]
    assert cli.main([*common, "--slots", "6", "--pilot", f"s01={model}", "--pilot", f"s01={model}"]) == 2
    assert "twice" in capsys.readouterr().err
    for slots in (["6", "6"], ["-1"]):
        assert cli.main([*common, "--slots", *slots, "--pilot", f"s01={model}"]) == 2
        assert "slot" in capsys.readouterr().err
    assert not (tmp_path / "campaign.json").exists()


# ----------------------------------------------------------------------------------------------------------------------
# launch: one job per free slot, 30 s apart; never on a live slot or one a running job of the campaign holds.
# ----------------------------------------------------------------------------------------------------------------------
def _campaign(tmp_path, jobs: list[tuple[str, int]]) -> Path:
    """A campaign file whose jobs run harmless commands: (name, slot) each."""
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps({"version": 1, "data_root": str(tmp_path / "data"), "records_dir": str(tmp_path / "rec"),
                                "jobs": [{"scene": name.split("_c")[0], "batch": int(name.split("_c")[1]), "name": name,
                                          "job": f"collect_{name}", "slot": slot, "seed_slice": 1, "episodes": 2,
                                          "argv": ["true"], "command": "true"} for name, slot in jobs]}))
    return path


def _launch(cli, monkeypatch, tmp_path, campaign, *extra):
    started, slept = [], []
    monkeypatch.setattr(cli, "start_job", lambda job, argv, **kw: started.append((job, argv)) or (True, "started"))
    monkeypatch.setattr(cli.time, "sleep", slept.append)
    code = cli.main(["launch", "--campaign", str(campaign), "--sim-root", str(tmp_path / "sim"), "--jobs-dir",
                     str(tmp_path / "jobs"), *extra])
    return code, [job for job, _argv in started], slept


def _sim_record(tmp_path, slot: int, *, pid: int, owner_pid: int | None = None) -> None:
    d = tmp_path / "sim" / f"inst{slot}"
    d.mkdir(parents=True)
    (d / "pid.json").write_text(json.dumps({
        "pid": pid, "pgid": pid, "instance": slot, "topics_port": 0, "services_port": 0, "cmd": ["x"], "log_path": "x",
        "started_unix": 0.0, "owner_pid": owner_pid, "owner_start_ticks": proc_start_ticks(owner_pid) if owner_pid else None}))


def _dead_pid() -> int:
    import subprocess

    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_launch_starts_one_job_per_free_slot_thirty_seconds_apart(cli, monkeypatch, tmp_path, capsys):
    campaign = _campaign(tmp_path, [("s01_c0", 6), ("s01_c1", 7), ("s02_c0", 6), ("s02_c1", 8)])
    code, started, slept = _launch(cli, monkeypatch, tmp_path, campaign)
    assert code == 0 and started == ["collect_s01_c0", "collect_s01_c1", "collect_s02_c1"]
    assert slept == [30.0, 30.0], "between starts, not after the last"
    out = capsys.readouterr().out
    assert "s02_c0" in out and "slot 6" in out, "the job waiting for its slot says so"


def test_launch_never_uses_a_slot_with_a_live_simulator_or_a_live_run(cli, monkeypatch, tmp_path):
    campaign = _campaign(tmp_path, [("s01_c0", 6), ("s01_c1", 7), ("s01_c2", 8), ("s01_c3", 9)])
    _sim_record(tmp_path, 6, pid=os.getpid())  # a live simulator
    _sim_record(tmp_path, 7, pid=_dead_pid(), owner_pid=os.getpid())  # down between a live run's evaluations
    _sim_record(tmp_path, 8, pid=_dead_pid(), owner_pid=_dead_pid())  # a finished run's: free
    (tmp_path / "sim" / "inst9").mkdir(parents=True)
    (tmp_path / "sim" / "inst9" / "pid.json").write_text("")  # left at 0 bytes by a freeze: look before using it
    code, started, _slept = _launch(cli, monkeypatch, tmp_path, campaign)
    assert code == 0 and started == ["collect_s01_c2"]


def test_launch_skips_collected_running_and_finished_jobs_and_retries_on_request(cli, monkeypatch, tmp_path):
    campaign = _campaign(tmp_path, [("s01_c0", 6), ("s01_c1", 7), ("s01_c2", 8), ("s01_c3", 8)])
    _store(tmp_path / "data", "s01_c0", [401_000_000])  # collected
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "collect_s01_c1.pid.json").write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpid()}))  # running
    (jobs / "collect_s01_c2.pid.json").write_text(json.dumps({"pid": _dead_pid(), "pgid": 1}))
    (jobs / "collect_s01_c2.exit").write_text("2\n")  # refused at startup: nothing claimed
    code, started, _slept = _launch(cli, monkeypatch, tmp_path, campaign)
    assert code == 0 and started == ["collect_s01_c3"], "c2 finished: not again unless asked; c3 shares slot 8"
    (jobs / "collect_s01_c3.pid.json").write_text(json.dumps({"pid": _dead_pid(), "pgid": 1}))
    (jobs / "collect_s01_c3.exit").write_text("0\n")
    _store(tmp_path / "data", "s01_c3", [401_000_003])
    code, started, _slept = _launch(cli, monkeypatch, tmp_path, campaign, "--retry")
    assert started == ["collect_s01_c2"], "a finished job without a store, on --retry"


def test_launch_dry_run_starts_nothing(cli, monkeypatch, tmp_path, capsys):
    campaign = _campaign(tmp_path, [("s01_c0", 6), ("s01_c1", 7)])
    code, started, slept = _launch(cli, monkeypatch, tmp_path, campaign, "--dry-run")
    assert code == 0 and started == [] and slept == []
    out = capsys.readouterr().out
    assert "would start collect_s01_c0" in out and "would start collect_s01_c1" in out


def test_the_job_starter_hands_the_command_to_run_job_sh(tmp_path, monkeypatch):
    import scripts.collect_campaign as cli

    monkeypatch.setenv("AUTOFLY_JOBS_DIR", str(tmp_path / "jobs"))
    ok, output = cli.start_job("t_campaign", ["bash", "-c", "echo started-by-campaign"])
    assert ok, output
    record = json.loads((tmp_path / "jobs" / "t_campaign.pid.json").read_text())
    assert record["cmd"] == ["bash", "-c", "echo started-by-campaign"]


# ----------------------------------------------------------------------------------------------------------------------
# status: per store, kept / flown / rejected by reason, from its manifest and its rejects.
# ----------------------------------------------------------------------------------------------------------------------
def test_status_counts_each_store_s_kept_flown_and_rejects_by_reason(cli, tmp_path, capsys):
    campaign = _campaign(tmp_path, [("s01_c0", 6), ("s01_c1", 7), ("s02_c0", 8)])
    _store(tmp_path / "data", "s01_c0", [401_000_000, 401_000_002],
           {401_000_001: "collision", 401_000_003: "collision", 401_000_004: "out_of_bounds"})
    _store(tmp_path / "data", "s02_c0", [421_000_000])
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "collect_s01_c0.pid.json").write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpid()}))
    assert cli.main(["status", "--campaign", str(campaign), "--jobs-dir", str(jobs)]) == 0
    out = capsys.readouterr().out
    rows = {line.split()[0]: line for line in out.splitlines() if line.split() and line.split()[0].startswith("s0")}
    assert "running" in rows["s01_c0"] and "kept 2/2" in rows["s01_c0"] and "flown 5" in rows["s01_c0"]
    assert "collision 2" in rows["s01_c0"] and "out_of_bounds 1" in rows["s01_c0"]
    assert "pending" in rows["s01_c1"] and "kept 0/2" in rows["s01_c1"] and "flown 0" in rows["s01_c1"]
    assert "kept 1/2" in rows["s02_c0"]
    assert "s01: kept 2/4, flown 5" in out and "all: kept 3/6, flown 6" in out
    summary = cli.campaign_status(json.loads(campaign.read_text()), jobs_dir=jobs)
    assert summary["jobs"][0]["rejected_by_reason"] == {"collision": 2, "out_of_bounds": 1}
