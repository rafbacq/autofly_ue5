import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_job.sh"


def run_job(jobs_dir: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["AUTOFLY_JOBS_DIR"] = str(jobs_dir)
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=90)


def test_start_records_the_job_and_wait_reports_success(tmp_path):
    started = run_job(tmp_path, "start", "ok", "--", "bash", "-c", "echo hello; sleep 1")
    assert started.returncode == 0, started.stdout + started.stderr
    record = json.loads((tmp_path / "ok.pid.json").read_text())
    assert record["pid"] == record["pgid"]
    assert record["cmd"] == ["bash", "-c", "echo hello; sleep 1"]
    waited = run_job(tmp_path, "wait", "ok", "30")
    assert waited.returncode == 0
    assert "hello" in waited.stdout and "job ok finished: exit=0" in waited.stdout


def test_wait_reports_a_failing_exit_code(tmp_path):
    run_job(tmp_path, "start", "bad", "--", "bash", "-c", "exit 3")
    waited = run_job(tmp_path, "wait", "bad", "30")
    assert waited.returncode == 1 and "job bad finished: exit=3" in waited.stdout


def test_wait_times_out_duplicate_start_is_refused_and_stop_ends_the_group(tmp_path):
    run_job(tmp_path, "start", "slow", "--", "bash", "-c", "sleep 300 & wait")
    waited = run_job(tmp_path, "wait", "slow", "1")
    assert waited.returncode == 124 and "still running" in waited.stdout
    again = run_job(tmp_path, "start", "slow", "--", "true")
    assert again.returncode == 1 and "still running" in again.stdout
    pgid = json.loads((tmp_path / "slow.pid.json").read_text())["pgid"]
    stopped = run_job(tmp_path, "stop", "slow")
    assert stopped.returncode == 0 and stopped.stdout.strip() == "terminated"
    assert subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True).returncode == 1
    assert run_job(tmp_path, "wait", "slow", "5").returncode == 3


def test_stop_refuses_a_process_it_did_not_launch(tmp_path):
    # A real process started outside run_job.sh: its own session/process group,
    # so a bug that let `stop` reach it could not also reach the test runner.
    foreign = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        foreign_pgid = os.getpgid(foreign.pid)
        (tmp_path / "foreign.pid.json").write_text(
            json.dumps(
                {
                    "pid": foreign.pid,
                    "pgid": foreign_pgid,
                    "started": "2026-01-01T00:00:00+0000",
                    "log": str(tmp_path / "foreign.log"),
                    "cmd": ["sleep", "60"],
                }
            )
        )
        stopped = run_job(tmp_path, "stop", "foreign")
        assert stopped.returncode == 1
        assert "does not match job foreign" in stopped.stdout + stopped.stderr
        assert foreign.poll() is None  # refused: the foreign process is still alive
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)
