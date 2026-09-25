"""The shell scripts must work from any clone location and on a host without this machine's extras."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RUN_JOB_TOOLS = ("bash", "mkdir", "rm", "setsid", "ps", "tr", "seq", "sleep", "env", "python3", "tail", "cat",
                 "date", "touch", "sed", "dirname")


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
def test_no_script_hardcodes_an_absolute_repo_path_or_interpreter(script):
    text = (SCRIPTS / script).read_text()
    assert "/home/" not in text, f"{script} hard-codes a home directory; derive ROOT from the script's location"
    assert "/usr/bin/python3.12" not in text, f"{script} hard-codes an interpreter path"


def _path_without_jq(tmp_path: Path) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in RUN_JOB_TOOLS:
        found = shutil.which(tool, path="/usr/local/bin:/usr/bin:/bin")
        assert found, f"{tool} is needed by this test"
        (bin_dir / tool).symlink_to(found)
    return str(bin_dir)


def test_run_job_start_and_wait_work_without_jq(tmp_path):
    jobs = tmp_path / "jobs"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(AUTOFLY_JOBS_DIR=str(jobs), PATH=_path_without_jq(tmp_path))
    run = lambda *a: subprocess.run(["bash", str(SCRIPTS / "run_job.sh"), *a], env=env, capture_output=True,
                                    text=True, timeout=60)
    started = run("start", "nojq", "--", "bash", "-c", "sleep 3; echo hi")
    assert started.returncode == 0, started.stdout + started.stderr
    waited = run("wait", "nojq", "30")
    assert waited.returncode == 0, waited.stdout + waited.stderr
    assert "job nojq finished: exit=0" in waited.stdout


def test_setup_venv_refuses_an_interpreter_that_is_not_python_312(tmp_path):
    fake = tmp_path / "python3.11"
    fake.write_text("#!/bin/sh\necho 3.11\n")
    fake.chmod(0o755)
    venv_marker = SCRIPTS.parent / ".venv" / "pyvenv.cfg"
    before = venv_marker.stat().st_mtime if venv_marker.exists() else None
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHON"] = str(fake)
    result = subprocess.run(["bash", str(SCRIPTS / "setup_venv.sh")], env=env, capture_output=True, text=True,
                            timeout=60)
    assert result.returncode != 0
    assert "3.12" in result.stdout + result.stderr
    after = venv_marker.stat().st_mtime if venv_marker.exists() else None
    assert before == after, "a refused interpreter must not touch the existing .venv"
