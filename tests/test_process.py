import json
import logging
import os
import socket
import sys
import time

import pytest

from autofly_ue5.paths import DDC_DIR, ZEN_DATA_DIR
from autofly_ue5.sim.process import (
    NotOwnedError,
    SimExitedError,
    SimPorts,
    editor_game_command,
    instance_dir,
    launch_process,
    listening_pids,
    own_running_instances,
    packaged_command,
    ports_for_instance,
    route_client_log,
    sim_environment,
    stop,
    wait_ready,
)

FAKE_SERVER = (
    "import socket, sys, time\n"
    "held = []\n"
    "for port in sys.argv[1:]:\n"
    "    s = socket.socket()\n"
    "    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "    s.bind(('127.0.0.1', int(port)))\n"
    "    s.listen()\n"
    "    held.append(s)\n"
    "time.sleep(600)\n"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ports_for_instance():
    assert ports_for_instance(0) == SimPorts(8989, 8990)
    assert ports_for_instance(1) == SimPorts(9001, 9002)
    with pytest.raises(ValueError):
        ports_for_instance(-1)


def test_editor_game_command(tmp_path):
    log = tmp_path / "sim.log"
    cmd = editor_game_command("/Game/BlocksMap", SimPorts(9001, 9002), log, 1)
    assert cmd[0].endswith("engine/Engine/Binaries/Linux/UnrealEditor")
    assert cmd[1].endswith("ue_project/Blocks.uproject")
    assert cmd[2] == "/Game/BlocksMap"
    for flag in ("-game", "-RenderOffScreen", "-vulkan", "-nosound", "-unattended", "-notraceserver", "-topicsport=9001",
                 "-servicesport=9002", f"-abslog={log}", "-saveddirsuffix=inst1", f"-ZenDataPath={ZEN_DATA_DIR}"):
        assert flag in cmd


def test_packaged_command(tmp_path):
    cmd = packaged_command("/Game/AutoFly/Maps/S01", SimPorts(8989, 8990), tmp_path / "sim.log", 0)
    assert cmd[0].endswith("Packaged/Development/Linux/Blocks/Binaries/Linux/Blocks")
    assert cmd[1:3] == ["Blocks", "/Game/AutoFly/Maps/S01"]
    assert "-game" not in cmd and "-notraceserver" in cmd
    assert f"-ZenDataPath={ZEN_DATA_DIR}" in cmd
    assert not any(c.startswith("-saveddirsuffix") for c in cmd)


def test_sim_environment(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/jazzy/lib/python3.12/site-packages")
    editor_env = sim_environment(editor_mode=True)
    assert "PYTHONPATH" not in editor_env
    assert editor_env["DISPLAY"] == ":1" and editor_env["SDL_VIDEODRIVER"] == "x11"
    assert editor_env["PROJECTAIRSIM_CI"] == "1"
    # Underscore, not hyphen: FUnixPlatformMisc::GetEnvironmentVariable rewrites "-" to "_" before calling
    # secure_getenv (UnixPlatformMisc.cpp:289-317), so only the underscore form in the process environment
    # is ever seen by the engine (confirmed by runs/m0/engine_check.json's zen_data_path_redirected check,
    # which matched "Found environment variable UE_ZenDataPath=..."). A hyphenated key here would be inert.
    assert editor_env["UE_ZenDataPath"] == str(ZEN_DATA_DIR)
    packaged_env = sim_environment(editor_mode=False)
    assert "PROJECTAIRSIM_CI" not in packaged_env
    assert packaged_env["UE_LocalDataCachePath"] == str(DDC_DIR)


def test_route_client_log_replaces_the_working_directory_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = logging.getLogger("projectairsim")
    saved = (list(logger.handlers), logger.level)
    try:
        route_client_log(tmp_path / "client" / "inst1.client.log")
        logger.info("hello from the client")
        for handler in logger.handlers:
            handler.flush()
        assert "hello from the client" in (tmp_path / "client" / "inst1.client.log").read_text()
        assert logger.hasHandlers()  # projectairsim_log() will not add ./projectairsim_client.log
        assert not (tmp_path / "projectairsim_client.log").exists()
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers[:] = saved[0]
        logger.setLevel(saved[1])


@pytest.fixture
def fake_sim(tmp_path):
    ports = SimPorts(free_port(), free_port())
    cmd = [sys.executable, "-c", FAKE_SERVER, str(ports.topics), str(ports.services)]
    sp = launch_process(cmd, 3, ports, dict(os.environ), run_root=tmp_path)
    yield sp, ports, tmp_path
    try:
        stop(3, grace_s=5.0, run_root=tmp_path)
    except Exception:
        pass


def test_launch_records_pid_and_becomes_ready(fake_sim):
    sp, ports, root = fake_sim
    data = json.loads((instance_dir(3, root) / "pid.json").read_text())
    assert data["pid"] == sp.pid and data["pgid"] == sp.pid
    assert wait_ready(sp, timeout_s=20.0, poll_s=0.2) < 20.0
    assert sp.pid in listening_pids(ports.topics)
    assert [p.pid for p in own_running_instances(root)] == [sp.pid]


def test_stop_terminates_owned_process(fake_sim):
    sp, ports, root = fake_sim
    wait_ready(sp, timeout_s=20.0, poll_s=0.2)
    assert stop(3, grace_s=10.0, run_root=root) == "terminated"
    assert not (instance_dir(3, root) / "pid.json").exists()
    deadline = time.monotonic() + 5.0
    while listening_pids(ports.topics) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert listening_pids(ports.topics) == set()


def test_stop_refuses_a_process_it_did_not_launch(tmp_path):
    d = instance_dir(5, tmp_path)
    d.mkdir(parents=True)
    foreign = {"pid": os.getpid(), "pgid": os.getpgid(os.getpid()), "instance": 5, "topics_port": 1,
               "services_port": 2, "cmd": ["/not/our/UnrealEditor"], "log_path": "x", "started_unix": 0.0}
    (d / "pid.json").write_text(json.dumps(foreign))
    with pytest.raises(NotOwnedError):
        stop(5, run_root=tmp_path)
    assert (d / "pid.json").exists()


def test_wait_ready_reports_early_exit(tmp_path):
    ports = SimPorts(free_port(), free_port())
    sp = launch_process([sys.executable, "-c", "import sys; sys.exit(3)"], 4, ports, dict(os.environ), run_root=tmp_path)
    with pytest.raises(SimExitedError):
        wait_ready(sp, timeout_s=10.0, poll_s=0.1)
    assert stop(4, run_root=tmp_path) == "not_running"


def test_launch_refuses_a_busy_port(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        with pytest.raises(RuntimeError, match="already in use"):
            launch_process([sys.executable, "-c", "pass"], 6, SimPorts(busy, free_port()), dict(os.environ), run_root=tmp_path)
