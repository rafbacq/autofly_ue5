import json
import logging
import os
import socket
import subprocess
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
    is_alive,
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


def _sleeper(instance: int, root, owner=None):
    """A real, owned process in slot `instance` under `root` (no ports needed: `sleep` never listens)."""
    from autofly_ue5.sim.process import ports_for_instance

    return launch_process(["sleep", "300"], instance, ports_for_instance(40 + instance), dict(os.environ),
                          run_root=root, owner=owner)


def _dead_owner():
    from autofly_ue5.sim.process import RunOwner, proc_start_ticks

    done = subprocess.Popen(["true"])
    ticks = proc_start_ticks(done.pid)
    done.wait()
    return RunOwner(done.pid, ticks if ticks is not None else 1)


def _init_owner():
    from autofly_ue5.sim.process import RunOwner, proc_start_ticks

    return RunOwner(1, proc_start_ticks(1))  # pid 1: always alive, never this test's run


def test_launch_records_the_run_owner_token(tmp_path):
    from autofly_ue5.sim.process import proc_start_ticks, read_pid_file

    _sleeper(1, tmp_path)
    try:
        record = read_pid_file(instance_dir(1, tmp_path) / "pid.json")
        assert record.owner_pid == os.getpid()
        assert record.owner_start_ticks == proc_start_ticks(os.getpid())
    finally:
        stop(1, grace_s=2.0, run_root=tmp_path)


def test_an_old_pid_file_without_owner_fields_still_loads(tmp_path):
    from autofly_ue5.sim.process import read_pid_file

    d = instance_dir(2, tmp_path)
    d.mkdir(parents=True)
    (d / "pid.json").write_text(json.dumps({"pid": 1, "pgid": 1, "instance": 2, "topics_port": 1, "services_port": 2,
                                            "cmd": ["x"], "log_path": "x", "started_unix": 0.0}))
    assert read_pid_file(d / "pid.json").owner_pid is None


def test_stop_with_a_superseded_expected_pid_signals_nothing_and_keeps_the_record(tmp_path):
    # C1: a close() abandoned by a bounded relaunch can return after the slot was relaunched; it must not stop
    # the successor recorded in the same pid.json.
    sp = _sleeper(1, tmp_path)
    try:
        assert stop(1, run_root=tmp_path, expected_pid=sp.pid + 100_000) == "superseded"
        assert (instance_dir(1, tmp_path) / "pid.json").exists()
        assert is_alive(sp.pid)
    finally:
        stop(1, grace_s=2.0, run_root=tmp_path)


def test_orphan_sweep_stops_a_dead_owners_instance_and_skips_a_live_foreign_owner(tmp_path):
    from autofly_ue5.sim.process import sweep_orphaned_instances

    orphan = _sleeper(1, tmp_path, owner=_dead_owner())
    foreign = _sleeper(2, tmp_path, owner=_init_owner())
    try:
        swept = sweep_orphaned_instances(tmp_path)
        assert [s["instance"] for s in swept] == [1]
        assert not is_alive(orphan.pid)
        assert is_alive(foreign.pid), "a live run's simulator must survive another run's startup sweep"
    finally:
        stop(2, grace_s=2.0, run_root=tmp_path)


def test_orphan_sweep_treats_a_legacy_record_without_an_owner_as_orphaned(tmp_path):
    from autofly_ue5.sim.process import sweep_orphaned_instances

    sp = _sleeper(1, tmp_path)
    pid_file = instance_dir(1, tmp_path) / "pid.json"
    record = json.loads(pid_file.read_text())
    record.pop("owner_pid"), record.pop("owner_start_ticks")
    pid_file.write_text(json.dumps(record))
    assert [s["instance"] for s in sweep_orphaned_instances(tmp_path)] == [1]
    assert not is_alive(sp.pid)


def test_stop_instance_refuses_a_slot_a_live_foreign_run_owns_and_stops_its_own(tmp_path):
    from autofly_ue5.sim.process import stop_instance

    foreign = _sleeper(1, tmp_path, owner=_init_owner())
    mine = _sleeper(2, tmp_path)
    try:
        assert stop_instance(1, tmp_path) == "owner_alive_elsewhere"
        assert is_alive(foreign.pid)
        assert stop_instance(2, tmp_path) in ("terminated", "killed")
        assert not is_alive(mine.pid)
        assert stop_instance(2, tmp_path) == "no_pid_file"
    finally:
        stop(1, grace_s=2.0, run_root=tmp_path)


def test_stop_instances_stops_only_the_listed_slots(tmp_path):
    from autofly_ue5.sim.process import stop_instances

    a, b = _sleeper(1, tmp_path), _sleeper(2, tmp_path)
    try:
        results = stop_instances([1], tmp_path)
        assert [r["instance"] for r in results] == [1]
        assert not is_alive(a.pid) and is_alive(b.pid)
    finally:
        stop(2, grace_s=2.0, run_root=tmp_path)


def test_wait_ready_timeout_is_a_typed_error(tmp_path):
    from autofly_ue5.sim.process import SimReadyTimeout

    sp = _sleeper(1, tmp_path)
    try:
        with pytest.raises(SimReadyTimeout):
            wait_ready(sp, timeout_s=0.3, poll_s=0.1)
        assert issubclass(SimReadyTimeout, TimeoutError)
    finally:
        stop(1, grace_s=2.0, run_root=tmp_path)


def test_stop_keeps_the_record_of_a_process_that_survives_sigkill(tmp_path, monkeypatch):
    import autofly_ue5.sim.process as process

    _sleeper(1, tmp_path)
    try:
        monkeypatch.setattr(process.os, "killpg", lambda pgid, sig: None)  # the signal "does nothing"
        monkeypatch.setattr(process, "KILL_WAIT_S", 0.2)
        assert stop(1, grace_s=0.2, run_root=tmp_path) == "unkillable"
        assert (instance_dir(1, tmp_path) / "pid.json").exists(), "an unkillable process must stay on the record"
    finally:
        monkeypatch.undo()
        stop(1, grace_s=2.0, run_root=tmp_path)


def test_stdout_log_keeps_every_launch_of_a_slot(tmp_path):
    # C7: stdout.log was reopened with "wb" on every launch, so a relaunch erased the engine output (a crash's
    # last words included) of every earlier session in that slot.
    _sleeper(1, tmp_path)
    stop(1, grace_s=2.0, run_root=tmp_path)
    _sleeper(1, tmp_path)
    stop(1, grace_s=2.0, run_root=tmp_path)
    text = (instance_dir(1, tmp_path) / "stdout.log").read_text()
    assert text.count("=== launch ") == 2



def test_stop_instance_acts_only_on_the_process_it_decided_about(tmp_path, monkeypatch):
    # Final review #5: the decision is made from an unlocked read; if the slot was relaunched in between, the stop
    # must not land on the successor.
    import dataclasses

    import autofly_ue5.sim.process as process
    from autofly_ue5.sim.process import stop_instance

    successor = _sleeper(1, tmp_path)
    real_read = process.read_pid_file
    stale = dataclasses.replace(real_read(instance_dir(1, tmp_path) / "pid.json"), pid=successor.pid + 100_000)
    reads = {"n": 0}

    def first_read_is_stale(path):
        reads["n"] += 1
        return stale if reads["n"] == 1 else real_read(path)

    monkeypatch.setattr(process, "read_pid_file", first_read_is_stale)
    try:
        assert stop_instance(1, tmp_path) == "superseded"
        assert is_alive(successor.pid)
    finally:
        monkeypatch.undo()
        stop(1, grace_s=2.0, run_root=tmp_path)


def test_a_half_written_pid_file_does_not_break_a_sweep(tmp_path):
    from autofly_ue5.sim.process import own_running_instances, sweep_orphaned_instances

    d = instance_dir(9, tmp_path)
    d.mkdir(parents=True)
    (d / "pid.json").write_text('{"pid": 12')
    assert own_running_instances(tmp_path) == []
    assert sweep_orphaned_instances(tmp_path) == []


def test_slot_busy_names_a_slot_a_live_run_holds_and_frees_the_rest(tmp_path):
    # A side job (gate, pilot, smoke) must not take a slot a live run holds: an --instances N training run also owns
    # slot N, its evaluation simulator (2026-10-03). A dead run's slot is the startup sweep's to clear, not busy.
    from autofly_ue5.sim.process import slot_busy

    foreign = _sleeper(1, tmp_path, owner=_init_owner())
    orphan = _sleeper(2, tmp_path, owner=_dead_owner())
    mine = _sleeper(3, tmp_path)
    try:
        assert "pid" in slot_busy(1, tmp_path) and str(foreign.pid) in slot_busy(1, tmp_path)
        assert slot_busy(2, tmp_path) is None, "a dead run's simulator is an orphan, not a reason to refuse"
        assert slot_busy(3, tmp_path) is None, "this run's own slot"
        assert slot_busy(4, tmp_path) is None, "an empty slot"
    finally:
        for instance in (1, 2, 3):
            stop(instance, grace_s=2.0, run_root=tmp_path)
    assert not any(is_alive(sp.pid) for sp in (foreign, orphan, mine))


# A record no reader can parse: 0 bytes (what the 2026-10-03 17:38 host freeze left in runs/sim/inst1), cut off
# mid-write, or missing fields.
UNREADABLE_RECORDS = ["", '{"pid": 12', '{"pid": 12}']


@pytest.mark.parametrize("content", UNREADABLE_RECORDS)
def test_a_launch_sets_aside_a_record_a_host_crash_left_unreadable(tmp_path, content):
    # The empty record made every later launch in its slot raise JSONDecodeError: in a SubprocVecEnv worker, the end
    # of the run, after the trainer had already claimed its session. It names no process, so a launch sets it aside
    # (kept for diagnosis); the port check after it still refuses a slot something is listening on.
    from autofly_ue5.sim.process import read_pid_file

    d = instance_dir(1, tmp_path)
    d.mkdir(parents=True)
    (d / "pid.json").write_text(content)
    sp = _sleeper(1, tmp_path)
    try:
        assert read_pid_file(d / "pid.json").pid == sp.pid
        assert (d / "pid.json.unreadable").read_text() == content
    finally:
        stop(1, grace_s=2.0, run_root=tmp_path)


@pytest.mark.parametrize("content", UNREADABLE_RECORDS)
def test_stop_and_stop_instance_signal_nothing_for_an_unreadable_record(tmp_path, content):
    # A relaunch stops its slot first (resilient.py), and a closing backend stops it with expected_pid: both raised.
    from autofly_ue5.sim.process import stop_instance

    d = instance_dir(2, tmp_path)
    d.mkdir(parents=True)
    (d / "pid.json").write_text(content)
    assert stop(2, run_root=tmp_path) == "unreadable_record"
    assert stop(2, run_root=tmp_path, expected_pid=12) == "unreadable_record"
    assert stop_instance(2, tmp_path) == "unreadable_record"
    assert (d / "pid.json").read_text() == content, "left for the slot's next launch to set aside"


def test_a_launch_record_reaches_the_disk_before_it_names_the_process(tmp_path, sync_events):
    # The root cause of the empty record: the rename was durable before the data was.
    _sleeper(1, tmp_path)
    try:
        assert sync_events.wrote_durably(instance_dir(1, tmp_path) / "pid.json")
    finally:
        stop(1, grace_s=2.0, run_root=tmp_path)
