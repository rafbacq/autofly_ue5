"""Launch, readiness and stop of simulator processes owned by this project.

Each launched process runs in its own session (PGID == PID) and is recorded in
runs/sim/inst<N>/pid.json. stop() only signals a process whose recorded PID is
alive, whose /proc cmdline[0] equals the recorded cmd[0], and whose process
group equals the recorded PGID. Unreal starts its helpers (ShaderCompileWorker,
zenserver, CrashReportClient) in their own process groups (UnixPlatformProcess.cpp:1048-1049),
so they are not signalled here; callers check for survivors with pgrep and report them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from autofly_ue5.paths import PACKAGED_BINARY, RUNS_DIR, UE_CACHE_ENV, UNREAL_EDITOR, UPROJECT, ZEN_DATA_DIR

SIM_RUN_DIR = RUNS_DIR / "sim"
BASE_TOPICS_PORT = 8989
BASE_SERVICES_PORT = 8990
PORT_STRIDE = 12
# -notraceserver: otherwise every non-Shipping UE process forks UnrealTraceServer, a daemon outside our process group
# that writes ~/UnrealEngine/UnrealTrace and /tmp/UnrealTraceServer.pid (TraceAuxiliary.cpp:1980-1986).
COMMON_FLAGS = ["-RenderOffScreen", "-nosound", "-unattended", "-nopause", "-nosplash", "-notraceserver", "-log",
                "-ResX=640", "-ResY=480"]


class NotOwnedError(RuntimeError):
    """The pid file points at a process this project did not launch."""


class SimExitedError(RuntimeError):
    """The simulator process exited before becoming ready."""


@dataclass(frozen=True)
class SimPorts:
    topics: int
    services: int


@dataclass
class SimProcess:
    pid: int
    pgid: int
    instance: int
    topics_port: int
    services_port: int
    cmd: list[str]
    log_path: str
    started_unix: float


def ports_for_instance(instance: int) -> SimPorts:
    if instance < 0:
        raise ValueError(f"instance must be >= 0, got {instance}")
    return SimPorts(BASE_TOPICS_PORT + PORT_STRIDE * instance, BASE_SERVICES_PORT + PORT_STRIDE * instance)


def instance_dir(instance: int, run_root: Path = SIM_RUN_DIR) -> Path:
    return Path(run_root) / f"inst{instance}"


def _port_flags(ports: SimPorts, log_path: Path, instance: int) -> list[str]:
    flags = [f"-topicsport={ports.topics}", f"-servicesport={ports.services}", f"-abslog={log_path}"]
    if instance > 0:
        flags.append(f"-saveddirsuffix=inst{instance}")
    return flags


def editor_game_command(
    map_path: str, ports: SimPorts, log_path: Path, instance: int, uproject: Path = UPROJECT, editor: Path = UNREAL_EDITOR
) -> list[str]:
    return [
        str(editor), str(uproject), map_path, "-game", "-vulkan", *COMMON_FLAGS, f"-ZenDataPath={ZEN_DATA_DIR}",
        *_port_flags(ports, log_path, instance),
    ]


def packaged_command(
    map_path: str, ports: SimPorts, log_path: Path, instance: int, binary: Path = PACKAGED_BINARY
) -> list[str]:
    return [
        str(binary), "Blocks", map_path, *COMMON_FLAGS, f"-ZenDataPath={ZEN_DATA_DIR}",
        *_port_flags(ports, log_path, instance),
    ]


def sim_environment(editor_mode: bool) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["DISPLAY"] = ":1"
    env["SDL_VIDEODRIVER"] = "x11"
    env.update(UE_CACHE_ENV)
    if editor_mode:
        env["PROJECTAIRSIM_CI"] = "1"
    else:
        env.pop("PROJECTAIRSIM_CI", None)
    return env


def route_client_log(path: Path) -> None:
    """Point the Project AirSim client logger at `path`; projectairsim_log() then skips its ./projectairsim_client.log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = logging.getLogger("projectairsim")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.DEBUG)


def read_pid_file(path: Path) -> SimProcess:
    return SimProcess(**json.loads(Path(path).read_text()))


def _proc_state(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return stat.rsplit(")", 1)[1].split()[0]


def is_alive(pid: int) -> bool:
    state = _proc_state(pid)
    return state is not None and state != "Z"


def _proc_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def is_owned(sp: SimProcess) -> bool:
    cmdline = _proc_cmdline(sp.pid)
    if not cmdline:
        return False
    try:
        pgid = os.getpgid(sp.pid)
    except ProcessLookupError:
        return False
    return cmdline[0] == sp.cmd[0] and pgid == sp.pgid


def _reap(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


def listening_pids(port: int) -> set[int]:
    out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True, check=True).stdout
    pids: set[int] = set()
    for line in out.splitlines():
        if not line.strip():
            continue
        found = [int(p) for p in re.findall(r"pid=(\d+)", line)]
        pids.update(found if found else [-1])
    return pids


def launch_process(
    cmd: list[str], instance: int, ports: SimPorts, env: dict[str, str], run_root: Path = SIM_RUN_DIR
) -> SimProcess:
    directory = instance_dir(instance, run_root)
    directory.mkdir(parents=True, exist_ok=True)
    pid_file = directory / "pid.json"
    if pid_file.exists():
        existing = read_pid_file(pid_file)
        if is_alive(existing.pid) and is_owned(existing):
            raise RuntimeError(f"instance {instance} is already running with pid {existing.pid}")
        pid_file.unlink()
    for port in (ports.topics, ports.services):
        if listening_pids(port):
            raise RuntimeError(f"port {port} is already in use")
    with open(directory / "stdout.log", "wb") as out:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
            start_new_session=True, cwd=directory,
        )
    log_path = next((c.split("=", 1)[1] for c in cmd if c.startswith("-abslog=")), str(directory / "stdout.log"))
    sp = SimProcess(
        pid=proc.pid, pgid=os.getpgid(proc.pid), instance=instance, topics_port=ports.topics,
        services_port=ports.services, cmd=list(cmd), log_path=log_path, started_unix=time.time(),
    )
    pid_file.write_text(json.dumps(asdict(sp), indent=2))
    return sp


def wait_ready(sp: SimProcess, timeout_s: float, poll_s: float = 1.0) -> float:
    start = time.monotonic()
    while True:
        if not is_alive(sp.pid):
            _reap(sp.pid)
            raise SimExitedError(f"simulator pid {sp.pid} exited before its ports opened; see {sp.log_path}")
        if sp.pid in listening_pids(sp.topics_port) and sp.pid in listening_pids(sp.services_port):
            return time.monotonic() - start
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"ports {sp.topics_port}/{sp.services_port} not open after {timeout_s} s; see {sp.log_path}")
        time.sleep(poll_s)


def stop(instance: int, grace_s: float = 30.0, run_root: Path = SIM_RUN_DIR) -> str:
    pid_file = instance_dir(instance, run_root) / "pid.json"
    if not pid_file.exists():
        return "no_pid_file"
    sp = read_pid_file(pid_file)
    if not is_alive(sp.pid):
        _reap(sp.pid)
        pid_file.unlink()
        return "not_running"
    if not is_owned(sp):
        raise NotOwnedError(
            f"pid {sp.pid} in {pid_file} does not match the recorded command/process group; not stopping it"
        )
    os.killpg(sp.pgid, signal.SIGTERM)
    result = "terminated"
    deadline = time.monotonic() + grace_s
    while is_alive(sp.pid) and time.monotonic() < deadline:
        _reap(sp.pid)
        time.sleep(0.2)
    if is_alive(sp.pid):
        os.killpg(sp.pgid, signal.SIGKILL)
        result = "killed"
        deadline = time.monotonic() + 10.0
        while is_alive(sp.pid) and time.monotonic() < deadline:
            _reap(sp.pid)
            time.sleep(0.2)
    pid_file.unlink()
    return result


def own_running_instances(run_root: Path = SIM_RUN_DIR) -> list[SimProcess]:
    running = []
    for pid_file in sorted(Path(run_root).glob("inst*/pid.json")):
        sp = read_pid_file(pid_file)
        if is_alive(sp.pid) and is_owned(sp):
            running.append(sp)
    return running


def handshake(ports: SimPorts, timeout_s: float = 120.0) -> float:
    """Connect a Project AirSim client, fetch the topic list and disconnect; retried until timeout."""
    from projectairsim import ProjectAirSimClient

    start = time.monotonic()
    last_error: Exception | None = None
    while time.monotonic() - start < timeout_s:
        client = ProjectAirSimClient(port_topics=ports.topics, port_services=ports.services)
        try:
            client.connect()
            client.get_topic_info()
            return time.monotonic() - start
        except Exception as err:  # the server may still be starting
            last_error = err
            time.sleep(2.0)
        finally:
            client.disconnect()
    raise TimeoutError(f"handshake on {ports} failed for {timeout_s} s: {last_error}")
