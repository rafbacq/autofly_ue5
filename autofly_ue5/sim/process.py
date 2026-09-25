"""Launch, readiness and stop of simulator processes owned by this project.

Each launched process runs in its own session (PGID == PID) and is recorded in
runs/sim/inst<N>/pid.json. stop() only signals a process whose recorded PID is
alive, whose /proc cmdline[0] equals the recorded cmd[0], and whose process
group equals the recorded PGID. Unreal starts its helpers (ShaderCompileWorker,
zenserver, CrashReportClient) in their own process groups (UnixPlatformProcess.cpp:1048-1049),
so they are not signalled here; callers check for survivors with pgrep and report them.

Ownership (C1, 2026-09-24 review): each record also names the *run* that launched it -- the main process of a
training/gate/measurement run, as a (pid, start time) token, even when a SubprocVecEnv worker did the launching.
A run stops only its own slots, or slots whose run is gone (`sweep_orphaned_instances`). The old global sweep
stopped every simulator on the host, which made a relaunch in one slot kill the eval simulator in another --
measured in Task 8's run: all 77 Timeout faults and 14 of its 23 relaunches.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import signal
import subprocess
import time
from contextlib import contextmanager
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
KILL_WAIT_S = 10.0  # how long stop() waits for a SIGKILLed process to disappear


class NotOwnedError(RuntimeError):
    """The pid file points at a process this project did not launch."""


class SimExitedError(RuntimeError):
    """The simulator process exited before becoming ready."""


class SimReadyTimeout(TimeoutError):
    """The simulator process stayed alive but never opened its ports within the ready timeout."""


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
    # The run that owns this simulator (see the module docstring). None in records written before ownership
    # existed; such a record counts as orphaned.
    owner_pid: int | None = None
    owner_start_ticks: int | None = None


def proc_start_ticks(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat (start time in clock ticks since boot): with the pid, it names one process
    for its whole life, so a recycled pid cannot impersonate a dead run."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return int(stat.rsplit(")", 1)[1].split()[19])


@dataclass(frozen=True)
class RunOwner:
    pid: int
    start_ticks: int | None

    @classmethod
    def of(cls, pid: int) -> "RunOwner":
        return cls(pid, proc_start_ticks(pid))

    def is_alive(self) -> bool:
        return is_alive(self.pid) and proc_start_ticks(self.pid) == self.start_ticks


_run_owner: RunOwner | None = None


def set_run_owner(owner: RunOwner | None) -> None:
    """Called in each SubprocVecEnv worker with the main process's token, so the simulators a worker launches are
    recorded as the run's, not the worker's (a SIGKILLed main can leave hung workers alive, which would otherwise
    keep their slots looking owned forever)."""
    global _run_owner
    _run_owner = owner


def current_run_owner() -> RunOwner:
    return _run_owner if _run_owner is not None else RunOwner.of(os.getpid())


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


@contextmanager
def _slot_lock(directory: Path):
    """Serialises every read-modify-write of one slot's pid.json (launch, stop): without it, a close() finishing
    late on an abandoned thread could read, and then delete, the record of the simulator relaunched in its place."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "pid.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


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
    cmd: list[str], instance: int, ports: SimPorts, env: dict[str, str], run_root: Path = SIM_RUN_DIR,
    owner: RunOwner | None = None,
) -> SimProcess:
    directory = instance_dir(instance, run_root)
    owner = owner if owner is not None else current_run_owner()
    with _slot_lock(directory):
        pid_file = directory / "pid.json"
        if pid_file.exists():
            existing = read_pid_file(pid_file)
            if is_alive(existing.pid) and is_owned(existing):
                raise RuntimeError(f"instance {instance} is already running with pid {existing.pid}")
            pid_file.unlink()
        for port in (ports.topics, ports.services):
            if listening_pids(port):
                raise RuntimeError(f"port {port} is already in use")
        # Appended, with a header per launch: a relaunch must not erase an earlier session's output (C7).
        with open(directory / "stdout.log", "ab") as out:
            out.write(f"=== launch {time.strftime('%Y-%m-%d %H:%M:%S %z')} {cmd[0]} ===\n".encode())
            out.flush()
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
                start_new_session=True, cwd=directory,
            )
        log_path = next((c.split("=", 1)[1] for c in cmd if c.startswith("-abslog=")), str(directory / "stdout.log"))
        sp = SimProcess(
            pid=proc.pid, pgid=os.getpgid(proc.pid), instance=instance, topics_port=ports.topics,
            services_port=ports.services, cmd=list(cmd), log_path=log_path, started_unix=time.time(),
            owner_pid=owner.pid, owner_start_ticks=owner.start_ticks,
        )
        # Written whole or not at all: readers outside the lock (own_running_instances) never see half a record.
        partial = pid_file.with_suffix(".json.tmp")
        partial.write_text(json.dumps(asdict(sp), indent=2))
        os.replace(partial, pid_file)
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
            raise SimReadyTimeout(f"ports {sp.topics_port}/{sp.services_port} not open after {timeout_s} s; see {sp.log_path}")
        time.sleep(poll_s)


def stop(instance: int, grace_s: float = 30.0, run_root: Path = SIM_RUN_DIR, expected_pid: int | None = None) -> str:
    """Stop the slot's recorded simulator. `expected_pid`: stop it only if it is still that process -- a caller
    closing a simulator it launched earlier gets "superseded" (and nothing is signalled or unlinked) when the slot
    has since been relaunched."""
    directory = instance_dir(instance, run_root)
    with _slot_lock(directory):
        pid_file = directory / "pid.json"
        if not pid_file.exists():
            return "no_pid_file"
        sp = read_pid_file(pid_file)
        if expected_pid is not None and sp.pid != expected_pid:
            return "superseded"
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
            deadline = time.monotonic() + KILL_WAIT_S
            while is_alive(sp.pid) and time.monotonic() < deadline:
                _reap(sp.pid)
                time.sleep(0.2)
        if is_alive(sp.pid):
            return "unkillable"  # keep the record: the process still holds its ports and VRAM
        pid_file.unlink()
    return result


def own_running_instances(run_root: Path = SIM_RUN_DIR) -> list[SimProcess]:
    running = []
    for pid_file in sorted(Path(run_root).glob("inst*/pid.json")):
        try:
            sp = read_pid_file(pid_file)
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as err:  # removed or mid-write: not a record yet
            logging.getLogger(__name__).warning("skipping unreadable %s: %s", pid_file, err)
            continue
        if is_alive(sp.pid) and is_owned(sp):
            running.append(sp)
    return running


def _recorded_owner(sp: SimProcess) -> RunOwner | None:
    return None if sp.owner_pid is None else RunOwner(sp.owner_pid, sp.owner_start_ticks)


def stop_instance(instance: int, run_root: Path = SIM_RUN_DIR, owner: RunOwner | None = None) -> str:
    """Stop one slot on behalf of `owner` (default: this run). Refuses ("owner_alive_elsewhere") a slot another
    live run owns; never raises for a slot that is already empty or not ours to signal."""
    owner = owner if owner is not None else current_run_owner()
    pid_file = instance_dir(instance, run_root) / "pid.json"
    try:
        sp = read_pid_file(pid_file)
    except FileNotFoundError:
        return "no_pid_file"
    recorded = _recorded_owner(sp)
    if recorded is not None and recorded != owner and recorded.is_alive():
        return "owner_alive_elsewhere"
    try:
        # expected_pid: the decision above came from an unlocked read; if the slot was relaunched since, leave it.
        return stop(instance, run_root=run_root, expected_pid=sp.pid)
    except NotOwnedError:
        return "not_owned"


def stop_instances(instances, run_root: Path = SIM_RUN_DIR, owner: RunOwner | None = None) -> list[dict]:
    """stop_instance() for each listed slot -- the teardown for a run's own simulators, and nothing else."""
    return [{"instance": i, "result": stop_instance(i, run_root, owner)} for i in instances]


def sweep_orphaned_instances(run_root: Path = SIM_RUN_DIR) -> list[dict]:
    """Stop every owned, running simulator whose run is gone (dead owner, or a record from before ownership) --
    what a crashed earlier run leaves holding VRAM. Simulators of a live run, including another concurrent one,
    are left alone. `scripts/launch_sim.py` exits after launching, so its simulators count as orphans here;
    `scripts/stop_sim.py` stops a slot explicitly."""
    swept = []
    for sp in own_running_instances(run_root):
        recorded = _recorded_owner(sp)
        if recorded is not None and recorded.is_alive():
            continue
        # expected_pid: the sweep stops slots one after another (up to ~40 s each), so its listing can go stale.
        result = stop(sp.instance, run_root=run_root, expected_pid=sp.pid)
        swept.append({"instance": sp.instance, "pid": sp.pid, "result": result})
    return swept


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
