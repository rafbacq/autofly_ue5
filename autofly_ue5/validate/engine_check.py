"""M0 engine readiness: versions, toolchain, plugins, platform checkout, GPU, Xid, editor opens on X11, out-of-ROOT paths.

Usage: env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.engine_check --out runs/m0/engine_check.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import (
    ENGINE_DIR,
    PLATFORM_DIR,
    RUNS_DIR,
    UE_CACHE_ENV,
    UNREAL_EDITOR,
    UNREAL_EDITOR_CMD,
    ZEN_DATA_DIR,
)

EPIC_CONFIG_DIR = Path.home() / ".config" / "Epic"
ZEN_DEFAULT_DATA = EPIC_CONFIG_DIR / "UnrealEngine" / "Common" / "Zen" / "Data"
EDITOR_PROBE_LOG = RUNS_DIR / "m0" / "editor_open.log"
# Written by UE unless prevented (-notraceserver, uebp_LogFolder) or cleaned up (the probe's empty project folder).
OUT_OF_ROOT_PATHS = {
    "unreal_trace_store": Path.home() / "UnrealEngine",
    "unreal_trace_server_pid": Path("/tmp/UnrealTraceServer.pid"),
    "documents_unreal_projects": Path.home() / "Documents" / "Unreal Projects",
    "documents_unreal_engine_logs": Path.home() / "Documents" / "Unreal Engine",
}
EXPECTED_BUILD = "5.7.4-51494982"
TOOLCHAIN = "v26_clang-20.1.8-rockylinux8"
FATAL_PATTERNS = ("Fatal error", "VK_ERROR_DEVICE_LOST", "Unhandled Exception", "Segmentation fault", "Critical error")
ENGINE_PLUGINS = {
    "PythonScriptPlugin": "Experimental/PythonScriptPlugin/PythonScriptPlugin.uplugin",
    "EditorScriptingUtilities": "Editor/EditorScriptingUtilities/EditorScriptingUtilities.uplugin",
    "SunPosition": "Runtime/SunPosition/SunPosition.uplugin",
    "ChaosVehiclesPlugin": "Experimental/ChaosVehiclesPlugin/ChaosVehiclesPlugin.uplugin",
}


def parse_build_version(text: str) -> str:
    data = json.loads(text)
    return f"{data['MajorVersion']}.{data['MinorVersion']}.{data['PatchVersion']}-{data['Changelist']}"


def count_xid(journal_text: str) -> int:
    return sum(1 for line in journal_text.splitlines() if "NVRM: Xid" in line)


def clang_version(text: str) -> str | None:
    match = re.search(r"clang version (\d+\.\d+\.\d+)", text)
    return match.group(1) if match else None


def find_fatal_lines(log_text: str) -> list[str]:
    return [line for line in log_text.splitlines() if any(p in line for p in FATAL_PATTERNS)]


def count_device_lost(log_paths: list[Path]) -> dict[str, int]:
    return {
        str(p): sum(1 for line in p.read_text(errors="replace").splitlines() if "VK_ERROR_DEVICE_LOST" in line)
        for p in log_paths
        if p.is_file()
    }


def find_zen_redirect_line(log_text: str, zen_data_dir: str) -> str | None:
    """The line proving Zen's data path was redirected to zen_data_dir, or None.

    -ZenDataPath=<path> (command line) outranks the UE_ZenDataPath env var (see resolution order in
    ZenServerInterface.cpp), so either counts as positive evidence; both quote the path back in the line.
    """
    needles = (
        f"Found environment variable UE_ZenDataPath={zen_data_dir}",
        f"Found command line override ZenDataPath={zen_data_dir}",
    )
    for line in log_text.splitlines():
        if any(needle in line for needle in needles):
            return line
    return None


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def xid_journal_command(since: str | None) -> list[str]:
    if since is None:
        return ["journalctl", "-k", "-b", "--no-pager"]
    # No -k here: journalctl's -k implies -b, which would hide Xid lines logged before a reboot.
    return ["journalctl", "_TRANSPORT=kernel", "--since", since, "--no-pager"]


def xid_count(since: str | None = None) -> int:
    return count_xid(_run(xid_journal_command(since)))


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def epic_config_usage() -> dict:
    out = _run(["du", "-sk", str(EPIC_CONFIG_DIR)]).split()
    return {"kib": int(out[0]) if out else None, "zen_default_data_exists": ZEN_DEFAULT_DATA.exists()}


def out_of_root_state() -> dict[str, bool]:
    return {name: path.exists() for name, path in OUT_OF_ROOT_PATHS.items()}


def open_editor_probe(wait_s: float) -> dict:
    """Open the editor (no project) on :1. It stays in this process's group, so stopping the job that runs
    engine_check (scripts/run_job.sh stop) also stops the editor; engine_check itself signals only the editor PID.
    The Project Browser may create an empty ~/Documents/Unreal Projects (SProjectDialog.cpp:1748-1752); it is removed
    afterwards only if this probe created it and it is still empty."""
    log = EDITOR_PROBE_LOG
    pid_file = RUNS_DIR / "m0" / "editor_probe.pid.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    # ue_project/ doesn't exist until Task 4; ValidateDataPath (ZenServerInterface.cpp) rejects a path it
    # can't create, so make ZEN_DATA_DIR ourselves rather than rely on the engine to create it.
    ZEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    projects_dir = OUT_OF_ROOT_PATHS["documents_unreal_projects"]
    projects_existed = projects_dir.exists()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update({"DISPLAY": ":1", "SDL_VIDEODRIVER": "x11", **UE_CACHE_ENV})
    # -ZenDataPath= is the highest-priority override in ZenServerInterface.cpp's resolution order, ahead of
    # the UE_ZenDataPath env var UE_CACHE_ENV also sets; passing both is belt-and-braces.
    cmd = [str(UNREAL_EDITOR), "-nosplash", "-notraceserver", f"-ZenDataPath={ZEN_DATA_DIR}", f"-abslog={log}"]
    with open(log.with_suffix(".stdout"), "wb") as out:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    pid_file.write_text(json.dumps({"pid": proc.pid, "pgid": os.getpgid(proc.pid), "cmd": cmd}, indent=2))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(2.0)
    alive = proc.poll() is None
    exit_code = None if alive else proc.returncode
    stop_result = "not_running"
    if alive:
        proc.terminate()
        stop_result = "terminated"
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
            stop_result = "killed"
    pid_file.unlink()
    projects_created = projects_dir.exists() and not projects_existed
    projects_removed = False
    if projects_created:
        try:
            projects_dir.rmdir()  # fails (and is reported) if the folder is not empty
            projects_removed = True
        except OSError:
            projects_removed = False
    text = log.read_text(errors="replace") if log.exists() else ""
    fatal = find_fatal_lines(text)
    return {
        "pid": proc.pid,
        "alive_after_wait": alive,
        "exit_code_before_stop": exit_code,
        "stop": stop_result,
        "fatal_lines": fatal[:20],
        "gpu_named_in_log": "RTX 4090" in text,
        "documents_unreal_projects": {"existed_before": projects_existed, "created_by_probe": projects_created,
                                      "removed_empty": projects_removed},
        "log": str(log),
        "pass": alive and not fatal and (not projects_created or projects_removed),
    }


def run_checks(editor_wait_s: float = 120.0) -> dict:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    epic_before = epic_config_usage()
    out_of_root_before = out_of_root_state()
    xid_boot_total = xid_count(None)
    checks: dict[str, dict] = {}
    build = parse_build_version((ENGINE_DIR / "Engine/Build/Build.version").read_text())
    checks["engine_version"] = {"value": build, "pass": build == EXPECTED_BUILD}
    checks["installed_build"] = {"pass": (ENGINE_DIR / "Engine/Build/InstalledBuild.txt").is_file()}
    clang = ENGINE_DIR / f"Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/{TOOLCHAIN}/x86_64-unknown-linux-gnu/bin/clang++"
    version = clang_version(_run([str(clang), "--version"])) if clang.is_file() else None
    checks["toolchain"] = {"clang": version, "pass": version == "20.1.8"}
    sdk = json.loads((ENGINE_DIR / "Engine/Config/Linux/Linux_SDK.json").read_text())
    checks["linux_sdk_json"] = {"main": sdk["MainVersion"], "pass": sdk["MainVersion"] == TOOLCHAIN}
    checks["editor_binaries"] = {
        "pass": os.access(UNREAL_EDITOR, os.X_OK) and os.access(UNREAL_EDITOR_CMD, os.X_OK)
    }
    present = {name: (ENGINE_DIR / "Engine/Plugins" / rel).is_file() for name, rel in ENGINE_PLUGINS.items()}
    checks["engine_plugins"] = {"present": present, "pass": all(present.values())}
    missing = [line.strip() for line in _run(["ldd", str(UNREAL_EDITOR)]).splitlines() if "not found" in line]
    checks["editor_shared_libs"] = {"missing": missing, "pass": not missing}
    head = _run(["git", "-C", str(PLATFORM_DIR), "rev-parse", "HEAD"]).strip()
    diff = subprocess.run(
        ["git", "-C", str(PLATFORM_DIR), "diff", "--quiet", "0975545", "4d878bf", "--",
         "unreal", "core_sim", "physics", "vehicle_apis", "simserver"]
    ).returncode
    checks["platform_checkout"] = {
        "head": head,
        "cpp_identical_to_v1_0_1": diff == 0,
        "pass": head.startswith("4d878bf") and diff == 0,
    }
    used, total = gpu_memory_mib()
    checks["gpu_idle"] = {"used_mib": used, "total_mib": total, "pass": used <= 2000}
    xid_before = xid_count(started_at)
    if checks["gpu_idle"]["pass"]:
        checks["editor_opens_x11"] = open_editor_probe(editor_wait_s)
        probe_log_text = EDITOR_PROBE_LOG.read_text(errors="replace") if EDITOR_PROBE_LOG.exists() else ""
        redirect_line = find_zen_redirect_line(probe_log_text, str(ZEN_DATA_DIR))
        zen_dir_ok = ZEN_DATA_DIR.is_dir() and any(ZEN_DATA_DIR.iterdir())
        checks["zen_data_path_redirected"] = {
            "matched_log_line": redirect_line,
            "zen_data_dir_exists_and_nonempty": zen_dir_ok,
            "pass": redirect_line is not None and zen_dir_ok,
        }
    else:
        checks["editor_opens_x11"] = {"pass": False, "skipped": "GPU busy"}
        checks["zen_data_path_redirected"] = {"pass": False, "skipped": "GPU busy"}
    xid_after = xid_count(started_at)
    # Xid lines from before this check (e.g. another program) are recorded, not failed on; counts from started_at on
    # span reboots, and M0 gates on the delta from this baseline plus an unchanged boot id.
    checks["nvidia_xid"] = {"since": started_at, "before": xid_before, "after": xid_after,
                            "current_boot_total_at_start": xid_boot_total, "pass": xid_after == xid_before}
    return {"started_at": started_at, "boot_id": boot_id(), "checks": checks, "epic_config_before": epic_before,
            "out_of_root_before": out_of_root_before, "pass": all(c["pass"] for c in checks.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--editor-wait", type=float, default=120.0)
    args = parser.parse_args(argv)
    report = run_checks(args.editor_wait)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({name: c["pass"] for name, c in report["checks"].items()}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
