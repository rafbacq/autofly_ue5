"""Watch a training run while it trains: a text dashboard every --interval seconds, and optionally a progress PNG.

Reads only what the run already writes (it never touches the run or its simulators):
  - <run-root>/sessions.json: the session, the scene and the observation it trains on;
  - <run-root>/monitor/session<k>/<instance>.monitor.csv: every finished training episode (return, length, success,
    and -- for runs since 2026-10-02 -- its outcome and collision source);
  - <run-root>/eval_logs/evaluations.npz: the periodic fixed-seed evaluations (FaultAwareEvalCallback);
  - <run-root>/checkpoints/: the newest checkpoint's step count;
  - the job log (runs/jobs/<job>.log): SB3's own progress table (total_timesteps, fps) and the fault, relaunch and
    mover-inference lines the wrappers print;
  - nvidia-smi, /proc/meminfo and the disk, for the host.

    env -u PYTHONPATH .venv/bin/python scripts/watch_training.py --run-root runs/expert/s01d_r1 --job m2d_train
    env -u PYTHONPATH .venv/bin/python scripts/watch_training.py --run-root runs/expert/s01d_r1 --job m2d_train \\
        --interval 0 --png runs/expert/s01d_r1/progress.png        # one snapshot, plus a plot

TensorBoard shows the same run's curves (rollout/success_rate, eval/success_rate, outcomes/*, train/*):
    .venv/bin/tensorboard --logdir runs/expert/s01d_r1/tensorboard --host 127.0.0.1 --port 6006
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from autofly_ue5.paths import RUNS_DIR  # noqa: E402

CHECKPOINT_RE = re.compile(r"^rl_model_(\d+)_steps\.zip$")
SB3_ROW_RE = re.compile(r"^\|\s*(\w+)\s*\|\s*([-+0-9.eE]+)\s*\|")
LOG_COUNTERS = {
    "faults": re.compile(r"^FAULT instance \d+: caught (\w+)"),
    "relaunches": re.compile(r"^RELAUNCH instance \d+: relaunching"),
    "mover_inferred": re.compile(r"^MOVER instance \d+:"),
    "move_refused": re.compile(r"^MOVE-REFUSED "),
    "fatal": re.compile(r"^FATAL instance \d+:"),
    "skipped_evaluations": re.compile(r"^WARNING: evaluation at \d+ timesteps could not finish"),
}


def _monitor_rows(run_root: Path) -> list[dict]:
    """Every training episode of every session, oldest first (the eval env's monitor is not training)."""
    rows = []
    for session_dir in sorted((run_root / "monitor").glob("session*")):
        for path in sorted(session_dir.glob("*.monitor.csv")):
            with open(path, newline="") as handle:
                lines = [line for line in handle if not line.startswith("#")]
            for row in csv.DictReader(lines):
                try:
                    rows.append({"r": float(row["r"]), "l": int(row["l"]), "t": float(row["t"]),
                                 "session": session_dir.name, "worker": path.name.split(".")[0],
                                 "is_success": row.get("is_success") == "True",
                                 "outcome": row.get("outcome") or None,
                                 "collision_source": row.get("collision_source") or None})
                except (KeyError, ValueError):
                    continue  # a row being written right now
    rows.sort(key=lambda r: (r["session"], r["t"]))
    return rows


def _rolling(rows: list[dict], window: int) -> dict:
    recent = rows[-window:]
    if not recent:
        return {"episodes": 0}
    out = {"episodes": len(recent), "success": sum(r["is_success"] for r in recent) / len(recent),
           "mean_return": float(np.mean([r["r"] for r in recent])), "mean_length": float(np.mean([r["l"] for r in recent]))}
    if any(r["outcome"] for r in recent):
        outcomes = Counter(r["outcome"] for r in recent)
        out["outcomes"] = {k: v / len(recent) for k, v in sorted(outcomes.items())}
        sources = Counter(r["collision_source"] or "sim" for r in recent if r["outcome"] == "collision")
        out["collision_sources"] = dict(sources)
    return out


def _evaluations(run_root: Path) -> list[dict]:
    path = run_root / "eval_logs" / "evaluations.npz"
    if not path.is_file():
        return []
    try:
        data = np.load(path)
        return [{"timesteps": int(t), "success": float(np.mean(s)), "mean_return": float(np.mean(r))}
                for t, s, r in zip(data["timesteps"], data["successes"], data["results"])]
    except Exception:  # being rewritten right now
        return []


def _newest_checkpoint_steps(run_root: Path) -> int | None:
    steps = [int(m.group(1)) for p in (run_root / "checkpoints").glob("rl_model_*_steps.zip") if (m := CHECKPOINT_RE.match(p.name))]
    return max(steps) if steps else None


def _job_log(path: Path | None) -> dict:
    out: dict = {"counts": {k: 0 for k in LOG_COUNTERS}, "fault_names": {}, "sb3": {}, "last_lines": [], "finished": None}
    if path is None or not path.is_file():
        return out
    faults: Counter[str] = Counter()
    with open(path, errors="replace") as handle:
        lines = handle.read().splitlines()
    for line in lines:
        for key, pattern in LOG_COUNTERS.items():
            m = pattern.match(line)
            if m:
                out["counts"][key] += 1
                if key == "faults":
                    faults[m.group(1)] += 1
        m = SB3_ROW_RE.match(line)
        if m:
            out["sb3"][m.group(1)] = float(m.group(2))
    exit_file = path.with_suffix(".exit")  # run_job.sh writes the exit code here when the job ends (and clears it at start)
    if exit_file.is_file():
        out["finished"] = f"job {path.stem} finished: exit={exit_file.read_text().strip()}"
    out["fault_names"] = dict(faults)
    out["last_lines"] = [line for line in lines if line.strip()][-3:]
    return out


def _host() -> dict:
    out: dict = {}
    try:
        used, total, util = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.splitlines()[0].split(", ")
        out["gpu"] = {"used_mib": int(used), "total_mib": int(total), "util_pct": int(util)}
    except Exception:
        out["gpu"] = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                out["ram_available_gb"] = int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    out["disk_free_gb"] = shutil.disk_usage(RUNS_DIR if RUNS_DIR.exists() else Path.cwd()).free / 1e9
    return out


def summarize(run_root: Path, job_log: Path | None = None, *, window: int = 200, host: bool = True) -> dict:
    run_root = Path(run_root)
    sessions = json.loads((run_root / "sessions.json").read_text())["sessions"] if (run_root / "sessions.json").is_file() else []
    rows = _monitor_rows(run_root)
    current = sessions[-1] if sessions else {}
    started = time.mktime(time.strptime(current["started"], "%Y-%m-%d %H:%M:%S")) if current.get("started") else None
    return {
        "run_root": str(run_root),
        "session": current.get("index"),
        "identity": current.get("identity"),
        "started": current.get("started"),
        "elapsed_s": (time.time() - started) if started else None,
        "episodes_total": len(rows),
        "rolling": _rolling(rows, window),
        "window": window,
        "evaluations": _evaluations(run_root),
        "newest_checkpoint_steps": _newest_checkpoint_steps(run_root),
        "final_saved": (run_root / "final.zip").is_file(),
        "log": _job_log(job_log),
        "host": _host() if host else None,
        "rolling_series": _series(rows, window),
    }


def _series(rows: list[dict], window: int) -> dict:
    """Rolling success rate and return over finished episodes, for the plot."""
    if not rows:
        return {"episode": [], "success": [], "return": []}
    success = np.array([r["is_success"] for r in rows], dtype=float)
    returns = np.array([r["r"] for r in rows], dtype=float)
    w = max(1, min(window, len(rows)))
    kernel = np.ones(w) / w
    return {"episode": list(range(w, len(rows) + 1)), "success": np.convolve(success, kernel, "valid").tolist(),
            "return": np.convolve(returns, kernel, "valid").tolist()}


def _fmt_s(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render_text(summary: dict) -> str:
    identity = summary["identity"] or {}
    obs = identity.get("obs_config") or {}
    sb3 = summary["log"]["sb3"]
    lines = [
        f"{summary['run_root']}  scene {identity.get('scene', '?')}  obs {obs.get('depth_frames', '?')}x"
        f"{obs.get('depth_dtype', '?')}  session {summary['session']}  started {summary['started']}  "
        f"elapsed {_fmt_s(summary['elapsed_s'])}",
        f"timesteps {int(sb3['total_timesteps']):,} at {sb3.get('fps', 0):.1f} steps/s (SB3)" if "total_timesteps" in sb3
        else "timesteps: no SB3 progress table in the log yet",
        f"newest checkpoint {summary['newest_checkpoint_steps'] or '-'}   final.zip {'saved' if summary['final_saved'] else 'not yet'}",
    ]
    r = summary["rolling"]
    if r["episodes"]:
        line = (f"episodes {summary['episodes_total']} | last {r['episodes']}: success {r['success']:.2f}  "
                f"return {r['mean_return']:.1f}  length {r['mean_length']:.0f}")
        if "outcomes" in r:
            line += "  | " + "  ".join(f"{k} {v:.2f}" for k, v in r["outcomes"].items())
            if r.get("collision_sources"):
                line += "  (collisions: " + ", ".join(f"{k} {v}" for k, v in sorted(r["collision_sources"].items())) + ")"
        lines.append(line)
    else:
        lines.append("episodes: none finished yet")
    evals = summary["evaluations"]
    if evals:
        best = max(evals, key=lambda e: e["mean_return"])
        recent = " | ".join(f"{e['timesteps'] // 1000}k {e['success']:.2f}" for e in evals[-6:])
        lines.append(f"evaluations (fixed seeds, deterministic): {recent}   best return at {best['timesteps'] // 1000}k "
                     f"(success {best['success']:.2f})")
    else:
        lines.append("evaluations: none yet")
    log = summary["log"]
    counts = log["counts"]
    faults = ", ".join(f"{k} {v}" for k, v in sorted(log["fault_names"].items())) or "none"
    lines.append(f"faults: {faults}   relaunches {counts['relaunches']}   mover-inferred collisions {counts['mover_inferred']}"
                 f"   refused moves {counts['move_refused']}   FATAL {counts['fatal']}"
                 f"   skipped evaluations {counts['skipped_evaluations']}")
    if log["finished"]:
        lines.append(log["finished"])
    host = summary["host"]
    if host:
        gpu = host.get("gpu")
        gpu_text = f"GPU {gpu['used_mib'] / 1024:.1f}/{gpu['total_mib'] / 1024:.1f} GiB, {gpu['util_pct']}%" if gpu else "GPU ?"
        lines.append(f"{gpu_text}   RAM available {host.get('ram_available_gb', 0):.1f} GB   disk free "
                     f"{host['disk_free_gb']:.0f} GB")
    return "\n".join(lines)


def plot(summary: dict, path: Path) -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11, 4.2), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)
    series = summary["rolling_series"]
    if series["episode"]:
        ax1.plot(series["episode"], series["success"], color="tab:green", label=f"success (rolling {summary['window']})")
        ax1b = ax1.twinx()
        ax1b.plot(series["episode"], series["return"], color="tab:blue", alpha=0.5, label="return")
        ax1b.set_ylabel("return")
    ax1.set_xlabel("training episodes")
    ax1.set_ylabel("success rate")
    ax1.set_ylim(0, 1)
    ax1.set_title("training episodes")
    evals = summary["evaluations"]
    if evals:
        ax2.plot([e["timesteps"] for e in evals], [e["success"] for e in evals], "o-", color="tab:orange")
    ax2.axhline(0.95, color="0.5", ls="--", lw=0.8)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("timesteps")
    ax2.set_ylabel("success rate")
    ax2.set_title("fixed-seed evaluations (gate bar 0.95)")
    identity = summary["identity"] or {}
    fig.suptitle(f"{summary['run_root']} (scene {identity.get('scene', '?')}), elapsed {_fmt_s(summary['elapsed_s'])}")
    fig.savefig(path, dpi=110)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job", default=None, help="the run_job.sh job name whose log to read (runs/jobs/<job>.log)")
    parser.add_argument("--interval", type=float, default=60.0, help="seconds between refreshes; 0 prints once")
    parser.add_argument("--window", type=int, default=200, help="episodes in the rolling statistics")
    parser.add_argument("--png", type=Path, default=None, help="also write a progress plot here at every refresh")
    args = parser.parse_args(argv)
    job_log = RUNS_DIR / "jobs" / f"{args.job}.log" if args.job else None
    while True:
        summary = summarize(args.run_root, job_log, window=args.window)
        print(time.strftime("%H:%M:%S"), "-" * 100)
        print(render_text(summary), flush=True)
        if args.png is not None:
            plot(summary, args.png)
        if args.interval <= 0 or summary["log"]["finished"]:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
