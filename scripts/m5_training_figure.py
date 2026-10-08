"""One M5 training run at a glance: its training episodes' outcomes and its eval-watch checkpoints, from the run's own files.

    env -u PYTHONPATH .venv/bin/python scripts/m5_training_figure.py --run-root runs/expert/s02_r1 \
        --train-record docs/gates/m5_s02_train.json \
        --out results/<date>/figures/s02_r1_training.png --json results/<date>/figures/s02_r1_training.json

- Left: the rolling success, collision and out-of-bounds rates of every training episode, from every worker's Monitor
  CSV (`monitor/session*/*.monitor.csv`) put in time order and placed on the run's cumulative step count. Since
  2026-10-02 Monitor writes `outcome` and `collision_source` per episode, so collisions under the training-only
  boundary (`footprint_margin`) can be told from those under the task's rule. Training episodes follow the run's
  training rules (a mesh scene's 0.9 m footprint rule plus its margin), so the curves sit below what the same policy
  scores under the task's rules.
- Right: the eval watch (`eval_watch/step_*.json`): 20 deterministic episodes per checkpoint on the training-time
  evaluation seeds, with each checkpoint's footprint and out-of-bounds endings. A record still being written
  (`*.partial.json`) or one that did not finish is skipped. Twenty-episode points swing between neighbours
  (MEMORY.md, run 4: a 0.85 checkpoint scored 0.55 on 40 episodes), so the panel prints the largest swing;
  checkpoints are chosen on 100 fresh episodes.

It launches nothing and writes only --out and --json.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BAR = 0.95
WINDOW = 300
# Categorical slots 1-3 of the dataviz reference palette (validated: CVD ΔE ≥ 9.2, normal-vision ΔE ≥ 24 on white);
# every series is also direct-labelled, since aqua sits under 3:1 contrast.
COLOURS = {"success": "#2a78d6", "collision": "#eb6834", "out_of_bounds": "#1baf7a"}
INK, MUTED = "#222222", "#6b6b6b"
_STEP_FILE = re.compile(r"^step_(\d+)\.json$")


def training_episodes(run_root: Path) -> dict:
    """Every training episode of the run, every session and worker, in order of the time it ended.

    Returns {"steps": cumulative env steps at each episode's end, "outcome": [...], "collision_source": [...]}.
    Monitor's `t` is seconds since its own `t_start` (the CSV's first line), so episodes from different workers are
    ordered by t_start + t. Rows without an `outcome` (older runs) fall back to `is_success`."""
    rows: list[tuple[float, int, str, str]] = []
    for session in sorted(run_root.glob("monitor/session*")):
        for path in sorted(session.glob("*.monitor.csv")):
            with open(path, newline="") as handle:
                first = handle.readline()
                t0 = json.loads(first[1:]).get("t_start", 0.0) if first.startswith("#") else 0.0
                if not first.startswith("#"):
                    handle.seek(0)
                for row in csv.DictReader(handle):
                    outcome = (row.get("outcome") or "").strip()
                    if not outcome:
                        outcome = "success" if str(row.get("is_success", "")).strip().lower() in ("true", "1", "1.0") else "other"
                    rows.append((t0 + float(row["t"]), int(float(row["l"])), outcome, (row.get("collision_source") or "").strip()))
    rows.sort(key=lambda r: r[0])
    return {
        "steps": np.cumsum([r[1] for r in rows]).astype(int),
        "outcome": [r[2] for r in rows],
        "collision_source": [r[3] for r in rows],
    }


def rolling_rate(flags: np.ndarray, window: int) -> np.ndarray:
    """Mean of each `window` consecutive flags (length len(flags) - window + 1; empty if there are fewer)."""
    if len(flags) < window:
        return np.array([])
    return np.convolve(np.asarray(flags, dtype=float), np.ones(window) / window, mode="valid")


def eval_watch(run_root: Path) -> list[dict]:
    """The finished deterministic eval-watch records, by checkpoint step: success rate and ending counts."""
    points = []
    for path in run_root.glob("eval_watch/step_*.json"):
        match = _STEP_FILE.match(path.name)  # step_N.partial.json does not match
        if not match:
            continue
        report = json.loads(path.read_text())
        for ck in report.get("checkpoints", {}).values():
            det = ck.get("deterministic", {})
            if det.get("status") != "ok":
                continue
            episodes = det.get("per_episode", [])
            outcomes = [e["outcome"] for e in episodes]
            sources = [e.get("collision_source") for e in episodes if e["outcome"] == "collision"]
            points.append({
                "step": int(match.group(1)),
                "episodes": det["n_episodes"],
                "success_rate": det["success_rate"],
                "successes": outcomes.count("success"),
                "footprint": sources.count("footprint"),
                "other_collisions": len(sources) - sources.count("footprint"),
                "out_of_bounds": outcomes.count("out_of_bounds"),
                "timeout": outcomes.count("timeout"),
            })
    return sorted(points, key=lambda p: p["step"])


def _record_line(record: dict | None) -> str:
    if not record:
        return ""
    faults = record.get("backend_faults", {})
    n_faults = sum(faults.get("fault_counts", {}).values())
    relaunches = faults.get("relaunch_count", 0)
    return (f"{record['num_timesteps']:,} steps in {record['wall_clock_s'] / 3600:.1f} h "
            f"({record['env_steps_per_s']:.1f} steps/s, {record['instances']} simulators), "
            f"{sum(record.get('outcome_histogram', {}).values()):,} episodes, "
            f"{n_faults} backend faults, {relaunches} relaunches")


def draw(run_root: Path, out: Path, record: dict | None = None, window: int = WINDOW) -> dict:
    """Draw the figure into `out` and return its numbers (what --json writes)."""
    eps = training_episodes(run_root)
    points = eval_watch(run_root)
    outcomes = np.array(eps["outcome"])
    sources = np.array(eps["collision_source"])
    steps = eps["steps"]
    flags = {
        "success": outcomes == "success",
        "collision": outcomes == "collision",
        "out_of_bounds": outcomes == "out_of_bounds",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.1, 1]})
    title = f"{run_root.name}: training and the eval watch"
    line = _record_line(record)
    fig.suptitle(title + (f"\n{line}" if line else ""), fontsize=11.5, color=INK)

    ax = axes[0]
    labels = {"success": "success", "collision": "collision", "out_of_bounds": "out of bounds"}
    final_rates = {}
    for key, flag in flags.items():
        rate = rolling_rate(flag, window)
        if not len(rate):
            continue
        x = steps[window - 1:] / 1e3
        ax.plot(x, rate, color=COLOURS[key], linewidth=2, label=labels[key])
        final_rates[key] = float(rate[-1])
    if final_rates:  # direct labels at the curves' ends, pushed apart so they never overlap
        placed: list[float] = []
        for key, value in sorted(final_rates.items(), key=lambda kv: kv[1]):
            y = max(value, placed[-1] + 0.05) if placed else value
            placed.append(y)
            ax.annotate(f"{labels[key]} {value:.2f}", (steps[-1] / 1e3, value), xytext=(steps[-1] / 1e3 * 1.01, y),
                        textcoords="data", va="center", fontsize=9, color=INK)
    n_coll = int(flags["collision"].sum())
    n_margin = int(((outcomes == "collision") & (sources == "footprint_margin")).sum())
    ax.set_title(f"Training episodes under the training rules (rolling {window} episodes)", fontsize=10.5, color=INK)
    if n_coll:
        ax.text(0.01, 0.98, f"{n_margin:,} of {n_coll:,} collisions ended at the training-only margin,\n"
                "not at the task's rule; training curves sit below the task's scores",
                transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
    ax.set_xlabel("environment steps (thousands)", color=INK)
    ax.set_ylabel("share of episodes", color=INK)
    ax.set_ylim(0, 1)
    if len(steps):
        ax.set_xlim(0, steps[-1] / 1e3 * 1.16)
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)

    ax = axes[1]
    if points:
        x = np.array([p["step"] for p in points]) / 1e3
        n = np.array([p["episodes"] for p in points], dtype=float)
        width = (x[1] - x[0]) * 0.3 if len(x) > 1 else 10.0
        ax.bar(x - width / 2, [p["footprint"] / k for p, k in zip(points, n)], width=width * 0.92,
               color=COLOURS["collision"], alpha=0.55, label="footprint-rule endings (share)")
        ax.bar(x + width / 2, [p["out_of_bounds"] / k for p, k in zip(points, n)], width=width * 0.92,
               color=COLOURS["out_of_bounds"], alpha=0.55, label="out-of-bounds endings (share)")
        ax.plot(x, [p["success_rate"] for p in points], "o-", color=COLOURS["success"], linewidth=2, markersize=6,
                label="success", zorder=3)
        for xi, p in zip(x, points):
            ax.annotate(f"{p['successes']}", (xi, p["success_rate"]), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.set_xlim(0, x[-1] + (x[1] - x[0] if len(x) > 1 else 50))
    ax.axhline(BAR, color=INK, linestyle="--", linewidth=1)
    ax.text(0.01, BAR, f" bar {BAR}", transform=ax.get_yaxis_transform(), va="bottom", fontsize=8.5, color=INK)
    ax.set_title("Eval watch: 20 deterministic episodes per checkpoint, under the task's rules", fontsize=10.5, color=INK)
    rates = [p["success_rate"] for p in points]
    swing = max((abs(a - b) for a, b in zip(rates, rates[1:])), default=0.0)
    ax.text(0.5, -0.2, f"Numbers above the points: successes of 20. Twenty-episode points are noisy (neighbours differ "
            f"by up to {swing:.2f} here);\ncheckpoints are chosen on 100 fresh episodes, and the pick is gated on 200.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.set_xlabel("checkpoint (thousands of steps)", color=INK)
    ax.set_ylabel("share of the 20 episodes", color=INK)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False)

    for a in axes:
        a.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    return {
        "run_root": str(run_root),
        "figure": str(out),
        "training": {
            "episodes": len(outcomes),
            "steps": int(steps[-1]) if len(steps) else 0,
            "window": window,
            "outcomes": {k: int((outcomes == k).sum()) for k in sorted(set(outcomes.tolist()))},
            "collision_sources": {k or "none": int(((outcomes == "collision") & (sources == k)).sum())
                                  for k in sorted(set(sources[outcomes == "collision"].tolist()))},
            "final_rolling_rates": final_rates,
        },
        "eval_watch": points,
        "train_record": line or None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--train-record", type=Path, default=None, help="the run's training record, for the subtitle")
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args(argv)
    record = json.loads(args.train_record.read_text()) if args.train_record else None
    summary = draw(args.run_root, args.out, record, args.window)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("training", "train_record")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
