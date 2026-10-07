"""Figures for a project status report, drawn from the records (docs/gates, runs/expert, data/): nothing typed by hand.

    env -u PYTHONPATH .venv/bin/python scripts/results_figures.py --out results/<date>/figures

- gates.png: every expert's 200-episode deterministic gate, outcome by outcome, against the 0.95 bar.
- training.png: training success (rolling over 300 episodes, every worker's monitor CSV in time order) and the 20-episode
  evaluations each run made along the way (evaluations.npz for runs with an in-training callback, eval_watch/ otherwise).
- rebalance.png: the detector's confidence for the pilot's target against the distance to it, the phase split per
  threshold, and where the transitions happen (data/<store>/detections.json and rebalance.json).
- path_efficiency.png: the pilot's flown lengths against the straight line and the PER interval
  (runs/rebalance/<store>_path_efficiency.json).
- layouts.png: s01's built layout beside one generated layout per new placement type (poisson, clusters, stacks).
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GATES = _ROOT / "docs" / "gates"
EXPERTS = _ROOT / "runs" / "expert"
BAR = 0.95
OUTCOMES = (("success", "#2a9d8f"), ("collision", "#e76f51"), ("out_of_bounds", "#f4a261"), ("timeout", "#8d99ae"))

# (gate file, checkpoint, label): the deterministic condition of each expert that was gated, in the order they were trained.
GATE_ROWS = (
    ("m2_gate.json", "best_model", "s01 run 2\nbest_model (M2)"),
    ("m2_gate.json", "final", "s01 run 2\nfinal"),
    ("m2d_gate.json", "best_model", "s01d run 1\nbest_model"),
    ("m2d_gate.json", "final", "s01d run 1\nfinal"),
    ("m2d_r5_gate.json", "final", "s01d run 5\nfinal"),
    ("m2d_r6_gate.json", "step_240000", "s01d run 6\nstep 240k"),
    ("m2d_r6_swa_gate.json", "avg_250k_550k", "s01d run 6\nmean 250k-550k"),
    ("m2d_r6s1_avg_gate.json", "avg_250k_end", "s01d run 6 s1\nmean 250k-end"),
)
# Training runs: (run directory, label).
RUNS = (("s01_r2", "s01 run 2"), ("s01d_r1", "s01d run 1"), ("s01d_r3", "s01d run 3"), ("s01d_r4", "s01d run 4"),
        ("s01d_r5", "s01d run 5"), ("s01d_r6", "s01d run 6"))


def gate_rates(path: Path, checkpoint: str) -> dict:
    report = json.load(open(path))["checkpoints"][checkpoint]["deterministic"]
    return {k: report[f"{k}_rate"] for k, _ in OUTCOMES} | {"n": report["n_episodes"]}


def fig_gates(out: Path) -> None:
    rows = [(label, gate_rates(GATES / f, ck)) for f, ck, label in GATE_ROWS if (GATES / f).is_file()]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for key, colour in OUTCOMES:
        values = np.array([r[key] for _, r in rows])
        ax.bar(x, values, bottom=bottom, color=colour, label=key.replace("_", " "), width=0.7)
        bottom += values
    for i, (_, r) in enumerate(rows):
        ax.text(i, r["success"] + 0.012, f"{r['success']:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(BAR, color="black", linestyle="--", linewidth=1)
    ax.text(-0.45, BAR + 0.012, f"gate bar {BAR}", ha="left", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in rows], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of 200 held-out episodes (deterministic)")
    ax.set_title("Expert exit gates: s01 passed (M2); s01d, with moving pillars, has climbed from 0.775 to 0.935 (M2d, open)")
    ax.legend(loc="lower left", ncol=4, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "gates.png", dpi=150)
    plt.close(fig)


def _monitor_episodes(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(cumulative env steps, success flag) of every training episode of a run, every session, in time order."""
    rows: list[tuple[float, int, int]] = []
    for session in sorted(run_dir.glob("monitor/session*")):
        base = 0.0
        for path in sorted(session.glob("*.csv")):
            with open(path) as handle:
                header = handle.readline()
                reader = csv.DictReader(handle)
                t0 = json.loads(header[1:]).get("t_start", 0.0) if header.startswith("#") else 0.0
                for row in reader:
                    flag = str(row["is_success"]).strip().lower()  # Monitor writes True/False, or 1.0/0.0 in older runs
                    rows.append((t0 + float(row["t"]), int(float(row["l"])), int(flag in ("true", "1", "1.0"))))
            base = max(base, t0)
    rows.sort()
    steps = np.cumsum([r[1] for r in rows])
    success = np.array([r[2] for r in rows], dtype=float)
    return steps, success


def _evaluations(run_dir: Path) -> tuple[list[int], list[float]]:
    npz = run_dir / "eval_logs" / "evaluations.npz"
    if npz.is_file():
        d = np.load(npz)
        return [int(t) for t in d["timesteps"]], [float(np.mean(row)) for row in d["successes"]]
    steps, rates = [], []
    for path in sorted(run_dir.glob("eval_watch/step_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        report = json.load(open(path))
        (name, ck), = report["checkpoints"].items()
        if ck["deterministic"].get("status") == "ok":
            steps.append(int(path.stem.split("_")[1]))
            rates.append(ck["deterministic"]["success_rate"])
    return steps, rates


def fig_training(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(RUNS)))
    for (run, label), colour in zip(RUNS, colours):
        run_dir = EXPERTS / run
        if not run_dir.is_dir():
            continue
        steps, success = _monitor_episodes(run_dir)
        if len(success) > 300:
            kernel = np.ones(300) / 300
            smooth = np.convolve(success, kernel, mode="valid")
            axes[0].plot(steps[299:] / 1e3, smooth, color=colour, label=f"{label} ({steps[-1] / 1e3:.0f}k steps)", linewidth=1.3)
        ev_steps, ev_rates = _evaluations(run_dir)
        if ev_steps:
            axes[1].plot(np.array(ev_steps) / 1e3, ev_rates, "o-", color=colour, label=label, markersize=3.5, linewidth=1)
    axes[0].set_title("Training success under each run's own rules (rolling 300 episodes;\nruns 5-6 end episodes at wider, training-only boundaries, so their curves sit lower)", fontsize=10)
    axes[0].set_xlabel("environment steps (thousands)")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Evaluations along the way: 20 fixed episodes, deterministic (noisy by design)")
    axes[1].set_xlabel("environment steps (thousands)")
    axes[1].axhline(BAR, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=150)
    plt.close(fig)


def fig_rebalance(out: Path, store: Path) -> None:
    from autofly_ue5.dataset.rebalance import build_rebalance

    manifest = json.load(open(store / "manifest.json"))
    detections = json.load(open(store / "detections.json"))
    rebalance = json.load(open(store / "rebalance.json"))
    states = {e["id"]: np.load(store / e["path"] / "steps.npz")["state"] for e in manifest["episodes"] if e["id"] in detections["episodes"]}
    dist = np.concatenate([states[eid][:, 0] for eid in states])
    score = np.concatenate([np.asarray(detections["episodes"][eid]["scores"]) for eid in states])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].scatter(dist, score, s=2, alpha=0.25, color="#264653")
    axes[0].axhline(rebalance["threshold"], color="#e76f51", linestyle="--", label=f"threshold {rebalance['threshold']}")
    axes[0].set_xlabel("distance to the target (m)")
    axes[0].set_ylabel("Grounding DINO confidence for the target phrase")
    axes[0].set_title(f"{len(score):,} records, {len(states)} episodes: confidence against distance", fontsize=10)
    axes[0].invert_xaxis()
    axes[0].legend(loc="lower left")
    thresholds = np.arange(0.5, 0.91, 0.05)
    p_seek, per_record = [], []
    for thr in thresholds:
        r = build_rebalance(manifest, detections, threshold=float(thr), alpha=0.0, allow_partial=True)
        p_seek.append(r["p0"][1])
        per_record.append(r["counts"]["records_above_threshold"] / sum(r["counts"]["records"]))
    axes[1].plot(thresholds, p_seek, "o-", color="#2a9d8f", label="P0(target seeking), one-way split (spec §10.3)")
    axes[1].plot(thresholds, per_record, "s--", color="#8d99ae", label="share of records above the threshold")
    axes[1].axhline(0.27, color="black", linestyle=":", linewidth=1)
    axes[1].text(0.505, 0.285, "the paper's 0.27", fontsize=8)
    axes[1].axvline(rebalance["threshold"], color="#e76f51", linestyle="--", linewidth=1)
    axes[1].set_xlabel("detection threshold")
    axes[1].set_ylabel("share of records")
    axes[1].set_title("Phase split against the threshold")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="upper right")
    transitions = [e["first_detection_distance_m"] for e in rebalance["episodes"] if e.get("first_detection_distance_m") is not None]
    fractions = [e["first_detection"] / e["records"] for e in rebalance["episodes"] if e["first_detection"] is not None]
    axes[2].hist(transitions, bins=20, color="#2a9d8f", alpha=0.85)
    axes[2].set_xlabel("distance to the target at the first confident detection (m)")
    axes[2].set_ylabel("episodes")
    axes[2].set_title(f"Transitions at {rebalance['threshold']}: median {np.median(transitions):.1f} m, "
                      f"{np.median(fractions):.0%} of the way\nP0 = ({rebalance['p0'][0]:.3f}, {rebalance['p0'][1]:.3f}), "
                      f"weights ({rebalance['weights'][0]:.3f}, {rebalance['weights'][1]:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "rebalance.png", dpi=150)
    plt.close(fig)


def fig_path_efficiency(out: Path, report_path: Path) -> None:
    report = json.load(open(report_path))
    eps = report["episodes"]
    flown = np.array([e["flown_m"] for e in eps])
    straight = np.array([e["straight_m"] for e in eps])
    lower = np.array([e["per_straight"] for e in eps])
    upper = np.array([e["per_physical"] for e in eps if e["per_physical"] is not None])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(straight, flown, s=14, color="#264653", alpha=0.8)
    lim = [min(straight.min(), flown.min()) - 2, max(straight.max(), flown.max()) + 2]
    axes[0].plot(lim, lim, "--", color="grey", linewidth=1, label="flown = straight line")
    axes[0].set_xlabel("straight line to the 5 m success disc (m)")
    axes[0].set_ylabel("flown trajectory (m, 3-D)")
    axes[0].set_title(f"{len(eps)} kept episodes: mean flown {flown.mean():.1f} m against {straight.mean():.1f} m straight")
    axes[0].legend()
    axes[1].hist(lower, bins=20, alpha=0.7, color="#8d99ae", label=f"lower bound (straight line): mean {lower.mean():.3f}")
    axes[1].hist(upper, bins=20, alpha=0.7, color="#2a9d8f", label=f"upper bound (grid optimum): mean {upper.mean():.3f}")
    axes[1].set_xlabel("path efficiency L_opt / max(L, L_opt) per episode")
    axes[1].set_ylabel("episodes")
    axes[1].set_title("The paper's PER for the pilot's expert (its models: 0.73-0.78)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "path_efficiency.png", dpi=150)
    plt.close(fig)


def fig_layouts(out: Path) -> None:
    import dataclasses

    from autofly_ue5.paths import SCENES_DIR
    from autofly_ue5.scenes.generate import generate_layout
    from autofly_ue5.scenes.model import load_registry, load_scene_file
    from autofly_ue5.scenes.paths import crossing_detours
    from autofly_ue5.scenes.reachability import check_reachability
    from autofly_ue5.scenes.resolve import resolve_scene

    registry = load_registry()
    cube = registry.assets["cube"]
    with_box = dataclasses.replace(registry, assets={**registry.assets, "crate": dataclasses.replace(cube, name="crate", footprint="box", role="obstacle")})
    base = json.loads((SCENES_DIR / "s01_white_pillars.json").read_text())

    def synthetic(sid, groups):
        import tempfile

        data = dict(base)
        data["id"] = sid
        data["obstacle_groups"] = groups
        path = Path(tempfile.mkdtemp()) / f"{sid}.json"
        path.write_text(json.dumps(data))
        return load_scene_file(path)

    panels = [("s01 (built): 80 pillars on a jittered grid", resolve_scene("s01").layout, registry)]
    panels.append(("poisson: 60 scattered obstacles, min 4 m apart (rocks)", generate_layout(synthetic("s03", [
        {"asset": "cylinder", "count": 60, "scale_range": {"xy": [1.0, 3.0], "z": [1.0, 2.5]}, "palette": ["white"],
         "placement": {"type": "poisson", "margin_m": 8.0, "min_distance_m": 4.0}}]), registry), registry))
    panels.append(("clusters: 6 clusters of 4-8 (tree clusters)", generate_layout(synthetic("s06", [
        {"asset": "cylinder", "count": 36, "scale_range": {"xy": [0.4, 0.8], "z": [6.0, 10.0]}, "palette": ["white"],
         "placement": {"type": "clusters", "margin_m": 8.0, "cluster_count": 6, "per_cluster": [4, 8], "radius_m": 3.5}}]), registry), registry))
    panels.append(("stacks: 12 stacks of 2-5 boxes (stacked crates)", generate_layout(synthetic("s07", [
        {"asset": "crate", "count": 40, "scale_range": {"xy": [1.0, 1.6], "z": [0.8, 1.2]}, "palette": ["white"],
         "placement": {"type": "stacks", "margin_m": 8.0, "stack_count": 12, "height_range": [2, 5]}}]), with_box), with_box))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.9))
    for ax, (title, layout, reg) in zip(axes, panels):
        scene_like = synthetic("s01", base["obstacle_groups"])
        reach = check_reachability(layout, scene_like.start_band, scene_like.target_band)
        detours = crossing_detours(layout, start_band=scene_like.start_band, target_band=scene_like.target_band, inflate_m=1.4, resolution_m=0.5)
        b = layout.bounds
        ax.add_patch(plt.Rectangle((b.x_min, b.y_min), b.width, b.height, fill=False, color="black", linewidth=1))
        for band, colour in ((scene_like.start_band, "#2a9d8f"), (scene_like.target_band, "#e9c46a")):
            for d in band:
                ax.add_patch(plt.Rectangle((b.x_min + d, b.y_min + d), b.width - 2 * d, b.height - 2 * d, fill=False, color=colour, linewidth=0.6, linestyle=":"))
        seen = set()
        for inst in layout.instances:
            if (inst.x, inst.y) in seen:
                continue
            seen.add((inst.x, inst.y))
            ax.add_patch(plt.Circle((inst.y, inst.x), inst.radius_m, color="#264653", alpha=0.85))
        ax.set_xlim(b.y_min - 2, b.y_max + 2)
        ax.set_ylim(b.x_min - 2, b.x_max + 2)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nreachability {'ok' if reach.ok else 'FAIL'}; detour max {detours['max']:.2f}, "
                     f"through-unreachable {detours['through_unreachable']}", fontsize=9)
        ax.set_xlabel("east (m)")
        ax.set_ylabel("north (m)")
    fig.suptitle("Layouts the generator produces (footprints drawn at the layout's radius; dotted: start and target bands)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "layouts.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--store", type=Path, default=_ROOT / "data" / "s01_pilot")
    p.add_argument("--path-efficiency", type=Path, default=_ROOT / "runs" / "rebalance" / "s01_pilot_path_efficiency.json")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_gates(args.out)
    fig_training(args.out)
    if (args.store / "rebalance.json").is_file():
        fig_rebalance(args.out, args.store)
    if args.path_efficiency.is_file():
        fig_path_efficiency(args.out, args.path_efficiency)
    fig_layouts(args.out)
    print(f"figures in {args.out}: {sorted(p.name for p in args.out.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
