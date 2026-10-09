"""What a rebalance document says, as figures for final_results/ (plan 6 step 7) and the numbers they draw: the phase
split of spec §10.3 (AutoFly App. A.2.4) overall and per scene, each target's first confident detection, and the
resampling weights against the paper's.

    env -u PYTHONPATH .venv/bin/python scripts/rebalance_figures.py --rebalance runs/rebalance/final/rebalance.json \\
        --out-dir runs/rebalance/final/figures [--raw data/<store> ...]

The document is scripts/rebalance_dataset.py's, in either of its forms: one store's (autofly_ue5_rebalance/1, the M3
pilot's data/s01_pilot/rebalance.json among them, written before by_target and the sparse search existed) or the union
of several stores (autofly_ue5_rebalance_union/1, a prototype's scenes and batches), dense or sparse. Writes into
--out-dir, and nowhere else:
- rebalance_phases.png: per scene and over all, the share of records in each phase (obstacle avoidance until the
  first detection above the threshold, target seeking from it on), against the paper's 0.73 / 0.27; and the weights
  that even the phases out, against the paper's 0.68 / 1.85.
- rebalance_targets.png: per target, the distance to it at each episode's first confident detection (state[0] there,
  recorded when the document was made with the stores' states), its median, and the share of its episodes the detector
  never confirmed it in, which the weights cannot see (plan 6 revision 1 item 6: the targets the detector never finds).
  A document without distances shows where in the episode the first detection fell instead.
- rebalance_summary.json: every number drawn, and where each came from; rebalance_summary.md: the same as tables.

What the document leaves out is not guessed. A union names every episode's target and store; one store's document
names neither the targets nor, beyond its ids, the scenes. --raw <store> reads that store's manifest.json (read only)
for both; without it an episode's scene is read from its id as the collector writes it (<scene>_<seed>), and its
target is "(target not recorded)". Refused: a file that is not a rebalance document, a document whose episodes do not
add up to its own record counts, and an output that is a committed record under docs/gates/.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from autofly_ue5 import evidence  # noqa: E402
from autofly_ue5.dataset.rebalance import FORMAT, PHASES, UNION_FORMAT  # noqa: E402
from autofly_ue5.paths import ROOT  # noqa: E402
from autofly_ue5.report.style import INK, INK_2, MUTED, SLOTS, STYLE, SURFACE, scene_label  # noqa: E402

OUTPUTS = ("rebalance_phases.png", "rebalance_targets.png", "rebalance_summary.json", "rebalance_summary.md")
# AutoFly App. A.2.4: its trajectories were 73 % obstacle avoidance; resampled towards uniform with weights 0.68 / 1.85
# at a detection confidence of 0.7 (dataset/rebalance.py's docstring; the status report, section 2).
PAPER = {"p0": [0.73, 0.27], "weights": [0.68, 1.85], "threshold": 0.7}
PHASE_COLOURS = (SLOTS[0], SLOTS[1])  # obstacle avoidance, target seeking: the same in both figures
PHASE_NAMES = ("obstacle avoidance", "target seeking")
UNRECORDED_TARGET = "(target not recorded)"
UNKNOWN_SCENE = "unknown"
COLLECTOR_ID = re.compile(r"(?P<scene>.+)_(?P<seed>\d+)")  # collect/collector.py: f"{scene.id}_{seed}"
ROW_IN = 0.27


class RebalanceFigureError(ValueError):
    """A document this tool cannot draw as it is."""


def _r(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _median(values) -> float | None:
    return float(np.median(values)) if len(values) else None


# ----------------------------------------------------------------------------------------------------------------------
# Reading.
# ----------------------------------------------------------------------------------------------------------------------
def load_document(path: Path) -> tuple[dict, str]:
    blob = Path(path).read_bytes()
    try:
        doc = json.loads(blob)
    except ValueError as err:
        raise RebalanceFigureError(f"{path}: not JSON ({err})") from None
    if not isinstance(doc, dict) or doc.get("format") not in (FORMAT, UNION_FORMAT):
        raise RebalanceFigureError(f"{path} is not a rebalance document (format {FORMAT} or {UNION_FORMAT}, "
                                   f"scripts/rebalance_dataset.py); its format is {doc.get('format')!r}"
                                   if isinstance(doc, dict) else f"{path} is not a rebalance document")
    for key in ("phases", "threshold", "p0", "weights", "counts", "episodes"):
        if key not in doc:
            raise RebalanceFigureError(f"{path}: a rebalance document without {key!r}")
    if list(doc["phases"]) != list(PHASES):
        raise RebalanceFigureError(f"{path}: phases {doc['phases']}, this tool draws {list(PHASES)}")
    return doc, hashlib.sha256(blob).hexdigest()


def read_manifests(raws) -> dict[str, dict]:
    """{store name: {episode id: {"scene", "target"}}} from each raw store's manifest.json."""
    out = {}
    for raw in raws:
        manifest = json.loads((Path(raw) / "manifest.json").read_text())
        out[manifest["name"]] = {e["id"]: {"scene": e.get("scene"), "target": e.get("target_name")}
                                 for e in manifest.get("episodes", [])}
    return out


def episode_table(doc: dict, manifests: dict) -> tuple[list[dict], str | None]:
    """Every episode with its store, scene and target, and where the scenes came from ("manifests", "episode ids",
    a mix of both, or None when no episode's scene is known)."""
    union = doc["format"] == UNION_FORMAT
    rows, sources = [], set()
    for e in doc["episodes"]:
        store = e["store"] if union else doc.get("dataset")
        known = manifests.get(store, {}).get(e["id"], {})
        scene, source = known.get("scene"), "manifests" if known.get("scene") else None
        if scene is None:
            match = COLLECTOR_ID.fullmatch(str(e["id"]))
            scene, source = (match.group("scene"), "episode ids") if match else (UNKNOWN_SCENE, None)
        sources.add(source)
        rows.append({"store": store, "id": e["id"], "scene": scene,
                     "target": e.get("target") or known.get("target") or UNRECORDED_TARGET,
                     "records": int(e["records"]), "first_detection": e.get("first_detection"),
                     "phase_records": [int(v) for v in e["phase_records"]],
                     "distance_m": e.get("first_detection_distance_m"),
                     "on_target": e.get("first_detection_on_target")})
    known_sources = sorted(s for s in sources if s)
    scene_source = None if not known_sources else (known_sources[0] if len(known_sources) == 1 else "mixed")
    return rows, scene_source


# ----------------------------------------------------------------------------------------------------------------------
# The numbers.
# ----------------------------------------------------------------------------------------------------------------------
def summarise_document(doc: dict, *, path: str, sha256: str, manifests: dict) -> dict:
    rows, scene_source = episode_table(doc, manifests)
    counts = doc["counts"]
    summed = [sum(r["phase_records"][k] for r in rows) for k in range(2)]
    if summed != list(counts["records"]):
        raise RebalanceFigureError(f"{path}: its episodes add up to {summed} records per phase, its counts say "
                                   f"{counts['records']}: the document is not one result")
    union = doc["format"] == UNION_FORMAT
    stores = doc["stores"] if union else {doc.get("dataset"): doc}
    search = {name: ("sparse" if (entry or {}).get("sparse") else "dense") for name, entry in stores.items()}
    notes = []

    scenes: dict[str, dict] = {}
    for r in rows:
        s = scenes.setdefault(r["scene"], {"episodes": 0, "records": [0, 0], "never_detected": 0, "stores": [],
                                           "_distances": []})
        s["episodes"] += 1
        s["records"] = [s["records"][k] + r["phase_records"][k] for k in range(2)]
        s["never_detected"] += r["first_detection"] is None
        if r["store"] not in s["stores"]:
            s["stores"].append(r["store"])
        if r["distance_m"] is not None:
            s["_distances"].append(float(r["distance_m"]))
    scene_out = {}
    for name in sorted(scenes, key=lambda s: (s == UNKNOWN_SCENE, s)):
        s = scenes[name]
        total = sum(s["records"])
        scene_out[name] = {"label": scene_label(name) if name != UNKNOWN_SCENE else name, "episodes": s["episodes"],
                           "records": s["records"], "shares": [_r(v / total) for v in s["records"]] if total else None,
                           "never_detected": s["never_detected"], "stores": s["stores"],
                           "first_detection_distance_m_median": _r(_median(s["_distances"]), 3)}

    targets: dict[str, dict] = {}
    for r in rows:
        t = targets.setdefault(r["target"], {"episodes": 0, "never_detected": 0, "_distances": [], "_fractions": [],
                                             "on_target": 0, "off_target": 0, "scenes": []})
        t["episodes"] += 1
        if r["first_detection"] is None:
            t["never_detected"] += 1
        else:
            t["_fractions"].append(r["first_detection"] / r["records"])
            if r["distance_m"] is not None:
                t["_distances"].append(float(r["distance_m"]))
            if r["on_target"] is not None:
                t["on_target" if r["on_target"] else "off_target"] += 1
        if r["scene"] not in t["scenes"]:
            t["scenes"].append(r["scene"])
    target_out = {}
    for name, t in targets.items():
        target_out[name] = {"episodes": t["episodes"], "never_detected": t["never_detected"],
                            "never_detected_share": _r(t["never_detected"] / t["episodes"]),
                            "distances_m": [_r(v, 3) for v in sorted(t["_distances"])],
                            "distance_m_median": _r(_median(t["_distances"]), 3),
                            "first_detection_fractions": [_r(v) for v in sorted(t["_fractions"])],
                            "first_detection_fraction_median": _r(_median(t["_fractions"])),
                            "on_target": t["on_target"], "off_target": t["off_target"], "scenes": sorted(t["scenes"])}

    def order(name: str) -> tuple:
        """Farthest median first (the detector confirms those early; without distances, the earliest in the episode),
        and the targets it never confirmed in any episode last."""
        t = target_out[name]
        never = t["distance_m_median"] is None and t["first_detection_fraction_median"] is None
        return never, -(t["distance_m_median"] or 0.0), t["first_detection_fraction_median"] or 0.0, name

    target_out = {n: target_out[n] for n in sorted(target_out, key=order)}
    for name, t in (doc.get("by_target") or {}).items():  # the document's own per-target tally, where it has one
        mine = target_out.get(name)
        if mine is None or (mine["episodes"], mine["never_detected"]) != (t.get("episodes"), t.get("never_detected")):
            notes.append(f"target {name}: the document's by_target says {t.get('episodes')} episodes, "
                         f"{t.get('never_detected')} never detected; its episodes say "
                         f"{None if mine is None else (mine['episodes'], mine['never_detected'])}")
    never = [n for n, t in target_out.items() if t["never_detected"] == t["episodes"]]
    if doc.get("targets_never_detected") is not None and sorted(doc["targets_never_detected"]) != sorted(never):
        notes.append(f"the document lists {doc['targets_never_detected']} as never detected, its episodes {never}")
    if scene_source is None:
        notes.append("no episode's scene is known: pass --raw <store> for its manifest")
    if any(r["target"] == UNRECORDED_TARGET for r in rows):
        notes.append("the document does not name the targets of some episodes (one store's document never does): "
                     "pass --raw <store> to read them from its manifest")
    store_out = {name: {"episodes": (entry.get("coverage") or {}).get("episodes"),
                        "records": (entry.get("counts") or {}).get("records"),
                        "p0": [_r(v) for v in entry.get("p0") or []], "weights": [_r(v) for v in entry.get("weights") or []
                                                                                  if v is not None],
                        "never_detected": (entry.get("counts") or {}).get("episodes_never_detected"),
                        "search": search[name]} for name, entry in stores.items()}
    return {
        "tool": "scripts/rebalance_figures.py", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "document": {"path": str(path), "sha256": sha256, "format": doc["format"]},
        "kind": "union of stores" if union else "single store",
        "datasets": list(doc["datasets"]) if union else [doc.get("dataset")],
        "detector": doc.get("detector"), "search": search, "method": doc.get("method"),
        "coverage": doc.get("coverage"),
        "overall": {"episodes": len(rows), "records": list(counts["records"]), "p0": list(doc["p0"]),
                    "p_target": doc.get("p_target"), "weights": list(doc["weights"]), "kl_nats": doc.get("kl_nats"),
                    "threshold": doc["threshold"], "alpha": doc.get("alpha"), "resample_sizes": doc.get("resample_sizes"),
                    "sub_trajectories": counts.get("sub_trajectories"),
                    "records_above_threshold": counts.get("records_above_threshold"),
                    "records_above_threshold_share": (_r(counts["records_above_threshold"] / sum(counts["records"]))
                                                      if counts.get("records_above_threshold") is not None else None),
                    "detection_persistence": doc.get("detection_persistence"),
                    "episodes_never_detected": counts.get("episodes_never_detected"),
                    "degenerate": doc.get("degenerate")},
        "first_detection": doc.get("first_detection"),
        "paper": PAPER,
        "scene_source": scene_source,
        "scenes": scene_out,
        "stores": store_out,
        "targets": target_out,
        "targets_never_detected": never,
        "notes": notes,
        "files": list(OUTPUTS),
    }


# ----------------------------------------------------------------------------------------------------------------------
# The figures.
# ----------------------------------------------------------------------------------------------------------------------
def draw_phases(path: Path, summary: dict) -> None:
    overall = summary["overall"]
    rows = [("all", "all", overall["episodes"], overall["records"])] + [
        (name, s["label"], s["episodes"], s["records"]) for name, s in summary["scenes"].items()]
    n = len(rows)
    top_in, bottom_in, row_in = 1.15, 0.95, 0.3  # a 0.6-row bar is 23 px at 130 dpi, inside the dataviz cap of 24
    panel_in = (max(n, 3) + 1.1) * row_in  # the weights panel takes four rows, beside a scene or two too
    height = top_in + bottom_in + panel_in
    with plt.rc_context(STYLE):
        fig, (ax, ax_w) = plt.subplots(1, 2, figsize=(14.5, height), gridspec_kw={
            "width_ratios": [1.9, 1.0], "wspace": 0.42, "left": 0.15, "right": 0.97, "top": 1 - top_in / height,
            "bottom": bottom_in / height})
        y = np.arange(n)
        for yi, (_key, _label, episodes, records) in zip(y, rows):
            total = sum(records) or 1
            start = 0.0
            for k in range(2):
                share = records[k] / total
                if share > 0:
                    ax.barh(yi, share, left=start, height=0.6, color=PHASE_COLOURS[k], edgecolor=SURFACE, linewidth=1.5,
                            zorder=2)
                    if share >= 0.08:  # a share is written inside its segment only where it fits
                        ax.text(start + share / 2, yi, f"{share:.2f}", ha="center", va="center", fontsize=8.5,
                                color="white", zorder=3)
                start += share
            ax.text(1.015, yi, f"{episodes:,} ep. · {sum(records):,} records", va="center", fontsize=8.5, color=INK_2,
                    transform=ax.get_yaxis_transform())
        # The paper's split as a reference behind the bars: it shows in the gaps between rows, never over a label.
        ax.axvline(PAPER["p0"][0], color=INK, lw=1.0, zorder=1)
        ax.text(PAPER["p0"][0] + 0.006, -0.62, f"the paper: {PAPER['p0'][0]:.2f} / {PAPER['p0'][1]:.2f}", ha="left",
                va="bottom", fontsize=8.5, color=INK_2)
        ax.set_yticks(y, [label for _key, label, _e, _r in rows])
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.set_ylim(panel_in / row_in - 0.9, -0.9)  # rows at row_in each, from the top
        ax.spines["left"].set_bounds(n - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("share of the records")
        ax.set_title(f"Records per phase, split at the first detection above {overall['threshold']}")
        weights, paper = overall["weights"], PAPER["weights"]
        yw = np.array([0.15, 2.15])  # in the left panel's row units, so both panels draw bars of one thickness
        for k in range(2):  # each phase named above its bar, the paper's weight marked under it
            ax_w.text(0.0, yw[k] - 0.42, PHASE_NAMES[k], ha="left", va="bottom", fontsize=9, color=INK)
            if weights[k] is not None:
                ax_w.barh(yw[k], weights[k], height=0.6, color=PHASE_COLOURS[k], edgecolor=SURFACE, linewidth=1.5)
                ax_w.text(weights[k] + 0.04, yw[k], f"{weights[k]:.3f}", va="center", fontsize=9, color=INK)
            ax_w.plot([paper[k]], [yw[k] + 0.62], marker="^", ms=6, color=INK, ls="")
            ax_w.text(paper[k] + 0.05, yw[k] + 0.62, f"paper {paper[k]:.2f}", ha="left", va="center", fontsize=8,
                      color=INK_2)
        ax_w.axvline(1.0, color=MUTED, lw=1.0, zorder=0)
        ax_w.set_yticks([])
        ax_w.spines["left"].set_visible(False)
        ax_w.set_ylim(panel_in / row_in - 0.9, -0.9)
        ax_w.set_xlim(0, max([2.0] + [w * 1.2 for w in weights if w is not None] + [p * 1.15 for p in paper]))
        ax_w.xaxis.grid(True)
        ax_w.set_axisbelow(True)
        ax_w.tick_params(axis="y", length=0)
        ax_w.set_xlabel("resampling weight (1 = kept as is)")
        alpha = overall.get("alpha")
        ax_w.set_title(f"Weights towards a uniform phase mix (alpha {alpha})" if alpha == 0 else
                       f"Weights (alpha {alpha})")
        handles = [Patch(color=PHASE_COLOURS[k], label=PHASE_NAMES[k]) for k in range(2)]
        handles.append(Line2D([0], [0], marker="^", color=INK, ls="", ms=7, label="the paper's weight"))
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.1 / height), ncol=3)
        p0 = overall["p0"]
        fig.suptitle(f"Rebalancing (spec §10.3, AutoFly App. A.2.4): P0 = ({p0[0]:.3f}, {p0[1]:.3f}) over "
                     f"{overall['episodes']:,} episodes and {sum(overall['records']):,} records, weights "
                     f"({', '.join('n/a' if w is None else f'{w:.3f}' for w in overall['weights'])})",
                     x=0.01, y=1 - 0.12 / height, ha="left", va="top", fontsize=12.5)
        detector = (summary.get("detector") or {}).get("model", "the detector")
        searches = sorted(set(summary["search"].values()))
        fig.text(0.01, 1 - 0.5 / height, f"{summary['kind']}: {', '.join(summary['datasets'])}; {detector}, "
                 f"{' and '.join(searches)} search; scenes from {summary['scene_source'] or 'nowhere'}",
                 ha="left", va="top", fontsize=8.5, color=INK_2)
        fig.savefig(path, dpi=130)
        plt.close(fig)


def draw_targets(path: Path, summary: dict) -> None:
    targets = summary["targets"]
    names = list(targets)
    n = len(names)
    use_distance = any(t["distances_m"] for t in targets.values())
    top_in, bottom_in = 1.2, 0.9
    height = top_in + bottom_in + (max(n, 2) + 0.5) * ROW_IN  # a strip of one target is still a readable panel
    left = (0.4 + 0.068 * max(len(name) for name in names)) / 14.5
    with plt.rc_context(STYLE):
        fig, (ax, ax_n) = plt.subplots(1, 2, figsize=(14.5, height), sharey=True, gridspec_kw={
            "width_ratios": [2.4, 1.0], "wspace": 0.06, "left": left, "right": 0.97, "top": 1 - top_in / height,
            "bottom": bottom_in / height})
        y = np.arange(n)
        rng = np.random.default_rng(0)  # a fixed jitter: the same document draws the same figure
        values_all = []
        for yi, name in zip(y, names):
            t = targets[name]
            # without distances: where in the episode the first detection fell, as a share of its records
            values = t["distances_m"] if use_distance else t["first_detection_fractions"]
            values_all += values
            if values:
                jitter = rng.uniform(-0.16, 0.16, len(values))
                ax.scatter(values, yi + jitter, s=12, color=SLOTS[0], alpha=0.55, linewidths=0, zorder=3)
                median = t["distance_m_median"] if use_distance else t["first_detection_fraction_median"]
                ax.plot([median, median], [yi - 0.32, yi + 0.32], color=INK, lw=2.0, zorder=4)
            else:
                ax.text(0.0, yi, "  never detected", transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                        color=INK_2)
            share = t["never_detected_share"]
            if share:
                ax_n.barh(yi, share, height=0.6, color=SLOTS[1], edgecolor=SURFACE, linewidth=1.0)
            if t["never_detected"]:
                ax_n.text(min(share, 1.0) + 0.02, yi, f"{t['never_detected']} of {t['episodes']}", va="center",
                          fontsize=8, color=INK_2)
        ax.set_yticks(y, names)
        ax.set_ylim(n - 0.5, -0.8)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        if use_distance:
            ax.set_xlim(0, max(values_all + [10.0]) * 1.05)
            ax.set_xlabel("distance to the target at the episode's first confident detection (m); bar: the median")
            ax.set_title("Where the detector first confirms each target (one dot per episode)")
        else:
            ax.set_xlim(0, 1)
            ax.set_xlabel("median share of the episode before its first confident detection (no distances recorded)")
            ax.set_title("Where in the episode the detector first confirms each target")
        ax_n.set_xlim(0, 1.3)
        ax_n.set_xticks(np.linspace(0, 1, 5))
        ax_n.xaxis.grid(True)
        ax_n.set_axisbelow(True)
        ax_n.tick_params(axis="y", length=0)
        ax_n.set_xlabel("share of the target's episodes")
        ax_n.set_title("Never confirmed (all obstacle avoidance)")
        overall = summary["overall"]
        never = summary["targets_never_detected"]
        fig.suptitle(f"First confident detection (above {overall['threshold']}) per target: {n} "
                     f"target{'s' if n != 1 else ''}, {overall['episodes']:,} episodes, "
                     f"{overall['episodes_never_detected'] or 0} never detected"
                     + (f"; never confirmed at all: {', '.join(never)}" if never else ""),
                     x=0.01, y=1 - 0.12 / height, ha="left", va="top", fontsize=12.5)
        first = summary.get("first_detection") or {}
        if first:
            fig.text(0.01, 1 - 0.5 / height, f"First detections whose box lies on the target's bearing: "
                     f"{first.get('on_target')} on, {first.get('off_target')} off "
                     f"({first.get('off_target_out_of_view')} with the target out of view); median distance "
                     f"{first.get('distance_m_median') or float('nan'):.1f} m", ha="left", va="top", fontsize=8.5,
                     color=INK_2)
        fig.savefig(path, dpi=130)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# The tables.
# ----------------------------------------------------------------------------------------------------------------------
def _f(value, fmt: str = ".3f") -> str:
    return "" if value is None else format(value, fmt)


def _shown(path) -> str:
    """A path as a README shows it: relative to the repository where it lies inside it."""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def markdown(summary: dict) -> str:
    o = summary["overall"]
    doc = summary["document"]
    lines = ["# Rebalancing (spec §10.3; AutoFly App. A.2.4)", "",
             f"`{_shown(doc['path'])}` (sha256 {doc['sha256'][:12]}, {summary['kind']}: {', '.join(summary['datasets'])}), drawn "
             f"by `scripts/rebalance_figures.py` ({summary['generated']}). A record is obstacle avoidance until the "
             f"episode's first detection of its target above {o['threshold']}, target seeking from it on; the weights "
             f"even the two phases out (alpha {o['alpha']}).", "",
             "| | this dataset | the paper |", "|---|---|---|",
             f"| phase split P0 (avoidance, seeking) | {_f(o['p0'][0])} / {_f(o['p0'][1])} | "
             f"{PAPER['p0'][0]:.2f} / {PAPER['p0'][1]:.2f} |",
             f"| resampling weights | {' / '.join(_f(w) or 'n/a' for w in o['weights'])} | "
             f"{PAPER['weights'][0]:.2f} / {PAPER['weights'][1]:.2f} |",
             f"| share of records with a confident detection | {_f(o['records_above_threshold_share']) or 'n/a (sparse)'} "
             f"| 0.27 |",
             f"| episodes never detected | {o['episodes_never_detected']} of {o['episodes']} | |",
             f"| detection persistence after the first detection | {_f(o['detection_persistence']) or 'n/a (sparse)'} | |",
             "", "## Per scene", "",
             "| Scene | Episodes | Records | Obstacle avoidance | Target seeking | Never detected | Median distance at "
             "first detection (m) |", "|---|---|---|---|---|---|---|",
             f"| all | {o['episodes']} | {sum(o['records'])} | {_f(o['p0'][0])} | {_f(o['p0'][1])} | "
             f"{o['episodes_never_detected']} | {_f((summary.get('first_detection') or {}).get('distance_m_median'), '.1f')} |"]
    for s in summary["scenes"].values():
        shares = s["shares"] or [None, None]
        lines.append(f"| {s['label']} | {s['episodes']} | {sum(s['records'])} | {_f(shares[0])} | {_f(shares[1])} | "
                     f"{s['never_detected']} | {_f(s['first_detection_distance_m_median'], '.1f')} |")
    lines += ["", "## Per target", "",
              "| Target | Episodes | Never detected | Median distance at first detection (m) | Median share of the episode "
              "before it | On target / off |", "|---|---|---|---|---|---|"]
    for name, t in summary["targets"].items():
        never = f"**{t['never_detected']}**" if t["never_detected"] == t["episodes"] else str(t["never_detected"])
        lines.append(f"| {name} | {t['episodes']} | {never} | {_f(t['distance_m_median'], '.1f')} | "
                     f"{_f(t['first_detection_fraction_median'], '.2f')} | {t['on_target']} / {t['off_target']} |")
    if summary["targets_never_detected"]:
        lines += ["", f"Never confirmed in any episode (all of their records are obstacle avoidance, which the weights "
                      f"cannot see): {', '.join(summary['targets_never_detected'])}."]
    if summary["notes"]:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in summary["notes"]]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------------------------------
# The whole set.
# ----------------------------------------------------------------------------------------------------------------------
def build(document: Path, out_dir: Path, *, raws=()) -> dict:
    doc, sha256 = load_document(Path(document))
    summary = summarise_document(doc, path=str(document), sha256=sha256, manifests=read_manifests(raws))
    out_dir = Path(out_dir)
    for name in OUTPUTS:
        evidence.refuse_existing_evidence(out_dir / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_phases(out_dir / OUTPUTS[0], summary)
    draw_targets(out_dir / OUTPUTS[1], summary)
    (out_dir / OUTPUTS[2]).write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / OUTPUTS[3]).write_text(markdown(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebalance", type=Path, required=True, help="a rebalance document (scripts/rebalance_dataset.py)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--raw", type=Path, action="append", default=[],
                   help="a raw store (data/<name>) whose manifest names its episodes' scenes and targets; repeat")
    args = p.parse_args(argv)
    try:
        summary = build(args.rebalance, args.out_dir, raws=args.raw)
    except (RebalanceFigureError, FileExistsError, FileNotFoundError, KeyError) as err:
        print(f"refused: {type(err).__name__}: {err}", file=sys.stderr)
        return 2
    print(json.dumps({"out_dir": str(args.out_dir), "files": summary["files"], "p0": summary["overall"]["p0"],
                      "weights": summary["overall"]["weights"], "targets_never_detected": summary["targets_never_detected"],
                      "notes": summary["notes"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
