"""Scene layouts side by side, drawn from their stored layouts (runs/levels/<id>.layout.json), for the status report.

    env -u PYTHONPATH .venv/bin/python scripts/layout_figure.py --scenes s02 s03 s06 --levels runs/levels \
        --out results/<date>/figures/layouts_c4.png --json results/<date>/figures/layouts_c4.json

Each panel draws a scene's 70 x 70 m bounds, its start band and target band (as rings inside the bounds), and every
instance's footprint circle (radius_m: the model's whole horizontal reach at its scale, which tree scenes end episodes
against) coloured by asset, with the pivot as a dot. Rock scales vary by group, so a rock scene writes each rock's height.
The title gives the worst crossing detour (scenes/paths.py:crossing_detours, the build's acceptance rule) and the line
under it each edge's worst. The JSON holds the same numbers, plus each scene's closest pair of footprint circles (a
negative gap means two circles overlap). It launches nothing and writes only --out and --json.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
from collections import Counter  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from autofly_ue5.paths import RUNS_DIR, SCENES_DIR  # noqa: E402
from autofly_ue5.scenes.model import load_scene_file  # noqa: E402

EDGES = ("x_min", "x_max", "y_min", "y_max")
DEFAULT_MAX_DETOUR = 1.25  # the crossing rule's bound when a record predates crossing.rule
PALETTE = ("#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f", "#76b7b2", "#edc948", "#ff9da7",
           "#bab0ac")
START_BAND_COLOUR, TARGET_BAND_COLOUR = "#9ecae1", "#fdd0a2"


def _scene_and_stored(scene_id: str, levels: Path):
    matches = sorted(SCENES_DIR.glob(f"{scene_id}_*.json"))
    if len(matches) != 1:
        raise SystemExit(f"scene {scene_id!r}: expected one file {SCENES_DIR}/{scene_id}_*.json, found {len(matches)}")
    scene = load_scene_file(matches[0])
    base_id = scene.level or scene.id
    layout_path = levels / f"{base_id}.layout.json"
    stored = json.loads(layout_path.read_text())
    if stored["layout"]["scene_id"] != base_id:
        raise SystemExit(f"{layout_path} holds scene {stored['layout']['scene_id']!r}'s layout, not {base_id!r}'s")
    if scene.level is None and stored.get("scene_sha256") not in (None, scene.sha256):
        raise SystemExit(f"{layout_path} was built from another version of {scene.path} (scene_sha256 differs)")
    return scene, stored


def _range(values) -> dict:
    return {"min": round(min(values), 4), "median": round(statistics.median(values), 4), "max": round(max(values), 4)}


def _closest_pair(instances) -> dict | None:
    """The two footprint circles with the smallest gap between them (negative: they overlap), and how many overlap."""
    best, overlapping = None, 0
    for i, a in enumerate(instances):
        for b in instances[i + 1:]:
            gap = math.hypot(a["x"] - b["x"], a["y"] - b["y"]) - a["radius_m"] - b["radius_m"]
            overlapping += gap < 0
            if best is None or gap < best["gap_m"]:
                best = {"gap_m": round(gap, 4), "tags": [a["tag"], b["tag"]], "assets": [a["asset"], b["asset"]]}
    return None if best is None else {**best, "overlapping_pairs": overlapping}


def summarise(scene, stored) -> dict:
    instances = stored["layout"]["instances"]
    crossing = stored["crossing"]
    return {
        "id": scene.id,
        "instruction_obstacle": scene.instruction_obstacle,
        "seed": stored["layout"]["seed"],
        "instances": len(instances),
        "per_asset": dict(sorted(Counter(i["asset"] for i in instances).items())),
        "radius_m": _range([i["radius_m"] for i in instances]),
        "height_m": _range([i["height_m"] for i in instances]),
        "crossing_max": round(crossing["max"], 4),
        "crossing_edge_max": {e: round(crossing["edges"][e]["max"], 4) for e in EDGES},
        "crossing_bound": crossing.get("rule", {}).get("max_detour", DEFAULT_MAX_DETOUR),
        "crossing_accepted": crossing.get("accepted"),
        "reachability_ok": stored["reachability"]["ok"],
        "closest_footprints": _closest_pair(instances),
    }


def _band(ax, bounds: dict, inner_m: float, outer_m: float, colour: str, alpha: float) -> None:
    """A ring between the rectangles inset by outer_m and inner_m from the bounds (even-odd fill)."""
    def rect(inset):
        x0, x1 = bounds["y_min"] + inset, bounds["y_max"] - inset  # east on the horizontal axis
        y0, y1 = bounds["x_min"] + inset, bounds["x_max"] - inset
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    outer, inner = rect(inner_m), rect(outer_m)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(outer + inner[::-1], codes + codes), facecolor=colour, edgecolor="none",
                           alpha=alpha, zorder=0))


def draw_panel(ax, scene, stored, summary: dict, show_heights: bool) -> None:
    bounds = stored["layout"]["bounds"]
    instances = stored["layout"]["instances"]
    _band(ax, bounds, scene.target_band[0], scene.target_band[1], TARGET_BAND_COLOUR, 0.7)
    _band(ax, bounds, scene.start_band[0], scene.start_band[1], START_BAND_COLOUR, 0.55)
    ax.add_patch(plt.Rectangle((bounds["y_min"], bounds["x_min"]), bounds["y_max"] - bounds["y_min"],
                               bounds["x_max"] - bounds["x_min"], fill=False, edgecolor="#333333", lw=1.2, zorder=1))
    colours = {asset: PALETTE[k % len(PALETTE)] for k, asset in enumerate(summary["per_asset"])}
    for inst in instances:
        c = colours[inst["asset"]]
        ax.add_patch(Circle((inst["y"], inst["x"]), inst["radius_m"], facecolor=c, edgecolor=c, alpha=0.45, lw=0.8,
                            zorder=2))
        ax.plot(inst["y"], inst["x"], "o", ms=2.2, color="#222222", zorder=3)
        if show_heights:
            ax.text(inst["y"], inst["x"] - 0.6, f"{inst['height_m']:.1f}", fontsize=6, ha="center", va="top",
                    color="#111111", zorder=4)
    pad = 2.0
    ax.set_xlim(bounds["y_min"] - pad, bounds["y_max"] + pad)
    ax.set_ylim(bounds["x_min"] - pad, bounds["x_max"] + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("y (east, m)", fontsize=8)
    ax.set_ylabel("x (north, m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_title(f"{scene.id} {scene.instruction_obstacle}: {summary['instances']} obstacles, worst crossing detour "
                 f"{summary['crossing_max']:.3f} (bound {summary['crossing_bound']:g})", fontsize=9)
    handles = [Patch(facecolor=colours[a], edgecolor=colours[a], alpha=0.6, label=f"{a}: {n}")
               for a, n in summary["per_asset"].items()]
    handles += [Patch(facecolor=START_BAND_COLOUR, alpha=0.55, label=f"start band {scene.start_band[0]:g}-"
                      f"{scene.start_band[1]:g} m inside"),
                Patch(facecolor=TARGET_BAND_COLOUR, alpha=0.7, label=f"target band {scene.target_band[0]:g}-"
                      f"{scene.target_band[1]:g} m inside")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.5, frameon=False,
              handlelength=1.2, columnspacing=1.0)
    edges = summary["crossing_edge_max"]
    note = "crossing max per edge: " + ", ".join(f"{e} {edges[e]:.3f}" for e in EDGES)
    if show_heights:
        note += "; labels: rock height (m)"
    ax.text(0.5, -0.105, note, transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#333333")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=["s02", "s03", "s06"], help="scene ids under scenes/")
    parser.add_argument("--levels", type=Path, default=RUNS_DIR / "levels", help="directory of <id>.layout.json")
    parser.add_argument("--out", type=Path, required=True, help="PNG to write")
    parser.add_argument("--json", type=Path, required=True, help="numbers to write")
    args = parser.parse_args(argv)

    loaded = [_scene_and_stored(s, args.levels) for s in args.scenes]
    summaries = [summarise(scene, stored) for scene, stored in loaded]
    fig, axes = plt.subplots(1, len(loaded), figsize=(6.0 * len(loaded), 6.5), squeeze=False)
    for ax, (scene, stored), summary in zip(axes[0], loaded, summaries):
        draw_panel(ax, scene, stored, summary, show_heights=scene.instruction_obstacle == "rocks")
    fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.24, wspace=0.12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"levels": str(args.levels), "scenes": summaries}, indent=2) + "\n")
    print(f"wrote {args.out} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
