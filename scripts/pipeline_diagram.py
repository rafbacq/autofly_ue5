"""An overview diagram of the system for the final README (plan 6 step 7): from the simulated scenes to a VLA-ready dataset
and its closed-loop evaluation, in eight stages read as a U (left to right along the top, right to left along the
bottom), with the next phase, a VLA, marked as not part of this one.

    env -u PYTHONPATH .venv/bin/python scripts/pipeline_diagram.py --out runs/report/pipeline/pipeline.png

The counts it prints are read, not typed: the training and test_seen scenes from the scene files' splits (scenes/*.json),
the target pool's size from assets/targets.json, and the dataset's name, record shapes, validation share and split
strata, the detection threshold and the evaluation conditions from the code that defines them (dataset/merge.py,
dataset/rebalance.py, evaluation/episodes.py). The rest is the architecture in words. matplotlib only; drawn in inches,
12 x 6.55 at 150 dpi (1800 px), so the body text is about 10 px high where GitHub scales it to the README's width.
`--out` ending in .svg writes a vector file instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import textwrap  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

from autofly_ue5 import evidence  # noqa: E402
from autofly_ue5.dataset.merge import DEFAULT_NAME, DEFAULT_VAL_FRACTION, RECORD_SHAPES, SPLITS, STRATIFY_BY  # noqa: E402
from autofly_ue5.dataset.rebalance import DEFAULT_THRESHOLD  # noqa: E402
from autofly_ue5.evaluation.episodes import CONDITIONS  # noqa: E402
from autofly_ue5.paths import ROOT, SCENES_DIR  # noqa: E402
from autofly_ue5.report.style import INK, INK_2, MUTED, SLOTS, SURFACE  # noqa: E402

TARGETS = ROOT / "assets" / "targets.json"
KEPT_RATE_BAR = 0.25  # plan 6 revision 1, item 3: a pilot's kept-rate needs a Wilson lower bound of at least this
DPI = 150
WIDTH_IN, HEIGHT_IN = 12.0, 6.55
MARGIN_IN, GAP_IN = 0.25, 0.62  # the gaps between boxes hold the arrows and their one-word labels
BOX_W = (WIDTH_IN - 2 * MARGIN_IN - 3 * GAP_IN) / 4
ROW_TOP = (0.95, 3.85)  # each row's top edge, inches from the top
BOX_H = 2.1
TITLE_PT, BODY_PT, LABEL_PT = 11.5, 10.5, 9.0
LINE_IN = 0.235
GROUPS = {"simulation": SLOTS[0], "dataset": SLOTS[2], "evaluation": SLOTS[1], "next": MUTED}
FILL = {"simulation": "#eef4fc", "dataset": "#ecf8f3", "evaluation": "#fdf0ea", "next": SURFACE}


def facts(scenes_dir: Path = SCENES_DIR, targets_path: Path = TARGETS) -> dict:
    """What the diagram states, read from the repository and the code."""
    train, test_seen = [], []
    for path in sorted(Path(scenes_dir).glob("*.json")):
        data = json.loads(path.read_text())
        {"train": train, "test_seen": test_seen}.get(data.get("split"), []).append(data.get("id"))
    pool = json.loads(Path(targets_path).read_text())["targets"]

    def dims(shape: str) -> int:
        return int(re.search(r"\[(\d+)\]", shape).group(1))

    return {"training_scenes": sorted(train), "test_seen_scenes": sorted(test_seen), "targets": len(pool),
            "target_categories": len({t.get("category") for t in pool}),
            "unseen_targets": sum(t.get("seen") is False for t in pool),
            "dataset_name": DEFAULT_NAME, "splits": list(SPLITS), "val_fraction": DEFAULT_VAL_FRACTION,
            "stratify_by": list(STRATIFY_BY), "detection_threshold": DEFAULT_THRESHOLD,
            "image": RECORD_SHAPES["observation.image_0"].replace(" PNG", ""),
            "state": f"state[{dims(RECORD_SHAPES['observation.state'])}]",
            "action": f"action[{dims(RECORD_SHAPES['action'])}]",
            "conditions": list(CONDITIONS), "kept_rate_bar": KEPT_RATE_BAR}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def stages(f: dict) -> list[dict]:
    """The eight stages in reading order: title, group (its colour) and lines."""
    val = f"{f['val_fraction'] * 100:g} %"
    strata = " x ".join(s.replace("target_name", "target") for s in f["stratify_by"])
    return [
        {"title": "Scenes", "group": "simulation",
         "lines": ["UE 5.7 + Project AirSim", _plural(len(f["training_scenes"]), "training scene"),
                   f"+ {_plural(len(f['test_seen_scenes']), 'test_seen layout')}", "procedural obstacles",
                   f"{f['targets']}-object target pool"]},
        {"title": "Expert pilots", "group": "simulation",
         "lines": ["SAC on privileged depth", "+ the target vector", "checkpoint averaging", "one pilot per scene",
                   f"kept-rate ≥ {f['kept_rate_bar']:g} (Wilson)"]},
        {"title": "Collector", "group": "simulation",
         "lines": ["a0-aligned start", "stochastic actions", "kept: success and", "collision-free",
                   "first-use warm-up"]},
        {"title": "Raw stores", "group": "dataset",
         "lines": ["per scene and batch", f"{f['image']}, instruction", f"{f['action']}, {f['state']}",
                   "provenance per episode"]},
        {"title": "Merged dataset", "group": "dataset",
         "lines": ["one RLDS / TFDS dataset", f["dataset_name"], f"{f['splits'][0]} + {val} {f['splits'][1]}",
                   f"split by {strata}", "OpenVLA statistics, card"]},
        {"title": "Rebalancing", "group": "dataset",
         "lines": ["Grounding DINO", f"first detection > {f['detection_threshold']:g}", "obstacle avoidance",
                   "then target seeking", "phase weights"]},
        {"title": "Next: a VLA", "group": "next", "lines": ["trains on the dataset", "plugs into the harness",
                                                            "(the next phase)"]},
        {"title": "Evaluation harness", "group": "evaluation",
         "lines": ["Policy interface:", "rgb, instruction, state", "→ action",
                   *textwrap.wrap(" / ".join(c.replace("_targets", " targets") for c in f["conditions"]), 24),
                   "SR, CR, PER (Wilson)"]},
    ]


def _origin(index: int) -> tuple[float, float]:
    """A stage's box, top-left corner in inches: the top row left to right, the bottom row right to left."""
    row, col = divmod(index, 4)
    col = col if row == 0 else 3 - col
    return MARGIN_IN + col * (BOX_W + GAP_IN), ROW_TOP[row]


# (from stage, to stage, label): the U's arrows, and the harness flying the same scenes.
ARROWS = ((0, 1, "fly"), (1, 2, "act"), (2, 3, "keep"), (3, 4, "merge, split"), (4, 5, "detect"), (5, 6, "weights"),
          (6, 7, "plug in"), (0, 7, "same scenes, new layouts, unseen targets"))


def render(f: dict):
    """The figure and, per stage, its box: {"title", "lines", "next", "patch", "artists"} (every text drawn in it)."""
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_IN)
    ax.set_ylim(HEIGHT_IN, 0)  # inches from the top-left corner
    ax.axis("off")
    boxes = []
    for i, stage in enumerate(stages(f)):
        x, y = _origin(i)
        colour, is_next = GROUPS[stage["group"]], stage["group"] == "next"
        patch = FancyBboxPatch((x, y), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.1", facecolor=FILL[stage["group"]],
                               edgecolor=colour, linewidth=1.6, linestyle=(0, (4, 3)) if is_next else "-", zorder=2)
        ax.add_patch(patch)
        ax.add_patch(Circle((x + 0.27, y + 0.3), 0.16, facecolor=colour, edgecolor="none", zorder=3))
        artists = [ax.text(x + 0.27, y + 0.305, str(i + 1), ha="center", va="center", fontsize=10, color="white",
                           fontweight="bold", zorder=4),
                   ax.text(x + 0.52, y + 0.3, stage["title"], ha="left", va="center", fontsize=TITLE_PT,
                           fontweight="bold", color=INK_2 if is_next else INK, zorder=4)]
        for k, line in enumerate(stage["lines"]):
            artists.append(ax.text(x + 0.16, y + 0.66 + k * LINE_IN, line, ha="left", va="center", fontsize=BODY_PT,
                                   color=INK_2 if is_next else INK, zorder=4))
        boxes.append({"title": stage["title"], "lines": stage["lines"], "next": is_next, "patch": patch,
                      "artists": artists})
    for a, b, label in ARROWS:
        (xa, ya), (xb, yb) = _origin(a), _origin(b)
        if ya == yb:  # along a row: from one box's side to the next box's, at mid-height
            left_to_right = xb > xa
            start = (xa + BOX_W if left_to_right else xa, ya + BOX_H / 2)
            end = (xb if left_to_right else xb + BOX_W, yb + BOX_H / 2)
            ax.text((start[0] + end[0]) / 2, start[1] - 0.12, label, ha="center", va="bottom", fontsize=LABEL_PT,
                    color=INK_2)
        else:  # down a column, from the top row's box to the bottom row's
            start, end = (xa + BOX_W / 2, ya + BOX_H), (xb + BOX_W / 2, yb)
            ax.text(start[0] + 0.12, (start[1] + end[1]) / 2, label, ha="left", va="center", fontsize=LABEL_PT,
                    color=INK_2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color=INK_2,
                                     shrinkA=2, shrinkB=2, zorder=3))
    ax.text(MARGIN_IN, 0.42, "autofly_ue5: from simulated scenes to a VLA-ready dataset and its closed-loop evaluation",
            ha="left", va="center", fontsize=14, color=INK)
    legend_y = HEIGHT_IN - 0.24
    x = MARGIN_IN
    for name, colour in GROUPS.items():
        ax.add_patch(FancyBboxPatch((x, legend_y - 0.07), 0.22, 0.14, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=FILL[name], edgecolor=colour, linewidth=1.3,
                                    linestyle=(0, (3, 2)) if name == "next" else "-"))
        shown = "next phase" if name == "next" else name
        ax.text(x + 0.3, legend_y, shown, ha="left", va="center", fontsize=LABEL_PT, color=INK_2)
        x += 0.42 + 0.075 * len(shown)
    ax.text(WIDTH_IN - MARGIN_IN, legend_y, "counts read from scenes/*.json, assets/targets.json and the code "
            "(scripts/pipeline_diagram.py)", ha="right", va="center", fontsize=8, color=MUTED)
    return fig, boxes


def draw(path: Path, f: dict) -> list[str]:
    fig, boxes = render(f)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return [b["title"] for b in boxes]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="the figure: .png, or .svg for a vector file")
    p.add_argument("--targets", type=Path, default=TARGETS, help="the target pool (assets/targets.json)")
    p.add_argument("--scenes-dir", type=Path, default=SCENES_DIR, help="the scene files (scenes/)")
    args = p.parse_args(argv)
    if args.out.suffix.lower() not in (".png", ".svg"):
        p.error(f"--out {args.out} must end in .png or .svg")
    evidence.refuse_existing_evidence(args.out)
    f = facts(args.scenes_dir, args.targets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    titles = draw(args.out, f)
    print(json.dumps({"out": str(args.out), "stages": titles, "facts": f}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
