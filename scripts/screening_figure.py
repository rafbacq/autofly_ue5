"""Draw plan 6's pilot screening (docs/gates/m5_pilot_screening.json) as a scene x candidate matrix.

Each cell is a candidate expert's stochastic success on a scene, with its Wilson 95 % interval; the chosen pilot per
scene is outlined. s01 and s01d are not screened: each flies its own accepted expert.

    env -u PYTHONPATH .venv/bin/python scripts/screening_figure.py \\
        --record docs/gates/m5_pilot_screening.json --out results/2026-10-07-project-status/figures/pilot_screening.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

CANDIDATES = (("s01_best", "s01 expert\n(pillars)"), ("s01d_mean", "s01d mean\n(pillars + movers)"),
              ("s02_pick", "s02 expert\n(sparse trees)"))
SCENE_NAMES = {"s02": "s02 sparse trees", "s03": "s03 scattered rocks", "s04": "s04 stone field",
               "s05": "s05 standing stones (snow)", "s06": "s06 tree clusters", "s07": "s07 stacked boxes",
               "s09": "s09 coloured poles"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", type=Path, default=Path("docs/gates/m5_pilot_screening.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rec = json.loads(args.record.read_text())
    cell = {(p["scene"], p["candidate"]): p for p in rec["pairs"]}
    scenes = sorted({p["scene"] for p in rec["pairs"]})
    fig, ax = plt.subplots(figsize=(9.5, 0.75 * len(scenes) + 1.9))
    cmap = plt.get_cmap("RdYlGn")
    for r, scene in enumerate(scenes):
        for c, (cand, _) in enumerate(CANDIDATES):
            p = cell.get((scene, cand))
            if p is None:
                ax.add_patch(Rectangle((c, r), 1, 1, color="#dddddd"))
                ax.text(c + 0.5, r + 0.5, "not run", ha="center", va="center", fontsize=9, color="#555555")
                continue
            ax.add_patch(Rectangle((c, r), 1, 1, color=cmap(p["success_rate"])))
            lo, hi = p["wilson_95"]
            ax.text(c + 0.5, r + 0.42, f"{p['successes']}/{p['episodes']} = {p['success_rate']:.2f}", ha="center",
                    va="center", fontsize=10.5, weight="bold")
            ax.text(c + 0.5, r + 0.72, f"95 % CI {lo:.2f}-{hi:.2f}", ha="center", va="center", fontsize=8.5)
        pilot = rec["pilot_by_scene"][scene]["candidate"]
        c = [k for k, _ in CANDIDATES].index(pilot)
        ax.add_patch(Rectangle((c + 0.03, r + 0.03), 0.94, 0.94, fill=False, lw=3, ec="black"))
    ax.set_xlim(0, len(CANDIDATES)); ax.set_ylim(len(scenes), 0)
    ax.set_xticks([i + 0.5 for i in range(len(CANDIDATES))], [label for _, label in CANDIDATES])
    ax.set_yticks([i + 0.5 for i in range(len(scenes))], [SCENE_NAMES.get(s, s) for s in scenes])
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("Which existing expert flies each new scene? (stochastic, task rules; black box = chosen pilot)",
                 fontsize=11, y=0.035, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
