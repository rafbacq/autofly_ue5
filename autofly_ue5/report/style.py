"""The report figures' look, shared by the plan 6 figure scripts (scripts/harness_figures.py, rebalance_figures.py,
pipeline_diagram.py): the dataviz skill's validated categorical slots in their fixed order, its neutral ink and chrome,
the matplotlib rcParams, and a scene's label as plan 6 names it.

scripts/dataset_figures.py holds the same values (it predates this module); tests/test_report_style.py keeps the two
from drifting apart, so a figure from either script reads as one system in final_results/.
"""

from __future__ import annotations

from autofly_ue5.paths import SCENES_DIR

# The dataviz skill's categorical slots (light surface), in their fixed order: a colour follows its entity, never a rank.
# Past three on one plot the skill's all-pairs check no longer holds, so a fourth series needs a second channel (position).
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BAND = "#e8e7e1"  # a reference interval behind marks (the pilot's kept-rate, the paper's PER): one step off the surface
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titlelocation": "left",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
    "grid.linewidth": 0.8, "grid.linestyle": "-", "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "text.color": INK, "font.size": 9, "legend.frameon": False, "legend.fontsize": 8.5,
}
PAPER_PER = (0.73, 0.78)  # the paper's models' path efficiency (results/2026-10-07-project-status, section 2)


def scene_label(scene_id: str) -> str:
    """'s02 sparse trees': the id and the rest of its scene file's name (scenes/<id>_<name>.json), as plan 6 names it;
    the bare id when no single scene file has it."""
    matches = sorted(SCENES_DIR.glob(f"{scene_id}_*.json"))
    return f"{scene_id} {matches[0].stem[len(scene_id) + 1:].replace('_', ' ')}" if len(matches) == 1 else scene_id
