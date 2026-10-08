"""The footprint contact rule on a real tree scene's layout, drawn from scripted flights on the FakeSimulator.

    env -u PYTHONPATH .venv/bin/python scripts/footprint_rule_figure.py --scene s02 \
        --out results/<date>/figures/s02_footprint_rule.png --json results/<date>/figures/s02_footprint_rule.json

A crash into a tree can wedge the simulator's actor (2 of 4 crashes into wood stayed wedged,
docs/gates/m4_s02_wood_crash_reset_probe.json), so tree scenes end an episode where the drone's swept step passes within
CIRCLE_CONTACT_M of a tree's footprint circle (expert/footprint_contact.py), and a training run may add a margin on
top. This flies each seed straight at its nearest tree twice, under the task's rule and with the margin, against a fake
whose only wood is a 0.3 m trunk, and shows where every flight ended: a "sim" ending would mean the rule let the body
reach wood. The rule is checked after a step is flown, so the ending step can go up to a full step (0.4 m) past the
boundary: with the first version's 0.5 m, every task-rule ending on s02 (40 of 40, gaps 0.11-0.48 m) had a rotor tip
inside the circle, which is why the rule is now 0.9 m; the histogram shades the zone a rotor tip would enter, and the
close-up draws the deepest ending step. It launches nothing and writes only --out and --json.

It draws the circle and flies the circle shape (`footprint_shape="circle"`, CIRCLE_CONTACT_M = 0.9 m): since 2026-10-08
(afternoon) the task's rule measures to each model's outline instead, at 1.05 m (expert/footprint_contact.py), and this
figure stays the record of the circle, the rule s02_r1 trained under. scripts/footprint_endings_figure.py measures the
outline's gap at real endings.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
from collections import Counter  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from autofly_ue5.expert.footprint_contact import CIRCLE_CONTACT_M, footprint_obstacles  # noqa: E402
from autofly_ue5.expert.seeds import PROBE_SEED_BASE  # noqa: E402
from autofly_ue5.paths import RUNS_DIR, SCENES_DIR  # noqa: E402
from autofly_ue5.scenes.level_spec import map_path_for  # noqa: E402
from autofly_ue5.scenes.model import (  # noqa: E402
    DRONE_HALF_SPAN_M,
    Bounds,
    Instance,
    Layout,
    SceneFile,
    load_registry,
    load_scene_file,
)
from autofly_ue5.sim.fake import FakeSimulator  # noqa: E402

WOOD_RADIUS_M = 0.3  # the fake's trunk per unit scale, as in tests/test_footprint_contact.py: inside every circle
MAX_STEPS = 300  # AutoFlyEnv's default episode length; it truncates there itself

# Ending colours (red = the task's rule, orange = the training margin, black = wood reached).
ENDING_STYLE = {
    "footprint": ("#d62828", "ended by the task rule (footprint)"),
    "footprint_margin": ("#f77f00", "ended by the training margin (footprint_margin)"),
    "sim": ("#000000", "reached the fake's wood (sim): must never happen"),
}
OTHER_COLOUR = "#8d99ae"
TREE_FILL, TREE_EDGE, TRUNK = "#b7e4a7", "#4f8a3c", "#7f4f24"
PATH_COLOUR = "#3a5a8c"


def _scene_and_layout(scene_arg: str, layout_arg: str | None) -> tuple[SceneFile, Layout, Path, str]:
    """The scene (an id from scenes/, or a scene JSON path) and its stored layout (runs/levels/<id>.layout.json format),
    checked against each other as scenes/resolve.py checks them: a layout built from another scene would draw the
    wrong trees under the right name."""
    path = Path(scene_arg)
    if not path.is_file():
        matches = sorted(SCENES_DIR.glob(f"{scene_arg}_*.json"))
        if len(matches) != 1:
            raise SystemExit(f"scene {scene_arg!r}: expected one file {SCENES_DIR}/{scene_arg}_*.json, found {len(matches)}")
        path = matches[0]
    scene = load_scene_file(path)
    base_id = scene.level or scene.id
    layout_path = Path(layout_arg) if layout_arg else RUNS_DIR / "levels" / f"{base_id}.layout.json"
    blob = layout_path.read_bytes()
    stored = json.loads(blob)
    raw = stored["layout"]
    if raw["scene_id"] != base_id:
        raise SystemExit(f"{layout_path} holds scene {raw['scene_id']!r}'s layout, not {base_id!r}'s")
    if scene.level is None and stored.get("scene_sha256") not in (None, scene.sha256):
        raise SystemExit(f"{layout_path} was built from another version of {scene.path} (scene_sha256 differs)")
    layout = Layout(scene_id=raw["scene_id"], seed=raw["seed"], bounds=Bounds(**raw["bounds"]),
                    instances=tuple(Instance(**i) for i in raw["instances"]))
    return scene, layout, layout_path, hashlib.sha256(blob).hexdigest()


def _wood(trees) -> list[tuple[float, float, float]]:
    return [(t.x, t.y, WOOD_RADIUS_M * t.scale[0]) for t in trees]


def _nearest_tree(trees, pose) -> Instance:
    return min(trees, key=lambda t: math.hypot(t.x - pose.x, t.y - pose.y) - t.radius_m)


def fly_at_nearest_tree(env, trees, seed: int) -> dict:
    """tests/test_footprint_contact.py::_fly_at_a_tree: turn to the nearest tree's pivot and fly straight at it at full
    speed, so every flight tests the rule head on. Returns the path flown and how it ended."""
    env.reset(seed=seed)
    start = env._last_pose
    target = _nearest_tree(trees, start)
    path = [[start.x, start.y]]
    info: dict = {}
    for _ in range(MAX_STEPS + 5):
        pose = env._last_pose
        bearing = math.atan2(target.y - pose.y, target.x - pose.x) - pose.yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        action = np.array([2.0 if abs(bearing) < 0.2 else 0.0, float(np.clip(2.0 * bearing, -1, 1)), 0.0], np.float32)
        _obs, _reward, terminated, truncated, info = env.step(action)
        path.append(info["pose"][:2])
        if terminated or truncated:
            break
    contact = info.get("footprint_contact")
    return {
        "seed": seed,
        "aimed_at": target.tag,
        "outcome": info.get("outcome"),
        "collision_source": info.get("collision_source"),
        "gap_m": contact["gap_m"] if contact else None,
        "ended_at_tag": contact["tag"] if contact else None,
        "ended_at_asset": contact["asset"] if contact else None,
        "steps": info.get("steps"),
        "end_pose": [round(v, 4) for v in info.get("pose", [])],
        "path": path,
    }


def _ending(ep: dict) -> str:
    return ep["collision_source"] or ep["outcome"] or "none"


def fly_all(scene, layout, trees, seeds, margin: float) -> list[dict]:
    from autofly_ue5.expert.env import AutoFlyEnv

    env = AutoFlyEnv(scene, layout, functools.partial(FakeSimulator, obstacles=_wood(trees)),
                     map_path=map_path_for(layout.scene_id), instance=0, max_episode_steps=MAX_STEPS,
                     footprint_contact_margin_m=margin, footprint_shape="circle")
    out = []
    for seed in seeds:
        try:
            out.append(fly_at_nearest_tree(env, trees, seed))
        except Exception as exc:  # a start the env refuses (StartCollisionError) is recorded, not hidden
            out.append({"seed": seed, "outcome": "reset_refused", "collision_source": None, "gap_m": None,
                        "steps": 0, "error": f"{type(exc).__name__}: {exc}", "path": []})
    return out


def summarise(episodes: list[dict], margin: float) -> dict:
    gaps = [ep["gap_m"] for ep in episodes if ep["gap_m"] is not None]
    rule = [ep["gap_m"] for ep in episodes if ep["collision_source"] == "footprint"]
    wide = [ep["gap_m"] for ep in episodes if ep["collision_source"] == "footprint_margin"]
    eps = 1e-9
    return {
        "margin_m": margin,
        "termination_m": CIRCLE_CONTACT_M + margin,
        "n": len(episodes),
        "endings": dict(sorted(Counter(_ending(ep) for ep in episodes).items())),
        "gap_m": ({"min": min(gaps), "median": round(statistics.median(gaps), 4), "max": max(gaps)} if gaps else None),
        # On the ending step the centre may already be a full step (0.4 m) inside the boundary: below the half span,
        # a rotor tip would be inside the circle at that step's closest approach.
        "endings_with_rotor_tip_inside_circle": sum(g < DRONE_HALF_SPAN_M for g in gaps),
        "checks": {
            "no_sim_endings": all(ep["collision_source"] != "sim" for ep in episodes),
            "footprint_gaps_le_contact_m": all(g <= CIRCLE_CONTACT_M + eps for g in rule),
            "footprint_margin_gaps_in_band": all(CIRCLE_CONTACT_M < g <= CIRCLE_CONTACT_M + margin + eps
                                                 for g in wide),
        },
    }


# ----------------------------------------------------------------------------------------------------------------------
# Drawing. Maps are top-down in NED: x (north) up the page, y (east) across it.
# ----------------------------------------------------------------------------------------------------------------------
def _ring(ax, b: Bounds, outer_inset: float, inner_inset: float, colour: str, label: str) -> None:
    def rect(inset, reverse):
        pts = [(b.y_min + inset, b.x_min + inset), (b.y_max - inset, b.x_min + inset),
               (b.y_max - inset, b.x_max - inset), (b.y_min + inset, b.x_max - inset)]
        pts = pts[::-1] if reverse else pts
        return pts + [pts[0]]

    verts = rect(outer_inset, False) + rect(inner_inset, True)
    codes = ([MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]) * 2
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=colour, edgecolor="none", alpha=0.55, label=label, zorder=0))


def _trees(ax, trees, margin: float, active: str, *, label_paths: list | None = None) -> None:
    for t in trees:
        ax.add_patch(Circle((t.y, t.x), t.radius_m, facecolor=TREE_FILL, edgecolor=TREE_EDGE, linewidth=0.6, zorder=1))
        ax.add_patch(Circle((t.y, t.x), WOOD_RADIUS_M * t.scale[0], facecolor=TRUNK, edgecolor="none", zorder=2))
        ax.add_patch(Circle((t.y, t.x), t.radius_m + CIRCLE_CONTACT_M, fill=False, edgecolor="#333333",
                            linestyle=(0, (4, 2)), linewidth=1.1 if active == "task" else 0.6, zorder=3))
        ax.add_patch(Circle((t.y, t.x), t.radius_m + CIRCLE_CONTACT_M + margin, fill=False, edgecolor="#333333",
                            linestyle=(0, (1, 1.5)), linewidth=1.1 if active == "margin" else 0.6, zorder=3))
    if label_paths is None:
        return
    for t, (cx, cy, _w, _h) in _label_spots(trees, label_paths):
        edge = (t.y + (cx - t.y) * t.radius_m / max(math.hypot(cx - t.y, cy - t.x), 1e-9),
                t.x + (cy - t.x) * t.radius_m / max(math.hypot(cx - t.y, cy - t.x), 1e-9))
        ax.annotate(t.asset, edge, xytext=(cx, cy), ha="center", va="center", fontsize=6.5, color="#24451a", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#24451a", linewidth=0.5, shrinkA=0, shrinkB=0),
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))


def _label_spots(trees, paths) -> list[tuple[Instance, tuple[float, float, float, float]]]:
    """One asset-name label per asset, on whichever of its instances and on whichever side overlaps the fewest other
    circles, flown paths and earlier labels (map coordinates: horizontal y, vertical x; a 6.5 pt character is about
    0.55 m wide on a 74 m panel)."""
    pts = np.concatenate([np.asarray(p)[:, ::-1] for p in paths if len(p)]) if any(len(p) for p in paths) else np.zeros((0, 2))
    placed: list[tuple[float, float, float, float]] = []
    out = []

    def overlap(a, b):
        return max(0.0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2)) * \
            max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))

    for asset in dict.fromkeys(t.asset for t in trees):
        best = None
        for t in (t for t in trees if t.asset == asset):
            w, h = 0.55 * len(asset) + 0.6, 1.6
            for k, ang in enumerate(np.radians([-90, 90, 0, 180, -45, -135, 45, 135])):
                d = t.radius_m + CIRCLE_CONTACT_M + 1.0 + abs(math.cos(ang)) * w / 2 + abs(math.sin(ang)) * h / 2
                box = (t.y + d * math.cos(ang), t.x + d * math.sin(ang), w, h)
                score = sum(overlap(box, (o.y, o.x, 2 * (o.radius_m + 1.0), 2 * (o.radius_m + 1.0))) for o in trees)
                score += sum(4 * overlap(box, q) for q in placed)
                if len(pts):
                    inside = (np.abs(pts[:, 0] - box[0]) < w / 2) & (np.abs(pts[:, 1] - box[1]) < h / 2)
                    score += 0.5 * float(inside.sum())
                score += 0.05 * k - 0.01 * t.radius_m  # ties: below, then the larger tree
                if best is None or score < best[0]:
                    best = (score, t, box)
        placed.append(best[2])
        out.append((best[1], best[2]))
    return out


def _map_panel(ax, layout: Layout, scene: SceneFile, trees, episodes, margin: float, active: str, title: str,
               label_paths) -> None:
    b = layout.bounds
    _ring(ax, b, scene.target_band[0], scene.target_band[1], "#cfe3f5", "target band (0-3 m in)")
    _ring(ax, b, scene.start_band[0], scene.start_band[1], "#f6e3b4", "start band (2-6 m in)")
    ax.add_patch(PathPatch(MplPath([(b.y_min, b.x_min), (b.y_max, b.x_min), (b.y_max, b.x_max), (b.y_min, b.x_max),
                                    (b.y_min, b.x_min)]), fill=False, edgecolor="#555555", linewidth=1.0, zorder=1))
    _trees(ax, trees, margin, active, label_paths=label_paths)
    for ep in episodes:
        if not ep["path"]:
            continue
        p = np.asarray(ep["path"])
        ax.plot(p[:, 1], p[:, 0], color=PATH_COLOUR, linewidth=0.7, alpha=0.75, zorder=4)
        ax.plot(p[0, 1], p[0, 0], "o", color=PATH_COLOUR, markersize=2.5, zorder=4)
        colour = ENDING_STYLE.get(_ending(ep), (OTHER_COLOUR, ""))[0]
        ax.plot(p[-1, 1], p[-1, 0], "x", color=colour, markersize=6, markeredgewidth=1.6, zorder=5)
    pad = 2.0
    ax.set_xlim(b.y_min - pad, b.y_max + pad)
    ax.set_ylim(b.x_min - pad, b.x_max + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m, east) →")
    ax.set_ylabel("x (m, north) →")
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(labelsize=8)


def _histogram(ax, task_eps, margin_eps, margin: float) -> None:
    hi = CIRCLE_CONTACT_M + margin + 0.1
    bins = np.arange(0.0, hi + 1e-9, 0.025)
    top = 0
    for eps, colour, label in ((task_eps, ENDING_STYLE["footprint"][0], "task rule"),
                               (margin_eps, ENDING_STYLE["footprint_margin"][0], f"with the {margin:g} m margin")):
        gaps = [ep["gap_m"] for ep in eps if ep["gap_m"] is not None]
        counts, _, _ = ax.hist(gaps, bins=bins, color=colour, alpha=0.85, edgecolor="white", linewidth=0.8,
                               label=f"{label} (n = {len(gaps)})", zorder=2)
        top = max(top, counts.max() if len(counts) else 0)
    ax.set_ylim(0, max(top, 1) * 1.75)
    ax.axvspan(0.0, DRONE_HALF_SPAN_M, color="#e9ecef", zorder=0)
    ax.text(0.015, 0.97, f"gap < drone half-span {DRONE_HALF_SPAN_M} m:\nrotor tips inside the circle\non the ending step",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#495057")
    for x, style, text in ((CIRCLE_CONTACT_M, (0, (4, 2)), f"task rule {CIRCLE_CONTACT_M:g} m"),
                           (CIRCLE_CONTACT_M + margin, (0, (1, 1.5)), f"training boundary {CIRCLE_CONTACT_M + margin:g} m")):
        ax.axvline(x, color="#333333", linestyle=style, linewidth=1.3, zorder=3)
        ax.text(x - 0.012, 0.5, text, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="center",
                fontsize=7, color="#333333")
    ax.set_xlim(0, hi)
    ax.set_xlabel("gap from the footprint circle at the ending step (m)", fontsize=8.5)
    ax.set_ylabel("flights", fontsize=8.5)
    ax.set_title("Where the rule ended each flight", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)


def _close_up(ax, tree: Instance, margin: float, min_task_gap: float | None) -> None:
    """The largest tree to scale, the drone parked on the task boundary (its tips 2 cm outside the circle), and the
    deepest ending step the task-rule flights reached: a step is 0.4 m at full speed, and the rule ends the episode
    after the step that crossed the boundary, not before it."""
    r, contact = tree.radius_m, CIRCLE_CONTACT_M
    trunk = WOOD_RADIUS_M * tree.scale[0]
    ax.add_patch(Circle((0, 0), r, facecolor=TREE_FILL, edgecolor=TREE_EDGE, linewidth=0.8))
    ax.add_patch(Circle((0, 0), trunk, facecolor=TRUNK, edgecolor="none"))
    ax.add_patch(Circle((0, 0), r + contact, fill=False, edgecolor="#333333", linestyle=(0, (4, 2)), linewidth=1.1))
    ax.add_patch(Circle((0, 0), r + contact + margin, fill=False, edgecolor="#333333", linestyle=(0, (1, 1.5)),
                        linewidth=1.2))
    centre = r + contact  # the drone due east of the pivot, its centre on the task boundary
    ax.add_patch(Circle((centre, 0), DRONE_HALF_SPAN_M, fill=False, edgecolor=PATH_COLOUR, linewidth=1.6, zorder=4))
    ax.plot(centre, 0, "+", color=PATH_COLOUR, markersize=6, zorder=4)
    box = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
    ax.text(0, -trunk - 0.1, f"trunk {trunk:.2f} m (the fake's wood)", ha="center", va="top", fontsize=6.5, color=TRUNK)
    ax.text(-0.9, 1.0, f"footprint circle\nr = {r:.2f} m", ha="left", va="center", fontsize=7, color="#24451a")
    if min_task_gap is not None:  # the deepest ending step the task-rule flights reached
        deep = r + min_task_gap
        ax.add_patch(Circle((deep, 0), DRONE_HALF_SPAN_M, fill=False, edgecolor=ENDING_STYLE["footprint"][0],
                            linewidth=0.9, linestyle=(0, (2, 1.5)), zorder=3))
        ax.annotate(f"deepest ending step seen (dashed red): centre {min_task_gap:.2f} m\nfrom the circle, tips "
                    f"{DRONE_HALF_SPAN_M - min_task_gap:.2f} m inside it (a full-speed step is 0.4 m)",
                    (deep, -DRONE_HALF_SPAN_M), xytext=(-0.9, -1.2), ha="left", va="center", fontsize=6.5,
                    color=ENDING_STYLE["footprint"][0], bbox=box,
                    arrowprops=dict(arrowstyle="-", color=ENDING_STYLE["footprint"][0], linewidth=0.6))
    ax.annotate(f"drone centred on the task boundary: rotor tips\n{DRONE_HALF_SPAN_M} m from its centre, "
                f"{contact - DRONE_HALF_SPAN_M:.2f} m outside the circle",
                (centre, -DRONE_HALF_SPAN_M), xytext=(-0.9, -1.85), ha="left", va="center", fontsize=6.5,
                color=PATH_COLOUR, bbox=box, arrowprops=dict(arrowstyle="-", color=PATH_COLOUR, linewidth=0.6))
    ax.set_xlim(-1.0, r + contact + margin + 0.7)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.set_xlabel("m (to scale)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"To scale: {tree.tag} ({tree.asset}), the largest circle", fontsize=10)
    # The 2 cm between the rotor tips and the circle is invisible at this scale: zoom on it, beside the drone.
    clearance = contact - DRONE_HALF_SPAN_M
    ins = ax.inset_axes([0.60, 0.70, 0.38, 0.28])
    ins.add_patch(Circle((0, 0), r, facecolor=TREE_FILL, edgecolor=TREE_EDGE, linewidth=1.0))
    ins.add_patch(Circle((centre, 0), DRONE_HALF_SPAN_M, fill=False, edgecolor=PATH_COLOUR, linewidth=1.6))
    w = 0.05
    ins.set_xlim(r - w, r + clearance + w)
    ins.set_ylim(-0.03, 0.03)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    ins.annotate("", (r, 0.0), (r + clearance, 0.0), arrowprops=dict(arrowstyle="<->", color="black", lw=0.8))
    ins.text(r + clearance / 2, 0.006, f"{clearance * 100:.0f} cm", ha="center", va="bottom", fontsize=7)
    ins.text(r - w * 0.9, -0.024, "circle", fontsize=6, color="#24451a", ha="left")
    ins.text(r + clearance + w * 0.9, -0.024, "rotor tip", fontsize=6, color=PATH_COLOUR, ha="right")
    ax.indicate_inset_zoom(ins, edgecolor="#555555")


def draw(out: Path, scene, layout, trees, task_eps, margin_eps, margin: float, summary: dict) -> None:
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.78], height_ratios=[1, 1], wspace=0.16, hspace=0.32,
                          left=0.045, right=0.985, top=0.88, bottom=0.14)
    ax_task, ax_wide = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    ax_hist, ax_tree = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    n = len(task_eps)

    def counts(run):
        e = summary["runs"][run]["endings"]
        return ", ".join(f"{k} {v}" for k, v in e.items())

    paths = [ep["path"] for ep in task_eps]
    _map_panel(ax_task, layout, scene, trees, task_eps, margin, "task",
               f"Task rule: ends at {CIRCLE_CONTACT_M:g} m from the circle (dashed)\n{n} flights: {counts('task_rule')}",
               paths)
    _map_panel(ax_wide, layout, scene, trees, margin_eps, margin, "margin",
               f"Training run: + {margin:g} m margin, ends at {CIRCLE_CONTACT_M + margin:g} m (dotted)\n"
               f"{n} flights: {counts('training_margin')}", paths)
    _histogram(ax_hist, task_eps, margin_eps, margin)
    biggest = max(trees, key=lambda t: t.radius_m)
    gaps = [ep["gap_m"] for ep in task_eps if ep["collision_source"] == "footprint"]
    _close_up(ax_tree, biggest, margin, min(gaps) if gaps else None)

    fig.suptitle(f"Footprint contact rule on {layout.scene_id} ({len(trees)} trees): an episode ends as a collision when "
                 f"the drone's swept step passes within {CIRCLE_CONTACT_M:g} m of a tree's footprint circle",
                 fontsize=13, y=0.975)
    fig.text(0.5, 0.928, f"Scripted flights straight at the nearest tree on the FakeSimulator (its only wood: a "
             f"{WOOD_RADIUS_M:g} m x scale trunk), seeds {task_eps[0]['seed']}-{task_eps[-1]['seed']}, each flown "
             f"under the task rule (left) and with a training margin (right). Maps: NED, x north up, y east across.",
             ha="center", fontsize=9, color="#444444")
    handles = [
        Patch(facecolor=TREE_FILL, edgecolor=TREE_EDGE, label="footprint circle (layout radius_m)"),
        Patch(facecolor=TRUNK, label=f"fake trunk ({WOOD_RADIUS_M:g} m x scale)"),
        Line2D([], [], color="#333333", linestyle=(0, (4, 2)), label=f"task contact boundary (+{CIRCLE_CONTACT_M:g} m)"),
        Line2D([], [], color="#333333", linestyle=(0, (1, 1.5)),
               label=f"training boundary (+{CIRCLE_CONTACT_M:g} + {margin:g} m)"),
        Patch(facecolor="#f6e3b4", label=f"start band ({scene.start_band[0]:g}-{scene.start_band[1]:g} m in)"),
        Patch(facecolor="#cfe3f5", label=f"target band ({scene.target_band[0]:g}-{scene.target_band[1]:g} m in)"),
        Line2D([], [], color=PATH_COLOUR, linewidth=1.0, marker="o", markersize=3, label="scripted flight (start dot)"),
        *[Line2D([], [], color=c, marker="x", linestyle="none", markersize=7, markeredgewidth=1.8, label=lab)
          for c, lab in ENDING_STYLE.values()],
        Line2D([], [], color=OTHER_COLOUR, marker="x", linestyle="none", markersize=7, markeredgewidth=1.8,
               label="other ending (timeout, out of bounds)"),
        Line2D([], [], color=PATH_COLOUR, marker="o", markersize=8, markerfacecolor="none", linestyle="none",
               label=f"drone rotor span (radius {DRONE_HALF_SPAN_M} m)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="s02", help="scene id (scenes/<id>_*.json) or a scene JSON path")
    ap.add_argument("--layout", default=None, help="stored layout (default runs/levels/<scene>.layout.json)")
    ap.add_argument("--out", type=Path, required=True, help="PNG to write")
    ap.add_argument("--json", type=Path, required=True, help="the figure's numbers")
    ap.add_argument("--margin", type=float, default=0.2, help="training margin for the right panel (m); the trainer allows at most 0.2")
    ap.add_argument("--seeds", type=int, default=40, help="episodes to fly under each rule")
    ap.add_argument("--seed-base", type=int, default=PROBE_SEED_BASE, help="first seed (expert/seeds.py's probe range)")
    args = ap.parse_args(argv)
    if not (math.isfinite(args.margin) and args.margin > 0):
        ap.error(f"--margin must be a positive number of metres, got {args.margin}")
    if args.seeds < 1:
        ap.error("--seeds must be at least 1")

    scene, layout, layout_path, layout_sha = _scene_and_layout(args.scene, args.layout)
    trees = footprint_obstacles(layout.instances, load_registry())
    if not trees:
        print(f"scene {scene.id} has no mesh obstacle: no footprint rule to draw", file=sys.stderr)
        return 2
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    task_eps = fly_all(scene, layout, trees, seeds, 0.0)
    margin_eps = fly_all(scene, layout, trees, seeds, args.margin)
    summary = {"runs": {"task_rule": summarise(task_eps, 0.0), "training_margin": summarise(margin_eps, args.margin)}}
    checks = {f"{run}.{k}": v for run, s in summary["runs"].items() for k, v in s["checks"].items()}
    summary["all_checks_passed"] = all(checks.values())

    draw(args.out, scene, layout, trees, task_eps, margin_eps, args.margin, summary)
    strip = lambda eps: [{k: v for k, v in ep.items() if k != "path"} for ep in eps]  # noqa: E731
    record = {
        "what": "scripted flights straight at the nearest tree on the FakeSimulator; footprint contact rule endings",
        "scene": {"id": scene.id, "path": str(scene.path), "sha256": scene.sha256},
        "layout": {"path": str(layout_path), "sha256": layout_sha, "trees": len(trees)},
        "rule": {"contact_m": CIRCLE_CONTACT_M, "shape": "circle", "training_margin_m": args.margin,
                 "drone_half_span_m": DRONE_HALF_SPAN_M, "fake_wood_radius_m_per_scale": WOOD_RADIUS_M},
        "seeds": {"base": args.seed_base, "n": args.seeds},
        "figure": str(args.out),
        "summary": summary,
        "episodes": {"task_rule": strip(task_eps), "training_margin": strip(margin_eps)},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=2) + "\n")
    for run, s in summary["runs"].items():
        print(f"{run}: endings {s['endings']}, gap {s['gap_m']}, rotor tip inside circle on "
              f"{s['endings_with_rotor_tip_inside_circle']}, checks {s['checks']}")
    print(f"wrote {args.out} and {args.json}")
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
