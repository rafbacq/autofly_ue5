"""Figures of what a list of raw dataset stores holds, for final_results/ (plan 6 step 7), and every number they draw.

    env -u PYTHONPATH .venv/bin/python scripts/dataset_figures.py --stores data/s01_c1 data/s02_c1 ... \\
        --out-dir runs/report/figures [--levels runs/levels] [--targets assets/targets.json] \\
        [--record <store>=<collector record> ...] [--records-dir runs/collect] [--no-rejects]

Writes into --out-dir, and nowhere else:
- composition.png: kept episodes per scene, per target category, per target (every target, most episodes first), seen
  against unseen targets, and stand-ins against the rest (where the provenance or the pool file marks them).
- kept_rates.png: per store, the flights kept and rejected by reason; the kept-rate with its Wilson 95 % interval, per
  store and per scene (a scene's batches pooled), against plan 6's bar of 0.25 for the interval's lower end.
- trajectories_<scene>.png: every kept trajectory from above, over the layout's obstacles, with the starts, the targets
  and where each rejected flight ended (by reason). A mesh obstacle is drawn as its outline (its footprint field's zero
  set, what the outline rule measures to), a pillar as its circle, a crate as its box; on a scene with movers, the
  pillars that moved in any kept episode are hatched at their homes.
- actions_states.png: per dimension of action[3] and state[9], each scene's distribution, one row per scene.
- lengths_per.png: records per episode, the flown path length, and the paper's PER as the interval between its
  straight-line (lower) and grid-optimum (upper) bounds, computed by scripts/dataset_stats.py's own functions. A scene
  with movers has the lower bound only: its layout's grid is not where the movers were.
- first_frames_<scene>.png: the first recorded frame of up to --first-frames episodes, the targets taken in turn so the
  sheet shows as many different targets as the scene holds, each captioned with the target its instruction names.
- targets_closeup.png: per target, the last recorded frame of the episode that ends with it nearest dead ahead (the last
  record lies within the 5 m success radius, so the target is large): the visual proof of the objects' diversity.
- dataset_figures.json: every number drawn (counts, rates, intervals, medians), so a README can quote the record.

What each number means is autofly_ue5/report/stores.py's to say (a target's category from its provenance or, for a
store from before the pool, the pool file by name; flights from the rejects, the collector's record or the manifest).
The colours are the dataviz skill's validated categorical slots in fixed order: kept is slot 1 everywhere, a reject
reason and a target category keep their colour in every figure. It launches nothing and reads the stores only.
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
import re  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon  # noqa: E402

from autofly_ue5.dataset.state import STATE_FIELDS  # noqa: E402
from autofly_ue5.dataset.targets import CATEGORIES, USABLE_STATUSES  # noqa: E402
from autofly_ue5.paths import ASSET_FOOTPRINTS, ROOT, RUNS_DIR, SCENES_DIR  # noqa: E402
from autofly_ue5.report.stores import (  # noqa: E402
    UNKNOWN,
    UNRECORDED,
    Episode,
    Pool,
    Reject,
    Store,
    load_pool,
    load_stores,
    wilson_interval,
)
from autofly_ue5.scenes.footprint_field import FootprintField, FootprintFields, read_manifest, to_world  # noqa: E402
from autofly_ue5.scenes.model import AssetEntry, AssetRegistry, Bounds, Instance, Layout, load_registry  # noqa: E402
from autofly_ue5.scenes.resolve import ResolvedScene, resolve_scene  # noqa: E402
from scripts.dataset_stats import episode_efficiency, summarise  # noqa: E402
from scripts.layout_figure import START_BAND_COLOUR, TARGET_BAND_COLOUR, _band  # noqa: E402

# The dataviz skill's validated categorical slots (light surface), in their fixed order; neutral chrome and ink.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OBSTACLE_FILL, OBSTACLE_EDGE, UNRECORDED_COLOUR = "#d6d4cc", "#77756f", "#b9b7af"
KEPT_COLOUR = SLOTS[0]
CATEGORY_COLOURS = {**{c: SLOTS[i] for i, c in enumerate(CATEGORIES)}, UNKNOWN: MUTED}
KNOWN_REASONS = ("collision", "out_of_bounds", "timeout", "exceeded_max_steps")  # env outcomes, then the collector's
ACTION_TITLES = ("forward (m/s)", "yaw rate (rad/s)", "vertical, up (m/s)")  # dataset/raw.py's record_fields["action"]
PRIVILEGED_STATE = (0, 1)  # from the target's true position: plan 6 revision 1, item 4
PLAN6_BAR = 0.25  # plan 6 revision 1, item 3: a pilot's kept-rate needs a Wilson lower bound of at least this
PAPER_PER = (0.73, 0.78)  # the paper's models' path efficiency (results/2026-10-07-project-status, section 2)
SCENE_BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)  # every scene's (scenes/model.py refuses any other): a missing layout's
BINS = 40
FIRST_FRAMES = 24
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titlelocation": "left",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
    "grid.linewidth": 0.8, "grid.linestyle": "-", "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "text.color": INK, "font.size": 9, "legend.frameon": False, "legend.fontsize": 8.5,
}


def _r(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _stats(values) -> dict | None:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    p5, q1, median, q3, p95 = np.percentile(a, [5, 25, 50, 75, 95])
    return {"n": int(a.size), "mean": _r(a.mean()), "std": _r(a.std()), "min": _r(a.min()), "p5": _r(p5), "q1": _r(q1),
            "median": _r(median), "q3": _r(q3), "p95": _r(p95), "max": _r(a.max())}


def scene_label(scene_id: str) -> str:
    """'s02 sparse trees': the id and the rest of its scene file's name (scenes/<id>_<name>.json), as plan 6 names it."""
    matches = sorted(SCENES_DIR.glob(f"{scene_id}_*.json"))
    return f"{scene_id} {matches[0].stem[len(scene_id) + 1:].replace('_', ' ')}" if len(matches) == 1 else scene_id


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width)) or text


# ----------------------------------------------------------------------------------------------------------------------
# What the stores hold.
# ----------------------------------------------------------------------------------------------------------------------
def composition(stores: Sequence[Store], pool: Pool | None) -> dict:
    episodes = [e for s in stores for e in s.episodes]
    by_category = {c: 0 for c in CATEGORIES}
    for e in episodes:
        by_category[e.target.category] = by_category.get(e.target.category, 0) + 1
    targets = Counter(e.target.name for e in episodes)
    category_of = {}
    for e in episodes:
        category_of.setdefault(e.target.name, e.target.category)
    by_scene_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in episodes:
        by_scene_category[e.scene][e.target.category] += 1
    never = None
    if pool is not None:  # what the pool could draw on a training scene (seen, spawnable), but no store kept
        never = sorted(name for name, entry in pool.entries.items()
                       if entry.get("status") in USABLE_STATUSES and entry.get("seen") is True and name not in targets)
    return {
        "stores": len(stores), "episodes": len(episodes), "records": sum(e.steps for e in episodes),
        "by_scene": {s: n for s, n in sorted(Counter(e.scene for e in episodes).items())},
        "by_category": by_category,
        "by_target": dict(sorted(targets.items(), key=lambda kv: (-kv[1], kv[0]))),
        "target_category": dict(sorted(category_of.items())),
        "by_scene_category": {s: dict(sorted(c.items())) for s, c in sorted(by_scene_category.items())},
        "seen": {"seen": sum(e.target.seen is True for e in episodes), "unseen": sum(e.target.seen is False for e in episodes),
                 "unknown": sum(e.target.seen is None for e in episodes)},
        "stand_in": {"stand-in": sum(e.target.stand_in is True for e in episodes),
                     "not a stand-in": sum(e.target.stand_in is False for e in episodes),
                     "unmarked": sum(e.target.stand_in is None for e in episodes)},
        "name_status": dict(sorted(Counter(e.target.name_status or "none" for e in episodes).items())),
        "filled_from_pool": dict(sorted(Counter(f for e in episodes for f in e.target.from_pool).items())),
        "pool_targets_never_kept": never,
    }


def kept_rates(stores: Sequence[Store]) -> dict:
    rows, notes = [], []
    pooled: dict[str, dict] = defaultdict(lambda: {"kept": 0, "flown": 0, "rejected": Counter(), "stores": []})
    for s in stores:
        a = s.attempts
        interval = a.wilson()
        rows.append({"store": s.name, "scenes": list(s.scenes), "kept": a.kept, "flown": a.flown, "rejected": a.rejected,
                     "kept_rate": _r(a.kept_rate), "wilson_95": [_r(v) for v in interval] if interval else None,
                     "source": a.source, "collision_sources": a.collision_sources, "fault_retries": a.fault_retries,
                     "disagreements": list(a.disagreements), "record": str(s.record_path) if s.record_path else None,
                     "rejects_dir": str(s.rejects_dir) if s.rejects_dir else None})
        if len(s.scenes) == 1:
            p = pooled[s.scenes[0]]
            p["kept"] += a.kept
            p["flown"] += a.flown
            p["rejected"].update(a.rejected)
            p["stores"].append(s.name)
        elif s.rejects is not None:  # a store of several scenes: split by each episode's and each reject's own scene
            for scene in s.scenes:
                p = pooled[scene]
                kept = sum(e.scene == scene for e in s.episodes)
                rejected = Counter(r.reason for r in s.rejects if r.scene == scene)
                p["kept"] += kept
                p["flown"] += kept + sum(rejected.values())
                p["rejected"].update(rejected)
                p["stores"].append(s.name)
        else:
            notes.append(f"{s.name} holds scenes {list(s.scenes)} and no rejects were read: its flights are not split "
                         f"by scene")
    scenes = []
    for scene, p in sorted(pooled.items()):
        interval = wilson_interval(p["kept"], p["flown"]) if p["flown"] else None
        scenes.append({"scene": scene, "stores": p["stores"], "kept": p["kept"], "flown": p["flown"],
                       "rejected": dict(sorted(p["rejected"].items())),
                       "kept_rate": _r(p["kept"] / p["flown"]) if p["flown"] else None,
                       "wilson_95": [_r(v) for v in interval] if interval else None})
    return {"stores": rows, "scenes": scenes, "bar_wilson_lower": PLAN6_BAR, "notes": notes}


def reason_colours(reasons) -> dict[str, str]:
    """A reject reason's colour, the same in every figure: the known outcomes take slots 2-5 in a fixed order, any other
    reason the next free slot by name, and the manifest-only count a neutral grey."""
    colours = {r: SLOTS[1 + i] for i, r in enumerate(KNOWN_REASONS)}
    spare = iter(SLOTS[1 + len(KNOWN_REASONS):])
    for r in sorted(set(reasons) - set(KNOWN_REASONS) - {UNRECORDED}):
        colours[r] = next(spare, MUTED)
    colours[UNRECORDED] = UNRECORDED_COLOUR
    return colours


def _reason_order(reasons) -> list[str]:
    extra = sorted(set(reasons) - set(KNOWN_REASONS) - {UNRECORDED})
    return [r for r in KNOWN_REASONS if r in reasons] + extra + ([UNRECORDED] if UNRECORDED in reasons else [])


# ----------------------------------------------------------------------------------------------------------------------
# Obstacles seen from above.
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Shape:
    kind: str  # "circle", "box" or "outline"
    tag: str
    asset: str
    polygons: tuple[np.ndarray, ...] = ()  # world (x, y) vertices, for a box or an outline
    centre: tuple[float, float] = (0.0, 0.0)  # a circle's
    radius_m: float = 0.0


def polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def outline_polygons(field: FootprintField, inst: Instance) -> list[np.ndarray]:
    """An instance's outline as world (x, y) polygons: the field's cells that hold geometry (a value of at most half a
    cell diagonal, scenes/footprint_field.py: every outline point lies within that of such a centre), traced along their
    outer boundary through the boundary cells' centres, simplified to within one cell, and placed by the instance's
    transform. Holes (gaps between leaf cards) are filled: from above, a canopy reads as one shape."""
    mask = (field.values <= field.cell_m * math.sqrt(2.0) / 2.0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        cells = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2).astype(np.float64)  # (column, row) = (glTF x, z)
        if len(cells) < 3:
            continue
        local = np.stack([field.origin[0] + (cells[:, 0] + 0.5) * field.cell_m,
                          field.origin[1] + (cells[:, 1] + 0.5) * field.cell_m], axis=1)
        out.append(to_world(inst, local))
    return out


def box_corners(inst: Instance, asset: AssetEntry) -> np.ndarray:
    """A box footprint's four corners, world (x, y): the asset's base size times the instance's scale, turned by its yaw
    (NED: positive yaw turns north toward east, as scenes/footprint_field.to_world)."""
    hx, hy = asset.base_size_m[0] / 2.0 * inst.scale[0], asset.base_size_m[1] / 2.0 * inst.scale[1]
    c, s = math.cos(inst.yaw), math.sin(inst.yaw)
    return np.array([(inst.x + c * dx - s * dy, inst.y + s * dx + c * dy) for dx, dy in
                     ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))])


def obstacle_shapes(layout: Layout, registry: AssetRegistry, fields_root: Path) -> tuple[list[Shape], list[str]]:
    """Every layout obstacle as drawn from above. A measured model (the registry gives it a radius profile: what the
    footprint rule covers) is its outline where its field exists, else its footprint circle; a box footprint is its box,
    once per stack; anything else its circle (`radius_m`)."""
    notes: list[str] = []
    try:
        available = set(read_manifest(fields_root).get("fields", {}))
    except ValueError as err:
        available = set()
        notes.append(f"no outline fields read: {err}")
    fields: dict[str, FootprintField] = {}
    shapes: list[Shape] = []
    boxes_seen, missing = set(), set()
    for inst in layout.instances:
        asset = registry.assets.get(inst.asset)
        if asset is not None and asset.footprint == "box":
            key = (round(inst.x, 3), round(inst.y, 3), round(inst.yaw % (2 * math.pi), 3), round(inst.scale[0], 3),
                   round(inst.scale[1], 3))
            if key not in boxes_seen:
                boxes_seen.add(key)
                shapes.append(Shape("box", inst.tag, inst.asset, (box_corners(inst, asset),)))
            continue
        if asset is not None and asset.radius_profile_m is not None:
            if inst.asset in available:
                if inst.asset not in fields:
                    fields[inst.asset] = FootprintFields.load([inst.asset], fields_root)[inst.asset]
                try:
                    polygons = outline_polygons(fields[inst.asset], inst)
                except ValueError as err:  # a field has one horizontal scale (footprint_field.horizontal_scale)
                    polygons = []
                    notes.append(f"{inst.tag} drawn as its footprint circle: {err}")
                if polygons:
                    shapes.append(Shape("outline", inst.tag, inst.asset, tuple(polygons)))
                    continue
            else:
                missing.add(inst.asset)
        shapes.append(Shape("circle", inst.tag, inst.asset, centre=(inst.x, inst.y), radius_m=inst.radius_m))
    if missing:
        notes.append(f"no outline field for {sorted(missing)} in {fields_root}: drawn as footprint circles")
    return shapes, notes


# ----------------------------------------------------------------------------------------------------------------------
# Episodes chosen for pictures.
# ----------------------------------------------------------------------------------------------------------------------
def pick_first_frames(episodes: Sequence[Episode], limit: int) -> list[Episode]:
    """Up to `limit` episodes, the targets taken in turn (by name), each target's episodes in store order: a sheet of
    first frames then shows as many different targets as there are."""
    queues: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        queues[e.target.name].append(e)
    picked: list[Episode] = []
    depth = 0
    while len(picked) < limit and any(len(q) > depth for q in queues.values()):
        for name in sorted(queues):
            if len(queues[name]) > depth and len(picked) < limit:
                picked.append(queues[name][depth])
        depth += 1
    return picked


def pick_closeups(episodes: Sequence[Episode], last_state: Mapping[tuple[str, str], np.ndarray]) -> dict[str, Episode]:
    """Per target, the episode whose last record has it nearest dead ahead (smallest |bearing|, state[1]; then the
    nearest, state[0]): the last record is inside the success radius, so that frame shows the target large and centred."""
    best: dict[str, tuple[tuple[float, float, str, str], Episode]] = {}
    for e in episodes:
        state = last_state[(e.store, e.id)]
        key = (abs(float(state[1])), float(state[0]), e.store, e.id)
        if e.target.name not in best or key < best[e.target.name][0]:
            best[e.target.name] = (key, e)
    order = list(CATEGORIES) + [UNKNOWN]
    return {name: best[name][1] for name in sorted(
        best, key=lambda n: (order.index(best[n][1].target.category) if best[n][1].target.category in order
                             else len(order), n))}


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path} does not decode as an image")
    return image[:, :, ::-1]


# ----------------------------------------------------------------------------------------------------------------------
# Numbers per scene.
# ----------------------------------------------------------------------------------------------------------------------
def _state_title(field: dict) -> str:
    name = field["name"]
    base, unit = name, ""
    for suffix, u in (("_rad_s", "rad/s"), ("_m_s", "m/s"), ("_rad", "rad"), ("_m", "m")):
        if name.endswith(suffix):
            base, unit = name[: -len(suffix)], u
            break
    base = re.sub(r"minus_(\d+)_(\d+)", r"minus \1.\2", base).replace("_", " ")
    return f"{base} ({unit})" if unit else base


def dimensions() -> dict[str, dict]:
    dims = {f"action[{i}]": {"title": t, "privileged": False, "autofly_verified": True} for i, t in enumerate(ACTION_TITLES)}
    for f in STATE_FIELDS:
        dims[f"state[{f['index']}]"] = {"title": _state_title(f), "name": f["name"], "status": f["status"],
                                        "privileged": f["index"] in PRIVILEGED_STATE,
                                        "autofly_verified": f["status"] != "not_autofly_verified"}
    return dims


def lengths_and_per(scene: str, episodes: Sequence[Episode], resolved: ResolvedScene | None) -> dict:
    """Records per episode, the flown length and the PER bounds, by scripts/dataset_stats.py's own functions. Without a
    static layout (a scene with movers, or none found) only the straight-line bound is kept: dataset_stats runs on the
    bounds alone and its grid optimum, which would ignore the obstacles, is discarded."""
    dynamic = resolved is not None and resolved.scene.dynamic is not None
    note = None
    if resolved is None:
        note = "no layout: the straight-line bound only"
    elif dynamic:
        note = "movers: the layout's grid is not where they flew, so the straight-line bound only"
    rows = []
    for e in episodes:
        prov = e.provenance()
        if note is None:
            row = episode_efficiency(resolved.layout, prov)
        else:
            bounds = resolved.layout.bounds if resolved is not None else SCENE_BOUNDS
            row = episode_efficiency(Layout(scene_id=scene, seed=0, bounds=bounds, instances=()), prov)
            row.update(l_opt_physical_m=None, per_physical=None, l_opt_clearance_m=None, per_clearance=None)
        rows.append({"id": e.id, **row})
    summary = summarise(rows)
    per_straight = summary["per_straight"] if note is None else _per_stats([r["per_straight"] for r in rows])
    upper = summary["per_physical"]["mean"] if note is None and summary["per_physical"] else None
    return {"episodes": len(rows), "records": _stats([e.steps for e in episodes]),
            "flown_m": _stats([r["flown_m"] for r in rows]), "straight_m": _stats([r["straight_m"] for r in rows]),
            "per_straight": per_straight, "per_physical": summary["per_physical"] if note is None else None,
            "per_clearance": summary["per_clearance"] if note is None else None,
            "per_interval": [per_straight["mean"] if per_straight else None, upper],
            "episodes_without_l_opt": summary["episodes_without_l_opt"] if note is None else None, "note": note,
            "_records_each": [e.steps for e in episodes], "_flown_each": [r["flown_m"] for r in rows]}


def _per_stats(values) -> dict | None:
    """dataset_stats.summarise's statistics (mean, median, min, max, n), for values it would not summarise."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(float(np.mean(values)), 4), "median": round(float(np.median(values)), 4),
            "min": round(float(np.min(values)), 4), "max": round(float(np.max(values)), 4), "n": len(values)}


# ----------------------------------------------------------------------------------------------------------------------
# Drawing.
# ----------------------------------------------------------------------------------------------------------------------
def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else plural or word + 's'}"


ROW_IN = 0.26  # a bar row's height in inches: at 130 dpi a 0.62-row bar is 21 px, inside the dataviz cap of 24


def _hbars(ax, labels: Sequence[str], values: Sequence[int], colours, title: str) -> None:
    """Horizontal bars, the first label on top, each value written at its bar's end. A panel taller than its rows need
    keeps them at ROW_IN from the top rather than stretching the bars."""
    fig = ax.get_figure()
    rows_fit = int(ax.get_position().height * fig.get_figheight() / ROW_IN)
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.62, color=colours, edgecolor=SURFACE, linewidth=1.0)
    top = max(values) if len(values) and max(values) > 0 else 1
    for yi, v in zip(y, values):
        ax.text(v + top * 0.012, yi, f"{v:,}", va="center", ha="left", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, labels)
    ax.set_ylim(max(len(labels), rows_fit) - 0.5, -0.5)
    ax.set_xlim(0, top * 1.14)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_bounds(len(labels) - 0.5, -0.5)
    ax.set_title(title)


def draw_composition(path: Path, comp: dict) -> None:
    by_target = comp["by_target"]
    names = list(by_target)
    scenes = list(comp["by_scene"])
    categories = list(CATEGORIES) + ([UNKNOWN] if comp["by_category"].get(UNKNOWN) else [])
    notes = []
    filled = comp["filled_from_pool"]
    if filled:
        notes.append("Filled in from the pool file by the target's name, where an episode's provenance predates the "
                     "field: " + ", ".join(f"{k} for {_plural(v, 'episode')}" for k, v in filled.items()) + ".")
    if comp["stand_in"]["unmarked"]:
        n = comp["stand_in"]["unmarked"]
        notes.append(f"{_plural(n, 'episode')} carr{'ies' if n == 1 else 'y'} no stand-in mark: neither the provenance "
                     f"nor the pool file says.")
    never = comp["pool_targets_never_kept"]
    if never:
        notes.append(f"{_plural(len(never), 'drawable pool target')} ha{'s' if len(never) == 1 else 've'} no kept episode "
                     f"here: {', '.join(never)}.")
    note = _wrap(" ".join(notes), 200)
    # The left column's four panels at ROW_IN per bar and GAP_IN between them (a title and a tick row); the per-target
    # panel spans them all. Gridspec's hspace is a fraction of the mean panel height, so it is derived from the inches.
    gap_in = 0.8
    panels = np.array([max(len(scenes), 1), len(categories), 3, 3], dtype=float) * ROW_IN + 0.5
    body_in = max(panels.sum() + 3 * gap_in, (len(names) + 0.5) * ROW_IN + 0.5)
    panels *= (body_in - 3 * gap_in) / panels.sum()
    top_in, bottom_in = 0.85, 0.5 + (0.16 * (note.count("\n") + 1) if notes else 0.0)
    height = body_in + top_in + bottom_in
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14.5, height))
        grid = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.25], height_ratios=list(panels),
                                hspace=4 * gap_in / panels.sum(), wspace=0.42, left=0.14, right=0.98,
                                top=1 - top_in / height, bottom=bottom_in / height)
        _hbars(fig.add_subplot(grid[0, 0]), [scene_label(s) for s in scenes], [comp["by_scene"][s] for s in scenes],
               KEPT_COLOUR, "Kept episodes per scene")
        _hbars(fig.add_subplot(grid[1, 0]), categories, [comp["by_category"].get(c, 0) for c in categories],
               [CATEGORY_COLOURS[c] for c in categories], "per target category")
        _hbars(fig.add_subplot(grid[2, 0]), list(comp["seen"]), list(comp["seen"].values()), KEPT_COLOUR,
               "seen and unseen targets (training scenes draw seen ones)")
        _hbars(fig.add_subplot(grid[3, 0]), list(comp["stand_in"]), list(comp["stand_in"].values()), KEPT_COLOUR,
               "low-poly stand-ins (Kenney) and the rest")
        ax = fig.add_subplot(grid[:, 1])
        pos = ax.get_position()  # as tall as its rows need, from the top: a few targets are not stretched down the page
        needed = min(pos.height, ((len(names) + 0.5) * ROW_IN + 0.5) / height)
        ax.set_position([pos.x0, pos.y1 - needed, pos.width, needed])
        _hbars(ax, names, list(by_target.values()), [CATEGORY_COLOURS.get(comp["target_category"][n], MUTED)
                                                     for n in names], f"per target ({_plural(len(names), 'target')})")
        present = [c for c in categories if any(comp["target_category"][n] == c for n in names)]
        handles = [Patch(color=CATEGORY_COLOURS[c], label=c) for c in present]
        if needed < pos.height - 1.2 / height:  # room below the panel
            ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, -0.5 / (needed * height)),
                      ncol=len(present), title="target category", title_fontsize=8.5)
        else:  # a full column: the shortest bars are at its foot, so its lower right corner is free
            ax.legend(handles=handles, loc="lower right", title="target category", title_fontsize=8.5)
        fig.suptitle(f"What the stores hold: {_plural(comp['episodes'], 'kept episode')} "
                     f"({_plural(comp['records'], 'record')}) from {_plural(comp['stores'], 'store')}, "
                     f"{_plural(len(scenes), 'scene')}, {_plural(len(names), 'target')}", x=0.01, y=1 - 0.12 / height,
                     ha="left", va="top", fontsize=12.5)
        if notes:
            fig.text(0.01, 0.1 / height, note, fontsize=8, color=INK_2, va="bottom")
        fig.savefig(path, dpi=130)
        plt.close(fig)


def draw_kept_rates(path: Path, rates: dict) -> None:
    rows = [{"label": f"{r['store']}  ({', '.join(r['scenes'])})", "kind": "store", **r} for r in rates["stores"]]
    pooled = [s for s in rates["scenes"] if len(s["stores"]) > 1]
    rows += [{"label": f"{scene_label(s['scene'])}: {len(s['stores'])} stores", "kind": "scene", **s} for s in pooled]
    reasons = {k for r in rows for k in r["rejected"]}
    colours, order = reason_colours(reasons), _reason_order(reasons)
    n = len(rows)
    width, row_in, top_in, bottom_in = 14.5, 0.36, 1.0, 1.0
    height = top_in + bottom_in + (n + 0.6) * row_in
    left = (0.35 + 0.066 * max(len(r["label"]) for r in rows)) / width
    with plt.rc_context(STYLE):
        fig, (ax_n, ax_r) = plt.subplots(1, 2, figsize=(width, height), sharey=True, gridspec_kw={
            "width_ratios": [1.1, 1.0], "wspace": 0.06, "left": left, "right": 0.98, "top": 1 - top_in / height,
            "bottom": bottom_in / height})
        y = np.arange(n)
        most = max((r["flown"] for r in rows), default=1) or 1
        for yi, r in zip(y, rows):
            start = 0
            for count, colour in [(r["kept"], KEPT_COLOUR)] + [(r["rejected"].get(k, 0), colours[k]) for k in order]:
                if count:
                    ax_n.barh(yi, count, left=start, height=0.6, color=colour, edgecolor=SURFACE, linewidth=1.2)
                    start += count
            ax_n.text(start + most * 0.012, yi, f"{r['kept']:,} / {r['flown']:,}", va="center", fontsize=8.5,
                      color=INK_2)
            if r["kept_rate"] is not None and r["wilson_95"]:
                lo, hi = r["wilson_95"]
                ax_r.errorbar(r["kept_rate"], yi, xerr=[[r["kept_rate"] - lo], [hi - r["kept_rate"]]],
                              fmt="o" if r["kind"] == "store" else "D", ms=6, color=INK, ecolor=INK_2, elinewidth=1.4,
                              capsize=3)
                ax_r.text(1.03, yi, f"{r['kept_rate']:.2f}  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=8.5,
                          color=INK_2)
        ax_n.set_yticks(y, [r["label"] for r in rows])
        ax_n.set_ylim(n - 0.5, -1.1)  # a headroom row above the first, for the bar line's label
        ax_n.set_xlim(0, most * 1.22)
        ax_n.xaxis.grid(True)
        ax_n.set_axisbelow(True)
        ax_n.tick_params(axis="y", length=0)
        ax_n.set_xlabel("flights")
        ax_n.set_title("Flights per store: kept, and rejected by reason (kept / flown)")
        fig.legend(handles=[Patch(color=KEPT_COLOUR, label="kept")]
                   + [Patch(color=colours[k], label=k.replace("_", " ")) for k in order],
                   loc="lower left", bbox_to_anchor=(left, 0.12 / height), ncol=1 + len(order))
        ax_r.axvline(PLAN6_BAR, color=MUTED, linewidth=1.0)
        ax_r.text(PLAN6_BAR + 0.012, -0.8, "plan 6's bar for the interval's lower end: 0.25", fontsize=8,
                  color=INK_2, va="center")
        ax_r.set_xlim(0, 1.32)
        ax_r.set_xticks(np.linspace(0, 1, 6))
        ax_r.xaxis.grid(True)
        ax_r.set_axisbelow(True)
        ax_r.set_xlabel("kept-rate, with its Wilson 95 % interval")
        ax_r.set_title("Kept-rate: dots per store, diamonds per scene (its stores pooled)")
        ax_r.tick_params(axis="y", length=0)
        kept, flown = sum(r["kept"] for r in rates["stores"]), sum(r["flown"] for r in rates["stores"])
        fig.suptitle(f"Kept against flown: {_plural(kept, 'episode')} kept of {_plural(flown, 'flight')} in "
                     f"{_plural(len(rates['stores']), 'store')}", x=0.01, y=1 - 0.12 / height, ha="left", va="top",
                     fontsize=12.5)
        fig.savefig(path, dpi=130)
        plt.close(fig)


def _trajectory_points(provenance: dict) -> np.ndarray:
    poses = [p[:2] for p in provenance.get("poses_per_record") or []]
    if provenance.get("final_pose"):
        poses.append(provenance["final_pose"][:2])
    return np.asarray(poses, dtype=float).reshape(-1, 2)


def draw_trajectories(path: Path, scene: str, resolved: ResolvedScene | None, shapes: Sequence[Shape],
                      episodes: Sequence[Episode], rejects: Sequence[Reject], stores: Sequence[str]) -> dict:
    moved: set[str] = set()
    starts, targets, lines = [], [], []
    for e in episodes:
        prov = e.provenance()
        moved |= {m.get("tag") for m in prov.get("movers") or [] if isinstance(m, dict) and m.get("tag")}
        points = _trajectory_points(prov)
        if len(points):
            lines.append(points)
            starts.append(points[0])
        xyz = (prov.get("target") or {}).get("xyz")
        if xyz:
            targets.append(xyz[:2])
    endings = [(r.reason, r.final_pose[:2]) for r in rejects if r.final_pose]
    colours = reason_colours({reason for reason, _ in endings})
    by_reason = Counter(reason for reason, _ in endings)
    kinds = Counter(s.kind for s in shapes)
    drawn = {"outline": ("outline", "outlines"), "circle": ("circle", "circles"), "box": ("box (a stack seen from above)",
                                                                                       "boxes (stacks seen from above)")}
    obstacle_label = ("obstacles: " + ", ".join(f"{n} {drawn[k][n != 1]}" for k, n in sorted(kinds.items()))
                      if shapes else "obstacles: no layout found")
    handles = [Line2D([], [], color=KEPT_COLOUR, lw=1.6, label=f"kept trajectory ({len(lines)})"),
               Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK, mew=1.0, ms=5, label="start"),
               Line2D([], [], marker="*", ls="", color=INK, ms=9, label="target"),
               Patch(facecolor=OBSTACLE_FILL, edgecolor=OBSTACLE_EDGE, label=obstacle_label)]
    if moved:
        handles.append(Patch(facecolor=OBSTACLE_FILL, edgecolor=OBSTACLE_EDGE, hatch="////",
                             label=f"pillars that moved in a kept episode ({len(moved)}), at home"))
    handles += [Line2D([], [], marker="x", ls="", color=colours[r], ms=7, mew=1.6,
                       label=f"rejected flight's end: {r.replace('_', ' ')} ({by_reason[r]})")
                for r in _reason_order(set(by_reason))]
    if resolved is not None:
        handles += [Patch(facecolor=START_BAND_COLOUR, alpha=0.5, label="start band"),
                    Patch(facecolor=TARGET_BAND_COLOUR, alpha=0.6, label="target band")]
    legend_rows = math.ceil(len(handles) / 2)
    width, height = 8.2, 8.6 + 0.24 * legend_rows
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(width, height))
        if resolved is not None:
            b = resolved.layout.bounds
            bounds = {"x_min": b.x_min, "x_max": b.x_max, "y_min": b.y_min, "y_max": b.y_max}
            _band(ax, bounds, resolved.scene.target_band[0], resolved.scene.target_band[1], TARGET_BAND_COLOUR, 0.45)
            _band(ax, bounds, resolved.scene.start_band[0], resolved.scene.start_band[1], START_BAND_COLOUR, 0.35)
            ax.add_patch(plt.Rectangle((b.y_min, b.x_min), b.y_max - b.y_min, b.x_max - b.x_min, fill=False,
                                       edgecolor=INK_2, lw=1.2, zorder=1))
            pad = 2.0
            ax.set_xlim(b.y_min - pad, b.y_max + pad)
            ax.set_ylim(b.x_min - pad, b.x_max + pad)
        for shape in shapes:
            hatch = "////" if shape.tag in moved else None
            style = dict(facecolor=OBSTACLE_FILL, edgecolor=OBSTACLE_EDGE, lw=0.6, zorder=2, hatch=hatch)
            if shape.kind == "circle":
                ax.add_patch(Circle((shape.centre[1], shape.centre[0]), shape.radius_m, **style))
            else:
                for poly in shape.polygons:
                    ax.add_patch(Polygon(poly[:, ::-1], closed=True, **style))
        alpha = float(np.clip(3.0 / math.sqrt(max(len(lines), 1)), 0.08, 0.7))
        for points in lines:
            ax.plot(points[:, 1], points[:, 0], color=KEPT_COLOUR, lw=0.9, alpha=alpha, zorder=3, solid_capstyle="round")
        if starts:
            s = np.asarray(starts)
            ax.scatter(s[:, 1], s[:, 0], s=18, facecolors=SURFACE, edgecolors=INK, linewidths=0.9, zorder=4)
        if targets:
            t = np.asarray(targets)
            ax.scatter(t[:, 1], t[:, 0], s=64, marker="*", color=INK, edgecolor=SURFACE, linewidth=0.5, zorder=5)
        for reason in _reason_order(set(by_reason)):
            pts = np.asarray([p for r, p in endings if r == reason])
            ax.scatter(pts[:, 1], pts[:, 0], s=36, marker="x", color=colours[reason], linewidth=1.6, zorder=6)
        ax.set_aspect("equal")
        ax.set_xlabel("y (east, m)")
        ax.set_ylabel("x (north, m)")
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        legend_in = 0.75 + 0.24 * legend_rows
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.6 / 7.0), ncol=2, fontsize=8.5)
        title = f"{scene_label(scene)}: {_plural(len(lines), 'kept trajectory', 'kept trajectories')}"
        if endings:
            title += f", {_plural(len(endings), 'end')} of rejected flights"
        ax.set_title(f"{title}\n{_plural(len(stores), 'store')}: {_wrap(', '.join(stores), 80)}", fontsize=10.5)
        fig.subplots_adjust(left=0.1, right=0.98, top=1 - 0.75 / height, bottom=legend_in / height)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    out = {"kept": len(lines), "rejected_endings": len(endings),
           "rejected_endings_by_reason": dict(sorted(by_reason.items())), "obstacles": dict(sorted(Counter(
               s.kind for s in shapes).items())), "layout": str(resolved.layout_path) if resolved else None,
           "layout_scene": resolved.base_id if resolved else None}
    if resolved is not None and resolved.scene.dynamic is not None:
        out["moving_pillars"] = {"moved_in_any_episode": len(moved), "of": len(resolved.layout.instances)}
    return out


def _ridges(ax, values: Sequence[np.ndarray], labels: Sequence[str], show_labels: bool) -> None:
    filled = [v for v in values if v.size]
    lo, hi = (float(min(v.min() for v in filled)), float(max(v.max() for v in filled))) if filled else (0.0, 1.0)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, BINS + 1)
    n = len(values)
    for row, v in enumerate(values):
        base = n - 1 - row
        if not v.size:
            continue
        h, _ = np.histogram(v, bins=edges)
        h = np.append(h / h.max(), 0.0) * 0.86 + base
        ax.fill_between(edges, base, h, step="post", color=KEPT_COLOUR, alpha=0.28, lw=0)
        ax.step(edges, h, where="post", color=KEPT_COLOUR, lw=1.1)
        median = float(np.median(v))
        ax.plot([median, median], [base, base + 0.86], color=INK, lw=1.1)
    ax.set_yticks(np.arange(n), list(labels)[::-1] if show_labels else [])
    ax.set_ylim(-0.12, n)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


def draw_actions_states(path: Path, per_scene: Mapping[str, np.ndarray], dims: Mapping[str, dict]) -> None:
    scenes = list(per_scene)
    keys = list(dims)
    rows_per_panel = max(len(scenes), 1)
    panel_h = 0.9 + 0.36 * rows_per_panel
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 4, figsize=(16.5, 3 * panel_h + 1.3), squeeze=False)
        for k, (ax, key) in enumerate(zip(axes.flat, keys)):
            d = dims[key]
            marks = (" · privileged" if d["privileged"] else "") + (" †" if not d["autofly_verified"] else "")
            _ridges(ax, [per_scene[s][:, k] for s in scenes], scenes, show_labels=k % 4 == 0)
            ax.set_title(f"{key}  {d['title']}{marks}", fontsize=9.5)
        fig.suptitle(f"Every record's action[3] and state[9], per scene ({sum(len(v) for v in per_scene.values()):,} "
                     f"records)", x=0.01, ha="left", fontsize=12.5)
        fig.text(0.01, 0.008, "Each row is one scene's records, as a histogram scaled to its own peak; the black tick is "
                 "its median. privileged: computed from the target's true position (plan 6 revision 1, item 4). "
                 "†: a definition not verified against AutoFly's release (dataset/state.py).", fontsize=8,
                 color=INK_2, va="bottom")
        fig.subplots_adjust(left=0.05, right=0.99, top=1 - 0.65 / (3 * panel_h + 1.3), bottom=0.75 / (3 * panel_h + 1.3),
                            hspace=0.55, wspace=0.12)
        fig.savefig(path, dpi=130)
        plt.close(fig)


def draw_lengths_per(path: Path, per: Mapping[str, dict]) -> None:
    scenes = list(per)
    n = len(scenes)
    y = np.arange(n)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 2.2 + 0.55 * n), sharey=True,
                                 gridspec_kw={"wspace": 0.08, "width_ratios": [1, 1, 1.15]})
        for ax, key, title, unit in ((axes[0], "_records_each", "Records per episode (5 Hz)", "records"),
                                     (axes[1], "_flown_each", "Flown path length (3-D)", "m")):
            ax.boxplot([per[s][key] for s in scenes], positions=y, vert=False, widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor="#cde2fb", edgecolor=KEPT_COLOUR), medianprops=dict(color=INK, lw=1.5),
                       whiskerprops=dict(color=KEPT_COLOUR), capprops=dict(color=KEPT_COLOUR),
                       flierprops=dict(marker=".", markersize=3, markerfacecolor=MUTED, markeredgecolor=MUTED))
            for yi, s in zip(y, scenes):
                stats = per[s]["records" if key == "_records_each" else "flown_m"]
                ax.text(1.0, yi - 0.36, f"median {stats['median']:.0f} {unit}", transform=ax.get_yaxis_transform(),
                        ha="right", fontsize=8, color=INK_2)
            ax.set_title(title)
            ax.set_xlabel(unit)
            ax.xaxis.grid(True)
            ax.set_axisbelow(True)
        ax = axes[2]
        ax.axvspan(*PAPER_PER, color=GRID, zorder=0)
        lows = []
        for yi, s in zip(y, scenes):
            lo, hi = per[s]["per_interval"]
            if lo is None:
                continue
            lows.append(lo)
            if hi is not None:  # the label sits above the interval and ends where it ends: inside the axes up to 1.0
                ax.plot([lo, hi], [yi, yi], color=KEPT_COLOUR, lw=3.5, solid_capstyle="round", zorder=3)
                ax.plot([lo, hi], [yi, yi], ls="", marker="o", ms=5, color=KEPT_COLOUR, zorder=4)
                ax.text(hi, yi - 0.2, f"{lo:.3f} to {hi:.3f}", ha="right", va="bottom", fontsize=8.5, color=INK_2)
            else:
                ax.plot([lo], [yi], ls="", marker=">", ms=7, color=KEPT_COLOUR, zorder=4)
                ax.text(lo, yi - 0.2, f"at least {lo:.3f} ({per[s]['note'].split(':')[0]}: no upper bound)", ha="right",
                        va="bottom", fontsize=8.5, color=INK_2)
        ax.set_xlim(min([PAPER_PER[0] - 0.03] + [v - 0.03 for v in lows]), 1.01)
        ax.set_xlabel(f"PER = L_opt / max(L, L_opt), mean over episodes\ngrey band: the paper's models, "
                      f"{PAPER_PER[0]:.2f} to {PAPER_PER[1]:.2f}")
        ax.set_title("Path efficiency: straight-line bound to grid-optimum bound")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        axes[0].set_yticks(y, [scene_label(s) for s in scenes])
        axes[0].set_ylim(n - 0.5, -0.75)
        for a in axes:
            a.tick_params(axis="y", length=0)
        fig.suptitle("How long the kept episodes are, and how directly they fly", x=0.01, ha="left", fontsize=12.5)
        fig.subplots_adjust(left=0.11, right=0.99, top=1 - 0.7 / (2.2 + 0.55 * n), bottom=0.75 / (2.2 + 0.55 * n))
        fig.savefig(path, dpi=130)
        plt.close(fig)


def draw_sheet(path: Path, tiles: Sequence[tuple[np.ndarray, str, str]], *, cols: int, title: str,
               tile_in: float = 2.25, dpi: int = 120) -> None:
    """A contact sheet: each tile an image with a caption (bold first line) and a muted second line below it."""
    rows = math.ceil(len(tiles) / cols)
    width = max(cols * tile_in + 0.3, 6.0)  # room for the title beside a sheet of one or two tiles
    title = _wrap(title, int(width * 9.5))  # about 9.5 characters an inch at 12 pt
    lines = max(_wrap(caption, 30).count("\n") + detail.count("\n") + 2 for _, caption, detail in tiles)
    caption_in, title_in = 0.32 + 0.15 * lines, 0.35 + 0.22 * (title.count("\n") + 1)
    height = rows * (tile_in + caption_in) + title_in
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width, height))
        for k, (image, caption, detail) in enumerate(tiles):
            r, c = divmod(k, cols)
            left = (0.15 + c * tile_in) / width
            bottom = 1 - (title_in + r * (tile_in + caption_in) + tile_in) / height
            ax = fig.add_axes([left, bottom, (tile_in - 0.08) / width, (tile_in - 0.08) / height])
            ax.imshow(image, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(0.5, -0.03, _wrap(caption, 30), transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                    color=INK, fontweight="bold")
            ax.text(0.5, -0.03 - 0.075 * (1 + _wrap(caption, 30).count("\n")), detail, transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.5, color=INK_2)
        fig.text(0.01, 1 - 0.12 / height, title, ha="left", va="top", fontsize=12)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# The whole set.
# ----------------------------------------------------------------------------------------------------------------------
def build(stores: Sequence[Store], out_dir: Path, *, levels: Path, fields_root: Path = ASSET_FOOTPRINTS,
          pool: Pool | None = None, first_frames: int = FIRST_FRAMES) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    notes: list[str] = []
    episodes = [e for s in stores for e in s.episodes]
    by_scene: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        by_scene[e.scene].append(e)
    rejects_by_scene: dict[str, list[Reject]] = defaultdict(list)
    for s in stores:
        for r in s.rejects or ():
            scene = r.scene or (s.scenes[0] if len(s.scenes) == 1 else None)
            if scene is not None:
                rejects_by_scene[scene].append(r)
    scenes = sorted(set(by_scene) | set(rejects_by_scene))
    resolved: dict[str, ResolvedScene | None] = {}
    for scene in scenes:
        try:
            resolved[scene] = resolve_scene(scene, levels_dir=levels)
        except (FileNotFoundError, ValueError) as err:
            resolved[scene] = None
            notes.append(f"{scene}: no layout ({err})")

    comp = composition(stores, pool)
    draw_composition(out_dir / "composition.png", comp)
    files.append("composition.png")
    rates = kept_rates(stores)
    draw_kept_rates(out_dir / "kept_rates.png", rates)
    files.append("kept_rates.png")

    registry = load_registry()
    trajectories = {}
    for scene in scenes:
        shapes: list[Shape] = []
        if resolved[scene] is not None:
            shapes, shape_notes = obstacle_shapes(resolved[scene].layout, registry, fields_root)
            notes += [f"{scene}: {n}" for n in shape_notes]
        names = sorted({e.store for e in by_scene[scene]} | {r.store for r in rejects_by_scene[scene]})
        trajectories[scene] = draw_trajectories(out_dir / f"trajectories_{scene}.png", scene, resolved[scene], shapes,
                                                by_scene[scene], rejects_by_scene[scene], names)
        files.append(f"trajectories_{scene}.png")

    dims = dimensions()
    kept_scenes = [s for s in scenes if by_scene[s]]
    per_scene_values: dict[str, np.ndarray] = {}
    last_state: dict[tuple[str, str], np.ndarray] = {}
    for scene in kept_scenes:
        blocks = []
        for e in by_scene[scene]:
            rec = e.records()
            blocks.append(np.concatenate([rec["action"], rec["state"]], axis=1).astype(np.float64))
            last_state[(e.store, e.id)] = rec["state"][-1]
        per_scene_values[scene] = np.concatenate(blocks)
    if kept_scenes:
        draw_actions_states(out_dir / "actions_states.png", per_scene_values, dims)
        files.append("actions_states.png")
    actions_states = {"dimensions": dims, "records": {s: int(len(v)) for s, v in per_scene_values.items()},
                      "scenes": {s: {key: _stats(v[:, k]) for k, key in enumerate(dims)}
                                 for s, v in per_scene_values.items()}}

    per = {scene: lengths_and_per(scene, by_scene[scene], resolved[scene]) for scene in kept_scenes}
    if per:
        draw_lengths_per(out_dir / "lengths_per.png", per)
        files.append("lengths_per.png")
    lengths = {"scenes": {s: {k: v for k, v in p.items() if not k.startswith("_")} for s, p in per.items()},
               "paper_per": list(PAPER_PER),
               "bounds": "lower: the straight line to the success disc; upper: the 8-connected grid's shortest path at "
                         "the rotor half-span's inflation (scripts/dataset_stats.py per_physical)"}

    sheets = {}
    for scene in kept_scenes:
        picked = pick_first_frames(by_scene[scene], first_frames)
        tiles = [(read_rgb(e.frame(0)), e.target.name, f"{e.target.category} · {e.id}"
                  + (f"\nfirst frame: {e.first_frame.replace('_', ' ')}" if e.first_frame else "")) for e in picked]
        checked = Counter(e.first_frame or "not recorded" for e in by_scene[scene])
        title = (f"{scene_label(scene)}: the first recorded frame of {len(picked)} of {len(by_scene[scene])} kept "
                 f"episodes, captioned with the target the instruction names")
        if set(checked) != {"not recorded"}:
            title += " and whether the collector's check saw it in that frame"
        draw_sheet(out_dir / f"first_frames_{scene}.png", tiles, cols=min(6, len(tiles)), title=title)
        files.append(f"first_frames_{scene}.png")
        sheets[scene] = {"episodes": [e.id for e in picked], "targets": [e.target.name for e in picked],
                         "kept": len(by_scene[scene]),
                         "first_frame_visibility": None if set(checked) == {"not recorded"} else dict(checked)}

    closeups = pick_closeups(episodes, last_state)
    if closeups:
        tiles = []
        for name, e in closeups.items():
            mark = " · stand-in" if e.target.stand_in else ""
            tiles.append((read_rgb(e.frame(-1)), name, f"{e.target.category}{mark} · {e.scene}"))
        n = len(tiles)
        draw_sheet(out_dir / "targets_closeup.png", tiles, cols=min(10 if n > 24 else 6, n),
                   title=f"{_plural(n, 'target')} as the drone reaches {'it' if n == 1 else 'them'}: the last recorded "
                         f"frame of one episode per target (the one ending with its target nearest dead ahead)")
        files.append("targets_closeup.png")
    closeup_record = {"count": len(closeups), "targets": {
        name: {"episode": e.id, "store": e.store, "scene": e.scene, "category": e.target.category,
               "stand_in": e.target.stand_in, "bearing_rad": _r(last_state[(e.store, e.id)][1]),
               "distance_m": _r(last_state[(e.store, e.id)][0])} for name, e in closeups.items()}}

    summary = {
        "tool": "scripts/dataset_figures.py", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"stores": [{"name": s.name, "path": str(s.root), "rejects_dir": str(s.rejects_dir) if s.rejects_dir
                               else None, "record": str(s.record_path) if s.record_path else None} for s in stores],
                   "levels": str(levels), "fields": str(fields_root),
                   "targets": {"path": pool.path, "sha256": pool.sha256} if pool is not None else None},
        "scenes": {s: scene_label(s) for s in scenes},
        "composition": comp, "kept_rates": rates, "trajectories": trajectories, "actions_states": actions_states,
        "lengths_per": lengths, "first_frames": sheets, "targets_closeup": closeup_record, "notes": notes,
        "files": files,
    }
    (out_dir / "dataset_figures.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _record_arg(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected <store name>=<record path>, got {text!r}")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stores", type=Path, nargs="+", required=True, help="raw stores (data/<name>)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--levels", type=Path, default=RUNS_DIR / "levels", help="directory of <scene>.layout.json")
    p.add_argument("--targets", type=Path, default=ROOT / "assets" / "targets.json",
                   help="the target pool file, to complete a target its provenance predates (by name)")
    p.add_argument("--fields", type=Path, default=ASSET_FOOTPRINTS, help="the mesh obstacles' outline fields")
    p.add_argument("--record", type=_record_arg, action="append", default=[], metavar="STORE=PATH",
                   help="a store's collector record (scripts/collect_dataset.py --out)")
    p.add_argument("--records-dir", type=Path, default=RUNS_DIR / "collect",
                   help="where <store>.json records are looked for when no --record names one")
    p.add_argument("--no-rejects", action="store_true", help="ignore data/rejects/<store>/ (the manifest's count stays)")
    p.add_argument("--first-frames", type=int, default=FIRST_FRAMES, help="episodes per first-frame sheet")
    args = p.parse_args(argv)
    if args.first_frames < 1:
        p.error("--first-frames must be at least 1")
    records = dict(args.record)
    for root in args.stores:
        manifest = root / "manifest.json"
        name = (json.loads(manifest.read_text()).get("name") if manifest.is_file() else None) or root.name
        candidate = args.records_dir / f"{name}.json"
        if name not in records and candidate.is_file():
            records[name] = candidate
    pool = load_pool(args.targets) if args.targets is not None and args.targets.is_file() else None
    stores = load_stores(args.stores, pool=pool, read_rejects_dirs=not args.no_rejects, records=records)
    summary = build(stores, args.out_dir, levels=args.levels, fields_root=args.fields, pool=pool,
                    first_frames=args.first_frames)
    print(json.dumps({"out_dir": str(args.out_dir), "files": summary["files"], "notes": summary["notes"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
