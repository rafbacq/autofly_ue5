"""The closed-loop evaluation in one figure, the numbers it draws and a table for a README (plan 6 step 7), from any list
of evaluation harness records (autofly_ue5/evaluation, scripts/eval_harness.py run --out).

    env -u PYTHONPATH .venv/bin/python scripts/harness_figures.py --records runs/eval/final/*.json \\
        --out-dir runs/eval/final/figures [--pilots docs/gates/plan6_pilots.json | --no-done-check] \\
        [--drop <record>:<scene> ...]

Writes into --out-dir, and nowhere else:
- harness_results.png: per condition (seen, test_seen, unseen_targets) and scene, and over each condition's scenes, the
  success rate with its Wilson 95 % interval for every policy, the collision and out-of-bounds rates with theirs, and
  PER over the successes as its interval (the straight-line bound to the grid-optimum bound; evaluation/metrics.py). On
  the seen scenes the pilot's kept-rate interval (docs/gates/plan6_pilots.json) lies behind the expert's dot.
- harness_results.json: every number drawn, where each came from (records, their sha256 and status), and the done check.
- harness_results.md: the same as tables, ready to paste into a README.

A policy is what its record's description says (name, kind, checkpoint, and the rest of describe()) together with
whether the run handed it the privileged state; a deterministic expert is another policy than a stochastic one. The SAC
expert is the exception: every scene has its own pilot (plan 6), so its records, written per scene group with one
checkpoint each (seen_expert_s01.json, seen_expert_s01d_s07_s09.json, ...), merge into one policy per condition, and
each scene keeps the checkpoint that flew it. The rates are computed afresh over the merged episodes by the harness's own
`metrics.summarise`, so a scene or a condition's whole reads exactly as a single record of it would.

Refused before anything is written: a file that is not a harness record; records of one condition from different
condition files (other episodes) or flown under other rules (a0, step limit, contact rule shape); one policy's scene
flown twice over the same seeds (it would count twice: --drop <record>:<scene> says which copy counts), or by two
checkpoints; a --drop that matches nothing; an output that is a committed record under docs/gates/.

Plan 6's done check ("the expert flown through the harness lands within the CI of its kept-rate and the scripted
baseline below it"): on the seen condition, per scene, whether the expert's success interval overlaps the pilot's
kept-rate interval from the thin slice, whether the expert flew the pilot's checkpoint, and whether the scripted
baseline's success rate lies below the expert's (and "separated" where its whole interval does). A record still running
is drawn as far as it has flown and marked incomplete.
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
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from autofly_ue5 import evidence  # noqa: E402
from autofly_ue5.evaluation.metrics import RATE_KEYS, summarise  # noqa: E402
from autofly_ue5.paths import ROOT  # noqa: E402
from autofly_ue5.report.style import (BAND, GRID, INK, INK_2, MUTED, PAPER_PER, SLOTS, STYLE, SURFACE,  # noqa: E402
                                      scene_label)

HARNESS = "autofly_ue5.evaluation"
RECORD_VERSIONS = (1,)
DEFAULT_PILOTS = ROOT / "docs" / "gates" / "plan6_pilots.json"
OUTPUTS = ("harness_results.png", "harness_results.json", "harness_results.md")
CONDITION_ORDER = ("seen", "test_seen", "unseen_targets")
CONDITION_TITLES = {
    "seen": "seen: the nine training scenes, seen targets",
    "test_seen": "test_seen: the training scenes' elements in new layouts",
    "unseen_targets": "unseen targets: the training scenes, targets from the pool's unseen entries",
}
DONE_CHECK_CONDITION = "seen"  # the pilots' kept-rates were measured on the training scenes with seen targets
EXPERT_KIND = "sac_expert"
# The fields of an expert's description that are its scene's own: each scene flies its own pilot (plan 6).
PER_SCENE_FIELDS = ("checkpoint", "sha256", "obs_config", "trained_footprint_shape")
ROLES = {EXPERT_KIND: 0, "privileged_straight_line": 1, "floor": 2}  # the reference policies' places and colour slots
OTHER_ROLE = 3
DESCRIPTIONS = {
    0: "each scene's collection pilot: SAC on privileged depth and the target's position",
    1: "flies straight at the target's true bearing and avoids nothing",
    2: "commands nothing: the floor",
}
RATE_PANELS = (("success_rate", "Success rate (SR)"), ("collision_rate", "Collision rate (CR)"),
               ("out_of_bounds_rate", "Out of bounds"))
# A step up from the report's text sizes: this figure is the evaluation's headline and is read at README width.
LARGER = {"xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.titlesize": 11.5, "axes.labelsize": 9.5,
          "legend.fontsize": 9.5}


class HarnessFigureError(ValueError):
    """Records that cannot be drawn as one evaluation as they were given."""


def _r(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Reading records.
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Record:
    path: str
    sha256: str
    data: dict


def load_record(path: Path) -> Record:
    blob = Path(path).read_bytes()
    try:
        data = json.loads(blob)
    except ValueError as err:
        raise HarnessFigureError(f"{path}: not JSON ({err})") from None
    if not isinstance(data, dict) or data.get("harness") != HARNESS:
        raise HarnessFigureError(f"{path} is not a harness record (autofly_ue5/evaluation writes \"harness\": "
                                 f"\"{HARNESS}\")")
    if data.get("record_version") not in RECORD_VERSIONS:
        raise HarnessFigureError(f"{path}: record_version {data.get('record_version')!r}, this tool reads "
                                 f"{list(RECORD_VERSIONS)}")
    for key, kind in (("condition", dict), ("policy", dict), ("scenes", dict), ("episodes", list)):
        if not isinstance(data.get(key), kind):
            raise HarnessFigureError(f"{path}: a harness record without its {key!r}")
    return Record(str(path), _sha256(blob), data)


@dataclass(frozen=True)
class PolicyId:
    key: str  # the identity as canonical JSON: two records are one policy exactly when theirs are equal
    base_label: str
    name: str
    kind: str | None
    deterministic: bool | None
    privileged_state: bool
    per_scene: bool  # the checkpoint is each scene's own (the expert's pilots)
    role: int
    description: dict  # the record's description, less a per-scene policy's per-scene fields


def policy_id(record: Record) -> PolicyId:
    desc = record.data["policy"]
    name = str(desc.get("name") or "unnamed")
    kind = desc.get("kind")
    deterministic = desc.get("deterministic") if isinstance(desc.get("deterministic"), bool) else None
    privileged = bool(record.data.get("privileged_state"))
    per_scene = kind == EXPERT_KIND
    shared = {k: v for k, v in desc.items() if not (per_scene and k in PER_SCENE_FIELDS)}
    key = json.dumps({"policy": shared, "privileged_state": privileged}, sort_keys=True)
    notes = (["deterministic"] if deterministic else []) + (["privileged state"] if privileged else [])
    role = OTHER_ROLE if (per_scene and deterministic) else ROLES.get(kind, OTHER_ROLE)
    return PolicyId(key, name + (f" ({', '.join(notes)})" if notes else ""), name, kind, deterministic, privileged,
                    per_scene, role, shared)


def _labels(ids: list[PolicyId]) -> dict[str, str]:
    """{identity key: label}: the base label, and where several policies share one, a short id after it (the
    checkpoint's sha256 where the description has one)."""
    by_base: dict[str, list[PolicyId]] = {}
    for pid in {p.key: p for p in ids}.values():
        by_base.setdefault(pid.base_label, []).append(pid)
    labels = {}
    for base, group in by_base.items():
        for pid in group:
            if len(group) == 1:
                labels[pid.key] = base
            else:
                tag = str(pid.description.get("sha256") or "") or _sha256(pid.key.encode())
                labels[pid.key] = f"{base} #{tag[:8]}"
    return labels


# ----------------------------------------------------------------------------------------------------------------------
# Merging the records into conditions, policies and scenes.
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Part:
    """One record's share of a policy's scene: its scene entry and the episodes it flew."""

    record: Record
    entry: dict
    episodes: tuple


def _parse_drop(text: str) -> tuple[str, str]:
    path, sep, scene = str(text).rpartition(":")
    if not sep or not path or not scene:
        raise HarnessFigureError(f"--drop {text!r}: expected <record>:<scene>")
    return path, scene


def _metrics(episodes) -> dict:
    """The harness's own summary (evaluation/metrics.summarise) of `episodes`, rounded for the record; every value None
    when there are none."""
    if not episodes:
        return {"outcomes": None, **{key: None for key in RATE_KEYS.values()}, "per": None, "collision_sources": None,
                "mean_steps": None}
    s = summarise(list(episodes))
    out = {"outcomes": s["outcomes"]}
    for key in RATE_KEYS.values():
        out[key] = {"value": _r(s[key]["value"]), "wilson95": [_r(v) for v in s[key]["wilson95"]]}
    out["per"] = {k: _r(v) if isinstance(v, float) else v for k, v in s["per"].items()}
    out["collision_sources"] = s["collision_sources"]
    out["mean_steps"] = _r(s["mean_steps"], 2)
    return out


def _scene_order(listed: list[str], flown: set[str]) -> list[str]:
    """The condition file's order, then any scene it does not list."""
    return list(dict.fromkeys([s for s in listed if s in flown] + sorted(flown - set(listed))))


def collect(records: list[Record], drops=()) -> dict:
    """Every record's episodes merged per condition, policy and scene, checked as the module docstring says. Returns
    {"conditions", "policies", "dropped", "notes"} with the per-scene parts still attached (`_parts`)."""
    drop_keys = [_parse_drop(d) for d in drops]
    matched = {key: [] for key in drop_keys}
    ids = {id(r): policy_id(r) for r in records}
    labels = _labels(list(ids.values()))
    notes: list[str] = []

    by_condition: dict[str, list[Record]] = {}
    for r in records:
        by_condition.setdefault(str(r.data["condition"].get("name")), []).append(r)
    ordered = [c for c in CONDITION_ORDER if c in by_condition] + sorted(set(by_condition) - set(CONDITION_ORDER))

    conditions: dict[str, dict] = {}
    policies: dict[str, PolicyId] = {}
    dropped: list[dict] = []
    for cname in ordered:
        group = by_condition[cname]
        files = {(r.data["condition"].get("sha256"), r.data["condition"].get("file")) for r in group}
        if len({sha for sha, _ in files}) > 1:
            raise HarnessFigureError(
                f"the records of condition {cname} come from {len(files)} condition files "
                f"({', '.join(f'{f} sha256 {str(sha)[:12]}' for sha, f in sorted(files, key=str))}): their seeds are "
                f"other episodes, so they are not one evaluation; draw each set on its own")
        for field in ("a0", "max_steps", "contact_rule_shape"):
            values = sorted({json.dumps(r.data.get(field)) for r in group})
            if len(values) > 1:
                raise HarnessFigureError(f"the records of condition {cname} were flown under different {field}: "
                                         f"{', '.join(values)}; draw each set on its own")
        first = group[0].data
        listed = [str(s) for s in first["condition"].get("scenes") or []]
        per_scene_n = first["condition"].get("episodes_per_scene")
        parts: dict[str, dict[str, list[Part]]] = {}
        for r in group:
            pid = ids[id(r)]
            label = labels[pid.key]
            policies[label] = pid
            by_scene: dict[str, list[dict]] = {}
            for e in r.data["episodes"]:
                by_scene.setdefault(str(e.get("scene")), []).append(e)
            for scene in dict.fromkeys(list(r.data["scenes"]) + list(by_scene)):
                key = next((k for k in drop_keys if k[1] == scene and Path(k[0]).resolve() == Path(r.path).resolve()),
                           None)
                if key is not None:
                    matched[key].append(r.path)
                    dropped.append({"record": r.path, "scene": scene, "episodes": len(by_scene.get(scene, []))})
                    continue
                entry = r.data["scenes"].get(scene) or {"status": "ok", "reason": None, "error": None}
                parts.setdefault(label, {}).setdefault(scene, []).append(Part(r, entry, tuple(by_scene.get(scene, []))))
        for label, scenes in parts.items():
            for scene, shares in scenes.items():
                flown = [p for p in shares if p.episodes]
                for i, a in enumerate(flown):
                    for b in flown[i + 1:]:
                        both = {e.get("seed") for e in a.episodes} & {e.get("seed") for e in b.episodes}
                        if both:
                            raise HarnessFigureError(
                                f"policy {label} flew scene {scene} of condition {cname} in {a.record.path} and in "
                                f"{b.record.path} over the same seeds ({sorted(both)[:3]}...): its episodes would count "
                                f"twice; pass --drop {a.record.path}:{scene} or --drop {b.record.path}:{scene} to say "
                                f"which one counts")
                if policies[label].per_scene:
                    shas = {p.record.path: str(p.record.data["policy"].get("sha256")) for p in flown}
                    if len(set(shas.values())) > 1:
                        flew = "; ".join(f"{path}: {sha[:12]}" for path, sha in shas.items())
                        raise HarnessFigureError(
                            f"policy {label} flew scene {scene} of condition {cname} with two checkpoints ({flew}): a "
                            f"scene has one pilot; pass --drop <record>:{scene} for the one that does not count")
        seen_scenes = {s for scenes in parts.values() for s in scenes}
        conditions[cname] = {
            "file": first["condition"].get("file"), "sha256": first["condition"].get("sha256"),
            "episodes_per_scene": per_scene_n, "scenes": _scene_order(listed, seen_scenes),
            "listed_scenes": listed, "a0": first.get("a0"), "max_steps": first.get("max_steps"),
            "contact_rule_shape": first.get("contact_rule_shape"),
            "target_pools": sorted({str((r.data.get("target_pool") or {}).get("sha256")) for r in group}),
            "_parts": parts,
        }
        if len(conditions[cname]["target_pools"]) > 1:
            notes.append(f"{cname}: the records drew targets from {len(conditions[cname]['target_pools'])} pool files "
                         f"(sha256 {', '.join(p[:12] for p in conditions[cname]['target_pools'])}); only runs on one "
                         f"pool compare target for target")
        never = [s for s in listed if s not in seen_scenes]
        if never:
            notes.append(f"{cname}: no record flew {', '.join(never)}")
    unmatched = [f"{path}:{scene}" for (path, scene), hits in matched.items() if not hits]
    if unmatched:
        raise HarnessFigureError(f"--drop {', '.join(unmatched)} matches no record's scene (a typo would drop nothing "
                                 f"and keep a duplicate)")
    for r in records:
        if r.data.get("status") != "ok":
            notes.append(f"{r.path}: status {r.data.get('status')}"
                         + (f" ({r.data.get('error')})" if r.data.get("error") else "")
                         + ": drawn as far as it has flown")
    return {"conditions": conditions, "policies": policies, "dropped": dropped, "notes": notes}


def _scene_summary(shares: list[Part], condition_episodes) -> dict:
    flown = [p for p in shares if p.episodes]
    used = flown or shares
    statuses = [str(p.entry.get("status") or "ok") for p in used]
    status = "ok" if all(s == "ok" for s in statuses) else next(s for s in statuses if s != "ok")
    reason = next((p.entry.get("reason") or p.entry.get("error") for p in used
                   if p.entry.get("reason") or p.entry.get("error")), None)
    episodes = [e for p in flown for e in p.episodes]
    desc = used[0].record.data["policy"]
    complete = bool(episodes) and status == "ok" and (condition_episodes is None or len(episodes) >= condition_episodes)
    return {"status": status, "reason": reason, "complete": complete, "episodes": len(episodes),
            "condition_episodes": condition_episodes, "records": [p.record.path for p in used],
            "checkpoint": desc.get("checkpoint"), "sha256": desc.get("sha256"), **_metrics(episodes)}


def _success(entry: dict | None) -> dict | None:
    if not entry or not entry.get("episodes"):
        return None
    return {"value": entry["success_rate"]["value"], "wilson95": entry["success_rate"]["wilson95"],
            "episodes": entry["episodes"]}


def _reference_labels(policies: dict) -> tuple[str | None, str | None]:
    """The stochastic expert's label and the scripted baseline's, where the condition has them."""
    expert = next((label for label, p in policies.items() if p["role"] == ROLES[EXPERT_KIND]), None)
    scripted = next((label for label, p in policies.items() if p["role"] == ROLES["privileged_straight_line"]), None)
    return expert, scripted


def _versus(expert: dict | None, scripted: dict | None) -> dict:
    if expert is None or scripted is None:
        return {"below": None, "separated": None}
    return {"below": scripted["value"] < expert["value"], "separated": scripted["wilson95"][1] < expert["wilson95"][0]}


def summarise_records(collected: dict) -> dict:
    """The JSON-ready numbers: per condition, policy and scene, and each condition's whole per policy."""
    pids: dict[str, PolicyId] = collected["policies"]
    order = sorted(pids, key=lambda label: (pids[label].role, label))
    colours, spare, taken = {}, iter(SLOTS[OTHER_ROLE:]), set()
    for label in order:  # a reference policy's slot is its role's, so it keeps its colour whoever else is drawn
        role = pids[label].role
        colours[label] = SLOTS[role] if role < OTHER_ROLE and role not in taken else next(spare, MUTED)
        taken.add(role)
    policies = {label: {"name": pids[label].name, "kind": pids[label].kind, "deterministic": pids[label].deterministic,
                        "privileged_state": pids[label].privileged_state, "per_scene_checkpoint": pids[label].per_scene,
                        "role": pids[label].role, "colour": colours[label],
                        "description": DESCRIPTIONS.get(pids[label].role, pids[label].kind or pids[label].name),
                        "identity": pids[label].description} for label in order}
    conditions = {}
    for cname, c in collected["conditions"].items():
        parts = c["_parts"]
        out_policies = {}
        for label in [label for label in order if label in parts]:
            scenes = {s: _scene_summary(parts[label][s], c["episodes_per_scene"]) for s in c["scenes"]
                      if s in parts[label]}
            episodes = [e for s in c["scenes"] if s in parts[label] for p in parts[label][s] for e in p.episodes]
            flown = [s for s, entry in scenes.items() if entry["episodes"]]
            out_policies[label] = {"scenes": scenes, "all": {"scenes": flown, "episodes": len(episodes),
                                                             **_metrics(episodes)}}
        expert, scripted = _reference_labels({label: policies[label] for label in out_policies})
        versus = {}
        if expert is not None and scripted is not None:
            for s in c["scenes"]:
                e = _success(out_policies[expert]["scenes"].get(s))
                k = _success(out_policies[scripted]["scenes"].get(s))
                if e is not None and k is not None:
                    versus[s] = {"expert": expert, "scripted": scripted, "expert_success": e, "scripted_success": k,
                                 **_versus(e, k)}
        conditions[cname] = {**{k: v for k, v in c.items() if k != "_parts"}, "policies": out_policies,
                             "scripted_vs_expert": versus}
        coverage = {label: len(p["all"]["scenes"]) for label, p in out_policies.items()}
        if len(set(coverage.values())) > 1:
            collected["notes"].append(f"{cname}: each policy's whole is over the scenes it flew "
                                      f"({', '.join(f'{label} {n}' for label, n in coverage.items())})")
    return {"policies": policies, "conditions": conditions}


# ----------------------------------------------------------------------------------------------------------------------
# Plan 6's done check.
# ----------------------------------------------------------------------------------------------------------------------
def load_pilots(path: Path) -> dict:
    blob = Path(path).read_bytes()
    data = json.loads(blob)
    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(scenes, dict):
        raise HarnessFigureError(f"{path}: not a pilots record (no 'scenes'; docs/gates/plan6_pilots.json)")
    for scene, entry in scenes.items():
        interval = entry.get("wilson_95") if isinstance(entry, dict) else None
        if not (isinstance(interval, list) and len(interval) == 2 and isinstance(entry.get("kept_rate"), (int, float))):
            raise HarnessFigureError(f"{path}: scene {scene} has no kept_rate and wilson_95 interval")
    return {"path": str(path), "sha256": _sha256(blob), "scenes": scenes}


def done_check(numbers: dict, pilots: dict) -> dict:
    condition = numbers["conditions"].get(DONE_CHECK_CONDITION)
    policies = condition["policies"] if condition else {}
    expert, scripted = _reference_labels({label: numbers["policies"][label] for label in policies})
    scenes, without = {}, []
    flown_scenes = []
    if expert is not None:
        for s, entry in policies[expert]["scenes"].items():
            e = _success(entry)
            if e is None:
                continue
            flown_scenes.append(s)
            pilot = pilots["scenes"].get(s)
            if pilot is None:
                without.append(s)
                continue
            lo_p, hi_p = (float(v) for v in pilot["wilson_95"])
            lo_e, hi_e = e["wilson95"]
            k = _success(policies[scripted]["scenes"].get(s)) if scripted is not None else None
            versus = _versus(e, k)
            scenes[s] = {
                "expert_success": e, "expert_checkpoint": entry["checkpoint"], "expert_sha256": entry["sha256"],
                "complete": entry["complete"],
                "pilot": {"name": pilot.get("pilot"), "checkpoint": pilot.get("checkpoint"), "sha256": pilot.get("sha256"),
                          "kept": pilot.get("kept"), "flown": pilot.get("flown"), "kept_rate": pilot["kept_rate"],
                          "wilson95": [lo_p, hi_p]},
                "intervals_overlap": lo_e <= hi_p and lo_p <= hi_e,
                "expert_rate_in_pilot_interval": lo_p <= e["value"] <= hi_p,
                "same_checkpoint": (None if not entry["sha256"] or not pilot.get("sha256")
                                    else entry["sha256"] == pilot.get("sha256")),
                "scripted_success": k, "scripted_below": versus["below"], "scripted_separated": versus["separated"],
            }
    checked = list(scenes.values())
    return {
        "condition": DONE_CHECK_CONDITION, "policy": expert, "scripted": scripted,
        "rule": "plan 6 revision 1: the expert flown through the harness lands within the CI of its kept-rate and the "
                "scripted baseline below it. Here: the expert's Wilson 95 % success interval overlaps the pilot's "
                "kept-rate interval (thin slice), and the scripted baseline's success rate is below the expert's "
                "('separated' when its whole interval is)",
        "pilots": {"path": pilots["path"], "sha256": pilots["sha256"]},
        "scenes": scenes,
        "overlap_all": all(c["intervals_overlap"] for c in checked) if checked else None,
        "scripted_below_all": (None if not checked or any(c["scripted_below"] is None for c in checked)
                               else all(c["scripted_below"] for c in checked)),
        "same_checkpoint_all": (None if not checked or any(c["same_checkpoint"] is None for c in checked)
                                else all(c["same_checkpoint"] for c in checked)),
        "scenes_without_pilot": without,
        "pilot_scenes_not_flown": [s for s in pilots["scenes"] if s not in flown_scenes],
    }


# ----------------------------------------------------------------------------------------------------------------------
# The figure.
# ----------------------------------------------------------------------------------------------------------------------
def _rows(summary: dict) -> list[dict]:
    rows, y = [], 0.0
    for cname, c in summary["conditions"].items():
        scenes = [s for s in c["scenes"] if any(s in p["scenes"] for p in c["policies"].values())]
        if not scenes:
            continue
        rows.append({"kind": "header", "y": y, "condition": cname})
        y += 0.85
        for s in scenes:
            rows.append({"kind": "scene", "y": y, "condition": cname, "scene": s})
            y += 1.0
        if any(p["all"]["episodes"] for p in c["policies"].values()):
            rows.append({"kind": "all", "y": y, "condition": cname})
            y += 1.0
        y += 0.45
    return rows


def _entry(summary: dict, row: dict, label: str) -> dict | None:
    policy = summary["conditions"][row["condition"]]["policies"].get(label)
    if policy is None:
        return None
    return policy["all"] if row["kind"] == "all" else policy["scenes"].get(row["scene"])


def draw(path: Path, summary: dict) -> None:
    rows = _rows(summary)
    labels = list(summary["policies"])
    check = summary.get("done_check") or {}
    k = max(1, len(labels))
    step = min(0.24, 0.72 / k)
    offsets = {label: (i - (k - 1) / 2) * step for i, label in enumerate(labels)}
    units = rows[-1]["y"] + 1.0 if rows else 1.0
    row_in = max(0.3, 0.12 * k + 0.1)
    top_in, bottom_in = 1.15, 1.25 + 0.2 * k
    height = top_in + bottom_in + units * row_in
    width = 14.5
    pers = [v for c in summary["conditions"].values() for p in c["policies"].values()
            for e in list(p["scenes"].values()) + [p["all"]] if e.get("per")
            for v in (e["per"].get("lower"), e["per"].get("upper")) if v is not None]
    with plt.rc_context({**STYLE, **LARGER}):
        fig, axes = plt.subplots(1, 4, figsize=(width, height), sharey=True, gridspec_kw={
            "width_ratios": [1.25, 1.0, 1.0, 1.1], "wspace": 0.1, "left": 0.185, "right": 0.985,
            "top": 1 - top_in / height, "bottom": bottom_in / height})
        ax_per = axes[3]
        for row in rows:
            if row["kind"] == "header":
                continue
            y = row["y"]
            for ax in axes:
                ax.axhspan(y - 0.5, y + 0.5, color=SURFACE, zorder=0)
            if row["kind"] == "scene" and row["condition"] == check.get("condition"):
                pilot = (check.get("scenes") or {}).get(row["scene"], {}).get("pilot")
                if pilot is not None:
                    lo, hi = pilot["wilson95"]
                    axes[0].add_patch(Rectangle((lo, y - 0.44), hi - lo, 0.88, facecolor=BAND, edgecolor="none", zorder=1))
                    axes[0].plot([pilot["kept_rate"]] * 2, [y - 0.44, y + 0.44], color=MUTED, lw=1.2, zorder=2)
            for label in labels:
                entry = _entry(summary, row, label)
                if not entry or not entry.get("episodes"):
                    continue
                colour, yy = summary["policies"][label]["colour"], y + offsets[label]
                for ax, (key, _title) in zip(axes[:3], RATE_PANELS):
                    rate = entry[key]
                    lo, hi = rate["wilson95"]
                    ax.errorbar(rate["value"], yy, xerr=[[rate["value"] - lo], [hi - rate["value"]]], fmt="o", ms=5.5,
                                color=colour, ecolor=colour, elinewidth=1.5, capsize=2.5, markeredgecolor=SURFACE,
                                markeredgewidth=1.0, zorder=4)
                if row["kind"] == "all":  # the one value labelled: each policy's whole, beside its interval
                    sr = entry["success_rate"]
                    lo, hi = sr["wilson95"]
                    right = hi <= 0.85
                    axes[0].text(hi + 0.02 if right else lo - 0.02, yy, f"{sr['value']:.2f}", va="center",
                                 ha="left" if right else "right", fontsize=7.5, color=INK_2, zorder=5)
                per = entry.get("per") or {}
                if per.get("lower") is not None:
                    upper = per.get("upper")
                    ax_per.plot([per["lower"], upper if upper is not None else per["lower"]], [yy, yy], color=colour,
                                lw=3.0, solid_capstyle="round", zorder=4)
                    ax_per.plot([per["lower"]] + ([upper] if upper is not None else []), [yy] * (1 + (upper is not None)),
                                ls="", marker="o", ms=4, color=colour, markeredgecolor=SURFACE, markeredgewidth=0.8,
                                zorder=5)
            if row["kind"] == "scene" and not any((_entry(summary, row, label) or {}).get("episodes") for label in labels):
                states = sorted({f"{(_entry(summary, row, label) or {}).get('status')}" for label in labels
                                 if _entry(summary, row, label)})
                axes[0].text(0.02, y, f"not flown ({', '.join(states)})", va="center", fontsize=8, color=MUTED)
        ticks, ticklabels = [], []
        for row in rows:
            if row["kind"] == "header":
                continue
            ticks.append(row["y"])
            if row["kind"] == "all":
                ticklabels.append("all scenes")
                continue
            text = scene_label(row["scene"])
            verdict = (check.get("scenes") or {}).get(row["scene"]) if row["condition"] == check.get("condition") else None
            if verdict is not None and not verdict["intervals_overlap"]:
                text += "  [outside pilot CI]"
            incomplete = [label for label in labels if (_entry(summary, row, label) or {}).get("episodes")
                          and not _entry(summary, row, label)["complete"]]
            if incomplete:
                text += "  *"
            ticklabels.append(text)
        axes[0].set_yticks(ticks, ticklabels)
        for tick, row in zip(axes[0].get_yticklabels(), [r for r in rows if r["kind"] != "header"]):
            if row["kind"] == "all":
                tick.set_fontweight("bold")
                tick.set_color(INK)
        header = blended_transform_factory(fig.transFigure, axes[0].transData)
        for row in rows:
            if row["kind"] == "header":
                fig.text(0.008, 0, CONDITION_TITLES.get(row["condition"], row["condition"]), transform=header,
                         fontsize=10, fontweight="bold", color=INK, va="center").set_y(row["y"])
                for ax in axes:
                    ax.axhline(row["y"] + 0.4, color=GRID, lw=0.8, zorder=1)
        for ax, (_key, title) in zip(axes[:3], RATE_PANELS):
            ax.set_xlim(-0.03, 1.08)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_title(title)
            ax.xaxis.grid(True)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
        axes[0].set_xlabel("share of the scene's episodes, with the Wilson 95 % interval")
        ax_per.axvspan(*PAPER_PER, color=BAND, zorder=0)
        ax_per.set_xlim(min([PAPER_PER[0] - 0.03] + [v - 0.02 for v in pers]), 1.01)
        ax_per.set_title("PER over successes")
        ax_per.set_xlabel("straight-line bound to grid-optimum bound")
        ax_per.xaxis.grid(True)
        ax_per.set_axisbelow(True)
        ax_per.tick_params(axis="y", length=0)
        axes[0].set_ylim(units - 0.4, -0.6)
        handles = [Line2D([0], [0], marker="o", color=summary["policies"][label]["colour"], lw=1.5, ms=5.5,
                          label=f"{label}: {summary['policies'][label]['description']}") for label in labels]
        band = "grey band: the paper's models' PER, 0.73-0.78 (PER panel)"
        if check.get("scenes"):
            band = ("grey band and tick: the pilot's kept-rate, 95 % interval, from the thin slice (success panel, seen "
                    "scenes); " + band.replace("grey band: ", ""))
        handles.append(Patch(color=BAND, label=band))
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.008, 0.12 / height), ncol=1, fontsize=8.5)
        n = sum(p["all"]["episodes"] for c in summary["conditions"].values() for p in c["policies"].values())
        fig.suptitle(f"Closed-loop evaluation: {n:,} episodes, {len(labels)} "
                     f"polic{'y' if len(labels) == 1 else 'ies'}, {len(summary['conditions'])} "
                     f"condition{'' if len(summary['conditions']) == 1 else 's'}", x=0.008, y=1 - 0.12 / height,
                     ha="left", va="top", fontsize=12.5)
        subtitle = ("Dots: the rate over the scene's episodes (a condition's whole: over all of them); bars: Wilson 95 % "
                    "intervals. The task's rules, lock-step, a0-aligned starts. * a scene not yet flown in full.")
        fig.text(0.008, 1 - 0.5 / height, subtitle, ha="left", va="top", fontsize=8.5, color=INK_2)
        fig.savefig(path, dpi=130)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# The tables.
# ----------------------------------------------------------------------------------------------------------------------
def _interval(rate: dict | None) -> str:
    if not rate:
        return ""
    lo, hi = rate["wilson95"]
    return f"{rate['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def _value(rate: dict | None) -> str:
    return "" if not rate else f"{rate['value']:.3f}"


def _per(per: dict | None) -> str:
    if not per or per.get("lower") is None:
        return ""
    upper = "?" if per.get("upper") is None else f"{per['upper']:.3f}"
    return f"{per['lower']:.3f}-{upper} (n={per['episodes']})"


def _yes(value) -> str:
    return "n/a" if value is None else ("yes" if value else "**no**")


def _shown(path) -> str:
    """A path as a README shows it: relative to the repository where it lies inside it."""
    if path is None:
        return "?"
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def markdown(summary: dict) -> str:
    lines = ["# Closed-loop evaluation", "",
             f"From {len(summary['inputs']['records'])} harness records by `scripts/harness_figures.py` "
             f"({summary['generated']}); every number is in harness_results.json. Success, collision, out-of-bounds and "
             f"timeout rates are shares of the scene's episodes with Wilson 95 % intervals; PER = L_opt / max(L, L_opt) "
             f"over successful episodes, between the straight-line and the grid-optimum bounds.", ""]
    for cname, c in summary["conditions"].items():
        lines += [f"## {CONDITION_TITLES.get(cname, cname)}", "",
                  f"Condition file `{_shown(c['file'])}` (sha256 {str(c['sha256'])[:12]}), {c['episodes_per_scene']} episodes "
                  f"per scene; a0 {c['a0']}, step limit {c['max_steps']}, mesh scenes' contact rule: the "
                  f"{c['contact_rule_shape']}.", "",
                  "| Scene | Policy | Episodes | Success [95 % CI] | Collision [95 % CI] | Out of bounds | Timeout | "
                  "PER over successes |", "|---|---|---|---|---|---|---|---|"]
        for scene in c["scenes"] + ["all"]:
            first = True
            for label, policy in c["policies"].items():
                entry = policy["all"] if scene == "all" else policy["scenes"].get(scene)
                if entry is None or (scene == "all" and not entry["episodes"]):
                    continue
                cell = ("**all**" if scene == "all" else scene_label(scene)) if first else ""
                first = False
                if not entry["episodes"]:
                    why = entry["status"] + (f": {entry['reason']}" if entry.get("reason") else "")
                    lines.append(f"| {cell} | {label} | 0 | {why} | | | | |")
                    continue
                n = str(entry["episodes"])
                if scene != "all" and not entry["complete"]:
                    n = f"{entry['episodes']} of {entry['condition_episodes']} ({entry['status']})"
                lines.append(f"| {cell} | {label} | {n} | {_interval(entry['success_rate'])} | "
                             f"{_interval(entry['collision_rate'])} | {_value(entry['out_of_bounds_rate'])} | "
                             f"{_value(entry['timeout_rate'])} | {_per(entry['per'])} |")
        lines.append("")
    check = summary.get("done_check")
    if check is not None:
        lines += ["## Plan 6's done check: the expert against its pilot's kept-rate", "",
                  f"The {check['condition']} condition's expert ({check['policy']}) against each scene's pilot in "
                  f"`{_shown(check['pilots']['path'])}` (sha256 {check['pilots']['sha256'][:12]}): {check['rule']}.", "",
                  "| Scene | Expert success [95 % CI] | Pilot kept-rate [95 % CI] | Intervals overlap | Same checkpoint | "
                  "Scripted success [95 % CI] | Scripted below |", "|---|---|---|---|---|---|---|"]
        for scene, c in check["scenes"].items():
            pilot = c["pilot"]
            lo, hi = pilot["wilson95"]
            scripted = _interval(c["scripted_success"])
            below = _yes(c["scripted_below"]) + (" (separated)" if c["scripted_separated"] else "")
            lines.append(f"| {scene_label(scene)} | {_interval(c['expert_success'])}"
                         f"{'' if c['complete'] else ' (incomplete)'} | {pilot['kept_rate']:.3f} [{lo:.3f}, {hi:.3f}] | "
                         f"{_yes(c['intervals_overlap'])} | {_yes(c['same_checkpoint'])} | {scripted} | "
                         f"{below if c['scripted_success'] else 'n/a'} |")
        outside = [s for s, c in check["scenes"].items() if not c["intervals_overlap"]]
        above = [s for s, c in check["scenes"].items() if c["scripted_below"] is False]
        lines += ["", f"- Every scene's expert interval overlaps its pilot's: {_yes(check['overlap_all'])}"
                  + (f" (outside: {', '.join(outside)})" if outside else ""),
                  f"- The scripted baseline below the expert on every scene: {_yes(check['scripted_below_all'])}"
                  + (f" (not below: {', '.join(above)})" if above else ""),
                  f"- The expert flew each scene's pilot checkpoint: {_yes(check['same_checkpoint_all'])}"]
        if check["pilot_scenes_not_flown"]:
            lines.append(f"- Pilots whose scene the expert has not flown here: {', '.join(check['pilot_scenes_not_flown'])}")
        if check["scenes_without_pilot"]:
            lines.append(f"- Scenes without a pilot entry: {', '.join(check['scenes_without_pilot'])}")
        lines.append("")
    if summary["notes"]:
        lines += ["## Notes", ""] + [f"- {note}" for note in summary["notes"]] + [""]
    lines += ["## Records", ""] + [f"- `{_shown(r['path'])}` (sha256 {r['sha256'][:12]}): {r['policy']} on "
                                   f"{r['condition']}, status {r['status']}" for r in summary["inputs"]["records"]]
    if summary["inputs"]["dropped"]:
        lines += [f"- dropped: {d['scene']} of `{_shown(d['record'])}` ({d['episodes']} episodes)"
                  for d in summary["inputs"]["dropped"]]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------------------------------
# The whole set.
# ----------------------------------------------------------------------------------------------------------------------
def build(record_paths, out_dir: Path, *, pilots_path: Path | None = DEFAULT_PILOTS, drops=()) -> dict:
    """Read the records (and the pilots file unless None), check them, then write the three outputs into `out_dir`;
    nothing is written when anything is refused."""
    records = [load_record(Path(p)) for p in record_paths]
    if not records:
        raise HarnessFigureError("no records given")
    collected = collect(records, drops)
    numbers = summarise_records(collected)
    pilots = load_pilots(Path(pilots_path)) if pilots_path is not None else None
    out_dir = Path(out_dir)
    for name in OUTPUTS:
        evidence.refuse_existing_evidence(out_dir / name)
    record_policy = {}
    for r in records:
        pid = policy_id(r)
        record_policy[r.path] = next(label for label, p in collected["policies"].items() if p.key == pid.key)
    summary = {
        "tool": "scripts/harness_figures.py", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"records": [{"path": r.path, "sha256": r.sha256, "status": r.data.get("status"),
                                "condition": r.data["condition"].get("name"), "policy": record_policy[r.path],
                                "run_started": r.data.get("run_started"), "run_finished": r.data.get("run_finished"),
                                "scenes": {s: {"status": e.get("status"),
                                               "episodes": sum(1 for x in r.data["episodes"] if x.get("scene") == s)}
                                           for s, e in r.data["scenes"].items()}} for r in records],
                   "pilots": None if pilots is None else {"path": pilots["path"], "sha256": pilots["sha256"]},
                   "dropped": collected["dropped"]},
        **numbers,
        "done_check": done_check(numbers, pilots) if pilots is not None else None,
        "notes": collected["notes"],
        "files": list(OUTPUTS),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    draw(out_dir / OUTPUTS[0], summary)
    (out_dir / OUTPUTS[1]).write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / OUTPUTS[2]).write_text(markdown(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", type=Path, nargs="+", required=True, help="harness records (.json)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pilots", type=Path, default=DEFAULT_PILOTS,
                   help="the pilots' kept-rates for the done check (default docs/gates/plan6_pilots.json)")
    p.add_argument("--no-done-check", action="store_true", help="draw without the pilots' kept-rates")
    p.add_argument("--drop", action="append", default=[], metavar="RECORD:SCENE",
                   help="leave one record's scene out (a scene flown twice: say which copy counts)")
    args = p.parse_args(argv)
    try:
        summary = build(args.records, args.out_dir, pilots_path=None if args.no_done_check else args.pilots,
                        drops=args.drop)
    except (HarnessFigureError, FileExistsError, FileNotFoundError) as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    check = summary["done_check"]
    print(json.dumps({"out_dir": str(args.out_dir), "files": summary["files"],
                      "done_check": None if check is None else {k: check[k] for k in (
                          "policy", "overlap_all", "scripted_below_all", "same_checkpoint_all", "pilot_scenes_not_flown")},
                      "notes": summary["notes"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
