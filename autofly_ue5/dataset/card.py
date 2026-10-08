"""The merged dataset's card (plan 6, revision 1, items 1 and 6): what is in it, how its records are defined, how it was
made, and what it cannot be used for. Built from an export's merge.json and dataset_statistics.json (dataset/merge.py),
and written beside them as card.json (for tools) and dataset_card.md (for people).

The record definitions come from the stores (their record_fields and state_fields, which the merge requires to agree)
and from the env's action space. state[0] and state[1] are marked PRIVILEGED: AutoFly's own definition computes them
from the target's true position, which the drone cannot observe (revision 1, item 4).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from autofly_ue5.dataset.merge import MERGE_RECORD, METADATA_FIELDS, SPLITS
from autofly_ue5.dataset.statistics import STATISTICS_FILE
from autofly_ue5.durable import write_text_durably
from autofly_ue5.validate.dataset import ACTION_HIGH, ACTION_LOW

CARD_JSON = "card.json"
CARD_MD = "dataset_card.md"
PAPER_SUCCESS = 0.95  # the paper's expert bar (spec §12), which several of these scenes' experts do not reach
ACTION_FIELDS = (
    ("forward_speed_m_s", "forward speed in the body frame, m/s"),
    ("yaw_rate_rad_s", "yaw rate, rad/s, positive turning right (clockwise seen from above)"),
    ("vertical_speed_m_s", "vertical speed, m/s, positive up"),
)
PRIVILEGED_REASON = ("computed from the target's true position, which the drone cannot observe: a policy that reads it "
                     "can fly to the target without the image or the instruction")
PRIVILEGED_SUMMARY = ("state[0] (distance) and state[1] (bearing) are computed from the target's true position, which "
                      "the drone cannot observe: a policy that reads them can fly to the target without the image or "
                      "the instruction")
PRIVILEGED_STATE = (0, 1)
METADATA_MEANINGS = {
    "file_path": "<store>/episodes/<episode id>, as the release has it",
    "episode_id": "the episode's id in its store: <scene>_<seed>",
    "store": "the raw store (one scene and collection batch) it came from",
    "scene": "the scene id",
    "split_of_scene": "the scene's own split (train, test_seen, test_unseen); not the dataset split",
    "target_name": "the target the instruction names",
    "target_category": "vehicle, geometry, furniture, equipment, human or other (the paper's Fig. 3c mix)",
    "target_seen": "whether the target is in the seen pool (unseen targets are for test_unseen scenes)",
    "target_status": "the pool entry's status: primitive (engine shape), imported (a real model) or placeholder",
    "target_stand_in": "a low-poly stand-in (Kenney) for the object its name describes; filter on it to drop them",
    "seed": "the episode seed (start, target, distractors and layout follow from it and the scene)",
    "pilot": "sha256 of the expert checkpoint that flew it",
}
BREAKDOWNS = (
    ("scene", lambda m: m["scene"]),
    ("target", lambda m: m["target_name"]),
    ("category", lambda m: m["target_category"]),
    ("seen", lambda m: "seen" if m["target_seen"] else "unseen"),
    ("stand_in", lambda m: "stand-in" if m["target_stand_in"] else "not a stand-in"),
    ("target_status", lambda m: m["target_status"]),
    ("split_of_scene", lambda m: m["split_of_scene"]),
    ("store", lambda m: m["store"]),
)


def _counts(episodes, key) -> dict:
    out: dict[str, dict] = {}
    for e in episodes:
        value = out.setdefault(key(e["metadata"]), {s: {"episodes": 0, "records": 0} for s in (*SPLITS, "all")})
        for s in (e["split"], "all"):
            value[s]["episodes"] += 1
            value[s]["records"] += e["steps"]
    return dict(sorted(out.items(), key=lambda kv: (-kv[1]["all"]["episodes"], kv[0])))


def _provenance(record: dict) -> dict:
    pilots: dict[str, dict] = {}
    configs: dict[str, list] = {}
    heads: Counter = Counter()
    for store in record["stores"]:
        prov = store["provenance"]
        for sha, pilot in prov["pilots"].items():
            entry = pilots.setdefault(sha, {"paths": [], "stores": [], "episodes": 0, "deterministic": []})
            entry["episodes"] += pilot["episodes"]
            entry["stores"].append(store["name"])
            for k in ("paths", "deterministic"):
                entry[k] += [v for v in pilot[k] if v not in entry[k]]
        for file, shas in prov["scene_configs"].items():
            configs.setdefault(file, [])
            configs[file] += [s for s in shas if s not in configs[file]]
        heads.update(prov["collector_git_heads"])
    stores = [{k: store[k] for k in ("name", "root", "created", "episodes", "records", "rejects", "kept_rate", "scenes",
                                     "filled_from_pool", "manifest_sha256")}
              | {"pilots": sorted(store["provenance"]["pilots"]),
                 "collector_tree_dirty": store["provenance"]["collector_tree_dirty"],
                 "platform": store["provenance"]["platform"], "target_pools": store["provenance"]["target_pools"]}
              for store in record["stores"]]
    return {"stores": stores, "pilots": pilots, "scene_configs": configs, "collector_git_heads": dict(heads),
            "exporter": record["exporter"], "target_pool": record.get("target_pool")}


def _known_limits(record: dict, counts: dict) -> list[str]:
    templates = record["instruction_templates"]
    limits = [
        f"Language: the two instruction templates are AutoFly's, verbatim, including the original's spelling "
        f"\"avioding\" ({templates[1]!r}); with one obstacle phrase per scene, every instruction is a function of the "
        f"scene and the target. No paraphrases.",
        f"The state is partly privileged: {PRIVILEGED_SUMMARY}. They are AutoFly's own definition, kept as the "
        f"record's; zero or drop them for a policy that should navigate from the image and the instruction.",
    ]
    unverified = [f"state[{f['index']}] {f['name']}" for f in record["state_fields"]
                  if f["status"] == "not_autofly_verified"]
    if unverified:
        limits.append(f"{', '.join(unverified)} follow best-fit definitions that could not be verified against AutoFly's "
                      f"released episodes (status not_autofly_verified; dataset/state.py).")
    stand_ins = counts["stand_in"].get("stand-in", {}).get("all", {}).get("episodes", 0)
    limits.append(f"Kenney low-poly stand-ins stand for vehicles and people where no photoreal model was available "
                  f"(target_stand_in; {stand_ins} episodes here). A later Fab session may replace them.")
    rates = ", ".join(f"{s['name']} {s['kept_rate']}" for s in record["stores"] if s["kept_rate"] is not None)
    limits.append(f"Only successful, collision-free episodes are kept; failures stay in each store's rejects. The SAC "
                  f"experts are below the paper's {PAPER_SUCCESS} success on several scenes, so each scene's data covers "
                  f"the starts its expert could fly (kept-rates: {rates}). The actions are the expert's (stochastic "
                  f"where the pilot record says deterministic false), not a human's.")
    filled = [s for s in record["stores"] if s.get("filled_from_pool")]
    for s in filled:
        limits.append(f"Store {s['name']}: its provenance predates the target pool, so "
                      f"{', '.join(s['filled_from_pool']['fields'])} of {s['filled_from_pool']['episodes']} episodes "
                      f"come from the pool entry with the same asset, material and scale "
                      f"({s['filled_from_pool']['pool']['path']}).")
    limits.append("The val split is a stratified random sample of the same scenes and targets: an in-distribution "
                  "validation set, not a test of unseen scenes or targets (test_seen and test_unseen are evaluated "
                  "in closed loop).")
    limits.append("Per-step phase weights from the rebalancing stage (spec §10.3) are not part of the records.")
    return limits


def build_card(export_dir: Path) -> dict:
    export_dir = Path(export_dir)
    record = json.loads((export_dir / MERGE_RECORD).read_text())
    statistics = json.loads((export_dir / STATISTICS_FILE).read_text())[record["name"]]
    episodes = record["episodes"]
    counts = {name: _counts(episodes, key) for name, key in BREAKDOWNS}
    totals = {measure: {s: sum((e["steps"] if measure == "records" else 1) for e in episodes if e["split"] == s)
                        for s in SPLITS} for measure in ("episodes", "records")}
    for measure in totals:
        totals[measure]["all"] = sum(totals[measure][s] for s in SPLITS)
    distinct: dict[str, dict] = {}
    for e in episodes:
        d = distinct.setdefault(e["instruction"], {"instruction": e["instruction"], "episodes": 0, "records": 0,
                                                   "train": 0, "val": 0})
        d["episodes"] += 1
        d["records"] += e["steps"]
        d[e["split"]] += 1
    obstacle_phrases = {scene: info["instruction_obstacle"] for s in record["stores"]
                        for scene, info in (s.get("scenes") or {}).items()}
    fields = record["record_fields"]
    return {
        "name": record["name"], "version": record["version"], "created": record["created"], "export": record["path"],
        "totals": totals, "counts": counts,
        "instructions": {"templates": record["instruction_templates"], "obstacle_phrases": obstacle_phrases,
                         "distinct": sorted(distinct.values(), key=lambda d: (-d["episodes"], d["instruction"]))},
        "records": {
            "image_0": f"{fields['observation.image_0']}: the forward camera, one frame per 0.2 s step",
            "action": [{"index": i, "name": name, "range": [float(ACTION_LOW[i]), float(ACTION_HIGH[i])],
                        "definition": definition} for i, (name, definition) in enumerate(ACTION_FIELDS)],
            "state": [{**f, "privileged": f["index"] in PRIVILEGED_STATE,
                       **({"privileged_reason": PRIVILEGED_REASON} if f["index"] in PRIVILEGED_STATE else {})}
                      for f in record["state_fields"]],
            "language_instruction": "the episode's instruction, repeated on every step",
            "step_definitions": fields,
        },
        "episode_metadata": [{"name": name, "type": kind, "meaning": METADATA_MEANINGS[name]}
                             for name, kind in METADATA_FIELDS],
        "split": {k: record["split"][k] for k in ("file", "sha256", "val_fraction", "seed", "stratify_by", "rule")},
        "statistics": {"file": STATISTICS_FILE, "computed_on": "train", "action": statistics["action"],
                       "proprio": statistics["proprio"], "num_transitions": statistics["num_transitions"],
                       "num_trajectories": statistics["num_trajectories"]},
        "provenance": _provenance(record),
        "known_limits": _known_limits(record, counts),
        "loading": {
            "tfds": f"tfds.builder_from_directory('<out>/{record['name']}/{record['version']}'), or "
                    f"tfds.builder('{record['name']}', data_dir='<out>') as OpenVLA calls it",
            "openvla": "image_obs_keys {'primary': 'image_0'}, state_obs_keys ['state'], language_key "
                       "'language_instruction'; the action is 3-dim, so pass action_normalization_mask [True]*3 and no "
                       "absolute_action_mask (OpenVLA's make_oxe_dataset_kwargs assumes 7-dim EEF actions); "
                       "dataset_statistics.json[name] is the normalisation (BOUNDS_Q99 uses q01 and q99)",
        },
    }


def _table(rows: list[list], header: list[str]) -> str:
    """A markdown table; a '|' inside a cell (state definitions quote |r|) is escaped, or it would split the cell."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(values) -> str:
    return "[" + ", ".join(f"{v:.4g}" for v in values) + "]"


def render_markdown(card: dict) -> str:
    t = card["totals"]
    out = [f"# {card['name']} {card['version']}", "",
           f"AutoFly-format UAV navigation episodes (UE 5.7 + Project AirSim), merged from "
           f"{len(card['provenance']['stores'])} raw stores. Created {card['created']}; export `{card['export']}`.", "",
           _table([[s, t["episodes"][s], t["records"][s]] for s in (*SPLITS, "all")], ["split", "episodes", "records"]),
           ""]
    titles = {"scene": "Scene", "target": "Target", "category": "Category", "seen": "Seen / unseen",
              "stand_in": "Stand-in status", "target_status": "Pool status", "split_of_scene": "Scene split",
              "store": "Store"}
    out.append("## Counts")
    for name, title in titles.items():
        rows = [[value, c["train"]["episodes"], c["val"]["episodes"], c["all"]["episodes"], c["all"]["records"]]
                for value, c in card["counts"][name].items()]
        out += ["", f"### {title}", "", _table(rows, [name, "train", "val", "episodes", "records"])]
    ins = card["instructions"]
    out += ["", "## Instructions", "", "Templates (AutoFly's, verbatim; `{obstacle}` is the scene's phrase):", ""]
    out += [f"- `{tpl}`" for tpl in ins["templates"]]
    out += ["", _table([[s, p] for s, p in ins["obstacle_phrases"].items()], ["scene", "obstacle phrase"]), "",
            f"{len(ins['distinct'])} distinct instructions:", "",
            _table([[d["instruction"], d["train"], d["val"], d["episodes"], d["records"]] for d in ins["distinct"]],
                   ["instruction", "train", "val", "episodes", "records"])]
    rec = card["records"]
    out += ["", "## Records", "", "Each step holds `observation/image_0`, `observation/state` (float64 in TFDS, float32 "
            "values), `action` (float32) and `language_instruction`.", "", f"- image_0: {rec['image_0']}",
            f"- language_instruction: {rec['language_instruction']}", "", "Action:", "",
            _table([[a["index"], a["name"], a["range"], a["definition"]] for a in rec["action"]],
                   ["index", "name", "range", "definition"]), "", "State:", "",
            _table([[s["index"], s["name"] + (" **PRIVILEGED**" if s["privileged"] else ""), s["status"],
                     s["definition"]] for s in rec["state"]], ["index", "name", "status", "definition"]), "",
            f"**PRIVILEGED**: {PRIVILEGED_SUMMARY}.", "", "Episode metadata:", "",
            _table([[m["name"], m["type"], m["meaning"]] for m in card["episode_metadata"]], ["field", "type", "meaning"])]
    sp = card["split"]
    out += ["", "## Split", "", f"val_fraction {sp['val_fraction']}, seed {sp['seed']}, stratified by "
            f"{' x '.join(sp['stratify_by'])}; frozen in `{sp['file']}` (sha256 `{sp['sha256']}`).", "", sp["rule"]]
    st = card["statistics"]
    out += ["", "## Normalisation statistics", "", f"`{st['file']}` in OpenVLA's schema, over the train split only "
            f"({st['num_trajectories']} episodes, {st['num_transitions']} records):", "",
            _table([[k, _fmt(st["action"][k]), _fmt(st["proprio"][k])] for k in ("mean", "std", "min", "max", "q01", "q99")],
                   ["", "action[3]", "proprio = state[9]"])]
    prov = card["provenance"]
    out += ["", "## Provenance", "", _table(
        [[s["name"], s["episodes"], s["records"], s["rejects"], s["kept_rate"], ", ".join(s["scenes"] or {}),
          s["created"], "<br>".join(s["pilots"])] for s in prov["stores"]],
        ["store", "episodes", "records", "rejects", "kept-rate", "scenes", "created", "pilot sha256"]), "",
        "Pilots:", "", _table([[sha, p["episodes"], ", ".join(p["stores"]), ", ".join(map(str, p["paths"])),
                                ", ".join(map(str, p["deterministic"]))] for sha, p in prov["pilots"].items()],
                              ["sha256", "episodes", "stores", "checkpoint", "deterministic"]), "",
        "Scene configs:", "", _table([[f, ", ".join(map(str, shas))] for f, shas in prov["scene_configs"].items()],
                                     ["config", "sha256"]), "",
        "Collector git heads:", "", _table([[h, n] for h, n in prov["collector_git_heads"].items()], ["head", "episodes"]),
        "", f"Exported at git head {prov['exporter']['git_head']} (tree dirty: {prov['exporter']['tree_dirty']})."]
    out += ["", "## Known limits", ""] + [f"- {limit}" for limit in card["known_limits"]]
    out += ["", "## Loading", "", f"- TFDS: {card['loading']['tfds']}", f"- OpenVLA-style: {card['loading']['openvla']}", ""]
    return "\n".join(out)


def write_card(export_dir: Path) -> dict:
    """Build the card and write card.json and dataset_card.md beside the export (derived files: rewritten each time)."""
    card = build_card(export_dir)
    write_text_durably(Path(export_dir) / CARD_JSON, json.dumps(card, indent=1) + "\n")
    write_text_durably(Path(export_dir) / CARD_MD, render_markdown(card))
    return card
