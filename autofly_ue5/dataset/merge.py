"""One TFDS dataset from several raw stores, with a frozen train/val split (plan 6, revision 1, item 1).

`rlds.export_rlds` writes one dataset per store, split "train" only, with `episode_metadata` holding only the file path:
nine to eighteen per-scene datasets would each need their own OpenVLA registration and normalisation key, and none would
have a validation split. The raw stores stay per scene and batch; this exports them as ONE dataset:

    <out>/<name>/1.0.0/                    <name>-{train,val}.tfrecord-XXXXX-of-YYYYY, features.json, dataset_info.json
    <out>/<name>/merge.json                what was exported: the stores and their provenance, every episode's split,
                                           shard and metadata (the dataset card and the read-back checks work from it)
    <out>/<name>/dataset_statistics.json   OpenVLA's normalisation statistics over the train split (dataset/statistics.py)

- Steps are exactly AutoFly's four fields (rlds.py). `episode_metadata` keeps the release's `file_path` and adds
  METADATA_FIELDS, taken from the store's manifest entry and the episode's provenance (`episode_metadata`). M3's pilot
  predates the target pool, so its provenance has no category, seen or status: those come from the pool entry of the
  same name, and only if that entry is the same object (asset, material, scale); the record says which were filled.
- The stand-in mark: a provenance `target.stand_in` where the collector writes one, otherwise the pool's status
  "placeholder", which is how assets/targets.json marks the Kenney stand-ins (2026-10-08).
- The val split is `val_fraction` of the episodes, stratified by scene x target name (`assign_splits`) and frozen in a
  split file: an existing file is reused, never redrawn, and a top-up only adds the new episodes' assignments.
- A store is refused if its step definitions differ from the first's, if the validator fails it, or if it repeats
  another store's scene and seed (the same start, target and layout twice: a top-up collects on a new seed slice).
- Each split's episodes are shuffled by a keyed hash, so a shard or a `train[:N]` slice mixes the scenes.
- The version directory is written under a temporary name and renamed into place: a failed export leaves none.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from autofly_ue5.dataset.raw import FORMAT
from autofly_ue5.dataset.rlds import TEMPLATES, VERSION, episode_example, write_split_shards
from autofly_ue5.dataset.statistics import STATISTICS_FILE, dataset_statistics
from autofly_ue5.durable import write_text_durably
from autofly_ue5.paths import ROOT

DEFAULT_NAME = "autofly_ue5_proto"
SPLITS = ("train", "val")
DEFAULT_VAL_FRACTION = 0.05
DEFAULT_SPLIT_SEED = 0
SPLIT_FORMAT = "autofly_ue5_split/1"
MERGE_FORMAT = "autofly_ue5_merged/1"
MERGE_RECORD = "merge.json"
STRATIFY_BY = ("scene", "target_name")
SPLIT_RULE = ("val takes round(val_fraction x episodes), at least one, given out one at a time to the scene x target "
              "stratum furthest below its share (val_fraction x its episodes; ties by sha256 of seed and stratum), "
              "never a stratum's last train episode; within a stratum its new episodes go in sha256(seed, key) order. "
              "Episodes already in the file keep their split.")
MERGED_FEATURES_TEMPLATE = TEMPLATES / "merged_features.json"
STEP_KEYS = ("steps/observation/image_0", "steps/observation/state", "steps/action", "steps/language_instruction")
# (name, kind): kind "text" is a TFDS Text, "bool" and "int64" are Scalars. scripts/make_rlds_templates.py generates the
# features template from the same list (tests/test_merge.py holds them together).
METADATA_FIELDS = (
    ("file_path", "text"),  # the release's: <store>/episodes/<id>
    ("episode_id", "text"),
    ("store", "text"),
    ("scene", "text"),
    ("split_of_scene", "text"),  # the scene's own split: train, test_seen, test_unseen
    ("target_name", "text"),
    ("target_category", "text"),
    ("target_seen", "bool"),
    ("target_status", "text"),  # the pool entry's status: primitive, imported, placeholder
    ("target_stand_in", "bool"),  # a low-poly stand-in (Kenney) for the object the name describes
    ("seed", "int64"),
    ("pilot", "text"),  # the sha256 of the expert checkpoint that flew it
)
POOL_FILLABLE = {"target_category": "category", "target_seen": "seen", "target_status": "status"}
PLACEHOLDER_STATUS = "placeholder"
# The shapes the merged features declare; a store's manifest must describe its records the same way.
RECORD_SHAPES = {"observation.image_0": "RGB 256x256 PNG", "observation.state": "float32[9]", "action": "float32[3]",
                 "language_instruction": "string"}


class MergeError(ValueError):
    """Stores that cannot be merged as they stand."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rank(seed: int, kind: str, text: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\x1f{kind}\x1f{text}".encode()).digest()[:8], "big")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------------------------------------
# Per-episode metadata
# --------------------------------------------------------------------------------------------------------
def stand_in_mark(target: Mapping) -> bool:
    """Whether the target is a stand-in for the object its name describes: an explicit `stand_in` in its record wins;
    otherwise the pool's placeholder status (the Kenney models, assets/targets.json)."""
    if target.get("stand_in") is not None:
        return bool(target["stand_in"])
    return target.get("status") == PLACEHOLDER_STATUS


def load_pool_index(path: Path | None) -> tuple[dict[str, dict], dict | None]:
    """The target pool file's entries by name (plain JSON: only the fields the merge reads are needed), and its record."""
    if path is None:
        return {}, None
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("targets"), list):
        raise MergeError(f"{path}: not a target pool file ({{'version', 'targets': [...]}})")
    return {t["name"]: t for t in data["targets"] if isinstance(t, dict) and "name" in t}, {
        "path": str(path), "sha256": _sha256(path)}


def _same_object(target: Mapping, entry: Mapping) -> bool:
    scale_a, scale_b = target.get("scale"), entry.get("scale")
    return (target.get("asset") is not None and target.get("asset") == entry.get("asset")
            and target.get("material") == entry.get("material")
            and scale_a is not None and scale_b is not None and len(scale_a) == len(scale_b)
            and all(math.isclose(float(a), float(b), abs_tol=1e-9) for a, b in zip(scale_a, scale_b)))


def episode_metadata(store: str, entry: Mapping, provenance: Mapping, pool: Mapping[str, Mapping]) -> tuple[dict, list[str]]:
    """The episode's METADATA_FIELDS values, and which of them were filled from the pool."""
    key = f"{store}/{entry['id']}"
    target = provenance.get("target")
    if not isinstance(target, dict):
        raise MergeError(f"{key}: the provenance holds no target record")
    if target.get("name") != entry["target_name"]:
        raise MergeError(f"{key}: the provenance's target {target.get('name')!r} is not the manifest's "
                         f"{entry['target_name']!r}")
    values = {field: target.get(source) for field, source in POOL_FILLABLE.items()}
    filled = [field for field, value in values.items() if value is None]
    if filled:
        pool_entry = pool.get(entry["target_name"])
        if pool_entry is None:
            raise MergeError(f"{key}: the provenance gives no {', '.join(filled)} for target {entry['target_name']!r}, "
                             f"and no target pool entry of that name was given to take them from (--targets)")
        if not _same_object(target, pool_entry):
            raise MergeError(f"{key}: the provenance gives no {', '.join(filled)} for target {entry['target_name']!r}, "
                             f"and the pool's entry of that name is not the same asset, material and scale "
                             f"({target.get('asset')!r}, {target.get('material')!r}, {target.get('scale')!r} against "
                             f"{pool_entry.get('asset')!r}, {pool_entry.get('material')!r}, {pool_entry.get('scale')!r})")
        for field in filled:
            values[field] = pool_entry[POOL_FILLABLE[field]]
    if not isinstance(values["target_seen"], bool) or not isinstance(values["target_category"], str) \
            or not isinstance(values["target_status"], str):
        raise MergeError(f"{key}: target category {values['target_category']!r}, seen {values['target_seen']!r}, "
                         f"status {values['target_status']!r}: expected a string, a bool and a string")
    pilot = (provenance.get("expert") or {}).get("sha256")
    if not isinstance(pilot, str) or not pilot:
        raise MergeError(f"{key}: the provenance names no expert checkpoint sha256")
    seed = entry["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise MergeError(f"{key}: seed {seed!r} is not an integer")
    meta = {"file_path": f"{store}/{entry['path']}", "episode_id": entry["id"], "store": store, "scene": entry["scene"],
            "split_of_scene": entry["split"], "target_name": entry["target_name"],
            "target_category": values["target_category"], "target_seen": values["target_seen"],
            "target_status": values["target_status"],
            "target_stand_in": stand_in_mark({**target, "status": values["target_status"]}),
            "seed": seed, "pilot": pilot}
    return meta, filled


def _example_metadata(meta: Mapping) -> dict[str, tuple[str, list]]:
    out = {}
    for name, kind in METADATA_FIELDS:
        if name == "file_path":
            continue  # written by rlds.episode_example, as the release has it
        value = meta[name]
        out[name] = ("bytes", [value.encode()]) if kind == "text" else ("int64", [int(value)])
    return out


# --------------------------------------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------------------------------------
def assign_splits(keys_by_stratum: Mapping[str, Sequence[str]], frozen: Mapping[str, str], *, val_fraction: float,
                  seed: int) -> dict[str, str]:
    """Every key's split ("train" or "val"), keeping `frozen`'s and giving out the rest by SPLIT_RULE: val holds
    round(val_fraction x all episodes), at least one, where the strata allow, each stratum as close to its own share as
    whole episodes allow (largest remainder); a stratum never gives up its last train episode. At least one because TFDS
    cannot read an empty split ("Shard empty", 4.9.3 and 4.9.10)."""
    if not 0.0 < val_fraction < 1.0:
        raise MergeError(f"val_fraction {val_fraction} is not in (0, 1): TFDS cannot read an empty val split")
    strata = {s: list(keys) for s, keys in keys_by_stratum.items()}
    assignment: dict[str, str] = {}
    val: dict[str, int] = {}
    new: dict[str, list[str]] = {}
    for s, keys in strata.items():
        for k in keys:
            if k in frozen:
                if frozen[k] not in SPLITS:
                    raise MergeError(f"the split file gives {k} the split {frozen[k]!r}, not one of {SPLITS}")
                assignment[k] = frozen[k]
        val[s] = sum(1 for k in keys if assignment.get(k) == "val")
        new[s] = sorted((k for k in keys if k not in frozen), key=lambda k: _rank(seed, "episode", k))
    total = sum(len(keys) for keys in strata.values())
    want = max(1, math.floor(val_fraction * total + 0.5))
    have = sum(val.values())
    taken = dict.fromkeys(strata, 0)
    while have < want:
        eligible = [s for s in strata if taken[s] < len(new[s]) and len(strata[s]) - val[s] >= 2]
        if not eligible:
            break
        s = max(eligible, key=lambda s: (val_fraction * len(strata[s]) - val[s], _rank(seed, "stratum", s)))
        assignment[new[s][taken[s]]] = "val"
        taken[s] += 1
        val[s] += 1
        have += 1
    for s in strata:
        for k in new[s]:
            assignment.setdefault(k, "train")
    return assignment


def _stratum(meta: Mapping) -> str:
    return "|".join(str(meta[f]) for f in STRATIFY_BY)


def _read_split_file(path: Path, val_fraction: float | None, seed: int | None) -> dict | None:
    if not Path(path).is_file():
        return None
    data = json.loads(Path(path).read_text())
    if data.get("format") != SPLIT_FORMAT:
        raise MergeError(f"{path}: format {data.get('format')!r}, expected {SPLIT_FORMAT!r}")
    if val_fraction is not None and val_fraction != data["val_fraction"]:
        raise MergeError(f"{path} was drawn with val_fraction {data['val_fraction']}; it is reused as it is, never "
                         f"redrawn (omit --val-fraction, or start a new split file)")
    if seed is not None and seed != data["seed"]:
        raise MergeError(f"{path} was drawn with seed {data['seed']}; it is reused as it is, never redrawn (omit "
                         f"--split-seed, or start a new split file)")
    if list(data.get("stratify_by", [])) != list(STRATIFY_BY):
        raise MergeError(f"{path} is stratified by {data.get('stratify_by')}, not {list(STRATIFY_BY)}")
    return data


def _freeze_split(split_file: Path, existing: dict | None, metas: Sequence[Mapping], *, val_fraction: float, seed: int,
                  stores: Sequence[str]) -> tuple[dict[str, str], dict]:
    """Assign every episode a split, keeping the file's, and write the file durably if anything was added."""
    keys_by_stratum: dict[str, list[str]] = {}
    for meta in metas:
        keys_by_stratum.setdefault(_stratum(meta), []).append(f"{meta['store']}/{meta['episode_id']}")
    frozen_entries = dict(existing["episodes"]) if existing else {}
    frozen = {k: v["split"] for k, v in frozen_entries.items()}
    assignment = assign_splits(keys_by_stratum, frozen, val_fraction=val_fraction, seed=seed)
    stratum_of = {k: s for s, keys in keys_by_stratum.items() for k in keys}
    if not any(assignment[k] == "val" for k in stratum_of):  # checked before the file is written: nothing is frozen
        raise MergeError(f"the val split would be empty: {len(stratum_of)} episodes, and no scene x target stratum has "
                         f"a second episode to give it; TFDS cannot read an empty split")
    added = {k: assignment[k] for k in assignment if k not in frozen}
    warnings = [f"{k}: frozen in stratum {frozen_entries[k].get('stratum')!r}, now {stratum_of[k]!r}; its split is kept"
                for k in assignment if k in frozen and frozen_entries[k].get("stratum") != stratum_of[k]]
    content = existing or {"format": SPLIT_FORMAT, "rule": SPLIT_RULE, "val_fraction": val_fraction, "seed": seed,
                           "stratify_by": list(STRATIFY_BY), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "updates": [], "episodes": {}}
    if added:
        content = {**content, "episodes": {**frozen_entries, **{k: {"split": v, "stratum": stratum_of[k]}
                                                                  for k, v in sorted(added.items())}},
                   "updates": [*content["updates"], {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stores": list(stores),
                                                     "added": dict(Counter(added.values()))}]}
        Path(split_file).parent.mkdir(parents=True, exist_ok=True)
        write_text_durably(Path(split_file), json.dumps(content, indent=1) + "\n")
    record = {"file": str(split_file), "sha256": _sha256(split_file), "val_fraction": content["val_fraction"],
              "seed": content["seed"], "stratify_by": list(STRATIFY_BY), "rule": content["rule"],
              "added": dict(Counter(added.values())),
              "absent": sorted(set(frozen) - set(assignment)), "warnings": warnings}
    return {k: assignment[k] for k in stratum_of}, record


# --------------------------------------------------------------------------------------------------------
# Stores
# --------------------------------------------------------------------------------------------------------
def _load_store(root: Path) -> dict:
    root = Path(root).resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise MergeError(f"{root} holds no manifest.json: not a raw store")
    manifest = json.loads(path.read_text())
    if manifest.get("format") != FORMAT:
        raise MergeError(f"{root}: manifest format {manifest.get('format')!r}, expected {FORMAT!r}")
    return {"root": root, "manifest": manifest, "manifest_sha256": _sha256(path)}


def _check_definitions(stores: Sequence[dict]) -> None:
    first = stores[0]["manifest"]
    fields = first.get("record_fields") or {}
    for key, shape in RECORD_SHAPES.items():
        if not str(fields.get(key, "")).startswith(shape):
            raise MergeError(f"{first['name']}: record_fields[{key!r}] = {fields.get(key)!r}; the merged features "
                             f"declare {shape}")
    for store in stores[1:]:
        manifest = store["manifest"]
        for key in ("record_fields", "state_fields"):
            if manifest.get(key) != first.get(key):
                raise MergeError(f"{manifest['name']}: its {key} differ from {first['name']}'s; stores whose step "
                                 f"definitions disagree are not merged")


def _summarise_provenance(provenances: Sequence[Mapping]) -> dict:
    """What produced a store's episodes, for the card: pilots, scene configs, collector heads, pools, the platform."""
    pilots: dict[str, dict] = {}
    scene_configs: dict[str, list] = {}
    pools: dict[str, list] = {}
    heads: Counter = Counter()
    dirty: Counter = Counter()
    rewards: Counter = Counter()
    layouts: dict[str, list] = {}
    platform: dict[str, list] = {}
    for p in provenances:
        expert = p.get("expert") or {}
        pilot = pilots.setdefault(expert.get("sha256"), {"paths": [], "episodes": 0, "deterministic": []})
        for k, v in (("paths", expert.get("path")), ("deterministic", expert.get("deterministic"))):
            if v not in pilot[k]:
                pilot[k].append(v)
        pilot["episodes"] += 1
        config = p.get("scene_config") or {}
        if config.get("sha256") not in scene_configs.setdefault(config.get("file"), []):
            scene_configs[config.get("file")].append(config.get("sha256"))
        pool = p.get("target_pool")
        if isinstance(pool, dict) and pool.get("path") not in pools.setdefault(pool.get("sha256"), []):
            pools[pool.get("sha256")].append(pool.get("path"))
        heads[p.get("collector_git_head")] += 1
        dirty[str(p.get("collector_tree_dirty"))] += 1
        rewards[str(p.get("reward_version"))] += 1
        scene_id = (p.get("scene") or {}).get("id")
        if p.get("layout_sha256") not in layouts.setdefault(scene_id, []):
            layouts[scene_id].append(p.get("layout_sha256"))
        for key, value in (p.get("platform") or {}).items():
            value = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            if value not in platform.setdefault(key, []):
                platform[key].append(value)
    return {"pilots": pilots, "scene_configs": scene_configs, "target_pools": pools, "collector_git_heads": dict(heads),
            "collector_tree_dirty": dict(dirty), "reward_versions": dict(rewards), "layouts": layouts,
            "platform": platform}


def _gather(stores: Sequence[dict], pool: Mapping, pool_record: dict | None, validate: bool) -> list[dict]:
    """Every episode of every store, with its metadata; refuses what cannot be merged."""
    from autofly_ue5.validate.dataset import validate_dataset

    episodes: list[dict] = []
    seen_seeds: dict[tuple[str, int], str] = {}
    for store in stores:
        manifest, root = store["manifest"], store["root"]
        name = manifest["name"]
        if validate:
            report = validate_dataset(root, decode_every_frame=False)
            if not report["pass"]:
                raise MergeError(f"store {name} ({root}) fails the dataset validator: {report['failures'][:5]}")
            store["validation"] = {"pass": True, "warnings": report["warnings"], "distributions": report["distributions"]}
        else:
            store["validation"] = None
        provenances, filled_count = [], 0
        for entry in manifest["episodes"]:
            provenance = json.loads((root / "provenance" / f"{entry['id']}.json").read_text())
            meta, filled = episode_metadata(name, entry, provenance, pool)
            other = seen_seeds.setdefault((meta["scene"], meta["seed"]), name)
            if other != name:
                raise MergeError(f"{name}/{entry['id']} and an episode of {other} fly the same scene and seed "
                                 f"({meta['scene']}, {meta['seed']}): the same start, target and layout twice; collect "
                                 f"a top-up on a new --seed-slice")
            filled_count += bool(filled)
            provenances.append(provenance)
            episodes.append({"key": f"{name}/{entry['id']}", "store_root": str(root), "path": entry["path"],
                             "steps": entry["steps"], "instruction": entry["instruction"], "metadata": meta,
                             "stratum": _stratum(meta), "filled_from_pool": filled,
                             "target_name_status": provenance["target"].get("name_status")})
        store["provenance"] = _summarise_provenance(provenances)
        store["filled_from_pool"] = ({"episodes": filled_count, "fields": sorted(
            {f for e in episodes if e["metadata"]["store"] == name for f in e["filled_from_pool"]}),
            "pool": pool_record} if filled_count else None)
    return episodes


def _store_record(store: dict) -> dict:
    manifest = store["manifest"]
    counts = manifest.get("counts", {})
    kept, rejects = counts.get("episodes", 0), counts.get("rejects", 0)
    return {"name": manifest["name"], "root": str(store["root"]), "manifest_sha256": store["manifest_sha256"],
            "created": manifest.get("created"), "episodes": len(manifest["episodes"]),
            "records": sum(e["steps"] for e in manifest["episodes"]), "rejects": rejects,
            "kept_rate": round(kept / (kept + rejects), 4) if kept + rejects else None,
            "scenes": manifest.get("scenes"), "card": manifest.get("card"), "provenance": store["provenance"],
            "filled_from_pool": store["filled_from_pool"], "validation": store["validation"]}


# --------------------------------------------------------------------------------------------------------
# The export
# --------------------------------------------------------------------------------------------------------
def export_merged(stores: Sequence[Path], out_root: Path, *, name: str = DEFAULT_NAME, split_file: Path,
                  val_fraction: float | None = None, split_seed: int | None = None, pool: Path | None = None,
                  episodes_per_shard: int = 16, validate: bool = True) -> dict:
    """Export the raw `stores` as one TFDS dataset <out_root>/<name>/1.0.0/ with train and val splits, beside its
    merge.json and dataset_statistics.json. `val_fraction` and `split_seed` draw a new split file (defaults 0.05 and 0);
    an existing `split_file` is reused with its own and they may only repeat them. `pool` (assets/targets.json) fills
    the category, seen and status of targets whose provenance predates the pool. Refuses an existing export."""
    from autofly_ue5.dataset.raw import NAME_RE
    from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES

    if not NAME_RE.match(name):
        raise MergeError(f"dataset name {name!r} is not lowercase snake case starting with a letter (a TFDS name)")
    if not stores:
        raise MergeError("no stores to merge")
    dataset_dir = Path(out_root) / name
    final = dataset_dir / VERSION
    for existing in (final, dataset_dir / MERGE_RECORD, dataset_dir / STATISTICS_FILE):
        if existing.exists():
            raise FileExistsError(f"{existing} exists; export into a new directory")
    loaded = [_load_store(root) for root in stores]
    names = Counter(s["manifest"]["name"] for s in loaded)
    twice = [n for n, c in names.items() if c > 1]
    if twice:
        raise MergeError(f"store {twice[0]} is given twice; each store is merged once")
    _check_definitions(loaded)
    pool_index, pool_record = load_pool_index(pool)
    episodes = _gather(loaded, pool_index, pool_record, validate)
    if not episodes:
        raise MergeError("the stores hold no episodes")

    existing_split = _read_split_file(split_file, val_fraction, split_seed)
    if existing_split is not None:
        val_fraction, split_seed = existing_split["val_fraction"], existing_split["seed"]
    val_fraction = DEFAULT_VAL_FRACTION if val_fraction is None else val_fraction
    split_seed = DEFAULT_SPLIT_SEED if split_seed is None else split_seed
    assignment, split_record = _freeze_split(split_file, existing_split, [e["metadata"] for e in episodes],
                                             val_fraction=val_fraction, seed=split_seed,
                                             stores=[s["manifest"]["name"] for s in loaded])
    by_split = {split: sorted((e for e in episodes if assignment[e["key"]] == split),
                              key=lambda e: _rank(split_seed, "order", e["key"])) for split in SPLITS}

    created_dir = not dataset_dir.exists()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tmp = dataset_dir / f".{VERSION}.partial-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        split_infos, splits_summary = [], {}
        for split in SPLITS:
            items = by_split[split]
            info = write_split_shards(
                tmp, name, split, items,
                lambda e: episode_example(Path(e["store_root"]) / e["path"], e["instruction"], e["metadata"]["file_path"],
                                          _example_metadata(e["metadata"])),
                episodes_per_shard=episodes_per_shard)
            n_shards = len(info["shardLengths"])
            for i, e in enumerate(items):
                e["split"] = split
                e["shard"] = f"{name}-{split}.tfrecord-{i // episodes_per_shard:05d}-of-{n_shards:05d}"
            split_infos.append(info)
            splits_summary[split] = {"episodes": len(items), "records": sum(e["steps"] for e in items),
                                     "shards": n_shards, "bytes": int(info["numBytes"])}
        (tmp / "features.json").write_text(MERGED_FEATURES_TEMPLATE.read_text())
        info = json.loads((TEMPLATES / "dataset_info.json").read_text())
        stores_text = ", ".join(s["manifest"]["name"] for s in loaded)
        info.update(name=name, version=VERSION, splits=split_infos, moduleName="autofly_ue5.dataset.merge",
                    description=(f"AutoFly-format UAV navigation episodes collected in UE5 + Project AirSim, merged from "
                                 f"the raw stores {stores_text}. Splits train and val (val stratified by scene x target, "
                                 f"frozen in {Path(split_file).name}). episode_metadata names each episode's store, "
                                 f"scene, target and pilot. state[0] and state[1] are the distance and bearing to the "
                                 f"target from its true position: privileged. See dataset_card.md."))
        (tmp / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
        train = by_split["train"]
        if not train:
            raise MergeError("the train split is empty")
        actions, states = [], []
        for e in train:
            with np.load(Path(e["store_root"]) / e["path"] / "steps.npz") as steps:  # closed: thousands of episodes
                actions.append(steps["action"])
                states.append(steps["state"])
        statistics = dataset_statistics(name, actions, states)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if created_dir and dataset_dir.is_dir() and not any(dataset_dir.iterdir()):
            dataset_dir.rmdir()
        raise
    write_text_durably(dataset_dir / STATISTICS_FILE, json.dumps(statistics, indent=2) + "\n")
    first = loaded[0]["manifest"]
    record = {
        "format": MERGE_FORMAT, "name": name, "version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "path": str(final), "exporter": {"git_head": _git("rev-parse", "HEAD"),
                                         "tree_dirty": bool(_git("status", "--porcelain"))},
        "record_fields": first["record_fields"], "state_fields": first["state_fields"],
        "instruction_templates": list(INSTRUCTION_TEMPLATES), "metadata_fields": [list(f) for f in METADATA_FIELDS],
        "split": split_record, "splits": splits_summary, "statistics": STATISTICS_FILE,
        "target_pool": pool_record, "stores": [_store_record(s) for s in loaded],
        "episodes": [{k: e[k] for k in ("key", "split", "shard", "steps", "instruction", "store_root", "path", "stratum",
                                        "metadata", "filled_from_pool", "target_name_status")}
                     for split in SPLITS for e in by_split[split]],
    }
    write_text_durably(dataset_dir / MERGE_RECORD, json.dumps(record, indent=1) + "\n")
    return {"path": str(final), "splits": {s: splits_summary[s]["episodes"] for s in SPLITS},
            "records": {s: splits_summary[s]["records"] for s in SPLITS},
            "shards": sum(splits_summary[s]["shards"] for s in SPLITS),
            "bytes": sum(splits_summary[s]["bytes"] for s in SPLITS),
            "merge_record": str(dataset_dir / MERGE_RECORD), "statistics": str(dataset_dir / STATISTICS_FILE),
            "split_file": split_record}
