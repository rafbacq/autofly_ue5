"""Plan 6 (revision 1, item 1): several raw stores become ONE TFDS dataset with train and val splits, AutoFly's four step
fields unchanged and the episode metadata extended; the split is stratified by scene x target, frozen in a file and kept
across top-ups; `dataset_statistics.json` follows OpenVLA's schema on the train split only.

The stores here are synthetic (RawDatasetWriter with fake frames, built to pass the validator) plus, where this host has
it, a few real episodes of the M3 pilot (data/s01_pilot, read only, through a symlinked view)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from autofly_ue5.paths import ROOT, SCENES_DIR

# data/ and runs/tools/ are git-ignored host artefacts; a worktree points this at the main checkout to reach them.
HOST_ROOT = Path(os.environ.get("AUTOFLY_HOST_ROOT", ROOT))
S01_PILOT = HOST_ROOT / "data" / "s01_pilot"
# s01_r2 best_model, which flew the M3 pilot (docs/gates/m3_gate.json)
PILOT_SHA256 = "8c2c7500041907a98d5d696b9007e30578731ff5136f63c384270c0f4fcbf8dc"

SYNTH_TARGETS = (
    {"name": "red sedan", "category": "vehicle", "seen": True, "asset": "tgt_sedan", "scale": [1.568, 1.568, 1.568],
     "material": None, "status": "placeholder"},  # a Kenney stand-in, marked as the pool marks it today
    {"name": "covered car", "category": "vehicle", "seen": True, "asset": "tgt_covered_car", "scale": [0.913, 0.913, 0.913],
     "material": None, "status": "imported"},
    {"name": "blue cylinder", "category": "geometry", "seen": True, "asset": "cylinder", "scale": [1.0, 1.0, 2.0],
     "material": "runtime_blue", "status": "primitive"},
    {"name": "man in red shirt", "category": "human", "seen": True, "asset": "tgt_character_b", "scale": [1.0, 1.0, 1.0],
     "material": None, "status": "imported", "stand_in": True},  # an explicit stand-in mark, should provenance gain one
    {"name": "grey armchair", "category": "furniture", "seen": False, "asset": "tgt_armchair", "scale": [1.0, 1.0, 1.0],
     "material": None, "status": "imported"},
)
# M3's target as its provenance records it: no category, seen or pool status (docs/decisions/2026-10-03-m3-a0-...).
LEGACY_TARGET = {"name": "orange cylinder", "name_status": "placeholder_until_M4", "asset": "cylinder", "material": "orange",
                 "scale": [1.0, 1.0, 2.0]}
POOL_ORANGE_CYLINDER = {"name": "orange cylinder", "category": "geometry", "seen": True, "asset": "cylinder",
                        "scale": [1.0, 1.0, 2.0], "material": "orange", "status": "primitive", "source": None,
                        "height_m": 2.0}


def _frame(i: int, k: int) -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:8, :8] = (i * 37 % 256, k * 11 % 256, 200)  # distinct per episode and record, and tiny as a PNG
    img[100:140, 60:200] = (30, 120 + (i % 100), 40)
    return img


def _episode_arrays(i: int, n: int):
    k = np.arange(n, dtype=np.float32)
    altitude = 2.0 + 0.01 * k + 0.001 * (i % 7)
    state = np.stack([
        20.0 - (20.0 - 4.0) * k / max(n - 1, 1),  # distance: ends 4 m out, inside the 5 m success radius
        0.01 * k - 0.02 * (i % 3),  # bearing
        altitude - 1.47,
        1.5 + 0.05 * k,  # speed
        np.zeros(n), 0.02 * (k % 3) - 0.02,  # vertical velocity, yaw rate
        0.4 * k, 0.05 * k * (i % 2), altitude,  # x, y from the start, altitude
    ], axis=1).astype(np.float32)
    action = np.stack([1.2 + 0.1 * (k % 5) + 0.01 * (i % 4), 0.1 * ((k + i) % 3) - 0.1, 0.05 * (k % 4) - 0.075],
                      axis=1).astype(np.float32)
    times = 1_000_000_000 + 200_000_000 * np.arange(n, dtype=np.int64)
    return state, action, times


def write_synthetic_store(data_root: Path, name: str, *, scene_file: str = "s09_coloured_poles.json",
                          targets=SYNTH_TARGETS, per_target: int = 3, seed_base: int = 498_000_000,
                          pilot_sha256: str = "1" * 64, steps: tuple[int, int] = (4, 8), legacy_target: bool = False,
                          first_index: int = 0) -> Path:
    """A raw store the validator passes: per_target episodes per target, steps[0]..steps[1] records each."""
    from autofly_ue5.dataset.raw import RawDatasetWriter
    from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES
    from autofly_ue5.scenes.model import load_scene_file

    scene = load_scene_file(SCENES_DIR / scene_file)
    targets = (LEGACY_TARGET,) if legacy_target else targets
    provenance = {"expert": {"path": f"runs/expert/{name}/best_model.zip", "sha256": pilot_sha256, "deterministic": False},
                  "platform": {"projectairsim_client": "synthetic", "package_binary_sha256": "5" * 64},
                  "scene_config": {"file": f"scene_autofly_{scene.id}_fast.jsonc", "sha256": "2" * 64},
                  "collector_git_head": "f" * 40, "collector_tree_dirty": False, "reward_version": "synthetic",
                  "dataset": name}
    if not legacy_target:
        provenance["target_pool"] = {"path": "assets/targets.json", "sha256": "3" * 64, "version": 1, "split": scene.split,
                                     "names": [t["name"] for t in targets]}
    card = {"target_names": {t["name"]: ("placeholder_until_M4" if legacy_target else "pool") for t in targets}}
    writer = RawDatasetWriter(data_root, name, scene=scene, provenance=provenance, card=card)
    for j in range(per_target * len(targets)):
        i = first_index + j
        target = targets[j % len(targets)]
        seed = seed_base + i
        n = steps[0] + (i * 7) % (steps[1] - steps[0] + 1)
        state, action, times = _episode_arrays(i, n)
        instruction = INSTRUCTION_TEMPLATES[i % 2].format(target=target["name"], obstacle=scene.instruction_obstacle)
        if legacy_target:
            target_record = {**target, "spawned_as": "target", "xyz": [20.0, 0.0, -1.0]}
        else:
            target_record = {**target, "material_applied": target["material"] or "mesh_default", "spawn_z_ned": 0.0,
                             "name_status": "pool", "spawned_as": f"tgt_{i}", "xyz": [20.0, 0.0, -1.0]}
        poses = [[float(state[r, 6]), float(state[r, 7]), -float(state[r, 8]), 0.0] for r in range(n)]
        writer.write_episode(
            episode_id=f"{scene.id}_{seed}", scene_id=scene.id, split=scene.split, seed=seed, instruction=instruction,
            target_name=target["name"], frames=[_frame(i, r) for r in range(n)], states=state, actions=action,
            sim_time_ns=times,
            provenance={"scene": {"id": scene.id, "file": scene.path, "sha256": scene.sha256},
                        "layout_sha256": "4" * 64, "seed": seed, "target": target_record, "distractors": [],
                        "start_pose": [0.0, 0.0, -2.0, 0.0], "a0": {"mode": "sector8", "start_yaw_rad": 0.0},
                        "movers": [], "mover_positions_per_record": [[] for _ in range(n)], "poses_per_record": poses,
                        "sim_time_ns": [int(t) for t in times], "final_pose": poses[-1], "termination": "success",
                        "collision_source": None, "oob_kind": None, "final_distance_m": 4.0, "deterministic": False,
                        "pilot": "sac_expert"})
    return Path(data_root) / name


def store_subset(src: Path, data_root: Path, n: int) -> Path:
    """A read-only view of the first n episodes of a real store: symlinked episodes and provenance, the manifest's
    counts recomputed for them. Its name stays the store's own."""
    manifest = json.loads((src / "manifest.json").read_text())
    entries = manifest["episodes"][:n]
    dst = Path(data_root) / manifest["name"]
    (dst / "episodes").mkdir(parents=True)
    (dst / "provenance").mkdir()
    for e in entries:
        (dst / e["path"]).symlink_to(src / e["path"], target_is_directory=True)
        (dst / "provenance" / f"{e['id']}.json").symlink_to(src / "provenance" / f"{e['id']}.json")
    counts = {"episodes": len(entries), "records": sum(e["steps"] for e in entries), "rejects": 0, "by_scene": {},
              "by_target": {}}
    for e in entries:
        counts["by_scene"][e["scene"]] = counts["by_scene"].get(e["scene"], 0) + 1
        counts["by_target"][e["target_name"]] = counts["by_target"].get(e["target_name"], 0) + 1
    (dst / "manifest.json").write_text(json.dumps({**manifest, "episodes": entries, "counts": counts}, indent=2))
    return dst


def write_pool(path: Path, entries=(POOL_ORANGE_CYLINDER,)) -> Path:
    path.write_text(json.dumps({"version": 1, "targets": list(entries)}))
    return path


# --------------------------------------------------------------------------------------------------------
# A minimal tf.train.Example decoder: BytesList, packed FloatList and packed Int64List (what TFDS reads).
# --------------------------------------------------------------------------------------------------------
def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value, i


def _fields(buf: bytes):
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        assert key & 7 == 2, "only length-delimited fields are written"
        n, i = _varint(buf, i)
        yield key >> 3, buf[i:i + n]
        i += n


def decode_example(raw: bytes) -> dict:
    out = {}
    (_, features), = list(_fields(raw))
    for _, entry in _fields(features):
        parts = dict(_fields(entry))
        key = parts[1].decode()
        (kind, payload), = list(_fields(parts[2]))
        if kind == 1:
            out[key] = [v for _, v in _fields(payload)]
        elif kind == 2:
            (_, packed), = list(_fields(payload)) if payload else [(1, b"")]
            out[key] = np.frombuffer(packed, dtype="<f4")
        else:
            (_, packed), = list(_fields(payload)) if payload else [(1, b"")]
            values, i = [], 0
            while i < len(packed):
                v, i = _varint(packed, i)
                values.append(v - (1 << 64) if v >= 1 << 63 else v)
            out[key] = values
    return out


def read_split(version_dir: Path, name: str, split: str) -> list[dict]:
    from autofly_ue5.dataset.rlds import read_records

    shards = sorted(version_dir.glob(f"{name}-{split}.tfrecord-*"))
    return [decode_example(r) for s in shards for r in read_records(s)]


def _export(tmp_path, stores, **kwargs):
    from autofly_ue5.dataset.merge import export_merged

    kwargs.setdefault("split_file", tmp_path / "split.json")
    kwargs.setdefault("pool", None)
    return export_merged(stores, tmp_path / "out", **kwargs)


# --------------------------------------------------------------------------------------------------------
# The writer
# --------------------------------------------------------------------------------------------------------
def test_int64_features_are_written_as_packed_varints_including_negative_values():
    from autofly_ue5.dataset.rlds import encode_example

    raw = encode_example({"a": ("int64", [0, 1, 300, 498_000_123, -5]), "b": ("bytes", [b"x"])})
    assert decode_example(raw) == {"a": [0, 1, 300, 498_000_123, -5], "b": [b"x"]}


def test_the_merged_template_keeps_the_release_steps_and_adds_exactly_the_metadata_fields():
    from autofly_ue5.dataset.merge import MERGED_FEATURES_TEMPLATE, METADATA_FIELDS
    from autofly_ue5.dataset.rlds import TEMPLATES

    merged = json.loads(MERGED_FEATURES_TEMPLATE.read_text())["featuresDict"]["features"]
    release = json.loads((TEMPLATES / "features.json").read_text())["featuresDict"]["features"]
    assert merged["steps"] == release["steps"], "AutoFly's four step fields, unchanged"
    meta = merged["episode_metadata"]["featuresDict"]["features"]
    assert meta["file_path"] == release["episode_metadata"]["featuresDict"]["features"]["file_path"]
    classes = {"text": ("tensorflow_datasets.core.features.text_feature.Text", None),
               "bool": ("tensorflow_datasets.core.features.scalar.Scalar", "bool"),
               "int64": ("tensorflow_datasets.core.features.scalar.Scalar", "int64")}
    assert set(meta) == {name for name, _ in METADATA_FIELDS}
    for name, kind in METADATA_FIELDS:
        cls, dtype = classes[kind]
        assert meta[name]["pythonClassName"] == cls, name
        if dtype is not None:
            assert meta[name]["tensor"] == {"dtype": dtype, "encoding": "none", "shape": {}}, name
    # scripts/make_rlds_templates.py generates the template with TFDS from its own copy of the list: the same list
    from scripts.make_rlds_templates import METADATA

    assert tuple(METADATA) == tuple(METADATA_FIELDS)


def test_a_merged_export_holds_every_store_s_episodes_with_the_release_steps_and_the_new_metadata(tmp_path):
    from autofly_ue5.dataset.merge import METADATA_FIELDS, STEP_KEYS
    from autofly_ue5.dataset.rlds import VERSION

    a = write_synthetic_store(tmp_path / "data", "synth_a", per_target=4)
    b = write_synthetic_store(tmp_path / "data", "synth_b", scene_file="s07_stacked_boxes.json", per_target=2,
                              seed_base=497_000_000, pilot_sha256="9" * 64)
    result = _export(tmp_path, [a, b], name="autofly_ue5_test", episodes_per_shard=4)
    version_dir = tmp_path / "out" / "autofly_ue5_test" / VERSION
    assert result["path"] == str(version_dir)
    n_a, n_b = 4 * len(SYNTH_TARGETS), 2 * len(SYNTH_TARGETS)
    assert result["splits"]["train"] + result["splits"]["val"] == n_a + n_b
    assert result["splits"]["val"] == math.floor(0.05 * (n_a + n_b) + 0.5)

    record = json.loads((tmp_path / "out" / "autofly_ue5_test" / "merge.json").read_text())
    by_path = {e["metadata"]["file_path"]: e for e in record["episodes"]}
    assert len(by_path) == n_a + n_b
    stores = {"synth_a": a, "synth_b": b}
    seen_paths = set()
    for split in ("train", "val"):
        examples = read_split(version_dir, "autofly_ue5_test", split)
        assert len(examples) == result["splits"][split]
        for ex in examples:
            assert set(ex) == set(STEP_KEYS) | {f"episode_metadata/{f}" for f, _ in METADATA_FIELDS}
            file_path = ex["episode_metadata/file_path"][0].decode()
            entry = by_path[file_path]
            assert entry["split"] == split
            seen_paths.add(file_path)
            store, _, episode_id = file_path.partition("/episodes/")
            root = stores[store]
            manifest_entry = next(e for e in json.loads((root / "manifest.json").read_text())["episodes"]
                                  if e["id"] == episode_id)
            n = manifest_entry["steps"]
            steps = np.load(root / manifest_entry["path"] / "steps.npz")
            frames = sorted((root / manifest_entry["path"] / "frames").glob("*.png"))
            assert ex["steps/observation/image_0"] == [f.read_bytes() for f in frames]
            assert np.array_equal(ex["steps/observation/state"].reshape(n, 9), steps["state"])
            assert np.array_equal(ex["steps/action"].reshape(n, 3), steps["action"])
            assert ex["steps/language_instruction"] == [manifest_entry["instruction"].encode()] * n
            meta = {f: ex[f"episode_metadata/{f}"][0] for f, _ in METADATA_FIELDS}
            assert meta["episode_id"].decode() == episode_id and meta["store"].decode() == store
            assert meta["scene"].decode() == manifest_entry["scene"] and meta["split_of_scene"] == b"train"
            assert meta["seed"] == manifest_entry["seed"]
            assert meta["target_name"].decode() == manifest_entry["target_name"]
            target = next(t for t in SYNTH_TARGETS if t["name"] == manifest_entry["target_name"])
            assert meta["target_category"].decode() == target["category"]
            assert meta["target_seen"] == int(target["seen"]) and meta["target_status"].decode() == target["status"]
            assert meta["target_stand_in"] == int(target["name"] in ("red sedan", "man in red shirt"))
            assert meta["pilot"].decode() == ("1" * 64 if store == "synth_a" else "9" * 64)
    assert seen_paths == set(by_path)

    info = json.loads((version_dir / "dataset_info.json").read_text())
    assert info["name"] == "autofly_ue5_test" and info["version"] == VERSION
    assert {s["name"] for s in info["splits"]} == {"train", "val"}
    for s in info["splits"]:
        shards = sorted(version_dir.glob(f"autofly_ue5_test-{s['name']}.tfrecord-*-of-*"))
        assert sum(int(n) for n in s["shardLengths"]) == result["splits"][s["name"]]
        assert len(shards) == len(s["shardLengths"]) and all(int(n) <= 4 for n in s["shardLengths"])
        assert int(s["numBytes"]) == sum(p.stat().st_size for p in shards)
    from autofly_ue5.dataset.merge import MERGED_FEATURES_TEMPLATE

    assert json.loads((version_dir / "features.json").read_text()) == json.loads(MERGED_FEATURES_TEMPLATE.read_text())


def test_the_merge_record_names_each_store_s_provenance(tmp_path):
    a = write_synthetic_store(tmp_path / "data", "synth_a", per_target=2)
    _export(tmp_path, [a], name="autofly_ue5_test")
    record = json.loads((tmp_path / "out" / "autofly_ue5_test" / "merge.json").read_text())
    (store,) = record["stores"]
    assert store["name"] == "synth_a" and store["root"] == str(a.resolve())
    assert store["episodes"] == 2 * len(SYNTH_TARGETS) and store["validation"]["pass"] is True
    assert store["provenance"]["pilots"] == {"1" * 64: {"paths": ["runs/expert/synth_a/best_model.zip"],
                                                       "episodes": 2 * len(SYNTH_TARGETS), "deterministic": [False]}}
    assert store["provenance"]["scene_configs"] == {"scene_autofly_s09_fast.jsonc": ["2" * 64]}
    assert store["provenance"]["collector_git_heads"] == {"f" * 40: 2 * len(SYNTH_TARGETS)}
    assert store["provenance"]["target_pools"] == {"3" * 64: ["assets/targets.json"]}
    assert record["state_fields"][0]["name"] == "distance_to_target_m" and len(record["state_fields"]) == 9
    assert record["name"] == "autofly_ue5_test" and record["split"]["val_fraction"] == 0.05


def test_the_metadata_s_stand_in_mark_is_read_when_present_and_derived_from_the_placeholder_status_otherwise():
    from autofly_ue5.dataset.merge import stand_in_mark

    assert stand_in_mark({"status": "placeholder"}) is True
    assert stand_in_mark({"status": "imported"}) is False
    assert stand_in_mark({"status": "imported", "stand_in": True}) is True
    assert stand_in_mark({"status": "placeholder", "stand_in": False}) is False, "an explicit mark wins"
    assert stand_in_mark({"status": "primitive"}) is False


def test_m3_s_target_takes_category_seen_and_status_from_the_pool_entry_with_the_same_object(tmp_path):
    legacy = write_synthetic_store(tmp_path / "data", "legacy", scene_file="s01_white_pillars.json", legacy_target=True,
                                   per_target=3)
    pool = write_pool(tmp_path / "targets.json")
    _export(tmp_path, [legacy], name="autofly_ue5_test", pool=pool)
    record = json.loads((tmp_path / "out" / "autofly_ue5_test" / "merge.json").read_text())
    for e in record["episodes"]:
        meta = e["metadata"]
        assert (meta["target_category"], meta["target_seen"], meta["target_status"], meta["target_stand_in"]) == (
            "geometry", True, "primitive", False)
        assert e["filled_from_pool"] == ["target_category", "target_seen", "target_status"]
        assert e["target_name_status"] == "placeholder_until_M4"
    (store,) = record["stores"]
    assert store["filled_from_pool"]["episodes"] == 3 and store["filled_from_pool"]["pool"]["path"] == str(pool)


def test_a_target_the_pool_cannot_vouch_for_is_refused(tmp_path):
    from autofly_ue5.dataset.merge import MergeError

    legacy = write_synthetic_store(tmp_path / "data", "legacy", scene_file="s01_white_pillars.json", legacy_target=True,
                                   per_target=1)
    with pytest.raises(MergeError, match="target_category"):
        _export(tmp_path, [legacy], name="autofly_ue5_test", pool=None)
    other = write_pool(tmp_path / "targets.json", [{**POOL_ORANGE_CYLINDER, "material": "runtime_orange"}])
    with pytest.raises(MergeError, match="same asset, material and scale"):
        _export(tmp_path, [legacy], name="autofly_ue5_test", pool=other)
    assert not (tmp_path / "out" / "autofly_ue5_test" / "1.0.0").exists()


def test_stores_whose_step_definitions_disagree_are_refused(tmp_path):
    from autofly_ue5.dataset.merge import MergeError

    a = write_synthetic_store(tmp_path / "data", "synth_a", per_target=1)
    b = write_synthetic_store(tmp_path / "data", "synth_b", scene_file="s07_stacked_boxes.json", per_target=1,
                              seed_base=497_000_000)
    manifest = json.loads((b / "manifest.json").read_text())
    manifest["state_fields"][2]["definition"] = "altitude above ground, m"
    (b / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(MergeError, match="state_fields"):
        _export(tmp_path, [a, b], name="autofly_ue5_test")
    manifest["state_fields"] = json.loads((a / "manifest.json").read_text())["state_fields"]
    manifest["record_fields"]["action"] = "float32[4]"
    (b / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(MergeError, match="record_fields"):
        _export(tmp_path, [a, b], name="autofly_ue5_test")
    assert not (tmp_path / "split.json").exists(), "nothing is frozen for a merge that never started"


def test_a_store_the_validator_fails_is_refused(tmp_path):
    from autofly_ue5.dataset.merge import MergeError

    a = write_synthetic_store(tmp_path / "data", "synth_a", per_target=1)
    entry = json.loads((a / "manifest.json").read_text())["episodes"][0]
    steps = dict(np.load(a / entry["path"] / "steps.npz"))
    steps["action"][0, 0] = 3.0  # outside the action space
    np.savez(a / entry["path"] / "steps.npz", **steps)
    with pytest.raises(MergeError, match="validator"):
        _export(tmp_path, [a], name="autofly_ue5_test")


def test_the_same_scene_and_seed_in_two_stores_is_refused(tmp_path):
    from autofly_ue5.dataset.merge import MergeError

    a = write_synthetic_store(tmp_path / "data", "synth_a", per_target=1)
    b = write_synthetic_store(tmp_path / "data", "synth_b", per_target=1)  # the same scene and seed slice
    with pytest.raises(MergeError, match="same scene and seed"):
        _export(tmp_path, [a, b], name="autofly_ue5_test")
    with pytest.raises(MergeError, match="twice"):
        _export(tmp_path, [a, a], name="autofly_ue5_test")


def test_an_existing_export_is_refused_and_a_failed_one_leaves_no_version_directory(tmp_path, monkeypatch):
    from autofly_ue5.dataset import merge

    a = write_synthetic_store(tmp_path / "data", "synth_a", per_target=2)
    _export(tmp_path, [a], name="autofly_ue5_test")
    with pytest.raises(FileExistsError):
        _export(tmp_path, [a], name="autofly_ue5_test")

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(merge, "episode_example", broken)
    with pytest.raises(OSError, match="disk full"):
        _export(tmp_path, [a], name="autofly_ue5_other")
    other = tmp_path / "out" / "autofly_ue5_other"
    assert not (other / "1.0.0").exists() and not (other / "merge.json").exists()
    assert not other.exists() or not any(p.name.startswith(".") for p in other.iterdir()), "no partial directory left"


# --------------------------------------------------------------------------------------------------------
# The split: stratified by scene x target, frozen in a file, kept across top-ups
# --------------------------------------------------------------------------------------------------------
def _strata(keys_by_stratum):
    return {s: list(v) for s, v in keys_by_stratum.items()}


def test_a_fresh_split_takes_the_fraction_spread_over_scene_x_target_as_evenly_as_whole_episodes_allow():
    from autofly_ue5.dataset.merge import assign_splits

    sizes = {"s01|orange cylinder": 100, "s09|red sedan": 23, "s09|blue cylinder": 17, "s07|covered car": 41,
             "s07|grey armchair": 2, "s02|man in red shirt": 9}
    strata = {s: [f"{s}#{i}" for i in range(n)] for s, n in sizes.items()}
    assignment = assign_splits(strata, {}, val_fraction=0.05, seed=0)
    total = sum(sizes.values())
    assert set(assignment) == {k for keys in strata.values() for k in keys}
    assert sum(v == "val" for v in assignment.values()) == math.floor(0.05 * total + 0.5)
    for s, keys in strata.items():
        val = sum(assignment[k] == "val" for k in keys)
        assert math.floor(0.05 * len(keys)) <= val <= math.ceil(0.05 * len(keys)), s
    again = assign_splits(strata, {}, val_fraction=0.05, seed=0)
    assert again == assignment, "deterministic for a seed"
    other = assign_splits(strata, {}, val_fraction=0.05, seed=1)
    assert other != assignment and sum(v == "val" for v in other.values()) == sum(v == "val" for v in assignment.values())


def test_a_stratum_never_gives_up_its_only_episode():
    from autofly_ue5.dataset.merge import assign_splits

    strata = {f"s0{i}|t": [f"s0{i}|t#0"] for i in range(40)}  # forty one-episode strata: 2 val would be due
    assignment = assign_splits(strata, {}, val_fraction=0.05, seed=0)
    assert set(assignment.values()) == {"train"}


def test_val_takes_at_least_one_episode_and_an_export_whose_val_would_be_empty_is_refused(tmp_path):
    # TFDS cannot read an empty split ("Shard empty", TFDS 4.9.3 and 4.9.10, 2026-10-08)
    from autofly_ue5.dataset.merge import MergeError, assign_splits

    strata = {"s09|a": ["s09|a#0", "s09|a#1"], "s09|b": ["s09|b#0", "s09|b#1"]}  # 4 x 0.05 rounds to 0
    assert sum(v == "val" for v in assign_splits(strata, {}, val_fraction=0.05, seed=0).values()) == 1
    single = write_synthetic_store(tmp_path / "data", "synth_a", per_target=1)  # five one-episode strata
    with pytest.raises(MergeError, match="val split would be empty"):
        _export(tmp_path, [single], name="autofly_ue5_test")
    assert not (tmp_path / "out" / "autofly_ue5_test" / "1.0.0").exists()
    with pytest.raises(MergeError, match="val_fraction"):
        assign_splits(strata, {}, val_fraction=0.0, seed=0)


def test_a_top_up_keeps_every_earlier_assignment_and_tops_the_fraction_up_from_the_new_episodes():
    from autofly_ue5.dataset.merge import assign_splits

    first = {"s01|a": [f"s01|a#{i}" for i in range(30)], "s09|b": [f"s09|b#{i}" for i in range(30)]}
    frozen = assign_splits(first, {}, val_fraction=0.05, seed=0)
    second = {"s01|a": first["s01|a"] + [f"s01|a#{i}" for i in range(30, 70)], "s09|b": first["s09|b"],
              "s07|c": [f"s07|c#{i}" for i in range(50)]}
    topped = assign_splits(second, frozen, val_fraction=0.05, seed=0)
    assert all(topped[k] == v for k, v in frozen.items()), "earlier assignments are frozen"
    assert sum(v == "val" for v in topped.values()) == math.floor(0.05 * 150 + 0.5)
    for s, keys in second.items():
        val = sum(topped[k] == "val" for k in keys)
        assert math.floor(0.05 * len(keys)) <= val <= math.ceil(0.05 * len(keys)), s


def test_an_existing_split_file_is_reused_never_redrawn_and_only_extended(tmp_path):
    from autofly_ue5.dataset.merge import MergeError

    a = write_synthetic_store(tmp_path / "data", "synth_a", per_target=4)
    split_file = tmp_path / "split.json"
    _export(tmp_path, [a], name="autofly_ue5_one", split_file=split_file)
    frozen = json.loads(split_file.read_text())
    assert frozen["val_fraction"] == 0.05 and frozen["seed"] == 0 and frozen["stratify_by"] == ["scene", "target_name"]
    assert len(frozen["episodes"]) == 4 * len(SYNTH_TARGETS)

    _export(tmp_path, [a], name="autofly_ue5_two", split_file=split_file)  # the same stores: the same file
    assert json.loads(split_file.read_text()) == frozen

    b = write_synthetic_store(tmp_path / "data", "synth_b", scene_file="s07_stacked_boxes.json", per_target=3,
                              seed_base=497_000_000)
    _export(tmp_path, [a, b], name="autofly_ue5_three", split_file=split_file)
    topped = json.loads(split_file.read_text())
    assert {k: v for k, v in topped["episodes"].items() if k in frozen["episodes"]} == frozen["episodes"]
    assert len(topped["episodes"]) == 7 * len(SYNTH_TARGETS) and len(topped["updates"]) == 2
    record = json.loads((tmp_path / "out" / "autofly_ue5_three" / "merge.json").read_text())
    assert {e["key"]: e["split"] for e in record["episodes"]} == {k: v["split"] for k, v in topped["episodes"].items()}

    with pytest.raises(MergeError, match="val_fraction"):
        _export(tmp_path, [a], name="autofly_ue5_four", split_file=split_file, val_fraction=0.1)
    with pytest.raises(MergeError, match="seed"):
        _export(tmp_path, [a], name="autofly_ue5_four", split_file=split_file, split_seed=7)


# --------------------------------------------------------------------------------------------------------
# dataset_statistics.json: OpenVLA's schema, the train split only
# --------------------------------------------------------------------------------------------------------
def test_statistics_follow_openvla_s_schema_over_the_train_split_only(tmp_path):
    a = write_synthetic_store(tmp_path / "data", "synth_a", per_target=6)
    result = _export(tmp_path, [a], name="autofly_ue5_test")
    stats = json.loads((tmp_path / "out" / "autofly_ue5_test" / "dataset_statistics.json").read_text())
    assert list(stats) == ["autofly_ue5_test"]
    s = stats["autofly_ue5_test"]
    assert set(s) == {"action", "proprio", "num_transitions", "num_trajectories"}
    record = json.loads((tmp_path / "out" / "autofly_ue5_test" / "merge.json").read_text())
    train = [e for e in record["episodes"] if e["split"] == "train"]
    assert result["splits"]["val"] >= 1 and s["num_trajectories"] == len(train) == result["splits"]["train"]
    # Accumulated in float64: numpy's float32 sums along axis 0 (OpenVLA's get_dataset_statistics) drift on long data
    actions = np.concatenate([np.load(a / e["path"] / "steps.npz")["action"] for e in train]).astype(np.float64)
    states = np.concatenate([np.load(a / e["path"] / "steps.npz")["state"] for e in train]).astype(np.float64)
    assert s["num_transitions"] == len(actions) and isinstance(s["num_transitions"], int)
    for key, values, dims in (("action", actions, 3), ("proprio", states, 9)):
        block = s[key]
        assert set(block) == {"mean", "std", "max", "min", "q01", "q99", "mask"}
        assert block["mask"] == [True] * dims
        for stat, ref in (("mean", values.mean(0)), ("std", values.std(0)), ("max", values.max(0)),
                          ("min", values.min(0)), ("q01", np.quantile(values, 0.01, axis=0)),
                          ("q99", np.quantile(values, 0.99, axis=0))):
            assert len(block[stat]) == dims and all(isinstance(v, float) for v in block[stat])
            assert np.allclose(block[stat], ref, rtol=1e-6, atol=1e-7), (key, stat)
    every = np.concatenate([np.load(a / e["path"] / "steps.npz")["action"] for e in record["episodes"]])
    assert len(every) > len(actions), "val's records are left out"


# --------------------------------------------------------------------------------------------------------
# The real M3 pilot (read only), merged with a synthetic store
# --------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (S01_PILOT / "manifest.json").is_file(), reason="data/s01_pilot is git-ignored host data")
def test_real_pilot_episodes_merge_with_a_synthetic_store(tmp_path):
    view = store_subset(S01_PILOT, tmp_path / "view", 2)
    synth = write_synthetic_store(tmp_path / "data", "synth_a", per_target=2)
    pool = write_pool(tmp_path / "targets.json")
    result = _export(tmp_path, [view, synth], name="autofly_ue5_test", pool=pool)
    assert result["splits"]["train"] + result["splits"]["val"] == 2 + 2 * len(SYNTH_TARGETS)
    record = json.loads((tmp_path / "out" / "autofly_ue5_test" / "merge.json").read_text())
    pilot = [e for e in record["episodes"] if e["metadata"]["store"] == "s01_pilot"]
    assert len(pilot) == 2
    for e in pilot:
        meta = e["metadata"]
        assert meta["scene"] == "s01" and meta["split_of_scene"] == "train" and meta["pilot"] == PILOT_SHA256
        assert meta["target_name"] == "orange cylinder" and meta["target_category"] == "geometry"
        assert meta["target_seen"] is True and meta["target_status"] == "primitive" and meta["target_stand_in"] is False
        assert meta["file_path"] == f"s01_pilot/episodes/{meta['episode_id']}"
    examples = {ex["episode_metadata/file_path"][0].decode(): ex for split in ("train", "val")
                for ex in read_split(Path(result["path"]), "autofly_ue5_test", split)}
    for e in pilot:
        ex = examples[e["metadata"]["file_path"]]
        frames = sorted((S01_PILOT / "episodes" / e["metadata"]["episode_id"] / "frames").glob("*.png"))
        assert ex["steps/observation/image_0"][0] == frames[0].read_bytes()
        assert len(ex["steps/observation/image_0"]) == len(frames)
    (s01,) = [s for s in record["stores"] if s["name"] == "s01_pilot"]
    assert s01["provenance"]["pilots"][PILOT_SHA256]["paths"] == ["runs/expert/s01_r2/best/best_model.zip"]
