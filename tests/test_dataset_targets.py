"""The target pool's file and its splits (plan 5 C5, "Design (2026-10-08)"; spec §6.4, §9 step 1), against a small pool
of engine primitives (and one imported base-pivot model) the registry already holds: the mechanism needs no downloaded
asset, and the real `assets/targets.json` is checked by its own tests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest


def _entry(name, category, asset, scale, material, *, seen=True, status="primitive", height_m=None, source=None):
    return {"name": name, "category": category, "seen": seen, "asset": asset, "scale": scale, "material": material,
            "status": status, "source": source, "height_m": height_m}


def primitive_pool_entries() -> list[dict]:
    """Six seen and one unseen usable entry, and one candidate with nothing imported yet. Materials are the registry's
    base `UMaterial`s ("orange", "grid") or none (the mesh's own); the palette's colours are material instances, which a
    runtime spawn cannot apply (expert/episode.py `_runtime_material_path`)."""
    return [
        _entry("orange cylinder", "geometry", "cylinder", [1.0, 1.0, 2.0], "orange", height_m=2.0),
        _entry("orange cube", "geometry", "crate", [1.2, 1.2, 1.2], "orange", height_m=1.2),
        _entry("grid cube", "geometry", "crate", [1.5, 1.5, 1.5], "grid", height_m=1.5),
        _entry("grid pole", "geometry", "cylinder", [0.6, 0.6, 3.0], "grid", height_m=3.0),
        _entry("white box", "other", "crate", [2.0, 1.0, 1.0], None, height_m=1.0),
        _entry("grey boulder", "other", "boulder_01", [1.5, 1.5, 1.5], None, status="imported", height_m=1.51,
               source={"source": "polyhaven", "id": "boulder_01", "licence": "CC0-1.0",
                       "page": "https://polyhaven.com/a/boulder_01"}),
        _entry("orange slab", "geometry", "crate", [2.0, 2.0, 0.5], "orange", seen=False, height_m=0.5),
        _entry("blue hatchback", "vehicle", None, None, None, status="candidate", height_m=1.45),
    ]


SEEN_USABLE = ("orange cylinder", "orange cube", "grid cube", "grid pole", "white box", "grey boulder")


def write_pool(path: Path, entries: list[dict] | None = None, *, version=1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": version, "targets": entries if entries is not None else primitive_pool_entries()},
                               indent=1) + "\n")
    return path


def primitive_pool(tmp_path: Path):
    from autofly_ue5.dataset.targets import TargetPool

    return TargetPool.load(write_pool(tmp_path / "targets.json"))


def test_a_pool_file_loads_its_entries_in_file_order_as_frozen_records(tmp_path):
    from autofly_ue5.dataset.targets import TargetEntry, TargetPool

    path = write_pool(tmp_path / "targets.json")
    pool = TargetPool.load(path)
    assert [e.name for e in pool.entries] == [e["name"] for e in primitive_pool_entries()]
    assert pool.entries[0] == TargetEntry(name="orange cylinder", category="geometry", seen=True, asset="cylinder",
                                          scale=(1.0, 1.0, 2.0), material="orange", status="primitive", source=None,
                                          height_m=2.0)
    assert pool.entries[5].source["licence"] == "CC0-1.0" and pool.entries[7].asset is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        pool.entries[0].name = "blue cone"
    assert pool.sha256 == hashlib.sha256(path.read_bytes()).hexdigest() and pool.path == str(path)
    assert pool.entries[0].to_json() == {"name": "orange cylinder", "category": "geometry", "seen": True,
                                         "asset": "cylinder", "scale": [1.0, 1.0, 2.0], "material": "orange",
                                         "status": "primitive"}


def test_each_split_draws_from_its_own_usable_entries_in_file_order(tmp_path):
    pool = primitive_pool(tmp_path)
    # Spec §9 step 1: seen targets for train and test_seen, seen or unseen for test_unseen; a candidate is never drawn.
    assert tuple(e.name for e in pool.for_split("train")) == SEEN_USABLE
    assert tuple(e.name for e in pool.for_split("test_seen")) == SEEN_USABLE
    assert tuple(e.name for e in pool.for_split("test_unseen")) == SEEN_USABLE + ("orange slab",)
    with pytest.raises(ValueError, match="split"):
        pool.for_split("validation")


def test_a_split_with_fewer_than_six_usable_entries_is_refused(tmp_path):
    # One target and up to five distractors, all different entries (spec §9 step 1: 3-5 distractors).
    from autofly_ue5.dataset.targets import TargetPool, TargetPoolError

    entries = [e for e in primitive_pool_entries() if e["name"] != "white box"]
    pool = TargetPool.load(write_pool(tmp_path / "targets.json", entries))
    with pytest.raises(TargetPoolError, match="6"):
        pool.for_split("train")
    assert len(pool.for_split("test_unseen")) == 6, "five seen plus the unseen one are enough there"
    assert issubclass(TargetPoolError, ValueError)


def _with(index: int, **changes):
    entries = primitive_pool_entries()
    entries[index] = {**entries[index], **changes}
    return entries


def _without_key(index: int, key: str):
    entries = primitive_pool_entries()
    del entries[index][key]
    return entries


@pytest.mark.parametrize("entries, version, match", [
    (None, 2, "version"),
    (_with(1, name="orange cylinder"), 1, "unique"),
    (_with(1, name="Orange Cylinder"), 1, "unique"),  # the detector lower-cases its query: one phrase, two entries
    (_with(0, category="animal"), 1, "category"),
    (_with(0, seen="yes"), 1, "seen"),
    (_with(0, seen=1), 1, "seen"),
    (_with(0, status="ready"), 1, "status"),
    (_without_key(0, "height_m"), 1, "height_m"),
    (_with(0, name=""), 1, "name"),
    (_with(0, name=" orange cylinder"), 1, "name"),
    (_with(0, name="orange cylinder. white pillar"), 1, "period"),
    (_with(0, asset=None), 1, "asset"),
    (_with(0, asset="sofa_01"), 1, "registry"),
    (_with(0, scale=None), 1, "scale"),
    (_with(0, scale=[1.0, 1.0]), 1, "scale"),
    (_with(0, scale=[1.0, -1.0, 1.0]), 1, "scale"),
    (_with(0, material="purple"), 1, "material"),
    (_with(0, material="red"), 1, "base UMaterial"),  # a palette colour: the runtime would leave the mesh's own
    (_with(0, height_m=-2.0), 1, "height_m"),
    (_with(0, height_m="2 m"), 1, "height_m"),
    (_with(0, source="polyhaven"), 1, "source"),
    (_with(0, asset="island_tree_02", status="imported"), 1, "cards"),
])
def test_a_malformed_pool_is_refused_with_a_reason(tmp_path, entries, version, match):
    from autofly_ue5.dataset.targets import TargetPool, TargetPoolError

    path = write_pool(tmp_path / "targets.json", entries, version=version)
    with pytest.raises(TargetPoolError, match=match):
        TargetPool.load(path)


def test_a_candidate_may_name_an_asset_not_imported_yet(tmp_path):
    from autofly_ue5.dataset.targets import TargetPool

    entries = _with(7, asset="hatchback_01", scale=[1.0, 1.0, 1.0], material="red")
    pool = TargetPool.load(write_pool(tmp_path / "targets.json", entries))
    assert pool.entries[7].asset == "hatchback_01" and not pool.entries[7].usable


def test_the_pool_s_record_names_its_file_and_a_split_s_entries(tmp_path):
    pool = primitive_pool(tmp_path)
    record = pool.to_json("test_unseen")
    assert record == {"path": pool.path, "sha256": pool.sha256, "version": 1, "split": "test_unseen",
                      "names": list(SEEN_USABLE) + ["orange slab"]}
    assert json.loads(json.dumps(record)) == record


def test_an_entry_waiting_for_a_manual_download_loads_and_is_never_drawn(tmp_path):
    # assets/targets.json marks the Fab-only targets "manual" (tests/test_targets_pool.py): a status, not a refusal.
    from autofly_ue5.dataset.targets import TargetPool

    entries = _with(7, status="manual")
    pool = TargetPool.load(write_pool(tmp_path / "targets.json", entries))
    assert not pool.entries[7].usable and "blue hatchback" not in {e.name for e in pool.for_split("test_unseen")}


# --------------------------------------------------------------------------------------------------------
# Per-scene exclusions (plan 6, step 2c and revision 1 item 7): a pool entry that looks like a scene's own obstacles is
# never drawn there, so the instruction's target is never one of the pillars. A table beside the pool file, keyed by
# registry asset (the shape), with a reason per asset.
# --------------------------------------------------------------------------------------------------------
def wide_pool_entries() -> list[dict]:
    """The primitive pool plus a cone and two spheres: nine seen usable entries over four meshes, so a scene can lose
    a shape and still fill an episode."""
    return primitive_pool_entries() + [
        _entry("orange cone", "geometry", "cone", [1.0, 1.0, 1.5], "orange", height_m=1.5),
        _entry("grid sphere", "geometry", "sphere", [1.2, 1.2, 1.2], "grid", height_m=1.2),
        _entry("orange sphere", "geometry", "sphere", [1.2, 1.2, 1.2], "orange", height_m=1.2),
    ]


WIDE_SEEN = SEEN_USABLE + ("orange cone", "grid sphere", "orange sphere")
TABLE = {"version": 1, "scenes": {"s07": {"crate": "s07's obstacles are crates"},
                                  "s01": {"cylinder": "s01's obstacles are cylinders"}}}


def write_table(path: Path, table: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(TABLE if table is None else table, indent=1) + "\n")
    return path


def wide_pool(tmp_path: Path, table: dict | None = None):
    """The wide pool with an exclusion table beside it (the pool's own file name, target_exclusions.json)."""
    from autofly_ue5.dataset.targets import TargetPool

    write_table(tmp_path / "target_exclusions.json", table)
    return TargetPool.load(write_pool(tmp_path / "targets.json", wide_pool_entries()))


def test_a_scene_s_exclusions_drop_the_named_assets_after_the_split_filter_in_file_order(tmp_path):
    pool = wide_pool(tmp_path)
    assert pool.for_split("train", scene_id="s07") == tuple(
        e for e in pool.for_split("train") if e.asset != "crate"), "file order kept: the sampler's draws index it"
    assert [e.name for e in pool.for_split("train", scene_id="s07")] == [
        "orange cylinder", "grid pole", "grey boulder", "orange cone", "grid sphere", "orange sphere"]
    assert [e.name for e in pool.for_split("train", scene_id="s01")] == [
        "orange cube", "grid cube", "white box", "grey boulder", "orange cone", "grid sphere", "orange sphere"]
    assert "orange slab" not in {e.name for e in pool.for_split("test_unseen", scene_id="s07")}, "unseen entries too"
    # a scene the table does not name, and no scene at all, exclude nothing
    assert tuple(e.name for e in pool.for_split("train", scene_id="s02")) == WIDE_SEEN
    assert tuple(e.name for e in pool.for_split("train")) == WIDE_SEEN
    # s01d flies s01's level but is its own scene: only an entry of its own excludes anything there
    assert tuple(e.name for e in pool.for_split("train", scene_id="s01d")) == WIDE_SEEN


def test_the_six_entry_floor_holds_after_a_scene_s_exclusions(tmp_path):
    from autofly_ue5.dataset.targets import TargetPoolError

    pool = wide_pool(tmp_path, {"version": 1, "scenes": {"s01": {"cylinder": "pillars", "sphere": "stones"}}})
    with pytest.raises(TargetPoolError, match="s01"):
        pool.for_split("train", scene_id="s01")  # 9 seen less 2 cylinders and 2 spheres: 5
    assert len(pool.for_split("test_unseen", scene_id="s01")) == 6, "the unseen slab makes six there"


def test_without_a_table_beside_the_pool_nothing_is_excluded(tmp_path):
    from autofly_ue5.dataset.targets import TargetPool

    pool = TargetPool.load(write_pool(tmp_path / "targets.json", wide_pool_entries()))
    assert pool.exclusions is None and tuple(e.name for e in pool.for_split("train", scene_id="s07")) == WIDE_SEEN


@pytest.mark.parametrize("table, match", [
    ({"version": 2, "scenes": {}}, "version"),
    ({"version": 1, "scenes": []}, "scenes"),
    ({"version": 1, "scenes": {"s42": {"crate": "no such scene"}}}, "s42"),
    ({"version": 1, "scenes": {"s07": {"tgt_barrel_01": "not an asset of this pool"}}}, "tgt_barrel_01"),
    ({"version": 1, "scenes": {"s07": {"crate": ""}}}, "reason"),
    ({"version": 1, "scenes": {"s07": {"crate": 3}}}, "reason"),
    ({"version": 1, "scenes": {"s07": {}}}, "s07"),
    ({"version": 1, "scenes": {"s07": ["crate"]}}, "s07"),
])
def test_a_malformed_exclusion_table_is_refused_with_a_reason(tmp_path, table, match):
    from autofly_ue5.dataset.targets import TargetPoolError

    with pytest.raises(TargetPoolError, match=match):
        wide_pool(tmp_path, table)


def test_the_pool_s_record_for_a_scene_names_what_it_draws_and_what_its_table_excluded(tmp_path):
    pool = wide_pool(tmp_path)
    table = tmp_path / "target_exclusions.json"
    record = pool.to_json("train", scene_id="s07")
    assert record == {"path": pool.path, "sha256": pool.sha256, "version": 1, "split": "train",
                      "names": ["orange cylinder", "grid pole", "grey boulder", "orange cone", "grid sphere",
                                "orange sphere"],
                      "scene_id": "s07", "excluded": {"crate": "s07's obstacles are crates"},
                      "exclusions": {"path": str(table), "sha256": hashlib.sha256(table.read_bytes()).hexdigest()}}
    assert pool.to_json("train", scene_id="s02")["excluded"] == {}
    assert json.loads(json.dumps(record)) == record
