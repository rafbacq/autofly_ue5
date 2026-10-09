"""autofly_ue5/report/stores.py reads raw stores for reports: each kept episode's target as fully as its provenance (or,
for a store from before the pool, the pool file by name) says, and how many flights each store took, from its rejects,
its collector's record or its manifest, with any disagreement between them reported."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.synthetic_stores import Flight, legacy_target, pool_target, write_store

GATES = Path(__file__).resolve().parents[1] / "docs" / "gates"


def _pool(tmp_path, entries: list[dict]) -> Path:
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"version": 1, "targets": entries}))
    return path


def _entry(name, category, seen=True, status="primitive", **extra):
    return {"name": name, "category": category, "seen": seen, "asset": "cylinder", "scale": [1, 1, 2], "material": None,
            "status": status, "source": None, "height_m": 2.0, **extra}


def test_a_pool_episode_s_target_comes_from_its_provenance_with_the_stand_in_mark(tmp_path):
    from autofly_ue5.report.stores import load_store

    root = write_store(tmp_path / "data", "s02_c1", "s02", [
        Flight(1, 5, pool_target("red sedan", "vehicle", status="imported", stand_in=True)),
        Flight(2, 6, pool_target("beige sofa", "furniture", stand_in=False)),
        Flight(3, 7, pool_target("wooden ladder", "equipment", seen=False)),  # drawn before the mark existed
    ])
    store = load_store(root)
    by_name = {e.target.name: e.target for e in store.episodes}
    assert [e.id for e in store.episodes] == ["s02_1", "s02_2", "s02_3"] and store.name == "s02_c1"
    assert by_name["red sedan"].category == "vehicle" and by_name["red sedan"].stand_in is True
    assert by_name["beige sofa"].stand_in is False and by_name["beige sofa"].seen is True
    assert by_name["wooden ladder"].seen is False and by_name["wooden ladder"].stand_in is None, "no mark, no pool: unknown"
    assert all(t.from_pool == () for t in by_name.values())
    assert [e.steps for e in store.episodes] == [5, 6, 7] and store.episodes[0].scene == "s02"
    assert store.episodes[1].frame(-1).name == "000005.png" and store.episodes[1].frame(-1).is_file()


def test_the_first_frame_s_visibility_status_is_read_where_the_provenance_has_one(tmp_path):
    from autofly_ue5.report.stores import load_store

    sofa = pool_target("beige sofa", "furniture", stand_in=False)
    root = write_store(tmp_path / "data", "s02_c1", "s02", [
        Flight(1, 3, sofa, extra={"prewarm_steps": 5, "first_frame_visibility": {"status": "visible", "range_m": 60.1}}),
        Flight(2, 3, sofa, extra={"first_frame_visibility": {"status": "occluded", "occluder": "obs_0003"}}),
        Flight(3, 3, sofa),  # collected before the check existed
        Flight(4, 3, sofa, extra={"first_frame_visibility": "visible"})])  # not the block's shape: no status read
    assert [e.first_frame for e in load_store(root).episodes] == ["visible", "occluded", None, None]


def test_a_store_from_before_the_pool_is_completed_from_the_pool_file_by_name_and_says_so(tmp_path):
    from autofly_ue5.report.stores import load_pool, load_store

    root = write_store(tmp_path / "data", "s01_pilot", "s01", [Flight(1, 4, legacy_target())])
    unmarked = load_pool(_pool(tmp_path, [_entry("orange cylinder", "geometry")]))
    (episode,) = load_store(root, pool=unmarked).episodes
    t = episode.target
    assert (t.name, t.category, t.seen, t.status, t.name_status) == ("orange cylinder", "geometry", True, "primitive",
                                                                      "placeholder_until_M4")
    assert t.stand_in is None, "a pool file without any stand-in mark cannot say"
    assert set(t.from_pool) == {"category", "seen", "status"}
    # Once the pool file marks its stand-ins, an unmarked entry is not one.
    marked = load_pool(_pool(tmp_path, [_entry("orange cylinder", "geometry"),
                                        _entry("red sedan", "vehicle", status="imported", stand_in=True)]))
    (episode,) = load_store(root, pool=marked).episodes
    assert episode.target.stand_in is False and "stand_in" in episode.target.from_pool
    (episode,) = load_store(root).episodes
    assert episode.target.category == "unknown" and episode.target.seen is None


def test_flown_is_kept_plus_rejects_and_the_reasons_come_from_the_rejects_directory(tmp_path):
    from autofly_ue5.report.stores import load_store

    t = pool_target("beige sofa", "furniture")
    root = write_store(tmp_path / "data", "s03_c1", "s03", [Flight(1, 5, t), Flight(2, 5, t)],
                       rejects=[(Flight(3, 3, t), "collision", "footprint"), (Flight(4, 3, t), "collision", "sim"),
                                (Flight(5, 3, t), "out_of_bounds", None)])
    store = load_store(root)
    a = store.attempts
    assert (a.kept, a.flown, a.source, a.disagreements) == (2, 5, "rejects", ())
    assert a.rejected == {"collision": 2, "out_of_bounds": 1}
    assert a.collision_sources == {"footprint": 1, "sim": 1}
    assert [r.reason for r in store.rejects] == ["collision", "collision", "out_of_bounds"]
    assert store.rejects[0].final_pose is not None and store.rejects[0].scene == "s03"
    assert a.kept_rate == pytest.approx(0.4)
    # The rejects can be left out: the manifest's count still gives the total, the reasons are then unrecorded.
    a = load_store(root, rejects_dir=None).attempts
    assert (a.flown, a.rejected, a.source) == (5, {"unrecorded": 3}, "manifest")


def test_the_collector_s_record_gives_the_reasons_without_rejects_and_any_disagreement_is_reported(tmp_path):
    from autofly_ue5.report.stores import load_store

    t = pool_target("beige sofa", "furniture")
    root = write_store(tmp_path / "data", "s04_c1", "s04", [Flight(1, 5, t)],
                       rejects=[(Flight(2, 3, t), "timeout", None)])
    record = tmp_path / "s04_c1.json"
    record.write_text(json.dumps({"dataset": {"name": "s04_c1"}, "collection": {
        "kept": 1, "rejected": 1, "attempted": 2, "fault_retries": 3, "outcomes": {"success": 1, "timeout": 1}}}))
    a = load_store(root, rejects_dir=None, record_path=record).attempts
    assert (a.flown, a.rejected, a.source, a.fault_retries, a.disagreements) == (2, {"timeout": 1}, "record", 3, ())
    a = load_store(root, record_path=record).attempts
    assert (a.source, a.disagreements) == ("rejects+record", ())
    record.write_text(json.dumps({"dataset": {"name": "s04_c1"}, "collection": {
        "kept": 1, "attempted": 3, "outcomes": {"success": 1, "collision": 2}}}))
    a = load_store(root, record_path=record).attempts
    assert a.rejected == {"timeout": 1}, "the rejects directory holds each reject: it decides"
    assert len(a.disagreements) == 2 and any("attempted 3" in d for d in a.disagreements)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"dataset": {"name": "s04_c2"}, "collection": {"kept": 1}}))
    with pytest.raises(ValueError, match="s04_c2"):
        load_store(root, record_path=other)


def test_several_stores_load_together_and_a_name_twice_or_a_record_for_no_listed_store_is_refused(tmp_path):
    from autofly_ue5.report.stores import load_stores

    t = pool_target("beige sofa", "furniture")
    a = write_store(tmp_path / "a", "s02_c1", "s02", [Flight(1, 3, t)], rejects=[(Flight(2, 3, t), "timeout", None)])
    b = write_store(tmp_path / "b", "s03_c1", "s03", [Flight(1, 3, t)])
    stores = load_stores([a, b])
    assert [s.name for s in stores] == ["s02_c1", "s03_c1"] and stores[0].attempts.rejected == {"timeout": 1}
    assert load_stores([a], read_rejects_dirs=False)[0].attempts.source == "manifest"
    twin = write_store(tmp_path / "c", "s02_c1", "s02", [Flight(1, 3, t)])
    with pytest.raises(ValueError, match="two stores are named 's02_c1'"):
        load_stores([a, twin])
    with pytest.raises(ValueError, match="s09_c1"):
        load_stores([a], records={"s09_c1": tmp_path / "s09_c1.json"})


def test_a_directory_that_is_not_a_raw_store_is_refused(tmp_path):
    from autofly_ue5.report.stores import load_store

    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="manifest"):
        load_store(tmp_path / "empty")


def test_wilson_interval_reproduces_the_screening_record_and_known_values():
    from autofly_ue5.report.stores import wilson_interval

    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_interval(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    record = GATES / "m5_pilot_screening.json"
    if not record.is_file():
        pytest.skip("docs/gates/m5_pilot_screening.json is not in this tree")
    for pair in json.loads(record.read_text())["pairs"]:
        assert wilson_interval(pair["successes"], pair["episodes"]) == pytest.approx(tuple(pair["wilson_95"]), abs=1e-4)
