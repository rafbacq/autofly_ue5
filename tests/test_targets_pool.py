"""assets/targets.json: the target pool of spec §6.4 (plan 5, C5). 60 instances, 50 seen and 10 unseen, in the paper's
Fig. 3c category mix, each with the name an instruction uses. Engine primitives must resolve in the registry; every
downloaded model must be CC0 and traceable to its page and its fetch record (assets/sources.json)."""

from __future__ import annotations

import json
import re

import pytest

from autofly_ue5.paths import ROOT

TARGETS = ROOT / "assets" / "targets.json"
REGISTRY = ROOT / "assets" / "registry.json"
SOURCES = ROOT / "assets" / "sources.json"
# The runtime colours' registry entries wait here until the editor has made the materials (scripts/apply_runtime_materials.py)
PENDING_MATERIALS = ROOT / "assets" / "pending" / "runtime_materials.registry.json"

# Fig. 3c: vehicles 22 %, geometry 20 %, furniture 18 %, equipment 18 %, human 17 %, others 5 %
MIX_PERCENT = {"vehicle": 22, "geometry": 20, "furniture": 18, "equipment": 18, "human": 17, "other": 5}
STATUSES = {"candidate", "imported", "placeholder", "manual", "primitive"}


@pytest.fixture(scope="module")
def pool() -> list[dict]:
    data = json.loads(TARGETS.read_text())
    assert data["version"] == 1
    return data["targets"]


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY.read_text())


def test_sixty_instances_fifty_seen_and_ten_unseen(pool):
    assert len(pool) == 60
    assert sum(1 for t in pool if t["seen"] is True) == 50 and sum(1 for t in pool if t["seen"] is False) == 10


def test_the_category_mix_is_the_paper_s_within_one_instance(pool):
    counts = {c: sum(1 for t in pool if t["category"] == c) for c in MIX_PERCENT}
    assert sum(counts.values()) == len(pool), "every entry has one of the six categories"
    for category, percent in MIX_PERCENT.items():
        assert abs(counts[category] - percent * len(pool) / 100.0) <= 1.0, (category, counts[category])


def test_names_are_unique_lower_case_instruction_phrases(pool):
    names = [t["name"] for t in pool]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[a-z]+( [a-z]+)*", n) for n in names), [n for n in names if not re.fullmatch(r"[a-z]+( [a-z]+)*", n)]


def test_every_entry_has_the_plan_s_fields(pool, registry):
    for t in pool:
        assert t["status"] in STATUSES, t["name"]
        assert len(t["scale"]) == 3 and all(s > 0 for s in t["scale"]), t["name"]
        assert isinstance(t["source"], dict) and {"source", "id", "licence", "page"} <= set(t["source"]), t["name"]
        if t["status"] != "primitive" and t["asset"] is not None:
            # set by `scripts/import_assets.py apply --targets` (docs/runbook-targets.md), never by hand
            entry = registry["assets"].get(t["asset"])
            assert entry is not None and entry["role"] == "target", f"{t['name']}: {t['asset']} is not an imported target"
            # Interchange's live layout puts the mesh under StaticMeshes/ (2026-10-08); either way the spawn table's short
            # name, the path's last segment, is the target's key
            assert entry["ue_path"] in (f"/Game/AutoFly/Targets/{t['asset']}/{t['asset']}",
                                        f"/Game/AutoFly/Targets/{t['asset']}/{t['asset']}/StaticMeshes/{t['asset']}"), t["name"]
        if t["status"] == "imported":
            assert t["asset"] is not None, t["name"]


def test_primitives_resolve_in_the_registry(pool, registry):
    """A primitive's material is in the registry, or waits in the runtime colours' pending patch (merged into the registry
    once the editor has made the base materials)."""
    pending = json.loads(PENDING_MATERIALS.read_text())["materials"]
    primitives = [t for t in pool if t["status"] == "primitive"]
    assert primitives and all(t["category"] == "geometry" for t in primitives)
    for t in primitives:
        assert t["asset"] in registry["assets"], t["name"]
        assert t["material"] in registry["materials"] or t["material"] in pending, t["name"]


def test_downloaded_targets_are_cc0_with_a_page_and_a_fetch_record(pool):
    fetched = {a["id"]: a for a in json.loads(SOURCES.read_text())["assets"]}
    for t in pool:
        if t["status"] == "primitive":
            continue
        source = t["source"]
        assert source["licence"] == "CC0-1.0" and re.match(r"https://", source["page"] or ""), t["name"]
        if t["status"] == "manual":
            continue
        record = fetched.get(source["id"])
        assert record is not None and record["licence"] == "CC0-1.0" and record["page"] == source["page"], t["name"]
        assert "targets" in record["use"], t["name"]
        if record["kind"] == "model_pack":
            assert source["model"] in [m["name"] for m in record["models"]], t["name"]


def test_measured_targets_are_half_a_metre_to_four_metres_at_their_scale(pool):
    """The drone succeeds within 5 m of a target: a target larger than 4 m would be reached before it is approached,
    one under 0.5 m would be a few pixels until the last metres."""
    for t in pool:
        if t.get("extent_m") is not None:
            assert 0.5 <= max(t["extent_m"]) <= 4.0 + 1e-6, (t["name"], t["extent_m"])
            assert t["height_m"] == pytest.approx(t["extent_m"][2]), t["name"]


def test_the_paper_s_unseen_objects_are_unseen(pool):
    unseen = {t["name"] for t in pool if not t["seen"]}
    for word in ("crane", "tiger", "elephant", "traffic cone", "ladder"):
        assert any(word in name for name in unseen), word


# Plan 6 step 2c / revision 1 item 7: what each scene must not draw because it looks like that scene's obstacles, by
# registry asset (the shape): cylinders and the 0.88 m barrels among pillars and poles, cubes and the box-shaped cabinet
# and tool chest among stacked boxes, spheres among rocks (s04 paints its stones in the spheres' palette).
EXCLUSIONS = ROOT / "assets" / "target_exclusions.json"
PILLARS = {"cylinder", "tgt_barrel_01", "tgt_barrel_02", "tgt_wine_barrel_01"}
EXPECTED_EXCLUSIONS = {"s01": PILLARS, "s01d": PILLARS, "s09": PILLARS,
                       "s07": {"cube", "tgt_painted_wooden_cabinet", "tgt_metal_tool_chest"},
                       "s03": {"sphere"}, "s04": {"sphere"}, "s05": {"sphere"}, "s05r": {"sphere"}}


def test_each_scene_excludes_the_targets_that_look_like_its_obstacles(pool):
    from autofly_ue5.dataset.targets import TargetPool

    table = json.loads(EXCLUSIONS.read_text())
    assert {scene: set(assets) for scene, assets in table["scenes"].items()} == EXPECTED_EXCLUSIONS
    assert all(isinstance(reason, str) and len(reason) > 20 for assets in table["scenes"].values()
               for reason in assets.values()), "every exclusion says why"
    loaded = TargetPool.load(TARGETS)  # the table beside the pool is the pool's own
    assert loaded.exclusions is not None and loaded.exclusions.path == str(EXCLUSIONS)
    by_asset = {}
    for t in pool:
        by_asset.setdefault(t["asset"], set()).add(t["name"])
    for scene, assets in EXPECTED_EXCLUSIONS.items():
        drawn = {e.name for e in loaded.for_split("test_unseen", scene_id=scene)}
        for asset in assets:
            assert by_asset[asset] and drawn.isdisjoint(by_asset[asset]), (scene, asset)
    assert "orange cylinder" in {e.name for e in loaded.for_split("train", scene_id="s07")}, "only the scene's own"


# Plan 6 step 2b (docs/decisions/2026-10-08-target-pool-diversity.md): the 25 Kenney stand-ins are drawn, for the
# vehicle and human categories nothing else covers yet, and each keeps a mark episodes record.
SEEN_DRAWN_PER_SCENE = {"s01": 43, "s01d": 43, "s09": 43,  # 50 less 4 cylinders and 3 barrels
                        "s07": 45,  # less 3 cubes, the cabinet and the tool chest
                        "s03": 48, "s04": 48, "s05": 48, "s05r": 48,  # less 2 spheres
                        "s02": 50, "s06": 50, "s06r": 50}


def test_the_kenney_stand_ins_are_drawn_and_marked_as_stand_ins(pool):
    kenney = [t for t in pool if t["source"]["source"] == "kenney"]
    assert len(kenney) == 25 and all(t["status"] == "imported" and t.get("stand_in") is True for t in kenney)
    assert not any("stand_in" in t for t in pool if t["source"]["source"] != "kenney")
    vehicles = [t for t in kenney if t["category"] == "vehicle"]
    humans = [t for t in kenney if t["category"] == "human"]
    assert len(vehicles) == 12 and len(humans) == 10, "what only the stand-ins cover until a Fab session"


def test_sixty_entries_are_drawn_fifty_seen_and_each_scene_draws_what_its_exclusions_leave():
    from autofly_ue5.dataset.targets import TargetPool

    loaded = TargetPool.load(TARGETS)
    assert sum(e.usable for e in loaded.entries) == 60
    assert len(loaded.for_split("train")) == 50 and len(loaded.for_split("test_unseen")) == 60
    for scene, seen in SEEN_DRAWN_PER_SCENE.items():
        assert len(loaded.for_split("train", scene_id=scene)) == seen, scene
        assert len(loaded.for_split("test_unseen", scene_id=scene)) == seen + 10, f"{scene}: no unseen entry excluded"
