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
            assert entry["ue_path"] == f"/Game/AutoFly/Targets/{t['asset']}/{t['asset']}", t["name"]
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
