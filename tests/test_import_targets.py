"""scripts/import_assets.py's target mode (plan 5 C6): `plan --targets` writes each model target of the pool as one merged
glTF and the editor job that imports it as ONE static mesh under /Game/AutoFly/Targets/<key>/<key>; `apply --targets`
checks the editor's report before the registry gets a role "target" entry and the pool entry its asset. Offline, on
synthetic downloads (tests/test_gltf_merge.py's scene); the editor side runs only in Unreal."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from autofly_ue5.paths import ROOT
from tests.test_gltf_merge import write_scene

REAL_REGISTRY = ROOT / "assets" / "registry.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(name, category, source, *, status="candidate", scale=(1.0, 1.0, 1.0), seen=True, extent=None, asset=None,
           material=None):
    out = {"name": name, "category": category, "seen": seen, "asset": asset, "scale": list(scale), "material": material,
           "status": status, "source": source, "height_m": None, "stand_in": source.get("source") == "kenney"}
    if extent is not None:
        out["extent_m"] = extent
    return out


def world(tmp_path: Path) -> dict:
    """Downloads, fetch record, measurements, pool, registry and content roots for two models: a Poly Haven-like chair
    (a .gltf with a turned, scaled seat and a glass pane, plus a far-away second box) and a Kenney-like car (a .glb)."""
    from autofly_ue5.scenes.gltf import combine, load_meshes, measure

    downloads = tmp_path / "downloads"
    chair_src = write_scene(tmp_path / "a")
    chair_dir = downloads / "polyhaven" / "chair_x"
    shutil.copytree(chair_src.parent, chair_dir)
    car_src = write_scene(tmp_path / "b", glb=True, unlit=True)
    car_dir = downloads / "kenney" / "kenney_demo" / "Models" / "GLB format"
    shutil.copytree(car_src.parent, car_dir)
    shutil.move(str(car_dir / "chair.glb"), str(car_dir / "car.glb"))
    chair, car = chair_dir / "chair.gltf", car_dir / "car.glb"
    sources = {"format": "autofly_ue5_asset_sources/1", "downloads_dir": str(downloads), "assets": [
        {"id": "chair_x", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"], "licence": "CC0-1.0",
         "authors": {"A": "All"}, "page": "https://polyhaven.com/a/chair_x",
         "files": [{"path": "chair.gltf", "sha256": _sha(chair)}, {"path": "chair.bin", "sha256": _sha(chair_dir / "chair.bin")}]},
        {"id": "kenney_demo", "source": "kenney", "kind": "model_pack", "resolution": "GLB", "use": ["targets"],
         "licence": "CC0-1.0", "authors": {"Kenney": "All"}, "page": "https://kenney.nl/assets/demo",
         "files": [{"path": "kenney_demo.zip", "sha256": "z"}],
         "models": [{"name": "car", "path": "Models/GLB format/car.glb", "sha256": _sha(car)}]}]}
    whole_chair = measure(combine(load_meshes(chair), name="chair_x"))
    whole_car = measure(combine(load_meshes(car), name="car"))
    measurements = {"format": "autofly_ue5_asset_measurements/1", "band_m": [1.0, 3.0], "meshes": [], "models": [
        {"asset": "chair_x", "model": "chair_x", "nodes": [], **whole_chair},
        {"asset": "kenney_demo", "model": "car", "nodes": [], **whole_car}]}
    seat = measure(combine([m for m in load_meshes(chair) if m.name == "chair_seat"], name="seat"))["extent_m"]
    pool = {"version": 1, "targets": [
        _entry("orange cylinder", "geometry", {"source": "engine", "id": "/Engine/BasicShapes/Cylinder", "licence": "UE", "page": None},
               status="primitive", scale=(1.0, 1.0, 2.0), asset="cylinder", material="orange"),
        _entry("wooden chair", "furniture", {"source": "polyhaven", "id": "chair_x", "file": "chair.gltf", "licence": "CC0-1.0",
                                             "page": "https://polyhaven.com/a/chair_x"}),
        _entry("red car", "vehicle", {"source": "kenney", "id": "kenney_demo", "model": "car", "file": "Models/GLB format/car.glb",
                                      "licence": "CC0-1.0", "page": "https://kenney.nl/assets/demo"},
               status="placeholder", scale=(2.0, 2.0, 2.0), seen=False),
        _entry("chair seat", "furniture", {"source": "polyhaven", "id": "chair_x", "file": "chair.gltf", "licence": "CC0-1.0",
                                           "page": "https://polyhaven.com/a/chair_x", "nodes": ["chair_seat"]},
               scale=(0.5, 0.5, 0.5), extent=[round(seat[0] * 0.5, 3), round(seat[2] * 0.5, 3), round(seat[1] * 0.5, 3)]),
        _entry("blue hatchback", "vehicle", {"source": "fab", "id": "x", "licence": "Fab Standard", "page": "https://fab.com/x"},
               status="manual")]}
    content = tmp_path / "content"
    (content / "Engine" / "BasicShapes").mkdir(parents=True)
    (content / "Engine" / "BasicShapes" / "Cone.uasset").write_bytes(b"")
    paths = {"sources": tmp_path / "sources.json", "measurements": tmp_path / "measurements.json",
             "pool": tmp_path / "targets.json", "registry": tmp_path / "registry.json"}
    paths["sources"].write_text(json.dumps(sources))
    paths["measurements"].write_text(json.dumps(measurements))
    paths["pool"].write_text(json.dumps(pool, indent=2) + "\n")
    shutil.copy(REAL_REGISTRY, paths["registry"])
    return {"sources": sources, "measurements": measurements, "pool": pool, "content": [content], "paths": paths,
            "registry": json.loads(REAL_REGISTRY.read_text()), "work": tmp_path / "work"}


def plan(w: dict) -> dict:
    from scripts.import_assets import plan_targets

    return plan_targets(w["pool"], w["sources"], w["measurements"], work_dir=w["work"], registry=w["registry"],
                        content_roots=w["content"])


def fake_report(spec: dict) -> dict:
    """What scenes/ue/import_assets.py reports for a clean import of the spec: UE's box is the glTF's (x, z, y up) in cm,
    centred over the base pivot."""
    models = []
    for model in spec["models"]:
        (node,) = model["nodes"]
        gx, gy, gz = node["extent_m"]
        key = node["key"]
        models.append({"id": model["id"], "destination": model["destination"], "gltf": model["gltf"], "missing_nodes": [],
                       "extra_meshes": [], "cards_meshes": [], "meshes": [{
                           "name": key, "imported_name": key, "node": key, "path": f"{model['destination']}/{key}",
                           "box_extent_cm": [gx * 50.0, gz * 50.0, gy * 50.0], "origin_cm": [0.0, 0.0, gy * 50.0],
                           "triangles": node["triangles"], "source_triangles": node["triangles"],
                           "fallback_triangles": node["triangles"], "vertices": 3 * node["triangles"], "nanite_enabled": True,
                           "collision_trace_flag": "CTF_USE_COMPLEX_AS_SIMPLE",
                           "materials": [f"{model['destination']}/MI_wood"],
                           "foliage": {"changed": [], "slots": [{"material": f"{model['destination']}/MI_wood",
                                                                  "blend_mode": "BLEND_OPAQUE"}]}}]})
    return {"pass": True, "error": None, "models": models, "materials": []}


def test_the_plan_merges_every_model_target_into_one_mesh_under_its_own_key(tmp_path):
    from autofly_ue5.scenes.gltf import load_meshes

    w = world(tmp_path)
    spec = plan(w)
    assert spec["kind"] == "targets" and spec["nanite"] is True and spec["collision"] == "complex_as_simple"
    assert spec["nanite_fallback_percent"] == 1.0, "a target's collision is its whole visible mesh, not a 10 % fallback"
    assert spec["materials"] == []
    keys = [m["id"] for m in spec["models"]]
    assert keys == ["tgt_chair_x", "tgt_car", "tgt_chair_seat"], "primitives and manual entries are not imported"
    for model in spec["models"]:
        key = model["id"]
        assert model["destination"] == f"/Game/AutoFly/Targets/{key}" and model["gltf_cards"] is None
        (node,) = model["nodes"]
        assert node["name"] == node["key"] == node["mesh"] == key
        merged = load_meshes(Path(model["gltf"]))
        assert len(merged) == 1 and merged[0].name == key and Path(model["gltf"]).parent == w["work"] / "merged" / key
        assert node["triangles"] == len(merged[0].triangles)
    chair, car, seat = spec["models"]
    assert chair["target"]["name"] == "wooden chair" and chair["target"]["category"] == "furniture"
    assert car["target"]["status"] == "placeholder" and car["target"]["seen"] is False and car["target"]["scale"] == [2.0, 2.0, 2.0]
    assert seat["merge"]["nodes"] == ["chair_seat"] and chair["merge"]["nodes"] == ["chair_seat", "chair_glass", "other_box"]
    assert chair["merge"]["rewrites"][0]["change"] == "alphaMode BLEND -> MASK"
    assert any(r["change"] == "KHR_materials_unlit -> lit" for r in car["merge"]["rewrites"])
    assert chair["measurement"]["radius_profile_m"] and chair["measurement"]["ground_radius_m"] > 0
    assert chair["target"]["sha256"] == w["sources"]["assets"][0]["files"][0]["sha256"]


def test_the_plan_refuses_a_download_that_is_not_the_fetched_file(tmp_path):
    from scripts.import_assets import TargetPlanError

    w = world(tmp_path)
    w["sources"]["assets"][1]["models"][0]["sha256"] = "0" * 64
    with pytest.raises(TargetPlanError, match="red car.*sha256"):
        plan(w)


def test_the_plan_refuses_a_key_that_another_cooked_asset_already_has_in_any_case(tmp_path):
    """Project AirSim keys its spawn table by short name in a case-insensitive map (UnrealHelpers.cpp
    GenerateAssetRegistryMap, TMap<FString>): a second "Tgt_Car" anywhere in the cooked content would silently replace
    ours. The engine already has two assets named "cone", hence the tgt_ prefix."""
    from scripts.import_assets import TargetPlanError

    w = world(tmp_path)
    (w["content"][0] / "Elsewhere").mkdir()
    (w["content"][0] / "Elsewhere" / "Tgt_Car.uasset").write_bytes(b"")
    with pytest.raises(TargetPlanError, match="tgt_car.*Tgt_Car.uasset"):
        plan(w)
    # the target's own earlier import is not a collision: it is what a re-import replaces
    shutil.rmtree(w["content"][0] / "Elsewhere")
    own = w["content"][0] / "AutoFly" / "Targets" / "tgt_car"
    own.mkdir(parents=True)
    (own / "tgt_car.uasset").write_bytes(b"")
    assert [m["id"] for m in plan(w)["models"]][1] == "tgt_car"


def test_the_plan_refuses_two_entries_that_would_share_a_key(tmp_path):
    from scripts.import_assets import TargetPlanError

    w = world(tmp_path)
    w["pool"]["targets"].append({**w["pool"]["targets"][1], "name": "second chair"})
    with pytest.raises(TargetPlanError, match="tgt_chair_x"):
        plan(w)


def test_the_plan_refuses_a_merge_whose_size_is_not_the_measured_one(tmp_path):
    from scripts.import_assets import TargetPlanError

    w = world(tmp_path)
    w["measurements"]["models"][0]["extent_m"] = [9.0, 9.0, 9.0]
    with pytest.raises(TargetPlanError, match="wooden chair.*extent"):
        plan(w)
    w = world(tmp_path / "again")
    w["pool"]["targets"][3]["extent_m"] = [5.0, 5.0, 5.0]
    with pytest.raises(TargetPlanError, match="chair seat.*extent"):
        plan(w)


def test_the_plan_refuses_a_model_that_is_mostly_alpha_cards(tmp_path):
    """A target is one mesh: a tree-like model would need its cards split off without collision (scripts/import_assets.py
    for obstacles), which a runtime spawn of one asset cannot carry."""
    from scripts.import_assets import TargetPlanError

    w = world(tmp_path)
    doc = json.loads((Path(w["sources"]["downloads_dir"]) / "polyhaven" / "chair_x" / "chair.gltf").read_text())
    doc["materials"][0]["alphaMode"] = "BLEND"
    path = Path(w["sources"]["downloads_dir"]) / "polyhaven" / "chair_x" / "chair.gltf"
    path.write_text(json.dumps(doc))
    w["sources"]["assets"][0]["files"][0]["sha256"] = _sha(path)
    with pytest.raises(TargetPlanError, match="wooden chair.*not opaque"):
        plan(w)


def _with_path(spec: dict, path_of) -> dict:
    """fake_report with each mesh at path_of(destination, key)."""
    report = fake_report(spec)
    for model, reported in zip(spec["models"], report["models"]):
        reported["meshes"][0]["path"] = path_of(model["destination"], model["id"])
    return report


def test_registry_entries_record_where_the_mesh_actually_is(tmp_path):
    from scripts.import_assets import target_registry_entries

    w = world(tmp_path)
    spec = plan(w)
    report = _with_path(spec, lambda d, k: f"{d}/{k}/StaticMeshes/{k}")
    entries = target_registry_entries(spec, report, w["sources"])
    for model in spec["models"]:
        key = model["id"]
        assert entries[key]["ue_path"] == f"{model['destination']}/{key}/StaticMeshes/{key}"
        assert entries[key]["ue_path"].rsplit("/", 1)[-1] == key, "the spawn table's short name stays the key"


def test_the_editor_report_is_checked_against_the_plan(tmp_path):
    from scripts.import_assets import check_target_report

    spec = plan(world(tmp_path))
    assert check_target_report(spec, fake_report(spec)) == []

    def broken(change) -> list[str]:
        report = fake_report(spec)
        change(report["models"][0]["meshes"][0], report)
        return check_target_report(spec, report)

    assert any("height" in p for p in broken(lambda m, r: m["box_extent_cm"].__setitem__(2, 500.0)))
    assert any("pivot" in p for p in broken(lambda m, r: m["origin_cm"].__setitem__(0, 40.0)))
    assert any("pivot" in p for p in broken(lambda m, r: m["origin_cm"].__setitem__(2, 0.0)))
    assert any("Nanite" in p for p in broken(lambda m, r: m.__setitem__("nanite_enabled", False)))
    assert any("collision" in p and "of" in p for p in broken(lambda m, r: m.__setitem__("fallback_triangles", 1)))
    assert any("collision is" in p for p in broken(lambda m, r: m.__setitem__("collision_trace_flag", "CTF_USE_DEFAULT")))
    assert any("translucent" in p for p in broken(lambda m, r: m["foliage"]["slots"][0].__setitem__("blend_mode", "BLEND_TRANSLUCENT")))
    assert any("material" in p for p in broken(lambda m, r: m["materials"].append(None)))
    assert any("one static mesh" in p for p in broken(lambda m, r: r["models"][0]["meshes"].append(dict(m, name="extra"))))
    assert any("tgt_chair_x" in p and "path" in p for p in broken(lambda m, r: m.__setitem__("path", "/Game/AutoFly/Targets/x/y")))
    # Interchange's live layout (2026-10-08): a folder per import with Materials/StaticMeshes/Textures inside. The spawn
    # table keys meshes by short name and the cook rule covers the destination recursively, so it is accepted; a mesh
    # outside the target's own destination, or under another name, is not.
    assert check_target_report(spec, _with_path(spec, lambda d, k: f"{d}/{k}/StaticMeshes/{k}")) == []
    assert any("path" in p for p in check_target_report(spec, _with_path(spec, lambda d, k: f"/Game/AutoFly/Elsewhere/{k}")))
    assert any("path" in p for p in check_target_report(spec, _with_path(spec, lambda d, k: f"{d}/{k}/StaticMeshes/other")))
    assert any("cards" in p for p in broken(lambda m, r: r["models"][0].__setitem__("cards_meshes", [m])))
    assert any("pass=False" in p for p in broken(lambda m, r: r.__setitem__("pass", False)))
    report = fake_report(spec)
    report["models"].pop(1)
    assert any("tgt_car" in p and "not in the report" in p for p in check_target_report(spec, report))


def test_registry_entries_are_role_target_with_the_ue_box_the_profile_and_the_source(tmp_path):
    from scripts.import_assets import target_registry_entries

    w = world(tmp_path)
    spec = plan(w)
    entries = target_registry_entries(spec, fake_report(spec), w["sources"])
    assert set(entries) == {"tgt_chair_x", "tgt_car", "tgt_chair_seat"}
    chair = entries["tgt_chair_x"]
    (node,) = spec["models"][0]["nodes"]
    gx, gy, gz = node["extent_m"]
    assert chair["ue_path"] == "/Game/AutoFly/Targets/tgt_chair_x/tgt_chair_x"
    assert chair["role"] == "target" and chair["category"] == "furniture" and chair["seen"] is True
    assert chair["pivot"] == "base" and chair["footprint"] == "circle" and "parts" not in chair
    assert chair["base_size_m"] == pytest.approx([gx, gz, gy], abs=1e-4)
    assert chair["measured_extent_cm_at_unit_scale"] == pytest.approx([gx * 50, gz * 50, gy * 50], abs=1e-3)
    assert chair["radius_profile_m"] == spec["models"][0]["measurement"]["radius_profile_m"]
    assert chair["ground_radius_m"] == spec["models"][0]["measurement"]["ground_radius_m"]
    assert chair["source"]["licence"] == "CC0-1.0" and chair["source"]["page"] == "https://polyhaven.com/a/chair_x"
    assert chair["source"]["sha256"] == w["sources"]["assets"][0]["files"][0]["sha256"]
    assert entries["tgt_car"]["seen"] is False and entries["tgt_car"]["source"]["model"] == "car"
    assert entries["tgt_chair_seat"]["source"]["nodes"] == ["chair_seat"]
    assert chair["target_name"] == "wooden chair" and chair["collision"] == "CTF_USE_COMPLEX_AS_SIMPLE"


def _run(w, *args) -> int:
    from scripts.import_assets import main

    p = w["paths"]
    return main([*args, "--targets", "--sources", str(p["sources"]), "--measurements", str(p["measurements"]),
                 "--registry", str(p["registry"]), "--pool", str(p["pool"]), "--work", str(w["work"]),
                 *sum((["--content-root", str(c)] for c in w["content"]), [])])


def test_apply_writes_the_registry_and_the_pool_and_the_pool_then_spawns_them(tmp_path):
    from autofly_ue5.dataset.targets import TargetPool
    from autofly_ue5.expert.episode import _spawn_asset_name, standing_z_ned
    from autofly_ue5.scenes.model import load_registry
    from autofly_ue5.sim.fake import FakeSimulator
    from autofly_ue5.sim.types import Pose

    w = world(tmp_path)
    assert _run(w, "plan") == 0
    spec = json.loads((w["work"] / "import_spec.json").read_text())
    (w["work"] / "import_report.json").write_text(json.dumps(fake_report(spec)))
    (w["work"] / "import_editor.log").write_text("LogInit: fine\n")
    assert _run(w, "apply") == 0
    registry = json.loads(w["paths"]["registry"].read_text())
    assert {"tgt_chair_x", "tgt_car", "tgt_chair_seat"} <= set(registry["assets"])
    pool = json.loads(w["paths"]["pool"].read_text())["targets"]
    by_name = {t["name"]: t for t in pool}
    assert by_name["wooden chair"]["asset"] == "tgt_chair_x" and by_name["wooden chair"]["status"] == "imported"
    assert by_name["red car"]["asset"] == "tgt_car" and by_name["red car"]["status"] == "placeholder", \
        "whether stand-ins are drawn is the user's call: without --use-placeholders they stay out of the draw"
    assert by_name["blue hatchback"]["asset"] is None and by_name["orange cylinder"]["status"] == "primitive"
    loaded = TargetPool.load(w["paths"]["pool"], registry=load_registry(w["paths"]["registry"]))
    usable = [e for e in loaded.entries if e.usable]
    assert [e.name for e in usable] == ["orange cylinder", "wooden chair", "chair seat"]
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 5)
    reg = load_registry(w["paths"]["registry"])
    for e in usable:
        asset = reg.assets[e.asset]
        sim.spawn("t", _spawn_asset_name(asset.ue_path), Pose(5.0, 0.0, standing_z_ned(asset, e.scale), 0.0), e.scale, None)
    assert _run(w, "apply", "--use-placeholders") == 0
    by_name = {t["name"]: t for t in json.loads(w["paths"]["pool"].read_text())["targets"]}
    assert by_name["red car"]["status"] == "imported"
    # drawn, a stand-in keeps its mark, so a dataset's metadata can filter its episodes later (plan 6 revision 1 item 7)
    assert by_name["red car"]["stand_in"] is True and by_name["wooden chair"]["stand_in"] is False
    loaded = {e.name: e for e in TargetPool.load(w["paths"]["pool"], registry=load_registry(w["paths"]["registry"])).entries}
    assert loaded["red car"].stand_in and loaded["red car"].usable and not loaded["wooden chair"].stand_in


def test_apply_refuses_a_report_with_a_material_that_failed_to_compile(tmp_path):
    w = world(tmp_path)
    assert _run(w, "plan") == 0
    spec = json.loads((w["work"] / "import_spec.json").read_text())
    (w["work"] / "import_report.json").write_text(json.dumps(fake_report(spec)))
    (w["work"] / "import_editor.log").write_text(
        "[AssetLog] /x/Content/AutoFly/Targets/tgt_car/MI_colormap.uasset: Failed to compile Material for platform SM6\n")
    before = (w["paths"]["registry"].read_bytes(), w["paths"]["pool"].read_bytes())
    assert _run(w, "apply") == 2
    assert (w["paths"]["registry"].read_bytes(), w["paths"]["pool"].read_bytes()) == before


def test_the_obstacle_plan_is_unchanged_by_the_target_mode(tmp_path):
    """`plan` without --targets still writes runs/assets/import_spec.json for the scene assets only (its own work dir)."""
    from scripts.import_assets import WORK_DIR, target_work_dir

    assert target_work_dir(WORK_DIR) == WORK_DIR / "targets"
