"""scripts/import_assets.py (host side): the editor job is planned from the sources and measurements, the editor's report
is checked against the measurements before anything reaches the registry, and the registry entries carry what the
generator needs. The editor-side script (scenes/ue/import_assets.py) runs only in Unreal and is not tested here."""

from __future__ import annotations

import json

import pytest


def _sources(tmp_path):
    (tmp_path / "downloads" / "polyhaven" / "tree_x").mkdir(parents=True, exist_ok=True)
    (tmp_path / "downloads" / "ambientcg" / "Paving1").mkdir(parents=True, exist_ok=True)
    for name in ("Paving1_2K-JPG_Color.jpg", "Paving1_2K-JPG_NormalDX.jpg", "Paving1_2K-JPG_Roughness.jpg", "Paving1_2K-JPG_AmbientOcclusion.jpg"):
        (tmp_path / "downloads" / "ambientcg" / "Paving1" / name).write_bytes(b"jpg")
    (tmp_path / "downloads" / "polyhaven" / "tree_x" / "tree_x_2k.gltf").write_text("{}")
    return {"format": "autofly_ue5_asset_sources/1", "downloads_dir": str(tmp_path / "downloads"), "assets": [
        {"id": "tree_x", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s02"], "licence": "CC0-1.0",
         "authors": {"A": "modeling"}, "files": [{"path": "tree_x_2k.gltf", "sha256": "a"}]},
        {"id": "Paving1", "source": "ambientcg", "kind": "material", "resolution": "2K-JPG", "use": ["s09"], "licence": "CC0-1.0",
         "dimensions_cm": [115, 115], "files": [{"path": "Paving1_2K-JPG.zip", "sha256": "b", "unpacked": [
             "Paving1_2K-JPG_Color.jpg", "Paving1_2K-JPG_NormalDX.jpg", "Paving1_2K-JPG_NormalGL.jpg", "Paving1_2K-JPG_Roughness.jpg",
             "Paving1_2K-JPG_AmbientOcclusion.jpg", "Paving1_2K-JPG_Displacement.jpg"]}]}]}


def _measurements():
    return {"format": "autofly_ue5_asset_measurements/1", "band_m": [1.0, 3.0], "meshes": [
        {"asset": "tree_x", "node": "tree_x_a_LOD0", "mesh": "Cube.070", "extent_m": [2.8, 3.5, 2.6], "height_m": 3.5, "base_m": 0.0, "top_m": 3.5,
         "band_radius_m": 1.4, "band_centre_offset_m": 0.2, "profile_step_m": 0.25, "radius_profile_m": [0.15] * 5 + [1.4] * 9,
         "ground_radius_m": 1.4, "triangles": 1000},
        {"asset": "tree_x", "node": "tree_x_b_LOD0", "mesh": "Cube.071", "extent_m": [1.0, 1.2, 0.9], "height_m": 1.2, "base_m": 0.0, "top_m": 1.2,
         "band_radius_m": 0.4, "band_centre_offset_m": 0.0, "profile_step_m": 0.25, "radius_profile_m": [0.5, 0.5, 0.5, 0.4, 0.4],
         "ground_radius_m": 0.5, "triangles": 200}]}


def _report(ok=True):
    # UE: Z up, centimetres; the importer turns glTF (x, y up, z) into (x or z, z or x, y). Bounds origin at half height: base pivot.
    a_extent = [140.0, 130.0, 175.0] if ok else [140.0, 130.0, 150.0]
    return {"pass": True, "models": [{"id": "tree_x", "destination": "/Game/AutoFly/Assets/tree_x", "meshes": [
        {"name": "tree_x_a", "imported_name": "Cube_070", "path": "/Game/AutoFly/Assets/tree_x/tree_x_a", "box_extent_cm": a_extent,
         "origin_cm": [0.0, 5.0, 87.5], "triangles": 1000, "vertices": 600, "nanite_enabled": True,
         "collision_trace_flag": "CTF_USE_COMPLEX_AS_SIMPLE", "materials": ["/Game/AutoFly/Assets/tree_x/M_bark"]},
        {"name": "Cube_071", "path": "/Game/AutoFly/Assets/tree_x/Cube_071", "box_extent_cm": [45.0, 50.0, 60.0],
         "source_triangles": 20000,
         "origin_cm": [0.0, 0.0, 60.0], "triangles": 200, "vertices": 120, "nanite_enabled": True,
         "collision_trace_flag": "CTF_USE_COMPLEX_AS_SIMPLE", "materials": []}]}],
        "materials": [{"id": "Paving1", "name": "M_Paving1", "path": "/Game/AutoFly/Materials/Ground/M_Paving1",
                       "connected": ["MP_BASE_COLOR", "MP_NORMAL", "MP_ROUGHNESS", "MP_AMBIENT_OCCLUSION"], "tiling": 173.913,
                       "textures": {"color": "/Game/AutoFly/Materials/Ground/Textures/Paving1/Paving1_2K-JPG_Color"}}]}


def test_the_plan_names_every_node_and_the_four_texture_maps_with_the_tiling(tmp_path):
    from scripts.import_assets import plan_import

    spec = plan_import(_sources(tmp_path), _measurements())
    (model,) = spec["models"]
    assert model["id"] == "tree_x" and model["gltf"].endswith("tree_x/tree_x_2k.gltf") and model["destination"] == "/Game/AutoFly/Assets/tree_x"
    assert [n["name"] for n in model["nodes"]] == ["tree_x_a_LOD0", "tree_x_b_LOD0"]
    assert [n["key"] for n in model["nodes"]] == ["tree_x_a", "tree_x_b"] and [n["mesh"] for n in model["nodes"]] == ["Cube_070", "Cube_071"]
    (material,) = spec["materials"]
    assert material["name"] == "M_Paving1" and material["tile_m"] == 1.15 and material["tiling"] == pytest.approx(200.0 / 1.15)
    assert set(material["textures"]) == {"color", "normal", "roughness", "ao"}
    assert material["textures"]["normal"].endswith("_NormalDX.jpg"), "UE's normal convention is DirectX's"
    assert spec["nanite"] is True and spec["collision"] == "complex_as_simple"


def test_the_report_is_checked_against_the_measurements_before_the_registry_is_written(tmp_path):
    from scripts.import_assets import check_report, plan_import

    spec = plan_import(_sources(tmp_path), _measurements())
    assert check_report(spec, _measurements(), _report()) == []
    problems = check_report(spec, _measurements(), _report(ok=False))
    assert any("tree_x_a_LOD0" in p and "height" in p for p in problems), problems
    missing = _report()
    missing["models"][0]["meshes"].pop()
    assert any("tree_x_b_LOD0" in p for p in check_report(spec, _measurements(), missing))
    no_nanite = _report()
    no_nanite["models"][0]["meshes"][0]["nanite_enabled"] = False
    assert any("nanite" in p.lower() for p in check_report(spec, _measurements(), no_nanite))
    half = _report()
    half["materials"][0]["connected"] = ["MP_BASE_COLOR"]
    assert any("M_Paving1" in p for p in check_report(spec, _measurements(), half))


def test_registry_entries_carry_the_profile_the_base_pivot_and_the_ue_bounds(tmp_path):
    from scripts.import_assets import plan_import, registry_entries, update_registry

    spec = plan_import(_sources(tmp_path), _measurements())
    assets, materials = registry_entries(spec, _sources(tmp_path), _measurements(), _report())
    assert set(assets) == {"tree_x_a", "tree_x_b"}, "one entry per node, the _LOD0 suffix dropped"
    a = assets["tree_x_a"]
    assert a["ue_path"] == "/Game/AutoFly/Assets/tree_x/tree_x_a" and a["pivot"] == "base" and a["footprint"] == "circle"
    assert assets["tree_x_b"]["ue_path"] == "/Game/AutoFly/Assets/tree_x/Cube_071", "matched through the glTF mesh name when not renamed"
    assert a["base_size_m"] == [2.8, 2.6, 3.5] and a["measured_extent_cm_at_unit_scale"] == [140.0, 130.0, 175.0]
    assert a["radius_profile_m"] == [0.15] * 5 + [1.4] * 9 and a["profile_step_m"] == 0.25 and a["ground_radius_m"] == 1.4
    assert assets["tree_x_b"]["triangles"] == 200, "the glTF's triangles (the measurements), not the Nanite fallback's"
    assert a["role"] == "obstacle" and a["category"] == "nature" and a["seen"] is None
    assert a["source"] == {"source": "polyhaven", "id": "tree_x", "node": "tree_x_a_LOD0", "licence": "CC0-1.0", "authors": {"A": "modeling"}}
    assert materials == {"paving1": {"kind": "textured", "ue_path": "/Game/AutoFly/Materials/Ground/M_Paving1", "tile_m": 1.15,
                                     "source": {"source": "ambientcg", "id": "Paving1", "licence": "CC0-1.0"}}}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "assets": {"cylinder": {"ue_path": "/Engine/BasicShapes/Cylinder"}},
                                         "materials": {"white": {"kind": "color_instance", "ue_path": "/Game/AutoFly/Materials/MI_White"}}}))
    update_registry(registry_path, assets, materials)
    merged = json.loads(registry_path.read_text())
    assert set(merged["assets"]) == {"cylinder", "tree_x_a", "tree_x_b"} and set(merged["materials"]) == {"white", "paving1"}
    update_registry(registry_path, assets, materials)  # the same entries again: fine
    changed = {**assets, "tree_x_a": {**assets["tree_x_a"], "ground_radius_m": 9.9}}
    with pytest.raises(ValueError, match="tree_x_a"):
        update_registry(registry_path, changed, materials)  # a different entry under an existing name is refused


def test_a_one_node_model_s_mesh_is_accepted_whatever_the_importer_named_it(tmp_path):
    from scripts.import_assets import check_report, plan_import, registry_entries

    sources = _sources(tmp_path)
    measurements = {"format": "x", "band_m": [1.0, 3.0], "meshes": [m for m in _measurements()["meshes"] if m["node"] == "tree_x_a_LOD0"]}
    spec = plan_import(sources, measurements)
    report = _report()
    report["models"][0]["meshes"] = [dict(report["models"][0]["meshes"][0], name="tree_x_2k", path="/Game/AutoFly/Assets/tree_x/tree_x_2k")]
    assert check_report(spec, measurements, report) == []
    assets, _materials = registry_entries(spec, sources, measurements, report)
    assert set(assets) == {"tree_x"} and assets["tree_x"]["ue_path"] == "/Game/AutoFly/Assets/tree_x/tree_x_2k"
    assert assets["tree_x"]["source"]["node"] == "tree_x_a_LOD0"


def test_the_material_key_for_a_ground_is_its_purpose_when_the_list_knows_one(tmp_path):
    from scripts.import_assets import GROUND_KEYS, material_key

    assert material_key("PavingStones070") == "paving" and GROUND_KEYS["Ground054"] == "sand"
    assert material_key("Paving1") == "paving1", "an unknown set falls back to its lower-cased id"
