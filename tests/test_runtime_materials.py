"""Runtime colours (plan 5 C5/C6; spec §7.1 "Runtime colour needs a base UMaterial"): the palette's red, blue, yellow,
green and white are material instances (MI_*), which Project AirSim's setMaterial cannot load, so 10 of the 12 geometry
targets would spawn grey. The editor makes base UMaterials of the same colours (scenes/ue/create_runtime_materials.py);
their registry entries wait in a pending patch (assets/pending/runtime_materials.registry.json) that
scripts/apply_runtime_materials.py merges only once the editor's report shows them made, base-class and compiled."""

from __future__ import annotations

import json
import shutil

import pytest

from autofly_ue5.paths import ROOT

PATCH = ROOT / "assets" / "pending" / "runtime_materials.registry.json"
REGISTRY = ROOT / "assets" / "registry.json"
TARGETS = ROOT / "assets" / "targets.json"
COLOURS = ("red", "blue", "yellow", "green", "white")


def _patch() -> dict:
    return json.loads(PATCH.read_text())


def _good_report(patch: dict) -> dict:
    """What scenes/ue/create_runtime_materials.py reports when every material came out right."""
    return {"pass": True, "error": None, "shading": {"roughness": {"value": 0.5, "from": "/Engine/BasicShapes/BasicShapeMaterial"}},
            "materials": [{"key": key, "path": entry["ue_path"], "class": "Material", "is_base_material": True,
                           "base_color_readback": list(entry["rgba"]), "connected": ["MP_BASE_COLOR", "MP_ROUGHNESS"],
                           "blend_mode": "BLEND_OPAQUE", "shading_model": "MSM_DEFAULT_LIT"}
                          for key, entry in patch["materials"].items()]}


def _registry_with_patch(tmp_path):
    """A copy of the real registry with the pending patch merged in, as apply would leave it."""
    from scripts.apply_runtime_materials import material_entries

    path = tmp_path / "registry.json"
    registry = json.loads(REGISTRY.read_text())
    registry["materials"].update(material_entries(_patch(), _good_report(_patch())))
    path.write_text(json.dumps(registry))
    return path


def test_the_pending_patch_holds_one_base_material_per_palette_colour_in_the_palette_s_rgba():
    patch = _patch()
    palette = json.loads(REGISTRY.read_text())["materials"]
    assert patch["format"] == "autofly_ue5_registry_patch/1"
    assert set(patch["materials"]) == {f"runtime_{c}" for c in COLOURS}
    for colour in COLOURS:
        entry = patch["materials"][f"runtime_{colour}"]
        assert entry["kind"] == "base_material" and entry["palette"] == colour
        assert entry["ue_path"] == f"/Game/AutoFly/Materials/Runtime/M_{colour.capitalize()}"
        assert entry["rgba"] == palette[colour]["rgba"], "the runtime colour must match the scene palette's"
        assert palette[colour]["kind"] == "color_instance", "the palette entry itself stays the editor's instance"


def test_the_patch_is_not_in_the_registry_until_the_editor_has_made_the_materials():
    """Merging before the editor ran would make the pool draw 'red cube' and spawn it grey: the registry gets the entries
    only from the apply step."""
    registry = json.loads(REGISTRY.read_text())
    if any(k in registry["materials"] for k in _patch()["materials"]):
        # applied in the live window: then every entry must be exactly as apply writes it from a report
        for key in _patch()["materials"]:
            assert registry["materials"][key]["kind"] == "base_material"
            assert registry["materials"][key]["made_by"].endswith("create_runtime_materials.py")


def test_a_base_material_is_runtime_spawnable_and_the_fake_accepts_its_path(tmp_path):
    from autofly_ue5.expert.episode import RUNTIME_SPAWNABLE_MATERIAL_KINDS, _runtime_material_path
    from autofly_ue5.scenes.model import load_registry
    from autofly_ue5.sim.fake import FakeSimulator
    from autofly_ue5.sim.types import Pose

    assert RUNTIME_SPAWNABLE_MATERIAL_KINDS == ("engine", "base_material")
    registry = load_registry(_registry_with_patch(tmp_path))
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    for colour in COLOURS:
        path = _runtime_material_path(registry, f"runtime_{colour}")
        assert path == f"/Game/AutoFly/Materials/Runtime/M_{colour.capitalize()}"
        sim.spawn("target", "Cube", Pose(5.0, 0.0, -0.6, 0.0), (1.2, 1.2, 1.2), path)
    assert _runtime_material_path(registry, "red") is None, "the palette's instance stays unspawnable"


def test_the_pool_loader_accepts_an_entry_painted_with_a_base_material(tmp_path):
    from autofly_ue5.dataset.targets import TargetPool, TargetPoolError
    from autofly_ue5.scenes.model import load_registry

    registry = load_registry(_registry_with_patch(tmp_path))
    entry = {"name": "red cube", "category": "geometry", "seen": True, "asset": "cube", "scale": [1.2, 1.2, 1.2],
             "material": "runtime_red", "status": "primitive", "source": None, "height_m": 1.2}
    pool = TargetPool.from_json({"version": 1, "targets": [entry]}, registry=registry)
    assert pool.entries[0].usable and pool.entries[0].material == "runtime_red"
    with pytest.raises(TargetPoolError, match="color_instance"):
        TargetPool.from_json({"version": 1, "targets": [{**entry, "material": "red"}]}, registry=registry)
    with pytest.raises(TargetPoolError, match="not in the registry"):
        TargetPool.from_json({"version": 1, "targets": [entry]}, registry=load_registry(_registry_without_patch(tmp_path)))


def _registry_without_patch(tmp_path):
    path = tmp_path / "registry_without.json"
    registry = json.loads(REGISTRY.read_text())
    for key in _patch()["materials"]:
        registry["materials"].pop(key, None)
    path.write_text(json.dumps(registry))
    return path


def test_the_real_pool_paints_its_geometry_with_base_materials_and_loads_once_the_patch_is_applied(tmp_path):
    """Every primitive of assets/targets.json names a material the runtime can apply (M_Orange, or a runtime_* base
    material), under its unchanged name; with the patch merged the whole pool loads and every primitive is drawable."""
    from autofly_ue5.dataset.targets import TargetPool
    from autofly_ue5.scenes.model import load_registry

    targets = json.loads(TARGETS.read_text())["targets"]
    primitives = [t for t in targets if t["status"] == "primitive"]
    assert len(primitives) == 12
    for t in primitives:
        colour = t["name"].split()[0]
        expected = "orange" if colour == "orange" else f"runtime_{colour}"
        assert t["material"] == expected, t["name"]
    pool = TargetPool.load(TARGETS, registry=load_registry(_registry_with_patch(tmp_path)))
    assert sum(1 for e in pool.entries if e.status == "primitive" and e.usable) == 12
    assert len(pool.for_split("train")) >= 6


def test_apply_checks_the_report_before_the_registry_is_written(tmp_path):
    from scripts.apply_runtime_materials import check_material_report

    patch = _patch()
    assert check_material_report(patch, _good_report(patch)) == []
    failed = {**_good_report(patch), "pass": False, "error": "boom"}
    assert any("pass=False" in p for p in check_material_report(patch, failed))
    missing = _good_report(patch)
    missing["materials"].pop(0)
    assert any("runtime_red" in p and "not in the report" in p for p in check_material_report(patch, missing))
    instance = _good_report(patch)
    instance["materials"][1].update({"class": "MaterialInstanceConstant", "is_base_material": False})
    assert any("runtime_blue" in p and "base UMaterial" in p for p in check_material_report(patch, instance))
    moved = _good_report(patch)
    moved["materials"][2]["path"] = "/Game/AutoFly/Materials/M_Yellow"
    assert any("runtime_yellow" in p and "path" in p for p in check_material_report(patch, moved))
    off = _good_report(patch)
    off["materials"][3]["base_color_readback"] = [0.5, 0.5, 0.5, 1.0]
    assert any("runtime_green" in p and "colour" in p for p in check_material_report(patch, off))
    unwired = _good_report(patch)
    unwired["materials"][4]["connected"] = ["MP_ROUGHNESS"]
    assert any("runtime_white" in p and "MP_BASE_COLOR" in p for p in check_material_report(patch, unwired))


def test_apply_refuses_a_material_the_editor_log_says_failed_to_compile(tmp_path):
    from scripts.apply_runtime_materials import main

    patch_path = tmp_path / "patch.json"
    shutil.copy(PATCH, patch_path)
    registry = tmp_path / "registry.json"
    shutil.copy(REGISTRY, registry)
    before = registry.read_bytes()
    report = tmp_path / "runtime_materials.json"
    report.write_text(json.dumps(_good_report(_patch())))
    log = tmp_path / "runtime_materials.log"
    log.write_text("LogMaterial: Warning: [AssetLog] /x/ue_project/Content/AutoFly/Materials/Runtime/M_Red.uasset: "
                   "Failed to compile Material for platform VULKAN_SM6, Default Material will be used in game.\n")
    assert main(["--report", str(report), "--patch", str(patch_path), "--registry", str(registry)]) == 2
    assert registry.read_bytes() == before


def test_apply_merges_the_entries_once_and_a_second_apply_changes_nothing(tmp_path):
    from scripts.apply_runtime_materials import main

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(json.loads(_registry_without_patch(tmp_path).read_text())))
    report = tmp_path / "runtime_materials.json"
    report.write_text(json.dumps(_good_report(_patch())))
    (tmp_path / "runtime_materials.log").write_text("LogInit: nothing failed\n")
    assert main(["--report", str(report), "--patch", str(PATCH), "--registry", str(registry)]) == 0
    merged = json.loads(registry.read_text())["materials"]
    assert merged["runtime_red"]["kind"] == "base_material" and merged["runtime_red"]["rgba"] == [0.8, 0.1, 0.1, 1.0]
    assert merged["runtime_red"]["roughness"] == 0.5 and merged["red"]["kind"] == "color_instance"
    once = registry.read_bytes()
    assert main(["--report", str(report), "--patch", str(PATCH), "--registry", str(registry)]) == 0
    assert registry.read_bytes() == once
    # a different editor result under the same key is refused unless asked for
    other = _good_report(_patch())
    other["shading"]["roughness"]["value"] = 0.8
    report.write_text(json.dumps(other))
    assert main(["--report", str(report), "--patch", str(PATCH), "--registry", str(registry)]) == 2
    assert registry.read_bytes() == once


def test_a_missing_editor_log_is_a_refusal_not_a_pass(tmp_path):
    from scripts.apply_runtime_materials import main

    registry = tmp_path / "registry.json"
    shutil.copy(_registry_without_patch(tmp_path), registry)
    report = tmp_path / "runtime_materials.json"
    report.write_text(json.dumps(_good_report(_patch())))
    assert main(["--report", str(report), "--patch", str(PATCH), "--registry", str(registry)]) == 2
