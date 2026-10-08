"""UE-side: make the scene palette's colours as base UMaterials for runtime spawns (plan 5 C6; spec §7.1).

    env -u PYTHONPATH .venv/bin/python scripts/ue_editor_run.py autofly_ue5/scenes/ue/create_runtime_materials.py \\
        --out runs/assets/runtime_materials.json [--arg patch=/abs/path/to/assets/pending/runtime_materials.registry.json]

Project AirSim's setMaterial loads a material with an exact UMaterial class filter, so the palette's MI_Red ... (instances
of BasicShapeMaterial, made by build_level.py for scenes) cannot colour a spawned target, which then shows the mesh's own
grey. For every entry of the pending registry patch this creates a plain Material at its ue_path: a Constant3Vector of the
palette's linear RGB into BaseColor, default lit, opaque, with the roughness (and metallic, specular) of the palette's
parent BasicShapeMaterial when its inputs are constants or parameters, else M_Orange's (the pool's orange target), else
UE's unconnected default 0.5 roughness. An existing asset at the path is deleted first (a re-run is a re-make), the
material is recompiled and saved, then read back from disk for the report: class, base colour, connected inputs, blend
mode and shading model. Whether it compiled shows only in the editor log ("Failed to compile Material", a warning), which
scripts/apply_runtime_materials.py reads before it merges the patch into assets/registry.json.

Reads env AUTOFLY_UE_OUT and AUTOFLY_UE_ARGS (scripts/ue_editor_run.py); stdlib + unreal only; quits the editor.
"""
import json
import os
import traceback

import unreal

OUT = os.environ["AUTOFLY_UE_OUT"]
ARGS = json.loads(os.environ.get("AUTOFLY_UE_ARGS", "{}"))
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PATCH_PATH = ARGS.get("patch") or os.path.join(_REPO, "assets", "pending", "runtime_materials.registry.json")
REPORT = {"pass": False, "error": None, "engine_version": unreal.SystemLibrary.get_engine_version(), "patch": PATCH_PATH,
          "shading": {}, "materials": [], "folder_contents": []}
MEL = unreal.MaterialEditingLibrary
EAL = unreal.EditorAssetLibrary
# Where the palette's look comes from: the MI_* colours are instances of BasicShapeMaterial (registry "parent"), and the
# pool's one runtime colour so far is M_Orange.
SHADING_SOURCES = ("/Engine/BasicShapes/BasicShapeMaterial", "/Game/Geometry/Materials/M_Orange")
SHADING_INPUTS = (("roughness", unreal.MaterialProperty.MP_ROUGHNESS, 0.5),
                  ("metallic", unreal.MaterialProperty.MP_METALLIC, None),
                  ("specular", unreal.MaterialProperty.MP_SPECULAR, None))
REPORTED_INPUTS = (unreal.MaterialProperty.MP_BASE_COLOR, unreal.MaterialProperty.MP_ROUGHNESS,
                   unreal.MaterialProperty.MP_METALLIC, unreal.MaterialProperty.MP_SPECULAR)


def enum_name(value):
    return getattr(value, "name", None) or str(value).split(".")[-1].split(":")[0]


def check(ok, what):
    if not ok:
        raise RuntimeError("AUTOFLY CHECK FAILED: " + what)


def constant_input(material, prop):
    """The value wired into `prop` when it is a constant or a scalar parameter's default; None when nothing is wired or
    the input is computed (a texture, a graph)."""
    node = MEL.get_material_property_input_node(material, prop)
    if node is None:
        return None
    if isinstance(node, unreal.MaterialExpressionConstant):
        return float(node.get_editor_property("r"))
    if isinstance(node, unreal.MaterialExpressionScalarParameter):
        return float(node.get_editor_property("default_value"))
    return None


def palette_shading():
    """name -> {"value", "from"}: the first source with a constant for that input, else the fallback (None: leave the
    input unconnected, UE's default)."""
    sources = []
    for path in SHADING_SOURCES:
        material = unreal.load_asset(path)
        if isinstance(material, unreal.Material):
            sources.append((path, material))
    shading = {}
    for name, prop, fallback in SHADING_INPUTS:
        chosen = {"value": fallback, "from": "UE default (unconnected)" if fallback is None else "fallback"}
        for path, material in sources:
            value = constant_input(material, prop)
            if value is not None:
                chosen = {"value": value, "from": path}
                break
        shading[name] = chosen
    return shading


def connected_inputs(material):
    return [enum_name(p) for p in REPORTED_INPUTS if MEL.get_material_property_input_node(material, p) is not None]


def make(key, entry, shading):
    path = entry["ue_path"]
    folder, name = path.rsplit("/", 1)
    info = {"key": key, "path": None, "requested_path": path}
    REPORT["materials"].append(info)
    if EAL.does_asset_exist(path):
        check(EAL.delete_asset(path), "delete the old " + path)
    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())
    check(material is not None, "create " + path)
    r, g, b, a = (float(v) for v in entry["rgba"])
    colour = MEL.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -500, 0)
    colour.set_editor_property("constant", unreal.LinearColor(r, g, b, a))
    check(MEL.connect_material_property(colour, "", unreal.MaterialProperty.MP_BASE_COLOR), "connect the colour of " + path)
    row = 1
    for input_name, prop, _fallback in SHADING_INPUTS:
        value = shading[input_name]["value"]
        if value is None:
            continue
        constant = MEL.create_material_expression(material, unreal.MaterialExpressionConstant, -500, 150 * row)
        constant.set_editor_property("r", float(value))
        check(MEL.connect_material_property(constant, "", prop), "connect %s of %s" % (input_name, path))
        row += 1
    MEL.recompile_material(material)
    check(EAL.save_loaded_asset(material, False), "save " + path)
    loaded = unreal.load_asset(path)
    check(loaded is not None, "load back " + path)
    node = MEL.get_material_property_input_node(loaded, unreal.MaterialProperty.MP_BASE_COLOR)
    readback = None
    if isinstance(node, unreal.MaterialExpressionConstant3Vector):
        c = node.get_editor_property("constant")
        readback = [c.r, c.g, c.b, c.a]
    info.update({"path": loaded.get_path_name().split(".")[0], "class": loaded.get_class().get_name(),
                 "is_base_material": isinstance(loaded, unreal.Material), "base_color_readback": readback,
                 "connected": connected_inputs(loaded), "blend_mode": enum_name(loaded.get_editor_property("blend_mode")),
                 "shading_model": enum_name(loaded.get_editor_property("shading_model")),
                 "expressions": MEL.get_num_material_expressions(loaded)})
    statistics = getattr(MEL, "get_statistics", None)
    if statistics is not None:  # information only: the editor log is what says whether it compiled
        try:
            stats = statistics(loaded)
            info["pixel_shader_instructions"] = int(stats.get_editor_property("num_pixel_shader_instructions"))
        except Exception as err:  # noqa: BLE001
            info["statistics_error"] = str(err)


def main():
    patch = json.load(open(PATCH_PATH))
    check(patch.get("format") == "autofly_ue5_registry_patch/1", "patch format " + str(patch.get("format")))
    shading = palette_shading()
    REPORT["shading"] = shading
    folders = set()
    for key, entry in patch["materials"].items():
        check(entry.get("kind") == "base_material", key + " is not a base_material")
        unreal.log("AUTOFLY runtime material " + key + " -> " + entry["ue_path"])
        make(key, entry, shading)
        folders.add(entry["ue_path"].rsplit("/", 1)[0])
    for folder in sorted(folders):
        REPORT["folder_contents"] += [str(p) for p in EAL.list_assets(folder, recursive=True, include_folder=False)]
    bad = [m["key"] for m in REPORT["materials"] if not m.get("is_base_material") or m.get("base_color_readback") is None]
    check(not bad, "not a base material with a constant base colour: %s" % bad)
    REPORT["pass"] = True


try:
    main()
except Exception as err:  # noqa: BLE001 -- the report must say what happened
    REPORT["error"] = "%s: %s\n%s" % (type(err).__name__, err, traceback.format_exc())
    unreal.log_error("AUTOFLY runtime materials failed: %s" % err)
finally:
    with open(OUT, "w") as handle:
        json.dump(REPORT, handle, indent=2)
    unreal.SystemLibrary.quit_editor()
