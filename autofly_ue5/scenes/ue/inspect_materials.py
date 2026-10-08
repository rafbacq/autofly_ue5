"""UE-side: list every material an imported AutoFly model uses, with its class, parent chain, blend mode, two-sidedness
and texture parameters (scripts/ue_editor_run.py; env AUTOFLY_UE_OUT). Read-only."""
import json
import os
import traceback

import unreal

OUT = os.environ["AUTOFLY_UE_OUT"]
ROOT = json.loads(os.environ.get("AUTOFLY_UE_ARGS", "{}")).get("root", "/Game/AutoFly/Assets")
REPORT = {"pass": False, "error": None, "meshes": []}
ASSETS = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
MEL = unreal.MaterialEditingLibrary


def enum_name(value):
    return str(value).split(".")[-1].split(":")[0]


def describe_material(material):
    chain = []
    m = material
    while m is not None:
        chain.append({"path": m.get_path_name(), "class": m.get_class().get_name()})
        m = m.get_editor_property("parent") if isinstance(m, unreal.MaterialInstance) else None
    base = chain[-1]
    info = {"chain": chain}
    if isinstance(material, unreal.MaterialInstance):
        ov = material.get_editor_property("base_property_overrides")
        info["overrides"] = {"override_blend_mode": ov.get_editor_property("override_blend_mode"),
                             "blend_mode": enum_name(ov.get_editor_property("blend_mode")),
                             "override_two_sided": ov.get_editor_property("override_two_sided"),
                             "two_sided": ov.get_editor_property("two_sided"),
                             "override_opacity_mask_clip_value": ov.get_editor_property("override_opacity_mask_clip_value"),
                             "opacity_mask_clip_value": ov.get_editor_property("opacity_mask_clip_value")}
        info["effective_blend_mode"] = enum_name(material.get_blend_mode())
        info["texture_parameters"] = [str(n) for n in MEL.get_texture_parameter_names(material)]
        info["scalar_parameters"] = [str(n) for n in MEL.get_scalar_parameter_names(material)]
        info["static_switch_parameters"] = [str(n) for n in MEL.get_static_switch_parameter_names(material)]
    else:
        info["effective_blend_mode"] = enum_name(material.get_editor_property("blend_mode"))
        info["two_sided"] = material.get_editor_property("two_sided")
    info["base"] = base
    return info


def main():
    for path in sorted(ASSETS.list_assets(ROOT, recursive=True, include_folder=False)):
        asset = unreal.load_asset(path.split(".")[0])
        if not isinstance(asset, unreal.StaticMesh):
            continue
        mats = []
        for i, slot in enumerate(asset.get_editor_property("static_materials")):
            mat = slot.get_editor_property("material_interface")
            mats.append({"slot": i, "slot_name": str(slot.get_editor_property("material_slot_name")),
                         "material": describe_material(mat) if mat is not None else None})
        REPORT["meshes"].append({"mesh": asset.get_path_name(), "materials": mats})
    REPORT["pass"] = True


try:
    main()
except Exception as err:  # noqa: BLE001
    REPORT["error"] = "%s: %s\n%s" % (type(err).__name__, err, traceback.format_exc())
finally:
    with open(OUT, "w") as handle:
        json.dump(REPORT, handle, indent=2)
    unreal.SystemLibrary.quit_editor()
