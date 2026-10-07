"""UE-side asset import (full editor, -ExecCmds="py ..."; UE embedded Python; stdlib + unreal only).

Reads the import spec (env AUTOFLY_IMPORT_SPEC, from scripts/import_assets.py plan) and writes a report (env
AUTOFLY_IMPORT_OUT). For every glTF model: import it under its destination, then for every static mesh that came out
enable Nanite, set complex-as-simple collision, save, and report its bounds, triangle count and materials. For every
ambientCG set: import the four texture maps, set each one's sampler kind (sRGB colour; linear, DirectX-convention
normal; linear masks), build one Material with a tiled TextureCoordinate feeding four TextureSamples into BaseColor,
Normal, Roughness and AmbientOcclusion, recompile, save, and report which inputs are connected.

Every failed check raises; the report is written in the finally block either way (pass False with the error), and the
editor is asked to quit, as scripts/build_level.sh's scripts do. Nothing here is a scene: no level is opened or saved.
"""
import json
import os
import traceback

import unreal

SPEC = json.load(open(os.environ["AUTOFLY_IMPORT_SPEC"]))
OUT = os.environ["AUTOFLY_IMPORT_OUT"]
REPORT = {"pass": False, "error": None, "engine_version": unreal.SystemLibrary.get_engine_version(), "models": [], "materials": [],
          "pipeline": {}}
NANITE_FALLBACK_PERCENT = float(SPEC.get("nanite_fallback_percent", 0.10))  # the fallback mesh is what complex collision uses


def enum_name(value):
    return getattr(value, "name", None) or str(value).split(".")[-1].split(":")[0]

INPUTS = {
    "color": ("RGB", unreal.MaterialProperty.MP_BASE_COLOR, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR),
    "normal": ("RGB", unreal.MaterialProperty.MP_NORMAL, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL),
    "roughness": ("R", unreal.MaterialProperty.MP_ROUGHNESS, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR),
    "ao": ("R", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR),
}


def check(ok, what):
    if not ok:
        raise RuntimeError("AUTOFLY CHECK FAILED: " + what)


def model_pipeline():
    """The pipeline stack override an AssetImportTask takes (`options`): the project's default glTF stack -- its generic
    assets pipeline and its glTF material pipeline -- with the generic one changed in memory so each glTF node becomes
    its own static mesh in its own frame, named after the node: no baking of node transforms (a Poly Haven set lays its
    variants out side by side), no combining, Nanite on. A plain pipeline object in `options` is ignored (measured:
    2026-10-07, the first two runs kept the file names and the baked offsets)."""
    generic = unreal.load_asset("/Interchange/Pipelines/DefaultGLTFAssetsPipeline")
    gltf = unreal.load_asset("/Interchange/Pipelines/DefaultGLTFPipeline")
    check(generic is not None and gltf is not None, "Interchange's default glTF pipelines")
    generic.set_editor_property("use_source_name_for_asset", False)
    settings = {"stack": ["DefaultGLTFAssetsPipeline (modified in memory)", "DefaultGLTFPipeline"], "use_source_name_for_asset": False}
    common = generic.get_editor_property("common_meshes_properties")
    for name in ("bake_meshes", "bake_pivot_meshes"):
        common.set_editor_property(name, False)
        settings[name] = False
    meshes = generic.get_editor_property("mesh_pipeline")
    meshes.set_editor_property("build_nanite", True)
    meshes.set_editor_property("combine_static_meshes", False)
    settings["build_nanite"] = True
    settings["combine_static_meshes"] = False
    override = unreal.InterchangePipelineStackOverride()
    override.add_pipeline(generic)
    override.add_pipeline(gltf)
    REPORT["pipeline"] = settings
    return override


def import_file(filename, destination, pipeline=None):
    """Run one AssetImportTask and return the object paths now under `destination` (listed, not trusted from the task:
    Interchange fills imported_object_paths late or not at all for some formats)."""
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", filename)
    task.set_editor_property("destination_path", destination)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("replace_existing_settings", True)
    if pipeline is not None:
        task.set_editor_property("options", pipeline)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    reported = [str(p) for p in (task.get_editor_property("imported_object_paths") or [])]
    listed = [str(p) for p in unreal.EditorAssetLibrary.list_assets(destination, recursive=True, include_folder=False)]
    return reported, listed


def static_meshes_under(paths):
    meshes = []
    for path in paths:
        asset = unreal.load_asset(path.split(".")[0]) if "." in path else unreal.load_asset(path)
        if isinstance(asset, unreal.StaticMesh):
            meshes.append(asset)
    return meshes


def enable_nanite(mesh):
    """Nanite on, with a fallback mesh of NANITE_FALLBACK_PERCENT of the triangles: the default relative-error target
    left a 2 M-triangle tree with a 2,300-triangle fallback, and complex collision runs on the fallback."""
    sms = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    settings = sms.get_nanite_settings(mesh)
    settings.set_editor_property("enabled", True)
    target = getattr(unreal, "NaniteFallbackTarget", None)
    if target is not None and hasattr(target, "PERCENT_TRIANGLES"):
        settings.set_editor_property("fallback_target", target.PERCENT_TRIANGLES)
    settings.set_editor_property("fallback_percent_triangles", NANITE_FALLBACK_PERCENT)
    settings.set_editor_property("fallback_relative_error", 0.0)
    sms.set_nanite_settings(mesh, settings, True)
    got = sms.get_nanite_settings(mesh)
    return {"nanite_enabled": bool(got.get_editor_property("enabled")),
            "nanite_fallback_target": enum_name(got.get_editor_property("fallback_target")),
            "nanite_fallback_percent_triangles": float(got.get_editor_property("fallback_percent_triangles")),
            "fallback_triangles": mesh.get_num_triangles(0)}


def set_complex_collision(mesh):
    body = mesh.get_editor_property("body_setup")
    check(body is not None, "body setup of " + mesh.get_path_name())
    body.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
    sms = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    return enum_name(sms.get_collision_complexity(mesh))


def describe_mesh(mesh):
    bounds = mesh.get_bounds()
    origin, extent = bounds.origin, bounds.box_extent
    materials = []
    for sm in mesh.get_editor_property("static_materials"):
        mi = sm.get_editor_property("material_interface")
        materials.append(mi.get_path_name() if mi is not None else None)
    return {"name": mesh.get_name(), "path": mesh.get_path_name().split(".")[0],
            "origin_cm": [origin.x, origin.y, origin.z], "box_extent_cm": [extent.x, extent.y, extent.z],
            "triangles": mesh.get_num_triangles(0), "vertices": mesh.get_num_vertices(0), "sections": mesh.get_num_sections(0),
            "materials": materials}


def import_model(model):
    entry = {"id": model["id"], "destination": model["destination"], "gltf": model["gltf"], "meshes": [], "imported_paths": []}
    REPORT["models"].append(entry)
    if unreal.EditorAssetLibrary.does_directory_exist(model["destination"]):
        # Ours alone (nothing else writes under /Game/AutoFly/Assets): a re-import must not leave an earlier run's meshes
        # behind under other names.
        check(unreal.EditorAssetLibrary.delete_directory(model["destination"]), "delete the old " + model["destination"])
    reported, listed = import_file(model["gltf"], model["destination"], model_pipeline())
    entry["imported_paths"] = listed
    meshes = static_meshes_under(listed)
    check(meshes, "no static mesh came out of " + model["gltf"] + " (imported: " + str(reported) + ")")
    for mesh in meshes:
        info = describe_mesh(mesh)
        info["source_triangles"] = info["triangles"]  # as imported, before the Nanite rebuild changes LOD0 to the fallback
        if SPEC.get("nanite", True):
            info.update(enable_nanite(mesh))
        else:
            info["nanite_enabled"] = False
        info["collision_trace_flag"] = set_complex_collision(mesh) if SPEC.get("collision") == "complex_as_simple" else None
        check(unreal.EditorAssetLibrary.save_loaded_asset(mesh, False), "save " + mesh.get_path_name())
        entry["meshes"].append(info)
    # Rename each mesh to its registry key: Interchange named it after the glTF mesh ("Cube_070"), the spec says which
    # node that is and what the scene files will call it.
    by_mesh = {n["mesh"]: n for n in model["nodes"]}
    if len(model["nodes"]) == 1 and len(entry["meshes"]) == 1:
        by_mesh = {entry["meshes"][0]["name"]: model["nodes"][0]}
    for info in entry["meshes"]:
        node = by_mesh.get(info["name"])
        info["imported_name"] = info["name"]
        info["node"] = node["name"] if node else None
        if node is not None and info["name"] != node["key"]:
            target = model["destination"] + "/" + node["key"]
            check(unreal.EditorAssetLibrary.rename_asset(info["path"], target), "rename %s to %s" % (info["path"], target))
            info["name"], info["path"] = node["key"], target
    wanted = {n["key"] for n in model["nodes"]}
    got = {m["name"] for m in entry["meshes"]}
    entry["missing_nodes"] = sorted(wanted - got)
    entry["extra_meshes"] = sorted(got - wanted)


def import_texture(path, destination, kind):
    reported, listed = import_file(path, destination)
    stem = os.path.splitext(os.path.basename(path))[0]
    candidates = [p for p in listed if p.split(".")[0].rsplit("/", 1)[-1] == stem]
    check(candidates, "texture %s not found under %s after import (%s)" % (stem, destination, listed))
    texture = unreal.load_asset(candidates[0].split(".")[0])
    check(isinstance(texture, unreal.Texture2D), "not a Texture2D: " + candidates[0])
    if kind == "normal":
        texture.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
        texture.set_editor_property("srgb", False)
        texture.set_editor_property("flip_green_channel", False)  # ambientCG's NormalDX is already UE's convention
    elif kind in ("roughness", "ao"):
        texture.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
        texture.set_editor_property("srgb", False)
    else:
        texture.set_editor_property("srgb", True)
    check(unreal.EditorAssetLibrary.save_loaded_asset(texture, False), "save " + texture.get_path_name())
    return texture


def build_material(spec):
    entry = {"id": spec["id"], "name": spec["name"], "path": None, "textures": {}, "connected": [], "tiling": spec["tiling"]}
    REPORT["materials"].append(entry)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    texture_folder = spec["destination"] + "/Textures/" + spec["id"]
    textures = {}
    for kind, path in spec["textures"].items():
        textures[kind] = import_texture(path, texture_folder, kind)
        entry["textures"][kind] = textures[kind].get_path_name().split(".")[0]
    path = spec["destination"] + "/" + spec["name"]
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        check(unreal.EditorAssetLibrary.delete_asset(path), "delete the old " + path)
    material = tools.create_asset(spec["name"], spec["destination"], unreal.Material, unreal.MaterialFactoryNew())
    check(material is not None, "create " + path)
    mel = unreal.MaterialEditingLibrary
    coord = mel.create_material_expression(material, unreal.MaterialExpressionTextureCoordinate, -900, 0)
    coord.set_editor_property("u_tiling", float(spec["tiling"]))
    coord.set_editor_property("v_tiling", float(spec["tiling"]))
    for row, (kind, (output, prop, sampler)) in enumerate(INPUTS.items()):
        sample = mel.create_material_expression(material, unreal.MaterialExpressionTextureSample, -500, 300 * row - 450)
        sample.set_editor_property("texture", textures[kind])
        sample.set_editor_property("sampler_type", sampler)
        check(mel.connect_material_expressions(coord, "", sample, "UVs"), "connect UVs of " + kind)
        check(mel.connect_material_property(sample, output, prop), "connect %s to %s" % (kind, prop))
        if mel.get_material_property_input_node(material, prop) is not None:
            entry["connected"].append(enum_name(prop))
    mel.recompile_material(material)
    check(unreal.EditorAssetLibrary.save_loaded_asset(material, False), "save " + path)
    entry["path"] = material.get_path_name().split(".")[0]
    entry["expressions"] = mel.get_num_material_expressions(material)


def main():
    for model in SPEC["models"]:
        unreal.log("AUTOFLY import model " + model["id"])
        import_model(model)
    for material in SPEC["materials"]:
        unreal.log("AUTOFLY build material " + material["name"])
        build_material(material)
    REPORT["pass"] = True


try:
    main()
except Exception as err:  # noqa: BLE001 -- the report must say what happened
    REPORT["error"] = "%s: %s\n%s" % (type(err).__name__, err, traceback.format_exc())
    unreal.log_error("AUTOFLY import failed: %s" % err)
finally:
    with open(OUT, "w") as handle:
        json.dump(REPORT, handle, indent=2)
    unreal.SystemLibrary.quit_editor()
