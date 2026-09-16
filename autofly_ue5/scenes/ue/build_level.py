"""UE-side map builder (UnrealEditor-Cmd -run=pythonscript; UE embedded Python; stdlib + unreal only).

Reads the level spec (env AUTOFLY_LEVEL_SPEC), creates the map with ground, obstacle actors, SunSky
lighting and the ProjectAirSim GameMode override, saves it and writes a report (env AUTOFLY_BUILD_OUT).
Every failed check raises, so the commandlet logs "Python script executed with errors".
"""
import json
import os

import unreal

SPEC = json.load(open(os.environ["AUTOFLY_LEVEL_SPEC"]))
OUT = os.environ["AUTOFLY_BUILD_OUT"]
TOL_CM = 1.0
REPORT = {"map_path": SPEC["map_path"], "materials": {}, "actors": [], "pass": False}


def check(ok, what):
    if not ok:
        raise RuntimeError("AUTOFLY CHECK FAILED: " + what)


def make_material(entry, ass):
    if entry["kind"] == "engine":
        material = unreal.load_asset(entry["ue_path"])
        check(material is not None, "engine material " + entry["ue_path"])
        return material
    check(entry["kind"] == "color_instance", "unknown material kind " + str(entry["kind"]))
    folder, name = entry["ue_path"].rsplit("/", 1)
    parent = unreal.load_asset(entry["parent"])
    check(parent is not None, "parent material " + entry["parent"])
    params = [str(n) for n in unreal.MaterialEditingLibrary.get_vector_parameter_names(parent)]
    REPORT["materials"][entry["name"]] = {"parent_vector_params": params}
    check(entry["parameter"] in params, "parameter %s not in %s" % (entry["parameter"], params))
    if ass.does_asset_exist(entry["ue_path"]):
        mic = ass.load_asset(entry["ue_path"])
    else:
        mic = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, folder, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    check(mic is not None, "create or load " + entry["ue_path"])
    unreal.MaterialEditingLibrary.set_material_instance_parent(mic, parent)
    got_parent = mic.get_editor_property("parent")
    check(got_parent is not None and got_parent.get_path_name() == parent.get_path_name(),
          "parent of %s is %s" % (entry["ue_path"], got_parent.get_path_name() if got_parent is not None else None))
    # The setter's bool is never set in UE 5.7.4 and is always False (MaterialEditingLibrary.cpp:1253-1262): read back.
    unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
        mic, entry["parameter"], unreal.LinearColor(*entry["rgba"]))
    unreal.MaterialEditingLibrary.update_material_instance(mic)
    got = unreal.MaterialEditingLibrary.get_material_instance_vector_parameter_value(mic, entry["parameter"])
    readback = [got.r, got.g, got.b, got.a]
    REPORT["materials"][entry["name"]]["readback_rgba"] = readback
    check(all(abs(a - b) < 1e-4 for a, b in zip(readback, entry["rgba"])),
          "set %s on %s: read back %s" % (entry["parameter"], entry["ue_path"], readback))
    check(ass.save_loaded_asset(mic, False), "save " + entry["ue_path"])
    return mic


def spawn_exposure_volume(spec, eas):
    """R16 (controller ruling): fixed MANUAL exposure via an unbound PostProcessVolume, not auto-exposure.
    Auto-exposure (eye adaptation) carries temporal state, so the same pose would render different pixels
    depending on where the camera looked before -- not reproducible for (image, action) pairs. With
    AutoExposureApplyPhysicalCameraExposure off, the engine's eye-adaptation math (EV100ToLuminance /
    EyeAdaptationCommon.ush) collapses the final image-intensity multiplier to exactly 2**bias_ev, a pure
    function of the one number in the level spec's `exposure` block -- independent of scene content, pose
    and camera history."""
    volume = eas.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0.0, 0.0, -500.0))
    check(volume is not None, "spawn PostProcessVolume")
    volume.set_actor_label("Exposure")
    volume.set_editor_property("tags", [spec["tag"]])
    volume.set_editor_property("unbound", True)
    volume.set_editor_property("enabled", True)
    check(spec["method"] == "manual", "exposure method " + str(spec["method"]) + " is not implemented (only manual)")
    settings = volume.get_editor_property("settings")
    settings.set_editor_property("override_auto_exposure_method", True)
    settings.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    settings.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
    settings.set_editor_property("auto_exposure_apply_physical_camera_exposure", bool(spec["apply_physical_camera_exposure"]))
    settings.set_editor_property("override_auto_exposure_bias", True)
    settings.set_editor_property("auto_exposure_bias", float(spec["bias_ev"]))
    volume.set_editor_property("settings", settings)
    # Struct properties are returned by value in UE python (confirmed live, R16 probe): read the actor's
    # settings back fresh rather than trusting the local `settings` object, to catch a silently-dropped set.
    readback = volume.get_editor_property("settings")
    got_method = readback.get_editor_property("auto_exposure_method")
    got_bias = readback.get_editor_property("auto_exposure_bias")
    got_physical_camera = readback.get_editor_property("auto_exposure_apply_physical_camera_exposure")
    check(readback.get_editor_property("override_auto_exposure_method"), "PostProcessVolume override_auto_exposure_method not set")
    check(got_method == unreal.AutoExposureMethod.AEM_MANUAL, "auto_exposure_method readback " + str(got_method))
    check(readback.get_editor_property("override_auto_exposure_bias"), "PostProcessVolume override_auto_exposure_bias not set")
    check(abs(got_bias - spec["bias_ev"]) < 1e-4, "auto_exposure_bias readback %s != %s" % (got_bias, spec["bias_ev"]))
    # R17 (Finding 2): this override was set alongside method/bias but never read back -- close the same
    # gap the surrounding checks already close for the other two properties (struct properties are
    # returned by value in UE python, so a silently-dropped set here would defeat the determinism claim
    # without anything in build, verify or the live gate catching it).
    check(readback.get_editor_property("override_auto_exposure_apply_physical_camera_exposure"),
          "PostProcessVolume override_auto_exposure_apply_physical_camera_exposure not set")
    check(got_physical_camera == bool(spec["apply_physical_camera_exposure"]),
          "auto_exposure_apply_physical_camera_exposure readback %s != %s" % (got_physical_camera, spec["apply_physical_camera_exposure"]))
    check(volume.get_editor_property("unbound") is True, "PostProcessVolume is not unbound")
    check(volume.get_editor_property("enabled") is True, "PostProcessVolume is not enabled")
    REPORT["exposure"] = {"tag": spec["tag"], "auto_exposure_method": str(got_method), "auto_exposure_bias": got_bias,
                          "apply_physical_camera_exposure": got_physical_camera,
                          "unbound": True, "enabled": True}
    return volume


def spawn_mesh(a, eas, materials):
    mesh = unreal.load_asset(a["mesh"])
    check(mesh is not None, "mesh " + a["mesh"])
    actor = eas.spawn_actor_from_object(mesh, unreal.Vector(*a["location_cm"]), unreal.Rotator(roll=0.0, pitch=0.0, yaw=a["yaw_deg"]))
    check(actor is not None, "spawn " + a["tag"])
    actor.set_actor_scale3d(unreal.Vector(*a["scale"]))
    component = actor.get_editor_property("static_mesh_component")
    component.set_mobility(unreal.ComponentMobility.MOVABLE)
    component.set_collision_profile_name("BlockAll")
    component.set_material(0, materials[a["material"]])
    actor.set_actor_label(a["tag"])
    actor.set_editor_property("tags", [a["tag"]])
    origin, extent = actor.get_actor_bounds(False)
    got_origin = [origin.x, origin.y, origin.z]
    got_extent = [extent.x, extent.y, extent.z]
    origin_err = max(abs(g - w) for g, w in zip(got_origin, a["location_cm"]))
    extent_err = max(abs(g - w) for g, w in zip(got_extent, a["expected_extent_cm"]))
    REPORT["actors"].append({"tag": a["tag"], "origin_cm": got_origin, "extent_cm": got_extent,
                             "origin_err_cm": origin_err, "extent_err_cm": extent_err})
    check(origin_err <= TOL_CM, "%s bounds origin %s != %s (mesh pivot not centred?)" % (a["tag"], got_origin, a["location_cm"]))
    check(extent_err <= TOL_CM, "%s bounds extent %s != %s (base size not 1 m?)" % (a["tag"], got_extent, a["expected_extent_cm"]))
    return actor


def main():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    ass = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    check(None not in (les, eas, ass, ues), "an editor subsystem is unavailable in this commandlet")
    check(les.new_level(SPEC["map_path"], False), "new_level " + SPEC["map_path"])
    materials = {m["name"]: make_material(m, ass) for m in SPEC["materials"]}
    spawn_mesh(SPEC["ground"], eas, materials)
    for a in SPEC["actors"]:
        spawn_mesh(a, eas, materials)
    sunsky_class = unreal.load_class(None, SPEC["sunsky_class"])
    check(sunsky_class is not None, "SunSky class " + SPEC["sunsky_class"])
    # below the ground slab so the SunSky helper meshes can never block or appear in front of the camera
    sunsky = eas.spawn_actor_from_class(sunsky_class, unreal.Vector(0.0, 0.0, -500.0))
    check(sunsky is not None, "spawn SunSky")
    sunsky.set_actor_label("SunSky")
    sunsky.set_editor_property("tags", ["SunSky"])
    spawn_exposure_volume(SPEC["exposure"], eas)
    game_mode = unreal.load_class(None, SPEC["game_mode_class"])
    check(game_mode is not None, "GameMode class " + SPEC["game_mode_class"])
    ues.get_editor_world().get_world_settings().set_editor_property("default_game_mode", game_mode)
    check(les.save_current_level(), "save_current_level")
    REPORT["actor_count"] = len(REPORT["actors"])
    REPORT["pass"] = True


try:
    main()
finally:
    with open(OUT, "w") as handle:
        json.dump(REPORT, handle, indent=2)
    # R15/Option C: run via the full editor (-ExecCmds="py ...") instead of the -run=pythonscript
    # commandlet, to avoid the UPlacementSubsystem null-pointer crash in EditorActorSubsystem's
    # spawn calls under the commandlet. The full editor keeps running after -ExecCmds finishes, so
    # the script must ask it to quit; harmless (a no-op on the way out) under the commandlet too.
    unreal.SystemLibrary.quit_editor()
