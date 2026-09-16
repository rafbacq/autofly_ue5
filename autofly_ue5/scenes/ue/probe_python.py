"""UE-side probe (UnrealEditor-Cmd -run=pythonscript): proves the editor Python APIs the level builder needs.

Writes JSON to the path in env AUTOFLY_PROBE_OUT. Runs on UE's embedded Python; stdlib + unreal only.
"""
import json
import os

import unreal

les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
ass = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
game_mode = unreal.load_class(None, "/Script/ProjectAirSim.ProjectAirSimGameMode")
sunsky = unreal.load_class(None, "/SunPosition/SunSky.SunSky_C")
cylinder = unreal.load_asset("/Engine/BasicShapes/Cylinder")
cube = unreal.load_asset("/Engine/BasicShapes/Cube")
basic_material = unreal.load_asset("/Engine/BasicShapes/BasicShapeMaterial")
grid_material = unreal.load_asset("/Engine/EngineMaterials/WorldGridMaterial")
params = []
mic_readback = None
mic_parent = None
mic_set_return = None
if basic_material is not None:
    params = [str(n) for n in unreal.MaterialEditingLibrary.get_vector_parameter_names(basic_material)]
    # Throwaway MaterialInstanceConstant, never saved (AssetTools CreateAsset only marks the package dirty), so nothing
    # is written to ue_project/Content. Same calls as the level builder in Task 14.
    probe_mic = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        "MI_ProbeColor", "/Game/AutoFly/Probe", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    if probe_mic is not None:
        unreal.MaterialEditingLibrary.set_material_instance_parent(probe_mic, basic_material)
        parent = probe_mic.get_editor_property("parent")
        mic_parent = parent.get_path_name() if parent is not None else None
        # UE 5.7.4 returns False here even when the value was set (MaterialEditingLibrary.cpp:1253-1262): read it back.
        mic_set_return = bool(unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
            probe_mic, "Color", unreal.LinearColor(0.1, 0.2, 0.3, 1.0)))
        got = unreal.MaterialEditingLibrary.get_material_instance_vector_parameter_value(probe_mic, "Color")
        mic_readback = [got.r, got.g, got.b, got.a]
result = {
    "engine_version": unreal.SystemLibrary.get_engine_version(),
    "level_editor_subsystem": les is not None,
    "editor_actor_subsystem": eas is not None,
    "editor_asset_subsystem": ass is not None,
    "game_mode_class": game_mode.get_path_name() if game_mode is not None else None,
    "sunsky_class": sunsky.get_path_name() if sunsky is not None else None,
    "cylinder_mesh": cylinder is not None,
    "cube_mesh": cube is not None,
    "world_grid_material": grid_material is not None,
    "basic_shape_material_vector_params": params,
    "mic_parent": mic_parent,
    "mic_set_vector_return_value": mic_set_return,
    "mic_color_readback": mic_readback,
    "mic_color_readback_ok": mic_readback is not None
    and all(abs(a - b) < 1e-4 for a, b in zip(mic_readback, (0.1, 0.2, 0.3, 1.0))),
}
with open(os.environ["AUTOFLY_PROBE_OUT"], "w") as handle:
    json.dump(result, handle, indent=2)
