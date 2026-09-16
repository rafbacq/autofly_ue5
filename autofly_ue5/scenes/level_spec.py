"""Layout (NED metres) -> UE level spec (centimetres, Z up) consumed by scenes/ue/build_level.py."""

from __future__ import annotations

import math

from autofly_ue5.frames import ned_to_ue_cm, ned_yaw_deg_to_ue_yaw_deg
from autofly_ue5.scenes.model import AssetRegistry, Layout, SceneFile

MAP_ROOT = "/Game/AutoFly/Maps"
GROUND_SIZE_M = 200.0
GROUND_THICKNESS_M = 0.2
SUNSKY_CLASS = "/SunPosition/SunSky.SunSky_C"
GAME_MODE_CLASS = "/Script/ProjectAirSim.ProjectAirSimGameMode"
EXPOSURE_TAG = "AF_Exposure"
# R16 (controller ruling, M1 gate fix): fixed MANUAL exposure, not auto-exposure. The SunSky's
# physically-based sun (~1e5 lux) saturates the default tonemapper's auto-exposure range to pure white
# (measured: mean RGB 255.0 on the packaged s01 build, Task 16's live gate). Auto-exposure (eye
# adaptation) also carries temporal state -- the same pose would render different pixels depending on
# where the camera looked before, which is not reproducible for (image, action) pairs or for RL training
# on these frames. build_level.py spawns an unbound PostProcessVolume with AutoExposureMethod=Manual and
# AutoExposureApplyPhysicalCameraExposure=False, which (per the engine's eye-adaptation math -- see
# EV100ToLuminance/EyeAdaptationCommon.usf) makes the final image-intensity multiplier exactly
# 2**EXPOSURE_BIAS_EV, a pure function of this one number: independent of scene content, pose and camera
# history. This is a design choice the AutoFly paper does not specify; the value below was chosen
# empirically (Task 16 fix, at most 3 build+package+probe iterations) against the S01 pillar field --
# see docs/gates/exposure_calibration.json for what was tried and the resulting mean brightness.
EXPOSURE_BIAS_EV = -11.0
EXPOSURE_APPLY_PHYSICAL_CAMERA = False


def map_path_for(scene_id: str) -> str:
    return f"{MAP_ROOT}/{scene_id.upper()}"


def _extent_cm(base_size_m: tuple[float, float, float], scale: tuple[float, float, float]) -> list[float]:
    return [round(b * s * 50.0, 4) for b, s in zip(base_size_m, scale)]


def layout_to_level_spec(layout: Layout, scene: SceneFile, registry: AssetRegistry) -> dict:
    cube = registry.assets["cube"]
    ground_scale = (GROUND_SIZE_M / cube.base_size_m[0], GROUND_SIZE_M / cube.base_size_m[1], GROUND_THICKNESS_M / cube.base_size_m[2])
    ground = {
        "tag": "AF_Ground",
        "mesh": cube.ue_path,
        "location_cm": list(ned_to_ue_cm(0.0, 0.0, GROUND_THICKNESS_M / 2.0)),
        "yaw_deg": 0.0,
        "scale": list(ground_scale),
        "material": scene.ground,
        "expected_extent_cm": _extent_cm(cube.base_size_m, ground_scale),
    }
    actors = []
    for inst in layout.instances:
        asset = registry.assets[inst.asset]
        actors.append({
            "tag": inst.tag,
            "mesh": asset.ue_path,
            "location_cm": [round(v, 4) for v in ned_to_ue_cm(inst.x, inst.y, inst.z_center)],
            "yaw_deg": ned_yaw_deg_to_ue_yaw_deg(math.degrees(inst.yaw)),
            "scale": list(inst.scale),
            "material": inst.material,
            "expected_extent_cm": _extent_cm(asset.base_size_m, inst.scale),
        })
    used = sorted({scene.ground} | {inst.material for inst in layout.instances})
    materials = []
    for name in used:
        m = registry.materials[name]
        materials.append({"name": name, "kind": m.kind, "ue_path": m.ue_path, "parent": m.parent,
                          "parameter": m.parameter, "rgba": list(m.rgba) if m.rgba is not None else None})
    return {
        "version": 1,
        "scene_id": scene.id,
        "scene_sha256": scene.sha256,
        "layout_seed": layout.seed,
        "map_path": map_path_for(scene.id),
        "game_mode_class": GAME_MODE_CLASS,
        "sunsky_class": SUNSKY_CLASS,
        "exposure": {
            "tag": EXPOSURE_TAG,
            "method": "manual",
            "bias_ev": EXPOSURE_BIAS_EV,
            "apply_physical_camera_exposure": EXPOSURE_APPLY_PHYSICAL_CAMERA,
        },
        "materials": materials,
        "ground": ground,
        "actors": actors,
    }
