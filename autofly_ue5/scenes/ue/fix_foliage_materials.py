"""UE-side: make every translucent material of the imported AutoFly models Masked (scripts/ue_editor_run.py; env
AUTOFLY_UE_OUT; --arg root=/Game/AutoFly/Assets --arg alpha_cutoff=0.5). The import script does the same for a fresh
import (ue_materials.mask_foliage_of); this applies it to a project imported before 2026-10-07. Fails if any slot is
still translucent afterwards."""
import json
import os
import sys
import traceback

import unreal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ue_materials  # noqa: E402

OUT = os.environ["AUTOFLY_UE_OUT"]
ARGS = json.loads(os.environ.get("AUTOFLY_UE_ARGS", "{}"))
ROOT = ARGS.get("root", "/Game/AutoFly/Assets")
ALPHA_CUTOFF = float(ARGS.get("alpha_cutoff", ue_materials.DEFAULT_ALPHA_CUTOFF))
REPORT = {"pass": False, "error": None, "root": ROOT, "alpha_cutoff": ALPHA_CUTOFF, "meshes": []}
ASSETS = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)


def main():
    for path in sorted(ASSETS.list_assets(ROOT, recursive=True, include_folder=False)):
        asset = unreal.load_asset(path.split(".")[0])
        if not isinstance(asset, unreal.StaticMesh):
            continue
        result = ue_materials.mask_foliage_of(asset, ALPHA_CUTOFF)
        result["mesh"] = asset.get_path_name()
        REPORT["meshes"].append(result)
        unreal.log("AUTOFLY foliage %s: %d material(s) made Masked" % (asset.get_name(), len(result["changed"])))
    still = [s["material"] for m in REPORT["meshes"] for s in m["slots"] if s["blend_mode"] == "BLEND_TRANSLUCENT"]
    if still:
        raise RuntimeError("still translucent: %s" % still)
    REPORT["pass"] = True


try:
    main()
except Exception as err:  # noqa: BLE001
    REPORT["error"] = "%s: %s\n%s" % (type(err).__name__, err, traceback.format_exc())
    unreal.log_error("AUTOFLY foliage fix failed: %s" % err)
finally:
    with open(OUT, "w") as handle:
        json.dump(REPORT, handle, indent=2)
    unreal.SystemLibrary.quit_editor()
