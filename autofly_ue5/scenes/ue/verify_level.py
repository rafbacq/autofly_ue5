"""UE-side reload check of a saved map (fresh commandlet process; stdlib + unreal only).

env AUTOFLY_LEVEL_SPEC (level spec JSON), AUTOFLY_VERIFY_OUT (report JSON).
"""
import json
import os

import unreal

SPEC = json.load(open(os.environ["AUTOFLY_LEVEL_SPEC"]))
OUT = os.environ["AUTOFLY_VERIFY_OUT"]
TOL_CM = 1.0
REPORT = {"map_path": SPEC["map_path"], "pass": False}


def check(ok, what):
    if not ok:
        raise RuntimeError("AUTOFLY CHECK FAILED: " + what)


def main():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    check(les.load_level(SPEC["map_path"]), "load_level " + SPEC["map_path"])
    by_tag = {}
    for actor in eas.get_all_level_actors():
        for tag in actor.get_editor_property("tags"):
            by_tag.setdefault(str(tag), []).append(actor)
    expected = [SPEC["ground"]] + SPEC["actors"]
    worst = 0.0
    for a in expected:
        found = by_tag.get(a["tag"], [])
        check(len(found) == 1, "tag %s found %d times" % (a["tag"], len(found)))
        loc = found[0].get_actor_location()
        err = max(abs(g - w) for g, w in zip([loc.x, loc.y, loc.z], a["location_cm"]))
        worst = max(worst, err)
        check(err <= TOL_CM, "%s location error %.3f cm" % (a["tag"], err))
    obstacle_tags = sorted(t for t in by_tag if t.startswith("obs_"))
    check(len(obstacle_tags) == len(SPEC["actors"]), "obstacle tag count %d != %d" % (len(obstacle_tags), len(SPEC["actors"])))
    check(len(by_tag.get("SunSky", [])) == 1, "SunSky actor missing")
    game_mode = ues.get_editor_world().get_world_settings().get_editor_property("default_game_mode")
    game_mode_path = game_mode.get_path_name() if game_mode is not None else None
    check(game_mode_path == SPEC["game_mode_class"], "GameMode override is %s" % game_mode_path)
    REPORT.update({"obstacles": len(obstacle_tags), "worst_location_error_cm": worst, "game_mode": game_mode_path, "pass": True})


try:
    main()
finally:
    with open(OUT, "w") as handle:
        json.dump(REPORT, handle, indent=2)
    # R15/Option C: run via the full editor, which stays open after -ExecCmds unless asked to quit.
    unreal.SystemLibrary.quit_editor()
