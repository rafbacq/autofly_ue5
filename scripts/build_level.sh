#!/usr/bin/env bash
# Build ue_project/Content/AutoFly/Maps/<ID>.umap from runs/levels/<id>.level.json, then verify it in a fresh process.
#
# R15 (Task 14 fix): runs the full editor with -ExecCmds="py <script>" instead of the
# -run=pythonscript commandlet. The commandlet crashes UE 5.7.4 with SIGSEGV the moment
# build_level.py calls unreal.EditorActorSubsystem.spawn_actor_from_object/spawn_actor_from_class:
# both funnel into InternalActorUtilitiesSubsystemLibrary::SpawnActor
# (EditorActorSubsystem.cpp:96-140), which unconditionally calls
# FLevelEditorViewportClient::TryPlacingActorFromObject -> UE::AssetPlacementUtil::PlaceAssetInCurrentLevel
# (AssetSelection.cpp:1080-1086), which dereferences GEditor->GetEditorSubsystem<UPlacementSubsystem>()
# with NO null check; that subsystem is null under the commandlet (confirmed live: ground-plate spike,
# runs/levels/spike.build.log). The full editor keeps the exact same spawn API and level-build code
# (no API change) and only needs the script to call unreal.SystemLibrary.quit_editor() when done,
# since -ExecCmds doesn't exit the editor on its own.
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
ID="${1:?usage: build_level.sh <scene_id>}"
LEVELS=$ROOT/runs/levels
SPEC=$LEVELS/$ID.level.json
test -f "$SPEC" || { echo "missing $SPEC (run scripts/build_scenes.py first)"; exit 1; }
MAP=$(jq -r .map_path "$SPEC")
UMAP=$ROOT/ue_project/Content/${MAP#/Game/}.umap
CMD=$ROOT/engine/Engine/Binaries/Linux/UnrealEditor
PROJECT=$ROOT/ue_project/Blocks.uproject
ZEN_DATA_DIR=$ROOT/ue_project/DerivedDataCache/Zen
DDC_DIR=$ROOT/ue_project/DerivedDataCache
mkdir -p "$ZEN_DATA_DIR"
rm -f "$UMAP" "$LEVELS/$ID.build.json" "$LEVELS/$ID.verify.json"

run_script() {  # $1 script, $2 log name, $3 report json path, remaining: env assignments
  local script=$1 log=$2 report=$3
  shift 3
  set +e
  # UNDERSCORE keys (R1): FUnixPlatformMisc::GetEnvironmentVariable replaces "-" with "_" before calling
  # secure_getenv (UnixPlatformMisc.cpp:289-317), so the engine actually looks up UE_ZenDataPath /
  # UE_LocalDataCachePath, never the hyphenated names (autofly_ue5.paths.UE_CACHE_ENV; confirmed live by
  # Task 4's python_probe job, runs/jobs/python_probe.pid.json). -ZenDataPath= is also passed on the command
  # line, the highest-priority override in ZenServerInterface.cpp's resolution order, belt-and-braces.
  env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 "UE_ZenDataPath=$ZEN_DATA_DIR" \
    "UE_LocalDataCachePath=$DDC_DIR" AUTOFLY_LEVEL_SPEC="$SPEC" "$@" \
    "$CMD" "$PROJECT" -ExecCmds="py $script" -unattended -nop4 -nosplash -notraceserver -RenderOffscreen \
    "-ZenDataPath=$ZEN_DATA_DIR" -stdout -FullStdOutLogOutput -abslog="$LEVELS/$log" > "$LEVELS/$log.stdout" 2>&1
  local code=$?
  set -e
  # information only: the full editor's exit code doesn't reflect a raised check() failure (the script
  # calls quit_editor() from its own finally block either way); gate on the JSON report instead.
  echo "$log exit code: $code (information), Error lines: $(grep -c "Error:" "$LEVELS/$log" || true)"
  if [ ! -f "$report" ] || ! jq -e '.pass == true' "$report" > /dev/null 2>&1; then
    echo "report $report missing or pass != true"
    grep "AUTOFLY CHECK FAILED" "$LEVELS/$log" | head -5 || true
    exit 1
  fi
}

run_script "$ROOT/autofly_ue5/scenes/ue/build_level.py" "$ID.build.log" "$LEVELS/$ID.build.json" AUTOFLY_BUILD_OUT="$LEVELS/$ID.build.json"
test -f "$UMAP" || { echo "map file $UMAP was not written"; exit 1; }
run_script "$ROOT/autofly_ue5/scenes/ue/verify_level.py" "$ID.verify.log" "$LEVELS/$ID.verify.json" AUTOFLY_VERIFY_OUT="$LEVELS/$ID.verify.json"
echo "level $MAP built and verified: $(jq -c '{obstacles, worst_location_error_cm, game_mode}' "$LEVELS/$ID.verify.json")"
