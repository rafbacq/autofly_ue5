#!/usr/bin/env bash
# Build BlocksEditor (Linux, Development) with the engine's bundled toolchain and check the outputs.
#
# Also verifies the Task 2 review's carry-over: ue_project/ did not exist when UE_LocalDataCachePath was
# last exercised, so this build is the first thing that can write derived data. We record du -sk before
# and after for both the in-ROOT DerivedDataCache and the out-of-ROOT ~/.config/Epic/.../Zen/Data, and
# stop (before writing build_result.json) if the out-of-ROOT directory grew.
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
LOG=$ROOT/runs/build/build_BlocksEditor_Development.log
RESULT=$ROOT/runs/build/build_result.json
DDC_DIR=$ROOT/ue_project/DerivedDataCache
CONFIG_ZEN_DATA=$HOME/.config/Epic/UnrealEngine/Common/Zen/Data
mkdir -p "$(dirname "$LOG")"
rm -f "$RESULT"

kib_of() { [ -e "$1" ] && du -sk "$1" 2>/dev/null | cut -f1 || echo 0; }

DDC_KIB_BEFORE=$(kib_of "$DDC_DIR")
CONFIG_ZEN_KIB_BEFORE=$(kib_of "$CONFIG_ZEN_DATA")

set +e
env -u PYTHONPATH "$ROOT/engine/Engine/Build/BatchFiles/Linux/Build.sh" BlocksEditor Linux Development \
  -project="$ROOT/ue_project/Blocks.uproject" -waitmutex > "$LOG" 2>&1
CODE=$?
set -e
echo "Build.sh exit code: $CODE"
tail -3 "$LOG"
test "$CODE" -eq 0

DDC_KIB_AFTER=$(kib_of "$DDC_DIR")
CONFIG_ZEN_KIB_AFTER=$(kib_of "$CONFIG_ZEN_DATA")
echo "ue_project/DerivedDataCache: ${DDC_KIB_BEFORE} KiB -> ${DDC_KIB_AFTER} KiB"
echo "~/.config/Epic/UnrealEngine/Common/Zen/Data: ${CONFIG_ZEN_KIB_BEFORE} KiB -> ${CONFIG_ZEN_KIB_AFTER} KiB"
if [ "$CONFIG_ZEN_KIB_AFTER" -gt "$CONFIG_ZEN_KIB_BEFORE" ]; then
  echo "build wrote derived data outside ROOT (~/.config/Epic/UnrealEngine/Common/Zen/Data grew during the build); stopping"
  exit 1
fi

for f in "$ROOT/ue_project/Binaries/Linux/libUnrealEditor-Blocks.so" \
         "$ROOT/ue_project/Plugins/ProjectAirSim/Binaries/Linux/libUnrealEditor-ProjectAirSim.so"; do
  test -f "$f" || { echo "missing $f"; exit 1; }
done
ldd "$ROOT/ue_project/Plugins/ProjectAirSim/Binaries/Linux/libUnrealEditor-ProjectAirSim.so" > "$ROOT/runs/build/ldd_ProjectAirSim.txt"
MISSING=$(grep "not found" "$ROOT/runs/build/ldd_ProjectAirSim.txt" | grep -v "libUnrealEditor-" || true)
test -z "$MISSING" || { echo "unresolved third-party libraries: $MISSING"; exit 1; }
printf '{"build_sh_exit": %d, "outputs_ok": true, "pass": true, "ddc_kib_before": %d, "ddc_kib_after": %d, "config_zen_data_kib_before": %d, "config_zen_data_kib_after": %d}\n' \
  "$CODE" "$DDC_KIB_BEFORE" "$DDC_KIB_AFTER" "$CONFIG_ZEN_KIB_BEFORE" "$CONFIG_ZEN_KIB_AFTER" > "$RESULT"
echo "BlocksEditor build OK"
