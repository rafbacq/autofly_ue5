#!/usr/bin/env bash
# Package the Development Linux game with every map under Content/AutoFly/Maps plus /Game/BlocksMap into
# ue_project/Packaged/Development. Only while no simulator of ours is running: the binary and pak are what they run from.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=$ROOT/ue_project/Packaged/Development
LOG=$ROOT/runs/package/package_dev.log
# Dedicated folder: UAT clears its log folder at startup and would otherwise use ~/Documents/Unreal Engine/LocalBuildLogs
# on an installed engine (CommandEnvironment.cs:104-122).
UAT_LOGS=$ROOT/runs/package/uat_logs
BIN=$OUT/Linux/Blocks/Binaries/Linux/Blocks
mkdir -p "$(dirname "$LOG")" "$UAT_LOGS"
test -f "$ROOT/ue_project/Content/AutoFly/Maps/S01.umap" || { echo "S01.umap missing (run scripts/build_level.sh s01)"; exit 1; }
MAPS=/Game/BlocksMap
for umap in "$ROOT"/ue_project/Content/AutoFly/Maps/*.umap; do MAPS="$MAPS+/Game/AutoFly/Maps/$(basename "$umap" .umap)"; done
echo "maps: $MAPS"
if ls "$ROOT"/runs/sim/inst*/pid.json > /dev/null 2>&1; then
  for rec in "$ROOT"/runs/sim/inst*/pid.json; do
    PID=$(env -u PYTHONPATH python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$rec" 2>/dev/null) || continue
    if kill -0 "$PID" 2>/dev/null; then echo "simulator $rec (pid $PID) is running; packaging would replace its binary"; exit 1; fi
  done
fi
FREE_GB=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
test "$FREE_GB" -ge 80 || { echo "only ${FREE_GB} GB free, need 80"; exit 1; }
set +e
# PATH=/usr/bin:/bin first: ~/.local/bin/env shadows /usr/bin/env in this account's PATH and is not
# executable (rw-rw-r--, not a real coreutils env), so RunUAT's internal `env -- chmod ...` helper calls
# fail with "Permission denied" unless /usr/bin is searched first.
env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 "PATH=/usr/bin:/bin:$PATH" "UE_ZenDataPath=$ROOT/ue_project/DerivedDataCache/Zen" \
  "UE_LocalDataCachePath=$ROOT/ue_project/DerivedDataCache" "uebp_LogFolder=$UAT_LOGS" \
  "$ROOT/engine/Engine/Build/BatchFiles/RunUAT.sh" BuildCookRun \
  -project="$ROOT/ue_project/Blocks.uproject" -noP4 -utf8output -unattended \
  -platform=Linux -clientconfig=Development -build -cook -stage -pak -compressed -archive \
  -archivedirectory="$OUT" -map="$MAPS" \
  -AdditionalCookerOptions=-notraceserver \
  -nocompileeditor -skipbuildeditor > "$LOG" 2>&1
CODE=$?
set -e
echo "RunUAT exit code: $CODE"
grep -E "BUILD SUCCESSFUL|AutomationTool exiting with ExitCode" "$LOG" | tail -2 || true
test "$CODE" -eq 0
if [ ! -x "$BIN" ]; then
  echo "expected binary $BIN not found; executables under $OUT:"
  find "$OUT" -maxdepth 6 -type f -perm -u+x -name 'Blocks*' | head
  exit 1
fi
echo "packaged binary: $BIN"
