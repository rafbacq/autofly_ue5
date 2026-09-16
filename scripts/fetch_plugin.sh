#!/usr/bin/env bash
# Download the Project AirSim 1.0.1 Linux UE5.7 plugin, check size and sha256, unzip, verify the manifest.
set -euo pipefail
ROOT=/home/jk_edge/research_uav/autofly_ue5
NAME=ProjectAirSim-Plugin-Linux-UE5_7-1.0.1.zip
URL=https://github.com/iamaisim/ProjectAirSim/releases/download/v1.0.1/$NAME
SHA=11f016ac7aa292a1a353dfde9ff7c4a1b5cebd660cf9e8f400f2ed4c030dcb49
SIZE=669317542
DL=$ROOT/downloads
mkdir -p "$DL"
cd "$DL"
if [ ! -f "$NAME" ] || [ "$(stat -c %s "$NAME")" != "$SIZE" ]; then
  curl -fL --retry 5 -C - -o "$NAME" "$URL"
fi
test "$(stat -c %s "$NAME")" = "$SIZE"
echo "$SHA  $NAME" | sha256sum -c -
sha256sum "$NAME" > "$NAME.sha256"
unzip -tq "$NAME"
rm -rf plugin_ue57_1.0.1
unzip -q "$NAME" -d plugin_ue57_1.0.1
env -u PYTHONPATH "$ROOT/.venv/bin/python" -m autofly_ue5.validate.plugin_manifest "$DL/plugin_ue57_1.0.1" > "$DL/plugin_manifest_report.json"
cat "$DL/plugin_manifest_report.json"
