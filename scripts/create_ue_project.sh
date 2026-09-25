#!/usr/bin/env bash
# init:    first creation of ue_project from platform/unreal/Blocks (Config, Source) plus the ignored parts.
# restore: copy only the git-ignored parts (Blocks content, prebuilt plugin) into an existing ue_project.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCKS=$ROOT/platform/unreal/Blocks
PLUGINS=$ROOT/downloads/plugin_ue57_1.0.1/Plugins
DST=$ROOT/ue_project
MODE="${1:?usage: create_ue_project.sh init|restore}"

copy_ignored() {
  test -d "$PLUGINS/ProjectAirSim/SimLibs" || { echo "missing $PLUGINS (run scripts/fetch_plugin.sh)"; exit 1; }
  mkdir -p "$DST/Content"
  rm -rf "$DST/Plugins" "$DST/Content/Geometry" "$DST/Content/BlocksMap.umap"
  cp -a "$PLUGINS" "$DST/Plugins"
  cp -a "$BLOCKS/Content/BlocksMap.umap" "$BLOCKS/Content/Geometry" "$DST/Content/"
}

case "$MODE" in
  init)
    if [ -e "$DST/Blocks.uproject" ]; then echo "ue_project already initialised; use restore"; exit 1; fi
    mkdir -p "$DST"
    cp -a "$BLOCKS/Config" "$BLOCKS/Source" "$DST/"
    cp -a "$BLOCKS/Blocks.uproject" "$DST/"
    copy_ignored
    ;;
  restore)
    test -f "$DST/Blocks.uproject" || { echo "ue_project not initialised; use init"; exit 1; }
    copy_ignored
    ;;
  *)
    echo "unknown mode $MODE"
    exit 1
    ;;
esac
echo "ue_project $MODE done"
