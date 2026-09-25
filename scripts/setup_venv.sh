#!/usr/bin/env bash
# Create ROOT/.venv from a Python 3.12 interpreter with the host PYTHONPATH removed, install the pinned
# requirements.lock and the package (editable), and check the imports M0-M2 need.
#   bash scripts/setup_venv.sh              install requirements.lock into .venv (creating it if missing)
#   bash scripts/setup_venv.sh --recreate   delete and rebuild .venv first (e.g. it was built from another Python)
#   bash scripts/setup_venv.sh --relock     install requirements.txt instead, then rewrite requirements.lock from it
# The interpreter is $PYTHON if set, else the first python3.12 on PATH. It must be 3.12: projectairsim 1.0.2 and
# numpy 1.26.4 are pinned for it. The lock is never rewritten except by --relock.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RECREATE=0
RELOCK=0
for arg in "$@"; do
  case "$arg" in
    --recreate) RECREATE=1 ;;
    --relock) RELOCK=1 ;;
    *) echo "usage: setup_venv.sh [--recreate] [--relock]"; exit 2 ;;
  esac
done

PY="${PYTHON:-$(command -v python3.12 || true)}"
[ -n "$PY" ] || { echo "no python3.12 on PATH; install Python 3.12 (uv, conda or deadsnakes) and set PYTHON=/path/to/python3.12"; exit 1; }
version_of() { env -u PYTHONPATH "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])'; }
PY_VERSION=$(version_of "$PY")
[ "$PY_VERSION" = "3.12" ] || { echo "$PY is Python $PY_VERSION; this project needs Python 3.12"; exit 1; }

if [ -x .venv/bin/python ]; then
  VENV_VERSION=$(version_of .venv/bin/python)
  if [ "$RECREATE" = 1 ]; then
    rm -rf .venv
  elif [ "$VENV_VERSION" != "3.12" ]; then
    echo ".venv is Python $VENV_VERSION, not 3.12; rerun with --recreate to rebuild it"
    exit 1
  fi
fi
[ -x .venv/bin/python ] || env -u PYTHONPATH "$PY" -m venv .venv

if [ "$RELOCK" = 1 ]; then REQ=requirements.txt; else REQ=requirements.lock; fi
env -u PYTHONPATH .venv/bin/python -m pip install -r "$REQ"
env -u PYTHONPATH .venv/bin/python -m pip install --no-deps -e .
if [ "$RELOCK" = 1 ]; then
  env -u PYTHONPATH .venv/bin/python -m pip freeze --exclude-editable > requirements.lock
  echo "rewrote requirements.lock from requirements.txt; review the diff before committing it"
fi

env -u PYTHONPATH .venv/bin/python - <<'PY'
import json
from importlib.metadata import version
from pathlib import Path

import gymnasium, jsonschema, numpy, PIL, projectairsim, stable_baselines3, tensorboard, torch  # noqa: F401

print(f"projectairsim {version('projectairsim')}  numpy {numpy.__version__}  torch {torch.__version__}  "
      f"stable-baselines3 {stable_baselines3.__version__}  gymnasium {gymnasium.__version__}  "
      f"cuda available: {torch.cuda.is_available()}")
manifest = Path("docs/gates/m2_env_manifest.json")
if manifest.is_file():
    recorded = json.loads(manifest.read_text())["packages"]
    drift = {name: (want, version(name)) for name, want in recorded.items() if version(name) != want}
    for name, (want, have) in drift.items():
        print(f"WARNING: {name} {have} differs from {want} recorded in {manifest}")
PY
