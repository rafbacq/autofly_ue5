#!/usr/bin/env bash
# Create ROOT/.venv from system python3.12 with the host PYTHONPATH removed,
# install the pinned requirements and the package (editable), and record the lock.
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  env -u PYTHONPATH /usr/bin/python3.12 -m venv .venv
fi
if [ -f requirements.lock ]; then REQ=requirements.lock; else REQ=requirements.txt; fi
env -u PYTHONPATH .venv/bin/python -m pip install -r "$REQ"
env -u PYTHONPATH .venv/bin/python -m pip install --no-deps -e .
env -u PYTHONPATH .venv/bin/python -m pip freeze --exclude-editable > requirements.lock
env -u PYTHONPATH .venv/bin/python -c "import projectairsim, numpy, PIL, jsonschema; print('projectairsim', projectairsim.__version__, 'numpy', numpy.__version__)"
