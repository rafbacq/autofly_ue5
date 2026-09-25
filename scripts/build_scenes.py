"""Expand a scene file into a layout, check reachability, write <id>.layout.json and <id>.level.json.

env -u PYTHONPATH .venv/bin/python scripts/build_scenes.py scenes/s01_white_pillars.json
"""

import argparse
import json
import sys
from pathlib import Path

from autofly_ue5.paths import RUNS_DIR
from autofly_ue5.scenes.build import build_scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR / "levels")
    args = parser.parse_args()
    summary = build_scene(args.scene, args.out_dir)
    print(json.dumps(summary))
    return 0 if summary["reachability"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
