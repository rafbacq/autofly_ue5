"""Expand a scene file into a layout, check reachability and write <out-dir>/<id>.layout.json.

env -u PYTHONPATH .venv/bin/python scripts/build_scenes.py scenes/s01_white_pillars.json
"""

import argparse
import json
import sys
from pathlib import Path

from autofly_ue5.paths import RUNS_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import load_registry, load_scene_file
from autofly_ue5.scenes.reachability import check_reachability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR / "levels")
    args = parser.parse_args()
    scene = load_scene_file(args.scene)
    layout = generate_layout(scene, load_registry())
    reach = check_reachability(layout, scene.start_band, scene.target_band)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{scene.id}.layout.json"
    out.write_text(json.dumps({"scene_path": str(args.scene), "scene_sha256": scene.sha256,
                               "layout": layout.to_json(), "reachability": reach.to_json()}, indent=2))
    print(json.dumps({"scene": scene.id, "instances": len(layout.instances), "reachability": reach.to_json(), "layout": str(out)}))
    return 0 if reach.ok else 1


if __name__ == "__main__":
    sys.exit(main())
