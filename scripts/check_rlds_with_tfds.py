"""Read an RLDS export back through TensorFlow Datasets and compare every episode with the raw store it came from.

Runs in a separate venv that has `tensorflow` and `tensorflow-datasets` (the project venv cannot: numpy 1.26.4 is pinned
for projectairsim). Self-contained on purpose: it imports nothing from autofly_ue5.

    <venv>/bin/python scripts/check_rlds_with_tfds.py --rlds data/<name>/rlds/<dataset>/1.0.0 --raw data/<name>

Prints one JSON report; exit code 0 only if every episode matches (frames exactly, state and action to float32).

A venv that works (2026-10-03; the newest tensorflow-metadata needs protobuf 6, which TF 2.18 refuses):

    uv venv <dir> --python 3.11
    uv pip install --python <dir>/bin/python "tensorflow-cpu==2.18.*" "tensorflow-datasets==4.9.*" \
        "tensorflow-metadata==1.16.1" pillow
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rlds", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    args = p.parse_args(argv)
    import tensorflow_datasets as tfds
    from PIL import Image

    manifest = json.loads((args.raw / "manifest.json").read_text())
    by_path = {f"{manifest['name']}/{e['path']}": e for e in manifest["episodes"]}
    builder = tfds.builder_from_directory(str(args.rlds))
    report = {"tfds_episodes": 0, "matched": 0, "mismatches": [], "splits": {k: v.num_examples for k, v in builder.info.splits.items()}}
    for split in builder.info.splits:
        for episode in tfds.as_numpy(builder.as_dataset(split=split)):
            report["tfds_episodes"] += 1
            key = episode["episode_metadata"]["file_path"].decode()
            entry = by_path.get(key)
            if entry is None:
                report["mismatches"].append(f"{key}: not in the raw manifest")
                continue
            ep = args.raw / entry["path"]
            raw = np.load(ep / "steps.npz")
            steps = episode["steps"]
            images = np.stack([s["observation"]["image_0"] for s in steps])
            states = np.stack([s["observation"]["state"] for s in steps])
            actions = np.stack([s["action"] for s in steps])
            texts = {s["language_instruction"].decode() for s in steps}
            frames = np.stack([np.asarray(Image.open(io.BytesIO(f.read_bytes())).convert("RGB")) for f in sorted((ep / "frames").glob("*.png"))])
            problems = []
            if images.shape != frames.shape or not np.array_equal(images, frames):
                problems.append(f"images {images.shape} differ from the raw frames {frames.shape}")
            if states.dtype != np.float64 or not np.allclose(states, raw["state"].astype(np.float64), rtol=0, atol=1e-6):
                problems.append(f"state ({states.dtype}) differs")
            if actions.dtype != np.float32 or not np.array_equal(actions, raw["action"]):
                problems.append("action differs")
            if texts != {entry["instruction"]}:
                problems.append(f"instructions {texts}")
            if problems:
                report["mismatches"].append(f"{key}: " + "; ".join(problems))
            else:
                report["matched"] += 1
    report["pass"] = not report["mismatches"] and report["matched"] == len(manifest["episodes"]) == report["tfds_episodes"]
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
