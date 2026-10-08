"""Read an RLDS export back through TensorFlow Datasets and compare every episode with the raw store(s) it came from.

Runs in a separate venv that has `tensorflow` and `tensorflow-datasets` (the project venv cannot: numpy 1.26.4 is pinned
for projectairsim). Self-contained on purpose: it imports nothing from autofly_ue5.

    <venv>/bin/python scripts/check_rlds_with_tfds.py --rlds data/<name>/1.0.0 --raw data/<name>
    <venv>/bin/python scripts/check_rlds_with_tfds.py --rlds <out>/<name>/1.0.0 --merged <out>/<name>/merge.json

`--raw`: one store's own export (dataset/rlds.py). `--merged`: a merged export (dataset/merge.py), checked against its
merge.json, which names each episode's store, split and metadata: every episode must come back in its split with its
frames, state, action and instruction as the store holds them and every metadata field as recorded, and each split
must list the recorded count. Prints one JSON report; exit code 0 only if every episode matches (frames exactly, state
and action to float32).

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


def _compare_steps(steps, ep: Path, instruction: str) -> list[str]:
    """Problems with one TFDS episode's steps against the raw episode directory `ep`."""
    from PIL import Image

    raw = np.load(ep / "steps.npz")
    images = np.stack([s["observation"]["image_0"] for s in steps])
    states = np.stack([s["observation"]["state"] for s in steps])
    actions = np.stack([s["action"] for s in steps])
    texts = {s["language_instruction"].decode() for s in steps}
    frames = np.stack([np.asarray(Image.open(io.BytesIO(f.read_bytes())).convert("RGB"))
                       for f in sorted((ep / "frames").glob("*.png"))])
    problems = []
    if images.shape != frames.shape or not np.array_equal(images, frames):
        problems.append(f"images {images.shape} differ from the raw frames {frames.shape}")
    if states.dtype != np.float64 or not np.allclose(states, raw["state"].astype(np.float64), rtol=0, atol=1e-6):
        problems.append(f"state ({states.dtype}) differs")
    if actions.dtype != np.float32 or not np.array_equal(actions, raw["action"]):
        problems.append("action differs")
    if texts != {instruction}:
        problems.append(f"instructions {texts}")
    return problems


def check_raw(builder, raw_root: Path) -> dict:
    import tensorflow_datasets as tfds

    manifest = json.loads((raw_root / "manifest.json").read_text())
    by_path = {f"{manifest['name']}/{e['path']}": e for e in manifest["episodes"]}
    report = {"tfds_episodes": 0, "matched": 0, "mismatches": [], "splits": {k: v.num_examples for k, v in builder.info.splits.items()}}
    for split in builder.info.splits:
        for episode in tfds.as_numpy(builder.as_dataset(split=split)):
            report["tfds_episodes"] += 1
            key = episode["episode_metadata"]["file_path"].decode()
            entry = by_path.get(key)
            if entry is None:
                report["mismatches"].append(f"{key}: not in the raw manifest")
                continue
            problems = _compare_steps(episode["steps"], raw_root / entry["path"], entry["instruction"])
            if problems:
                report["mismatches"].append(f"{key}: " + "; ".join(problems))
            else:
                report["matched"] += 1
    report["pass"] = not report["mismatches"] and report["matched"] == len(manifest["episodes"]) == report["tfds_episodes"]
    return report


def _metadata_value(value, kind: str):
    """A TFDS-decoded metadata value as merge.json records it; None if its type is not the declared one."""
    if kind == "text":
        return value.decode() if isinstance(value, bytes) else None
    if kind == "bool":
        return bool(value) if isinstance(value, np.bool_) else None
    return int(value) if isinstance(value, np.int64) else None


def check_merged(builder, record_path: Path) -> dict:
    import tensorflow_datasets as tfds

    record = json.loads(record_path.read_text())
    expected = {e["metadata"]["file_path"]: e for e in record["episodes"]}
    fields = [tuple(f) for f in record["metadata_fields"]]
    report = {"tfds_episodes": 0, "matched": 0, "mismatches": [],
              "splits": {k: v.num_examples for k, v in builder.info.splits.items()},
              "expected_splits": {k: v["episodes"] for k, v in record["splits"].items()},
              "metadata_fields": sorted(builder.info.features["episode_metadata"].keys())}
    if report["metadata_fields"] != sorted(name for name, _ in fields):
        report["mismatches"].append(f"episode_metadata declares {report['metadata_fields']}, merge.json "
                                    f"{sorted(name for name, _ in fields)}")
    returned = set()
    for split in builder.info.splits:
        for episode in tfds.as_numpy(builder.as_dataset(split=split, shuffle_files=False)):
            report["tfds_episodes"] += 1
            meta = episode["episode_metadata"]
            key = meta["file_path"].decode()
            entry = expected.get(key)
            if entry is None:
                report["mismatches"].append(f"{key}: not in merge.json")
                continue
            returned.add(key)
            problems = []
            if entry["split"] != split:
                problems.append(f"in split {split}, merge.json says {entry['split']}")
            for name, kind in fields:
                got = _metadata_value(meta[name], kind)
                if got is None or got != entry["metadata"][name]:
                    problems.append(f"metadata {name} = {meta[name]!r} ({type(meta[name]).__name__}), merge.json "
                                    f"{entry['metadata'][name]!r}")
            problems += _compare_steps(episode["steps"], Path(entry["store_root"]) / entry["path"], entry["instruction"])
            if problems:
                report["mismatches"].append(f"{key}: " + "; ".join(problems))
            else:
                report["matched"] += 1
    missing = sorted(set(expected) - returned)
    if missing:
        report["mismatches"].append(f"{len(missing)} episodes of merge.json did not come back, e.g. {missing[:3]}")
    report["pass"] = (not report["mismatches"] and report["matched"] == len(expected) == report["tfds_episodes"]
                      and report["splits"] == report["expected_splits"])
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rlds", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, help="the raw store a single-store export came from")
    source.add_argument("--merged", type=Path, help="the merge.json of a merged export")
    args = p.parse_args(argv)
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(args.rlds))
    report = check_raw(builder, args.raw) if args.raw is not None else check_merged(builder, args.merged)
    report["tfds_version"] = tfds.__version__
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
