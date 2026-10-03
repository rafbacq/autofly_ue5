"""RLDS / TFDS export of the raw store, mirroring the released `uavvlasplit_*_dataset/1.0.0` (spec §10.1; Plan 4 B5).

What the release stores, read from one of its records with TensorFlow on 2026-09-11 (the qwen_autofly pilot's task-1
report): one `tf.train.Example` per episode with exactly five keys --

    episode_metadata/file_path     BytesList, 1 value
    steps/observation/image_0      BytesList, one PNG (256 x 256 RGB) per step
    steps/language_instruction     BytesList, one per step
    steps/action                   FloatList, 3 per step, flattened
    steps/observation/state        FloatList, 9 per step, flattened (declared float64 in its features.json)

-- which is what TFDS writes for a FeaturesDict {steps: Dataset({observation: {image_0, state}, action,
language_instruction}), episode_metadata: {file_path}}. This module writes exactly that without TensorFlow, which the
venv cannot take (numpy 1.26.4 is pinned for projectairsim):
- TFRecord framing with masked CRC32C, from TensorBoard's `RecordWriter`;
- `tf.train.Example` with a small protobuf encoder (TensorBoard does not ship example.proto);
- `features.json` and `dataset_info.json` from templates TFDS itself generated (`rlds_templates/`), with this
  dataset's name, shard lengths and sizes filled in.
`scripts/export_rlds.py --check-python <python with tensorflow-datasets>` reads the result back through TFDS itself
(`scripts/check_rlds_with_tfds.py`); on 2026-10-03 TFDS 4.9 read a 5-episode export back exactly.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from tensorboard.summary.writer.record_writer import RecordWriter

TEMPLATES = Path(__file__).with_name("rlds_templates")
VERSION = "1.0.0"
FEATURE_KEYS = ("episode_metadata/file_path", "steps/observation/image_0", "steps/language_instruction", "steps/action",
                "steps/observation/state")


# --------------------------------------------------------------------------------------------------------
# Protobuf wire format (https://protobuf.dev/programming-guides/encoding/), just what tf.train.Example needs:
#   Example  { Features features = 1; }
#   Features { map<string, Feature> feature = 1; }      (a map is a repeated {string key = 1; Feature value = 2;})
#   Feature  { oneof kind { BytesList bytes_list = 1; FloatList float_list = 2; Int64List int64_list = 3; } }
#   BytesList { repeated bytes value = 1; }  FloatList { repeated float value = 1 [packed]; }
# --------------------------------------------------------------------------------------------------------
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _delimited(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _bytes_list(values) -> bytes:
    return b"".join(_delimited(1, v) for v in values)


def _float_list(values) -> bytes:
    packed = np.asarray(values, dtype="<f4").tobytes()
    return _delimited(1, packed) if packed else b""


def encode_example(features: dict[str, tuple[str, object]]) -> bytes:
    """features: {key: ("bytes", [bytes, ...]) | ("float", array-like)}, written in sorted key order."""
    entries = []
    for key in sorted(features):
        kind, values = features[key]
        if kind == "bytes":
            feature = _delimited(1, _bytes_list(values))
        elif kind == "float":
            feature = _delimited(2, _float_list(values))
        else:
            raise ValueError(f"unsupported feature kind {kind!r}")
        entries.append(_delimited(1, _delimited(1, key.encode()) + _delimited(2, feature)))
    return _delimited(1, b"".join(entries))


def episode_example(episode_dir: Path, instruction: str, file_path: str) -> bytes:
    steps = np.load(episode_dir / "steps.npz")
    frames = sorted((episode_dir / "frames").glob("*.png"))
    n = len(steps["state"])
    if len(frames) != n or len(steps["action"]) != n:
        raise ValueError(f"{episode_dir}: {len(frames)} frames, {n} states, {len(steps['action'])} actions")
    return encode_example({
        "episode_metadata/file_path": ("bytes", [file_path.encode()]),
        "steps/observation/image_0": ("bytes", [p.read_bytes() for p in frames]),
        "steps/language_instruction": ("bytes", [instruction.encode()] * n),
        "steps/action": ("float", steps["action"].reshape(-1)),
        "steps/observation/state": ("float", steps["state"].reshape(-1)),
    })


def export_rlds(raw_root: Path, out_root: Path, dataset_name: str, *, episodes_per_shard: int = 16) -> dict:
    """Write <out_root>/<dataset_name>/1.0.0/ (shards, features.json, dataset_info.json) from a raw store. Refuses an
    existing output directory."""
    raw_root = Path(raw_root)
    manifest = json.loads((raw_root / "manifest.json").read_text())
    out = Path(out_root) / dataset_name / VERSION
    if out.exists():
        raise FileExistsError(f"{out} exists; export into a new directory")
    out.mkdir(parents=True)
    by_split: dict[str, list[dict]] = {}
    for entry in manifest["episodes"]:
        by_split.setdefault(entry["split"], []).append(entry)
    splits = []
    for split, entries in sorted(by_split.items()):
        n_shards = max(1, -(-len(entries) // episodes_per_shard))
        lengths, total_bytes = [], 0
        for i in range(n_shards):
            chunk = entries[i * episodes_per_shard:(i + 1) * episodes_per_shard]
            path = out / f"{dataset_name}-{split}.tfrecord-{i:05d}-of-{n_shards:05d}"
            buffer = io.BytesIO()
            writer = RecordWriter(buffer)
            for entry in chunk:
                writer.write(episode_example(raw_root / entry["path"], entry["instruction"],
                                             f"{manifest['name']}/{entry['path']}"))
            path.write_bytes(buffer.getvalue())
            lengths.append(len(chunk))
            total_bytes += path.stat().st_size
        splits.append({"name": split, "shardLengths": [str(n) for n in lengths], "numBytes": str(total_bytes),
                       "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}"})
    (out / "features.json").write_text((TEMPLATES / "features.json").read_text())
    info = json.loads((TEMPLATES / "dataset_info.json").read_text())
    info.update(name=dataset_name, version=VERSION, splits=splits,
                description=f"AutoFly-format UAV navigation episodes collected in UE5 + Project AirSim ({manifest['name']}).")
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return {"path": str(out), "splits": {s["name"]: sum(int(n) for n in s["shardLengths"]) for s in splits},
            "shards": sum(len(s["shardLengths"]) for s in splits)}


def read_records(path: Path):
    """The raw records of one shard, CRCs checked (for tests and spot checks; TFDS is the real reader)."""
    from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

    data = Path(path).read_bytes()
    offset = 0
    while offset < len(data):
        header = data[offset:offset + 8]
        (length,) = struct.unpack("<Q", header)
        if struct.unpack("<I", data[offset + 8:offset + 12])[0] != masked_crc32c(header):
            raise ValueError(f"{path}: bad length CRC at {offset}")
        payload = data[offset + 12:offset + 12 + length]
        if struct.unpack("<I", data[offset + 12 + length:offset + 16 + length])[0] != masked_crc32c(payload):
            raise ValueError(f"{path}: bad data CRC at {offset}")
        yield payload
        offset += 16 + length
