"""autofly_ue5/dataset/rlds.py writes TFRecord shards of tf.train.Example episodes with exactly the release's five keys
(read back here by a minimal protobuf decoder; scripts/check_rlds_with_tfds.py does it with TFDS itself)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.test_collect import _collect


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value, i


def _fields(buf: bytes):
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        assert key & 7 == 2, "only length-delimited fields are written"
        n, i = _varint(buf, i)
        yield key >> 3, buf[i:i + n]
        i += n


def _decode_example(raw: bytes) -> dict:
    out = {}
    (_, features), = list(_fields(raw))
    for _, entry in _fields(features):
        parts = dict(_fields(entry))
        key = parts[1].decode()
        (kind, payload), = list(_fields(parts[2]))
        if kind == 1:
            out[key] = [v for _, v in _fields(payload)]
        else:
            (_, packed), = list(_fields(payload)) if payload else [(1, b"")]
            out[key] = np.frombuffer(packed, dtype="<f4")
    return out


def test_an_export_holds_the_release_s_five_keys_per_episode_and_its_metadata(tmp_path):
    from autofly_ue5.dataset.rlds import FEATURE_KEYS, export_rlds, read_records

    _summary, root = _collect(tmp_path, n_keep=3)
    result = export_rlds(root, root / "rlds", "autofly_ue5_test", episodes_per_shard=2)
    assert result["splits"] == {"train": 3} and result["shards"] == 2
    out = root / "rlds" / "autofly_ue5_test" / "1.0.0"
    shards = sorted(out.glob("autofly_ue5_test-train.tfrecord-*-of-00002"))
    assert len(shards) == 2
    manifest = json.loads((root / "manifest.json").read_text())
    examples = [_decode_example(r) for s in shards for r in read_records(s)]
    assert len(examples) == 3
    for ex, entry in zip(examples, manifest["episodes"]):
        assert set(ex) == set(FEATURE_KEYS)
        n = entry["steps"]
        steps = np.load(root / entry["path"] / "steps.npz")
        assert len(ex["steps/observation/image_0"]) == n and ex["steps/observation/image_0"][0][:8] == b"\x89PNG\r\n\x1a\n"
        assert ex["steps/observation/image_0"][0] == (root / entry["path"] / "frames" / "000000.png").read_bytes()
        assert np.array_equal(ex["steps/observation/state"].reshape(n, 9), steps["state"])
        assert np.array_equal(ex["steps/action"].reshape(n, 3), steps["action"])
        assert {v.decode() for v in ex["steps/language_instruction"]} == {entry["instruction"]}
        assert ex["episode_metadata/file_path"] == [f"pilot/{entry['path']}".encode()]
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["name"] == "autofly_ue5_test" and info["splits"][0]["shardLengths"] == ["2", "1"]
    assert int(info["splits"][0]["numBytes"]) == sum(s.stat().st_size for s in shards)
    assert "float64" in (out / "features.json").read_text()
    with pytest.raises(FileExistsError):
        export_rlds(root, root / "rlds", "autofly_ue5_test")


def test_a_corrupted_record_is_caught_by_its_crc(tmp_path):
    from autofly_ue5.dataset.rlds import export_rlds, read_records

    _summary, root = _collect(tmp_path, n_keep=1)
    export_rlds(root, root / "rlds", "autofly_ue5_test")
    shard = next((root / "rlds" / "autofly_ue5_test" / "1.0.0").glob("*.tfrecord-*"))
    data = bytearray(shard.read_bytes())
    data[100] ^= 0xFF
    shard.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="CRC"):
        list(read_records(shard))
