"""Generate the merged export's `features.json` with TensorFlow Datasets itself (autofly_ue5/dataset/merge.py).

Runs in a venv with `tensorflow-datasets` (the project venv cannot take TensorFlow: numpy 1.26.4 is pinned for
projectairsim); self-contained on purpose, like scripts/check_rlds_with_tfds.py. The steps are exactly the release's
four fields (rlds_templates/features.json, which TFDS 4.9 generated for M3's exporter); episode_metadata keeps the
release's `file_path` and adds the merged dataset's fields. tests/test_merge.py checks that the template and
merge.METADATA_FIELDS name the same fields with the same types, so neither can change alone.

    <venv>/bin/python scripts/make_rlds_templates.py --out autofly_ue5/dataset/rlds_templates/merged_features.json

TFDS stores a bool Scalar as int64 in the tf.train.Example and casts it back on reading; an int64 Scalar is an int64.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# (name, kind) in the order merge.METADATA_FIELDS lists them; kind is "text", "bool" or "int64".
METADATA = (
    ("file_path", "text"), ("episode_id", "text"), ("store", "text"), ("scene", "text"), ("split_of_scene", "text"),
    ("target_name", "text"), ("target_category", "text"), ("target_seen", "bool"), ("target_status", "text"),
    ("target_stand_in", "bool"), ("seed", "int64"), ("pilot", "text"),
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    import numpy as np
    import tensorflow_datasets as tfds

    kinds = {"text": tfds.features.Text, "bool": lambda: tfds.features.Scalar(dtype=np.bool_),
             "int64": lambda: tfds.features.Scalar(dtype=np.int64)}
    features = tfds.features.FeaturesDict({
        "steps": tfds.features.Dataset({
            "observation": tfds.features.FeaturesDict({
                "image_0": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="png"),
                "state": tfds.features.Tensor(shape=(9,), dtype=np.float64),
            }),
            "action": tfds.features.Tensor(shape=(3,), dtype=np.float32),
            "language_instruction": tfds.features.Text(),
        }),
        "episode_metadata": tfds.features.FeaturesDict({name: kinds[kind]() for name, kind in METADATA}),
    })
    with tempfile.TemporaryDirectory() as tmp:
        features.save_config(tmp)
        content = json.loads((Path(tmp) / "features.json").read_text())
    args.out.write_text(json.dumps(content, indent=4) + "\n")
    print(json.dumps({"written": str(args.out), "tfds": tfds.__version__}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
