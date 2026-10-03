"""Export a raw dataset store to RLDS/TFDS shards (spec §10.1), optionally checked by reading them back through TFDS.

    env -u PYTHONPATH .venv/bin/python scripts/export_rlds.py --raw data/<name> --dataset autofly_ue5_<name> \\
        [--check-python <venv with tensorflow-datasets>/bin/python]

Writes data/<name>/rlds/<dataset>/1.0.0/; refuses an existing one.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

from autofly_ue5.dataset.rlds import VERSION, export_rlds  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--dataset", required=True, help="the TFDS dataset name, e.g. autofly_ue5_s01_pilot")
    p.add_argument("--episodes-per-shard", type=int, default=16)
    p.add_argument("--check-python", type=Path, default=None, help="a python with tensorflow-datasets, to read it back")
    args = p.parse_args(argv)
    result = export_rlds(args.raw, args.raw / "rlds", args.dataset, episodes_per_shard=args.episodes_per_shard)
    print(json.dumps(result, indent=2))
    if args.check_python is None:
        return 0
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3")
    done = subprocess.run([str(args.check_python), str(_ROOT / "scripts" / "check_rlds_with_tfds.py"),
                           "--rlds", str(args.raw / "rlds" / args.dataset / VERSION), "--raw", str(args.raw)],
                          capture_output=True, text=True, env=env)
    out = done.stdout[done.stdout.find("{"):] if "{" in done.stdout else done.stdout
    (args.raw / "rlds" / "tfds_check.json").write_text(out)
    print(out or done.stderr[-2000:])
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
