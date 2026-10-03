"""Export a raw dataset store to RLDS/TFDS shards (spec §10.1), optionally checked by reading them back through TFDS.

    env -u PYTHONPATH .venv/bin/python scripts/export_rlds.py --raw data/<name> \\
        [--check-python <venv with tensorflow-datasets>/bin/python]

Writes data/<name>/1.0.0/ (spec §10.1: the TFDS dataset is named after the store); refuses an existing one.
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
    p.add_argument("--episodes-per-shard", type=int, default=16)
    p.add_argument("--check-python", type=Path, default=None, help="a python with tensorflow-datasets, to read it back")
    args = p.parse_args(argv)
    raw = args.raw.resolve()
    result = export_rlds(raw, raw.parent, raw.name, episodes_per_shard=args.episodes_per_shard)
    print(json.dumps(result, indent=2))
    if args.check_python is None:
        return 0
    report = check_with_tfds(args.check_python, raw / VERSION, raw)
    print(json.dumps(report, indent=2))
    return 0 if report.get("pass") else 1


def check_with_tfds(python: Path, rlds_dir: Path, raw: Path) -> dict:
    """Run scripts/check_rlds_with_tfds.py under `python` (a venv with TFDS, CPU only); its report, also saved beside
    the version directory as tfds_check.json."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3")
    done = subprocess.run([str(python), str(_ROOT / "scripts" / "check_rlds_with_tfds.py"), "--rlds", str(rlds_dir),
                           "--raw", str(raw)], capture_output=True, text=True, env=env)
    try:
        report = json.loads(done.stdout[done.stdout.find("{"):])
    except ValueError:
        report = {"pass": False, "error": f"exit {done.returncode}: {(done.stderr or done.stdout)[-2000:]}"}
    (Path(rlds_dir).parent / "tfds_check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    sys.exit(main())
