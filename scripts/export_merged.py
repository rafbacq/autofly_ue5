"""Export several raw stores as ONE TFDS dataset with train and val splits, its OpenVLA normalisation statistics and its
card (plan 6, revision 1, item 1; autofly_ue5/dataset/merge.py, statistics.py, card.py).

    env -u PYTHONPATH .venv/bin/python scripts/export_merged.py --store data/<store> [--store data/<store> ...] \\
        --out <dir> --split-file <file> [--name autofly_ue5_proto] [--val-fraction 0.05] [--split-seed 0] \\
        [--targets assets/targets.json] [--check-python runs/tools/tfds_venv/bin/python] \\
        [--vla-python runs/tools/vla_loader_venv/bin/python]

Writes <out>/<name>/1.0.0/ (the TFDS dataset), and beside it merge.json, dataset_statistics.json (train only),
dataset_card.md and card.json; refuses an existing export (export a top-up into a new --out, or delete the old export
first). The split file is required and is the split's only record: an existing one is reused and only extended, so keep
it with the dataset (e.g. <out>/<name>.split.json, outside the export directory). `--val-fraction` and `--split-seed`
draw a new one (defaults 0.05 and 0) and may only repeat an existing one's values.

`--targets` (default assets/targets.json) fills the category, seen and status of targets whose provenance predates the
pool (M3's pilot), from the entry of the same name and object. `--check-python` reads every episode back through TFDS
(scripts/check_rlds_with_tfds.py --merged -> tfds_check.json); `--vla-python` loads it through the OpenVLA-style
pipeline (scripts/check_vla_loader.py -> vla_loader_check.json). Exit 0 when the export and every requested check
passed, 1 when a check failed, 2 when refused before anything was written.
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

from autofly_ue5.dataset.card import CARD_MD, write_card  # noqa: E402
from autofly_ue5.dataset.merge import DEFAULT_NAME, MERGE_RECORD, MergeError, export_merged  # noqa: E402
from autofly_ue5.dataset.rlds import VERSION  # noqa: E402
from autofly_ue5.paths import ROOT  # noqa: E402

DEFAULT_POOL = ROOT / "assets" / "targets.json"


def run_check(python: Path, script: str, args: list[str], report_path: Path) -> dict:
    """Run one of the read-back checks under another venv's python (CPU only); its JSON report, saved at report_path."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3")
    done = subprocess.run([str(python), str(_ROOT / "scripts" / script), *args], capture_output=True, text=True, env=env)
    try:
        report = json.loads(done.stdout[done.stdout.find("{"):])
    except ValueError:
        report = {"pass": False, "error": f"exit {done.returncode}: {(done.stderr or done.stdout)[-2000:]}"}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, action="append", required=True, help="a raw store (data/<name>); repeat")
    p.add_argument("--out", type=Path, required=True, help="the export goes to <out>/<name>/")
    p.add_argument("--name", default=DEFAULT_NAME, help=f"the TFDS dataset name (default {DEFAULT_NAME})")
    p.add_argument("--split-file", type=Path, required=True, help="the frozen train/val split: reused if it exists")
    p.add_argument("--val-fraction", type=float, default=None, help="for a new split file (default 0.05)")
    p.add_argument("--split-seed", type=int, default=None, help="for a new split file (default 0)")
    p.add_argument("--targets", type=Path, default=DEFAULT_POOL, help="the target pool, for pre-pool provenance")
    p.add_argument("--episodes-per-shard", type=int, default=16)
    p.add_argument("--no-validate", action="store_true", help="skip the dataset validator on each store")
    p.add_argument("--check-python", type=Path, default=None, help="a python with tensorflow-datasets: TFDS read-back")
    p.add_argument("--vla-python", type=Path, default=None, help="runs/tools/vla_loader_venv's python: the loader check")
    p.add_argument("--vla-frames", type=int, default=300, help="frames per split the loader check reads")
    args = p.parse_args(argv)
    try:
        result = export_merged(args.store, args.out, name=args.name, split_file=args.split_file,
                               val_fraction=args.val_fraction, split_seed=args.split_seed, pool=args.targets,
                               episodes_per_shard=args.episodes_per_shard, validate=not args.no_validate)
    except (MergeError, FileExistsError, FileNotFoundError) as err:
        print(f"refusing to export: {err}", file=sys.stderr)
        return 2
    export_dir = args.out / args.name
    write_card(export_dir)
    summary = {**result, "card": str(export_dir / CARD_MD)}
    ok = True
    if args.check_python is not None:
        summary["tfds_check"] = run_check(args.check_python, "check_rlds_with_tfds.py",
                                          ["--rlds", str(export_dir / VERSION), "--merged", str(export_dir / MERGE_RECORD)],
                                          export_dir / "tfds_check.json")
        ok &= summary["tfds_check"].get("pass") is True
    if args.vla_python is not None:
        summary["vla_loader_check"] = run_check(
            args.vla_python, "check_vla_loader.py",
            ["--data-dir", str(args.out), "--name", args.name, "--frames", str(args.vla_frames),
             "--out", str(export_dir / "vla_loader_check.json")], export_dir / "vla_loader_check.json")
        ok &= summary["vla_loader_check"].get("pass") is True
    brief = {k: v for k, v in summary.items() if k not in ("tfds_check", "vla_loader_check")}
    brief.update({k: {"pass": summary[k].get("pass"), "error": summary[k].get("error")}
                  for k in ("tfds_check", "vla_loader_check") if k in summary})
    print(json.dumps(brief, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
