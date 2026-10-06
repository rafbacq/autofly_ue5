"""Turn a store's frame detections into phases and resampling weights (spec §10.3; AutoFly App. A.2.4).

    env -u PYTHONPATH .venv/bin/python scripts/rebalance_dataset.py --raw data/<name> \\
        [--detections <raw>/detections.json] [--threshold 0.7] [--alpha 0.0] [--out <raw>/rebalance.json] [--allow-partial]

Stage two of the rebalancing: `scripts/detect_targets.py` (in its own venv) has scored every frame for the episode's
target; this reads those scores with the store's manifest and steps.npz and writes `rebalance.json` beside them. It
refuses an existing rebalance.json, detections of another store, and -- unless `--allow-partial` with an `--out` away
from the store -- detections that do not cover every episode: a probe over a few episodes must never pose as the
store's own weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from autofly_ue5.dataset.rebalance import DEFAULT_THRESHOLD, build_rebalance, sha256_of, write_rebalance  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True, help="the raw store, data/<name>")
    p.add_argument("--detections", type=Path, default=None, help="default: <raw>/detections.json")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="confidence a detection must exceed")
    p.add_argument("--alpha", type=float, default=0.0, help="Eq. 11's balance: 0 targets uniform phases, 1 keeps P0")
    p.add_argument("--out", type=Path, default=None, help="default: <raw>/rebalance.json; refused if it exists")
    p.add_argument("--allow-partial", action="store_true", help="accept detections for a subset of the episodes (needs --out)")
    args = p.parse_args(argv)
    raw = args.raw.resolve()
    detections_path = args.detections if args.detections is not None else raw / "detections.json"
    out = args.out if args.out is not None else raw / "rebalance.json"
    if args.allow_partial and args.out is None:
        print("--allow-partial needs an --out outside the store: a partial result must not become its rebalance.json",
              file=sys.stderr)
        return 2
    try:
        manifest = json.loads((raw / "manifest.json").read_text())
        detections = json.loads(Path(detections_path).read_text())
        states = {entry["id"]: np.load(raw / entry["path"] / "steps.npz")["state"]
                  for entry in manifest["episodes"] if entry["id"] in detections.get("episodes", {})}
        result = build_rebalance(manifest, detections, threshold=args.threshold, alpha=args.alpha,
                                 allow_partial=args.allow_partial, states=states)
        result["detections"] = {"path": str(detections_path), "sha256": sha256_of(detections_path)}
        write_rebalance(out, result)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as err:
        print(f"refused: {type(err).__name__}: {err}", file=sys.stderr)
        return 2
    coverage = result["coverage"]
    print(f"{out}: {coverage['episodes']}/{coverage['of']} episodes{' (PARTIAL)' if coverage['partial'] else ''}, "
          f"records per phase {result['counts']['records']}, p0 {[round(v, 3) for v in result['p0']]}, "
          f"kl {result['kl_nats']:.3f} nats, weights {[None if w is None else round(w, 3) for w in result['weights']]}, "
          f"resample sizes {result['resample_sizes']}, never detected {result['counts']['episodes_never_detected']}, "
          f"persistence {result['detection_persistence']}, first detection {result.get('first_detection')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
