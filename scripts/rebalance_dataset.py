"""Turn the stores' frame detections into phases and resampling weights (spec §10.3; AutoFly App. A.2.4).

    env -u PYTHONPATH .venv/bin/python scripts/rebalance_dataset.py --raw data/<name> \\
        [--detections <raw>/detections.jsonl] [--threshold 0.7] [--alpha 0.0] [--out <raw>/rebalance.json] [--allow-partial]
    env -u PYTHONPATH .venv/bin/python scripts/rebalance_dataset.py --raw data/<a> --raw data/<b> ... --out <file> \\
        [--detections <a's> --detections <b's> ...] [--allow-mixed-search] [--threshold 0.7] [--allow-partial]

Stage two of the rebalancing: `scripts/detect_targets.py` (in its own venv) has scored each store's frames for the
episode's target, densely or with the sparse search; this reads those scores with the store's manifest and steps.npz.
One store: `rebalance.json` beside it by default. Several stores (a prototype's scenes and batches): one document for
their union at `--out` (required: the union is no single store's), with the records summed, per-store and per-target
summaries, and every episode keyed by store and id.

A store's detections are `<raw>/detections.jsonl` (the appendable cache) or `<raw>/detections.json` (the pilot's
whole-document form), whichever exists; both existing is refused rather than guessed between. It refuses an existing
output, detections of another store, a sparse file at a threshold other than its own, and -- unless `--allow-partial`
with an `--out` away from every store -- detections that do not cover every episode: a probe over a few episodes must
never pose as a store's own weights. Several stores scored by different searches (dense and sparse, or two sparse
searches) need `--allow-mixed-search`; by different detectors they are refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from autofly_ue5.dataset.detections import parse as parse_detections  # noqa: E402
from autofly_ue5.dataset.detections import store_file  # noqa: E402
from autofly_ue5.dataset.rebalance import (DEFAULT_THRESHOLD, build_rebalance, build_union_rebalance,  # noqa: E402
                                           write_rebalance)


def _load_store(raw: Path, detections_path: Path) -> tuple[dict, dict, dict, dict]:
    """The store's manifest, its detections (hashed and parsed from the same bytes: a scorer may still be appending),
    the states of the scored episodes, and where the detections came from."""
    manifest = json.loads((raw / "manifest.json").read_text())
    blob = Path(detections_path).read_bytes()
    detections = parse_detections(blob).doc
    states = {entry["id"]: np.load(raw / entry["path"] / "steps.npz")["state"]
              for entry in manifest["episodes"] if entry["id"] in detections.get("episodes", {})}
    return manifest, detections, states, {"path": str(detections_path), "sha256": hashlib.sha256(blob).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, action="append", required=True,
                   help="a raw store, data/<name>; repeat it for one document over several stores")
    p.add_argument("--detections", type=Path, action="append", default=None,
                   help="per --raw, in the same order; default <raw>/detections.jsonl or <raw>/detections.json")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="confidence a detection must exceed")
    p.add_argument("--alpha", type=float, default=0.0, help="Eq. 11's balance: 0 targets uniform phases, 1 keeps P0")
    p.add_argument("--out", type=Path, default=None,
                   help="one store: default <raw>/rebalance.json; several: required. Refused if it exists")
    p.add_argument("--allow-partial", action="store_true", help="accept detections for a subset of the episodes (needs --out)")
    p.add_argument("--allow-mixed-search", action="store_true",
                   help="several stores: accept stores scored densely and sparsely, or by different sparse searches")
    args = p.parse_args(argv)
    raws = [raw.resolve() for raw in args.raw]
    several = len(raws) > 1
    if args.detections is not None and len(args.detections) != len(raws):
        print(f"refused: {len(args.detections)} --detections for {len(raws)} --raw: give one per store, in the same order",
              file=sys.stderr)
        return 2
    if several and args.out is None:
        print("refused: several stores need an --out: their union is no single store's rebalance.json", file=sys.stderr)
        return 2
    out = args.out if args.out is not None else raws[0] / "rebalance.json"
    if args.allow_partial and (args.out is None or any(out.resolve().is_relative_to(raw) for raw in raws)):
        print("--allow-partial needs an --out outside every store: a partial result must not pose as a store's weights",
              file=sys.stderr)
        return 2
    try:
        paths = args.detections if args.detections is not None else [store_file(raw) for raw in raws]
        loaded = [_load_store(raw, path) for raw, path in zip(raws, paths)]
        if several:
            result = build_union_rebalance([(manifest, detections, states) for manifest, detections, states, _src in loaded],
                                           threshold=args.threshold, alpha=args.alpha, allow_partial=args.allow_partial,
                                           allow_mixed_search=args.allow_mixed_search)
            for (manifest, _d, _s, source) in loaded:
                result["stores"][manifest["name"]]["detections"] = source
        else:
            ((manifest, detections, states, source),) = loaded
            result = build_rebalance(manifest, detections, threshold=args.threshold, alpha=args.alpha,
                                     allow_partial=args.allow_partial, states=states)
            result["detections"] = source
            if result["degenerate"] and args.out is None:
                # A whole phase is empty: the detector never fired (or the threshold is wrong), and no weight can
                # rebalance that. A finding to inspect (--out elsewhere), not the store's weights.
                raise ValueError(f"degenerate result (weights {result['weights']}, {result['counts']['episodes_never_detected']} "
                                 f"of {result['coverage']['episodes']} episodes never detected): not written as the store's "
                                 f"rebalance.json; pass --out to inspect it")
        write_rebalance(out, result)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError, IndexError) as err:
        print(f"refused: {type(err).__name__}: {err}", file=sys.stderr)
        return 2
    coverage = result["coverage"]
    print(f"{out}: {coverage['episodes']}/{coverage['of']} episodes{' (PARTIAL)' if coverage['partial'] else ''}"
          f"{' of ' + str(len(raws)) + ' stores' if several else ''}, records per phase {result['counts']['records']}, "
          f"p0 {[round(v, 3) for v in result['p0']]}, kl {result['kl_nats']:.3f} nats, "
          f"weights {[None if w is None else round(w, 3) for w in result['weights']]}, "
          f"resample sizes {result['resample_sizes']}, never detected {result['counts']['episodes_never_detected']}, "
          f"persistence {result['detection_persistence']}, first detection {result.get('first_detection')}, "
          f"targets never detected {result['targets_never_detected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
