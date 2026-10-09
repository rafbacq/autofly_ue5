"""The sparse search replayed on dense scores: how many detector calls it saves and what it changes (plan 6 step 5;
Revision 1 item 6 asks for the sparse search to be validated against the s01 pilot's dense scores before any store is
scored sparsely: the same transition on at least 99 of 100 episodes).

    env -u PYTHONPATH .venv/bin/python scripts/sparse_search_study.py --raw data/s01_pilot [--with-states] \\
        [--detections <the store's>] [--threshold 0.7] [--strides 1-16] [--alerts 0.7,0.65,0.6,0.55,0.5,0.45] \\
        [--out runs/rebalance/sparse_study.json]
    env -u PYTHONPATH .venv/bin/python scripts/sparse_search_study.py --raw data/s01_pilot \\
        --check-sparse runs/rebalance/<probe>/detections.jsonl --out runs/rebalance/<probe>/check.json

Study mode. For each stride S and alert level, the search of `autofly_ue5/dataset/sparse.py` is replayed on every
episode's dense scores, and its result is weighed exactly as the rebalancing stage weighs a sparse file holding it
(`rebalance.build_rebalance` on the replayed records, which also verifies them). Per row: detector calls per episode
(mean and max, one record per call; and the mean at the detector's default --batch 4, whose look-ahead scores a few
records past the stop), the episodes whose first detection differs from the dense one and by how many records, P0 and
the weights against the dense ones, the share of records whose phase label changes, and -- to show how much of the row
is the luck of where its probes fell -- the differing episodes per 100 averaged over all S positions of the probe
grid. An alert equal to the threshold is the plain coarse scan with a fine scan before its first hit. The JSON goes to
--out and a markdown table beside it (.md); the recommendation is a rule's pick, written out with the rule.

Check mode (--check-sparse). A sparse file scored live on the same store is compared with the dense scores record by
record (the largest score and box differences over the visited records), and its first detections with the dense ones
and with the search replayed offline on the dense scores (the study's prediction).
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
import os  # noqa: E402

import numpy as np  # noqa: E402

from autofly_ue5.dataset import sparse  # noqa: E402
from autofly_ue5.dataset.detections import FORMAT as DETECTIONS_FORMAT  # noqa: E402
from autofly_ue5.dataset.detections import parse as parse_detections  # noqa: E402
from autofly_ue5.dataset.detections import store_file  # noqa: E402
from autofly_ue5.dataset.rebalance import (DEFAULT_THRESHOLD, box_on_target, build_rebalance,  # noqa: E402
                                           first_detection)

FORMAT = "autofly_ue5_sparse_study/1"
COST_BATCH = 4  # the detector's default --batch
PLAN_SHARE = 0.99  # plan 6, Revision 1 item 6: the same transition on at least 99 of 100 pilot episodes
OFFSET_MARGIN_PER_100 = 0.5  # half the plan's allowance, averaged over every probe grid: a margin for other targets
STRATEGIES = {
    "coarse_fine": ("the plain scan (alert = threshold): coarse probes at records 0, S, 2S, ... and the last; at the first "
                    "probe above the threshold, the records between the previous probe and it are scored in order, and "
                    "the first above the threshold is the first detection"),
    "coarse_alert_windows": sparse.RULE,
}
RECOMMENDATION_RULE = (f"the row with the fewest mean calls among those that keep the dense transition on at least "
                       f"{PLAN_SHARE:.0%} of the episodes on the detector's own grid and change at most "
                       f"{OFFSET_MARGIN_PER_100} per 100 averaged over every position of the probe grid; ties go to the "
                       f"larger stride, then to the lower alert")


def _covered(manifest: dict, dense: dict) -> list[dict]:
    if dense.get("sparse") is not None:
        raise ValueError("the study replays the search on dense scores; these detections are sparse")
    return [entry for entry in manifest["episodes"] if entry["id"] in dense["episodes"]]


def _effective(first: int | None, n: int) -> int:
    return n if first is None else first


def study_row(manifest: dict, dense: dict, dense_result: dict, *, stride: int, threshold: float, alert: float,
              states: dict | None = None) -> dict:
    """One strategy replayed on every covered episode, and weighed as the rebalancing stage would weigh its records."""
    entries = _covered(manifest, dense)
    dense_first = {e["id"]: e["first_detection"] for e in dense_result["episodes"]}
    episodes, calls, calls_batched, differences = {}, [], [], []
    label_changes = records = 0
    for entry in entries:
        ep = dense["episodes"][entry["id"]]
        scores, n = ep["scores"], entry["steps"]
        first, visited = sparse.replay(scores, stride=stride, threshold=threshold, alert=alert)
        _first, batched = sparse.search(n, stride=stride, threshold=threshold, alert=alert, batch=COST_BATCH,
                                        score=lambda indices, scores=scores: [scores[i] for i in indices])
        calls.append(len(visited))
        calls_batched.append(len(batched))
        d = dense_first[entry["id"]]
        records += n
        label_changes += abs(_effective(first, n) - _effective(d, n))
        if first != d:
            difference = {"id": entry["id"], "dense": d, "sparse": first,
                          "delta": None if first is None or d is None else first - d,
                          "dense_score": scores[d] if d is not None else None}
            if states is not None and d is not None:
                state = np.asarray(states[entry["id"]])
                difference.update(dense_distance_m=round(float(state[d, 0]), 2),
                                  dense_on_target=box_on_target(ep["boxes"][d], float(state[d, 1])))
            differences.append(difference)
        order = sorted(visited)
        episodes[entry["id"]] = {"query": ep["query"], "prompt": ep["prompt"], "records": n, "visited": order,
                                 "scores": [scores[i] for i in order], "boxes": [ep["boxes"][i] for i in order],
                                 "first_detection": first}
    replayed = {"format": DETECTIONS_FORMAT, "dataset": dense["dataset"], "dataset_created": dense["dataset_created"],
                "detector": dense.get("detector"), "sparse": sparse.describe(stride, threshold, alert), "episodes": episodes}
    result = build_rebalance(manifest, replayed, threshold=threshold, allow_partial=dense_result["coverage"]["partial"])
    offset_misses: dict[str, int] = {}  # per episode, at how many of the S grid positions the search changes it
    for e in entries:
        missed = sum(sparse.replay(dense["episodes"][e["id"]]["scores"], stride=stride, threshold=threshold, alert=alert,
                                   offset=offset)[0] != dense_first[e["id"]] for offset in range(stride))
        if missed:
            offset_misses[e["id"]] = missed
    misses = sum(offset_misses.values())
    dense_calls = sum(e["steps"] for e in entries)
    return {
        "strategy": "coarse_fine" if alert == threshold else "coarse_alert_windows",
        "stride": stride,
        "alert": alert,
        "calls_mean": float(np.mean(calls)),
        "calls_max": int(max(calls)),
        "calls_total": int(sum(calls)),
        "calls_mean_batch4": float(np.mean(calls_batched)),
        "fewer_calls_than_dense": dense_calls / sum(calls),
        "episodes_differing": len(differences),
        "same_transition_share": 1.0 - len(differences) / len(entries),
        "differences": differences,
        "p0": result["p0"],
        "weights": result["weights"],
        "p0_delta": result["p0"][0] - dense_result["p0"][0],
        "weights_delta": [None if a is None or b is None else a - b for a, b in zip(result["weights"], dense_result["weights"])],
        "label_changes": label_changes,
        "label_change_fraction": label_changes / records,
        "differing_over_offsets_per_100": 100.0 * misses / (stride * len(entries)),
        "offset_misses": offset_misses,
    }


def study(manifest: dict, dense: dict, *, threshold: float = DEFAULT_THRESHOLD, strides=range(1, 17),
          alerts=(0.7, 0.65, 0.6, 0.55, 0.5, 0.45), states: dict | None = None, allow_partial: bool = False) -> dict:
    """The study document: the dense reference and one row per (stride, alert)."""
    entries = _covered(manifest, dense)
    dense_result = build_rebalance(manifest, dense, threshold=threshold, allow_partial=allow_partial)
    rows = [study_row(manifest, dense, dense_result, stride=s, threshold=threshold, alert=a, states=states)
            for a in alerts for s in strides]
    meeting = [r for r in rows if r["same_transition_share"] >= PLAN_SHARE
               and r["differing_over_offsets_per_100"] <= OFFSET_MARGIN_PER_100]
    best = min(meeting, key=lambda r: (r["calls_mean"], -r["stride"], r["alert"])) if meeting else None
    return {
        "format": FORMAT,
        "threshold": threshold,
        "strategies": STRATEGIES,
        "dense": {"episodes": len(entries), "records": sum(e["steps"] for e in entries),
                  "calls_total": sum(e["steps"] for e in entries), "calls_mean": float(np.mean([e["steps"] for e in entries])),
                  "first_detection": {e["id"]: e["first_detection"] for e in dense_result["episodes"]},
                  "counts": dense_result["counts"]["records"], "p0": dense_result["p0"], "weights": dense_result["weights"]},
        "rows": rows,
        "recommendation": {"rule": RECOMMENDATION_RULE, "stride": best["stride"] if best else None,
                           "alert": best["alert"] if best else None, "calls_mean": best["calls_mean"] if best else None,
                           "note": "a rule's pick from the table, for the decision record; not a decision"},
    }


def _fmt(value, digits=3) -> str:
    return "none" if value is None else f"{value:.{digits}f}"


def markdown(doc: dict) -> str:
    rec = doc["recommendation"]
    dense = doc["dense"]
    lines = [f"Sparse search replayed on {dense['episodes']} episodes' dense scores at threshold {doc['threshold']}: dense "
             f"mode {dense['calls_mean']:.1f} calls per episode, P0 ({_fmt(dense['p0'][0])}, {_fmt(dense['p0'][1])}), "
             f"weights ({_fmt(dense['weights'][0])}, {_fmt(dense['weights'][1])}). Calls are records scored (one per "
             f"call); \"grid positions\" averages the differing episodes over every offset of the probe grid. "
             f"Recommended by rule (bold): stride {rec['stride']}, alert {rec['alert']}.", "",
             "| strategy | S | alert | calls mean (max) | at batch 4 | x fewer than dense | differing | per 100 over grid "
             "positions | differences (records) | P0 | weights | labels changed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        diffs = ", ".join(f"{d['id'][-3:]}: {d['dense']}->{d['sparse']}" for d in r["differences"][:6])
        if len(r["differences"]) > 6:
            diffs += f", ... ({len(r['differences'])})"
        cells = [r["strategy"], str(r["stride"]), f"{r['alert']:g}", f"{r['calls_mean']:.1f} ({r['calls_max']})",
                 f"{r['calls_mean_batch4']:.1f}", f"{r['fewer_calls_than_dense']:.2f}", str(r["episodes_differing"]),
                 f"{r['differing_over_offsets_per_100']:.2f}", diffs or "-",
                 f"({_fmt(r['p0'][0])}, {_fmt(r['p0'][1])})", f"({_fmt(r['weights'][0])}, {_fmt(r['weights'][1])})",
                 f"{r['label_change_fraction']:.2%}"]
        if (r["stride"], r["alert"]) == (rec["stride"], rec["alert"]):
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def check_sparse(manifest: dict, dense: dict, live: dict) -> dict:
    """A live sparse file against the dense scores of the same frames, and against the study's offline prediction."""
    search = live.get("sparse")
    if search is None:
        raise ValueError("--check-sparse takes a sparse detections file")
    if live.get("detector") != dense.get("detector"):
        raise ValueError(f"the sparse file was scored by {live.get('detector')}, the dense scores by {dense.get('detector')}")
    build_rebalance(manifest, live, threshold=search["threshold"], allow_partial=True)  # every record verified
    per_episode = []
    for eid, ep in live["episodes"].items():
        reference = dense["episodes"][eid]
        score_diffs = [abs(ep["scores"][k] - reference["scores"][i]) for k, i in enumerate(ep["visited"])]
        box_diffs = [max(abs(a - b) for a, b in zip(ep["boxes"][k], reference["boxes"][i])) for k, i in enumerate(ep["visited"])]
        predicted, needed = sparse.replay(reference["scores"], stride=search["stride"], threshold=search["threshold"],
                                          alert=search["alert"])
        per_episode.append({"id": eid, "records": ep["records"], "visited": len(ep["visited"]), "predicted_visited": len(needed),
                            "needed_frames_visited": set(needed) <= set(ep["visited"]),
                            "score_abs_diff_max": max(score_diffs), "box_abs_diff_max": max(box_diffs),
                            "first_detection": ep["first_detection"],
                            "dense": first_detection(reference["scores"], search["threshold"]), "predicted": predicted,
                            "seconds": ep.get("seconds")})
    diffs = [abs(ep["scores"][k] - dense["episodes"][eid]["scores"][i])
             for eid, ep in live["episodes"].items() for k, i in enumerate(ep["visited"])]
    return {"format": FORMAT.replace("study", "check"), "search": search, "detector": live.get("detector"),
            "runs": live.get("runs"), "episodes": len(per_episode), "frames_compared": len(diffs),
            "score_abs_diff_max": max(diffs), "score_abs_diff_mean": float(np.mean(diffs)),
            "box_abs_diff_max": max(e["box_abs_diff_max"] for e in per_episode),
            "first_detection_equal_dense": sum(e["first_detection"] == e["dense"] for e in per_episode),
            "first_detection_equal_predicted": sum(e["first_detection"] == e["predicted"] for e in per_episode),
            "per_episode": per_episode}


def _ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return out


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True, help="the store whose dense scores are replayed, data/<name>")
    p.add_argument("--detections", type=Path, default=None, help="its dense detections; default the store's own")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--strides", default="1-16", help="e.g. 1-16 or 1,2,4,8")
    p.add_argument("--alerts", default=None, help="comma-separated alert levels; default the threshold and 0.05 to 0.25 below it")
    p.add_argument("--with-states", action="store_true",
                   help="read the store's steps.npz to give each changed transition's distance and whether it was on target")
    p.add_argument("--check-sparse", type=Path, default=None, help="check mode: a sparse file scored live on this store")
    p.add_argument("--out", type=Path, default=None, help="default runs/rebalance/sparse_study.json (study mode)")
    args = p.parse_args(argv)
    raw = args.raw.resolve()
    manifest = json.loads((raw / "manifest.json").read_text())
    path = args.detections if args.detections is not None else store_file(raw)
    blob = Path(path).read_bytes()
    dense = parse_detections(blob).doc
    if args.check_sparse is not None:
        if args.out is None:
            p.error("--check-sparse needs an --out")
        live = parse_detections(args.check_sparse.read_bytes()).doc
        doc = check_sparse(manifest, dense, live)
        doc["source"] = {"dense": str(path), "dense_sha256": hashlib.sha256(blob).hexdigest(), "sparse": str(args.check_sparse)}
        _write(args.out, json.dumps(doc, indent=2) + "\n")
        print(f"{args.out}: {doc['frames_compared']} frames of {doc['episodes']} episodes, |score - dense| max "
              f"{doc['score_abs_diff_max']:.2e} (mean {doc['score_abs_diff_mean']:.2e}), box max {doc['box_abs_diff_max']:.2e}; "
              f"first detection = dense in {doc['first_detection_equal_dense']}, = prediction in "
              f"{doc['first_detection_equal_predicted']}")
        return 0
    alerts = ([float(a) for a in args.alerts.split(",")] if args.alerts is not None
              else [round(args.threshold - 0.05 * k, 2) for k in range(6) if args.threshold - 0.05 * k >= 0.0])
    states = None
    if args.with_states:
        states = {e["id"]: np.load(raw / e["path"] / "steps.npz")["state"] for e in manifest["episodes"]
                  if e["id"] in dense["episodes"]}
    doc = study(manifest, dense, threshold=args.threshold, strides=_ints(args.strides), alerts=alerts, states=states)
    doc["source"] = {"raw": str(raw), "dataset": manifest["name"], "detections": str(path),
                     "detections_sha256": hashlib.sha256(blob).hexdigest(), "detector": dense.get("detector")}
    out = args.out if args.out is not None else _ROOT / "runs" / "rebalance" / "sparse_study.json"
    _write(out, json.dumps(doc, indent=2) + "\n")
    _write(out.with_suffix(".md"), markdown(doc))
    rec = doc["recommendation"]
    print(f"{out} (+ .md): {len(doc['rows'])} rows; by rule: stride {rec['stride']}, alert {rec['alert']}, "
          f"{_fmt(rec['calls_mean'], 1)} calls per episode against {doc['dense']['calls_mean']:.1f} dense")
    return 0


if __name__ == "__main__":
    sys.exit(main())
