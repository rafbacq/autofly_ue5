"""The detections cache: Grounding DINO's scores for a raw store's frames, written by scripts/detect_targets.py and read
back by it on resume and by the rebalancing stage (spec §10.3; autofly_ue5/dataset/rebalance.py).

The form (FORMAT) is JSON Lines, appended and never rewritten:

    {"kind": "header", "format", "dataset", "dataset_created", "score", "detector", "sparse"}   the first line, once
    {"kind": "run", "started", "device", "threads", "torch", "batch", "episodes_total"}           each run's start
    {"kind": "episode", "id", "query", "prompt", "records", "scores", "boxes", "frames_scored", "seconds"}   dense
    {"kind": "episode", ..., "visited", "scores", "boxes", "first_detection", ...}                 sparse (sparse.py)
    {"kind": "run_end", "finished"}                                                                a run's clean end

`sparse` is null for a dense file (a score and a box for every record) and the search's parameters for a sparse one
(`sparse.describe`: scores and boxes for the `visited` records only, and the first detection the search found). A file
holds one kind: the detector continues a file only under its own header.

Why this form: the first one (FORMAT_V1, `data/s01_pilot/detections.json`) was one JSON document rewritten whole
after every episode, which is quadratic in the episodes (the 2026-10-06 review), and the prototype holds about 2,000.
Here each episode is one line, appended and synced; a host freeze mid-append (MEMORY.md) leaves at most a torn last
line, which `parse` sets aside and the detector cuts off before it appends again. Any other unreadable line is
corruption and is refused. FORMAT_V1 files are still read, never continued.

Standard library only: scripts/detect_targets.py imports this in the detector's own venv (tests/test_sparse_search.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FORMAT = "autofly_ue5_detections/2"
FORMAT_V1 = "autofly_ue5_detections/1"  # the pilot's whole-document form, read only
FILE_NAME = "detections.jsonl"  # a store's cache, beside its manifest
FILE_NAME_V1 = "detections.json"  # the pilot's (data/s01_pilot)
KINDS = ("header", "run", "episode", "run_end")


def store_file(raw: Path) -> Path:
    """A store's detections file: the appendable cache or the pilot's form, whichever it holds. Both is refused rather
    than guessed between; neither is a FileNotFoundError."""
    raw = Path(raw)
    cache, whole = raw / FILE_NAME, raw / FILE_NAME_V1
    if cache.is_file() and whole.is_file():
        raise ValueError(f"{raw} holds both {FILE_NAME} and {FILE_NAME_V1}: pass --detections to say which one is meant")
    if cache.is_file():
        return cache
    if whole.is_file():
        return whole
    raise FileNotFoundError(f"{raw} has no {FILE_NAME} or {FILE_NAME_V1}: score it with scripts/detect_targets.py")


@dataclass(frozen=True)
class Parsed:
    doc: dict              # header fields, "runs", "episodes" {id: line without kind and id}, "episodes_total", "complete"
    complete_bytes: int    # the length of the complete lines: where an appender continues
    torn: bool             # a last line without its newline was set aside


def encode(obj: dict) -> bytes:
    """One line of the cache."""
    return json.dumps(obj, separators=(",", ":")).encode() + b"\n"


def parse(blob: bytes) -> Parsed:
    """The cache in `blob` (a whole file's bytes, read once so a caller can hash what it parsed), in either form."""
    try:
        whole = json.loads(blob)
    except ValueError:  # several lines (or a torn one): the appendable form
        whole = None
    if isinstance(whole, dict) and whole.get("format") == FORMAT_V1:
        return Parsed({**whole, "sparse": whole.get("sparse")}, len(blob), False)

    end = blob.rfind(b"\n") + 1  # everything up to the last newline is complete lines; the rest is a torn append
    lines = blob[:end].split(b"\n")[:-1]
    if not lines:
        raise ValueError("the detections file holds no complete header line")
    objs = []
    for number, raw in enumerate(lines, 1):
        try:
            obj = json.loads(raw)
        except ValueError as err:
            raise ValueError(f"line {number} of the detections file is not JSON ({err}): only a torn last line, one "
                             f"without its newline, is set aside") from None
        if not isinstance(obj, dict) or obj.get("kind") not in KINDS:
            raise ValueError(f"line {number} of the detections file has no known kind ({str(obj)[:80]})")
        objs.append(obj)
    head = objs[0]
    if head["kind"] != "header" or head.get("format") != FORMAT:
        raise ValueError(f"the detections file does not start with a {FORMAT} header line: {str(head)[:120]}")

    doc = {k: v for k, v in head.items() if k != "kind"}
    doc.setdefault("sparse", None)
    runs: list[dict] = []
    episodes: dict[str, dict] = {}
    for number, obj in enumerate(objs[1:], 2):
        kind = obj["kind"]
        if kind == "header":
            raise ValueError(f"line {number} of the detections file is a second header")
        if kind == "run":
            runs.append({**{k: v for k, v in obj.items() if k != "kind"}, "episodes_scored": 0, "frames_scored": 0,
                         "seconds": 0.0})
        elif kind == "run_end":
            if runs:
                runs[-1]["finished"] = obj.get("finished")
        else:
            eid = obj.get("id")
            if not isinstance(eid, str):
                raise ValueError(f"line {number} of the detections file is an episode without an id")
            if eid in episodes:
                raise ValueError(f"episode {eid} appears twice in the detections file (line {number})")
            episodes[eid] = {k: v for k, v in obj.items() if k not in ("kind", "id")}
            if runs:
                runs[-1]["episodes_scored"] += 1
                runs[-1]["frames_scored"] += int(obj.get("frames_scored", 0))
                runs[-1]["seconds"] = round(runs[-1]["seconds"] + float(obj.get("seconds", 0.0)), 1)
    total = runs[-1].get("episodes_total") if runs else None
    doc.update(runs=runs, episodes=episodes, episodes_total=total,
               complete=total is not None and len(episodes) >= total)
    return Parsed(doc, end, end < len(blob))
