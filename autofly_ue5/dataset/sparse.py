"""A sparse search for an episode's first confident detection (plan 6 step 5; Revision 1 item 6), which is all the
one-way phase split of spec §10.3 needs: `rebalance.first_detection` is the first record whose score exceeds the
threshold, and nothing after it changes the split.

Scoring every frame costs a detector call per record, about 177 per pilot episode at 0.35 frames/s on this host's CPU,
so the 100-episode pilot took 13-14 h and the prototype's ~2,000 episodes would take about 12 days. A plain binary
search is unsafe because detections are not monotone. In 48 of the pilot's 100 episodes the first record above 0.7 is
a one-record flicker among records hovering just below 0.7, and the first run of five records above it comes a median
7 records later (up to 75). A coarse scan at stride S followed by a fine scan of the window before the first coarse
hit therefore misses the transition in 24 of 100 pilot episodes already at S = 2, and in 41 at S = 8
(scripts/sparse_search_study.py; docs/runbook-rebalance.md §4).

What works is the score's own warning: the records around a flicker score close to the threshold too. Here a window
between two coarse probes is scanned whenever either probe scored above an alert level below the threshold. The one
bet the search makes is that a window whose two probes both score at or below the alert level holds no record above
the threshold. At S = 8 and an alert of 0.5 that bet lost in none of the pilot's 100 episodes, and in 0.38 per 100
averaged over all eight positions of the probe grid, for about a third of the dense mode's calls (the study's table).

This file must stay standard-library only: scripts/detect_targets.py imports it in the detector's own venv, so that
the search the detector runs and the replay the rebalancing stage verifies it with are one piece of code
(tests/test_sparse_search.py holds it to the standard library).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

STRATEGY = "coarse_alert_windows"
RULE = ("coarse probes: records 0, S, 2S, ... and the episode's last record, scored in order. The records strictly between "
        "two consecutive probes form a window, scored in order before the probe that closes it is judged, whenever either "
        "of its two probes scored above the alert level. The first detection is the first record found above the "
        "threshold in that order; a window whose two probes both scored at or below the alert level is taken to hold no "
        "record above the threshold (the search's one bet). An alert level equal to the threshold is the plain coarse "
        "scan with a fine scan of the window before its first hit; stride 1 scores every record up to the first "
        "detection, which is exact.")


def _check_parameters(stride: int, threshold: float, alert: float) -> None:
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be an integer >= 1, got {stride!r}")
    for name, value in (("threshold", threshold), ("alert", alert)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be a confidence in [0, 1), got {threshold!r}")
    if not 0.0 <= alert <= threshold:
        raise ValueError(f"alert must lie in [0, threshold] = [0, {threshold}], got {alert!r}")


def describe(stride: int, threshold: float, alert: float) -> dict:
    """The search's parameters as the detections file records them: a file is continued, and its first detections
    accepted by the rebalancing stage, only under exactly these."""
    _check_parameters(stride, threshold, alert)
    return {"strategy": STRATEGY, "stride": stride, "threshold": threshold, "alert": alert, "rule": RULE}


def coarse_probes(n_records: int, stride: int, offset: int = 0) -> list[int]:
    """Records 0, offset, offset + stride, ... and the last record. The detector always uses offset 0; the study shifts
    the grid to see how much its result owes to where the probes happened to fall."""
    if isinstance(n_records, bool) or not isinstance(n_records, int) or n_records < 1:
        raise ValueError(f"an episode needs at least one record, got {n_records!r}")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be an integer >= 1, got {stride!r}")
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < stride:
        raise ValueError(f"offset must be an integer in [0, stride), got {offset!r}")
    return sorted({0, n_records - 1, *range(offset, n_records, stride)})


def _checked(values, indices: list[int]) -> list[float]:
    values = list(values)
    if len(values) != len(indices):
        raise ValueError(f"the scorer returned {len(values)} scores for {len(indices)} records")
    for i, v in zip(indices, values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
            raise ValueError(f"record {i}: a detection score must be a finite confidence in [0, 1], got {v!r}")
    return [float(v) for v in values]


def search(n_records: int, *, stride: int, threshold: float, alert: float, score: Callable[[list[int]], Sequence[float]],
           batch: int = 1, offset: int = 0) -> tuple[int | None, dict[int, float]]:
    """The episode's first detection under the rule above, and every record scored on the way ({index: score}, in the
    order scored).

    `score(indices)` scores the given records (ascending, never one already scored, at most `batch` of them) and
    returns their confidences. With `batch` > 1 the coarse probes are scored `batch` at a time ahead of the windows,
    and a window `batch` records at a time: the result is the same as with batch 1, and at most `batch` - 1 coarse
    probes and `batch` - 1 window records are scored past the point where the search stops.
    """
    _check_parameters(stride, threshold, alert)
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(f"batch must be an integer >= 1, got {batch!r}")
    probes = coarse_probes(n_records, stride, offset)
    scored: dict[int, float] = {}

    def need(index: int, upcoming: list[int]) -> float:
        if index not in scored:
            chunk = [i for i in upcoming if i not in scored][:batch]
            scored.update(zip(chunk, _checked(score(chunk), chunk)))
        return scored[index]

    if need(probes[0], probes) > threshold:
        return probes[0], scored
    for k in range(1, len(probes)):
        lo, hi = probes[k - 1], probes[k]
        closing = need(hi, probes[k:])
        if scored[lo] > alert or closing > alert:
            window = list(range(lo + 1, hi))
            for m, index in enumerate(window):
                if need(index, window[m:]) > threshold:
                    return index, scored
        if closing > threshold:
            return hi, scored
    return None, scored


def replay(scores: Sequence[float], *, stride: int, threshold: float, alert: float,
           offset: int = 0) -> tuple[int | None, dict[int, float]]:
    """The search run on an episode's dense scores, one record at a time: what it would have scored, and found."""
    return search(len(scores), stride=stride, threshold=threshold, alert=alert, offset=offset,
                  score=lambda indices: [scores[i] for i in indices])


def verify(n_records: int, visited: dict[int, float], *, stride: int, threshold: float, alert: float) -> int | None:
    """The first detection the search finds on the records a sparse run scored. Raises if the search needs a record the
    run did not score: such a record cannot stand for the episode. Extra records (a batch's overshoot) are fine."""
    for index in visited:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n_records:
            raise ValueError(f"record index {index!r} is outside an episode of {n_records} records")

    def lookup(indices: list[int]) -> list[float]:
        missing = [i for i in indices if i not in visited]
        if missing:
            raise ValueError(f"the search needs record {missing[0]}, which the sparse run did not score")
        return [visited[i] for i in indices]

    first, _ = search(n_records, stride=stride, threshold=threshold, alert=alert, score=lookup)
    return first
