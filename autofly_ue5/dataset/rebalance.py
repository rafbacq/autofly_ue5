"""Dataset rebalancing (spec §10.3; AutoFly App. A.2.4): which records are obstacle avoidance and which are target
seeking, and the per-phase resampling weights that even the two out. Pure: numpy and the two JSON documents.

AutoFly found its trajectories 73 % obstacle avoidance and 27 % target seeking, and resampled the phases towards
uniform (weights 0.68 and 1.85). The phase boundary is a detector's: a record is target seeking from the first one in
which Grounding DINO detects the episode's target with confidence above a threshold (0.7 in the paper), and every
record before that is obstacle avoidance (spec §10.3 reads the paper's "transitioning phases upon confident detection"
as a one-way transition; its per-record wording "2 if detected, otherwise 1" would make a target that flickers out of
detection flip phases, which no resampling weight can mean). The scores come from `scripts/detect_targets.py`, cached
per frame, so the threshold can be changed here without running the detector again.

The paper's Eq. 10 gives 0.110 nats for (0.73, 0.27) where the text says "approximately 0.36 nats"; the code follows
the equation (tests/test_rebalance.py pins both numbers), and the weights, which the paper does print consistently,
come out as its 0.68 / 1.85.

The result is written to `data/<name>/rebalance.json`: the weights, every episode's transition, and how the detector
behaved there. The raw store and the RLDS shards are never rewritten (spec §10.3); a training pipeline applies the
weights, or `stratified_resample` draws the paper's rebalanced set from them.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

FORMAT = "autofly_ue5_rebalance/1"
DETECTIONS_FORMAT = "autofly_ue5_detections/1"  # scripts/detect_targets.py's FORMAT, which cannot be imported here
PHASES = ("obstacle_avoidance", "target_seeking")  # the paper's k = 1, 2
DEFAULT_THRESHOLD = 0.7
METHOD = ("AutoFly App. A.2.4 (spec 10.3): a record is target seeking from the first record whose detection confidence "
          "for the target exceeds the threshold, obstacle avoidance before it; P0 is each phase's share of all records; "
          "P_target = alpha * P0 + (1 - alpha) / K; weights = P_target / P0; resample_sizes = round(weight * sub-trajectories)")

# The front camera: 256 px across 90 degrees (configs/robot_autofly_quadrotor.jsonc), so a pinhole with f = 128 px puts
# the target's bearing b at image x = 0.5 + 0.5 tan(b). A detection counts as on the target when its box covers that
# column within this margin: a rotor-span target at 15 m is 4 px wide, and the drone rolls a little in turns.
IMAGE_HFOV_RAD = math.radians(90.0)
ON_TARGET_MARGIN = 8.0 / 256.0


def first_detection(scores: Sequence[float], threshold: float) -> int | None:
    """Index of the first record whose score exceeds `threshold` (strictly: the paper's "exceeding"); None if none does."""
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"scores must be a non-empty 1-D sequence, got shape {a.shape}")
    if not np.all(np.isfinite(a)) or np.any(a < 0.0) or np.any(a > 1.0):
        raise ValueError("detection scores must be finite confidences in [0, 1]")
    above = np.flatnonzero(a > threshold)
    return int(above[0]) if above.size else None


def phase_records(n_records: int, first: int | None) -> tuple[int, int]:
    """How many of an episode's records fall in each phase: all avoidance without a detection, all seeking from a
    detection at the first record."""
    if first is None:
        return n_records, 0
    return first, n_records - first


def phase_distribution(record_counts: Sequence[int]) -> tuple[float, ...]:
    """P0(k): each phase's share of all records (Eq. 10's input)."""
    total = sum(record_counts)
    if total <= 0:
        raise ValueError("no records to distribute")
    return tuple(c / total for c in record_counts)


def kl_from_uniform(p: Sequence[float]) -> float:
    """D_KL(P0 || Uniform(K)) = sum_k P0(k) log(K P0(k)) in nats (Eq. 10), with 0 log 0 = 0."""
    k = len(p)
    return float(sum(pk * math.log(k * pk) for pk in p if pk > 0.0))


def target_distribution(p0: Sequence[float], alpha: float = 0.0) -> tuple[float, ...]:
    """P_target(k) = alpha P0(k) + (1 - alpha) / K (Eq. 11); alpha = 0 is the uniform target the paper used."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    k = len(p0)
    return tuple(alpha * pk + (1.0 - alpha) / k for pk in p0)


def resampling_weights(p0: Sequence[float], alpha: float = 0.0) -> tuple[float | None, ...]:
    """w_k = P_target(k) / P0(k); None for a phase with no records, which no weight can create."""
    return tuple((pt / pk) if pk > 0.0 else None for pt, pk in zip(target_distribution(p0, alpha), p0))


def resample_sizes(n_sub_trajectories: Sequence[int], weights: Sequence[float | None]) -> tuple[int, ...]:
    """n_k = round(w_k |D_k|), rounding half up (Python's round() would round half to even)."""
    sizes = []
    for n, w in zip(n_sub_trajectories, weights):
        if w is None:
            if n:
                raise ValueError(f"{n} sub-trajectories in a phase that has no weight: P0 and the groups disagree")
            sizes.append(0)
        else:
            sizes.append(int(math.floor(w * n + 0.5)))
    return tuple(sizes)


def stratified_resample(groups: Sequence[Sequence], weights: Sequence[float | None], rng: np.random.Generator) -> tuple[list, ...]:
    """The paper's stratified resampling: from each phase's sub-trajectories D_k draw n_k = round(w_k |D_k|), with
    replacement when w_k > 1 (the phase is upsampled) and without when w_k <= 1. Order within a draw is the rng's."""
    sizes = resample_sizes([len(g) for g in groups], weights)
    drawn = []
    for group, w, n in zip(groups, weights, sizes):
        if n == 0:
            drawn.append([])
        elif w is not None and w > 1.0:
            drawn.append([group[i] for i in rng.choice(len(group), size=n, replace=True)])
        else:
            drawn.append([group[i] for i in sorted(rng.choice(len(group), size=n, replace=False))])
    return tuple(drawn)


def box_on_target(box: Sequence[float], bearing_rad: float, *, margin: float = ON_TARGET_MARGIN) -> bool | None:
    """Whether a normalised (cx, cy, w, h) box covers the image column the target's bearing projects to; None when the
    target is outside the camera's horizontal field of view, where no detection of it is possible."""
    if abs(bearing_rad) >= IMAGE_HFOV_RAD / 2.0:
        return None
    column = 0.5 + 0.5 * math.tan(bearing_rad) / math.tan(IMAGE_HFOV_RAD / 2.0)
    cx, _cy, w, _h = (float(v) for v in box)
    return bool(abs(column - cx) <= w / 2.0 + margin)


def _check_detections(manifest: dict, detections: dict, *, allow_partial: bool) -> list[dict]:
    """The manifest episodes the detections cover, in manifest order; raises on anything that does not match the store."""
    if detections.get("format") != DETECTIONS_FORMAT:
        raise ValueError(f"detections format {detections.get('format')!r}, expected {DETECTIONS_FORMAT!r}")
    if detections.get("dataset") != manifest["name"] or detections.get("dataset_created") != manifest["created"]:
        raise ValueError(f"the detections are for store {detections.get('dataset')!r} created "
                         f"{detections.get('dataset_created')!r}, not {manifest['name']!r} created {manifest['created']!r}")
    scored = detections.get("episodes", {})
    ids = {e["id"] for e in manifest["episodes"]}
    unknown = sorted(set(scored) - ids)
    if unknown:
        raise ValueError(f"the detections score episodes the store does not have: {unknown[:5]}")
    missing = [e["id"] for e in manifest["episodes"] if e["id"] not in scored]
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} of {len(ids)} episodes have no detections (first: {missing[0]}); score them, or "
                         f"pass allow_partial for a probe that must not pose as the store's own rebalance.json")
    covered = []
    for entry in manifest["episodes"]:
        ep = scored.get(entry["id"])
        if ep is None:
            continue
        if len(ep["scores"]) != entry["steps"]:
            raise ValueError(f"episode {entry['id']}: {len(ep['scores'])} scores for {entry['steps']} records")
        if ep.get("query") != entry["target_name"]:
            raise ValueError(f"episode {entry['id']}: scored for query {ep.get('query')!r}, its target is {entry['target_name']!r}")
        covered.append(entry)
    return covered


def build_rebalance(manifest: dict, detections: dict, *, threshold: float = DEFAULT_THRESHOLD, alpha: float = 0.0,
                    allow_partial: bool = False, states: dict[str, np.ndarray] | None = None) -> dict:
    """The rebalance.json document for a store (its manifest) and its frames' detection scores.

    `states`: optional {episode id: state[9] array (T, 9)} from the store's steps.npz, adding what the simulator knew at
    each transition: the distance to the target (state[0]) and whether the detected box sat on the target's bearing
    (state[1]), so a detector that fires on a distractor or a pillar is caught rather than trusted.
    """
    covered = _check_detections(manifest, detections, allow_partial=allow_partial)
    episodes = []
    records = [0, 0]
    sub_trajectories = [0, 0]
    persistence_hits = persistence_total = 0
    records_above = 0  # the paper's other wording, "2 if detected, otherwise 1", counted per record for comparison
    transition_distances: list[float] = []
    on_target = {"on_target": 0, "off_target": 0, "target_out_of_view": 0}
    for entry in covered:
        scores = detections["episodes"][entry["id"]]["scores"]
        first = first_detection(scores, threshold)
        records_above += int((np.asarray(scores, dtype=float) > threshold).sum())
        split = phase_records(entry["steps"], first)
        for k in range(2):
            records[k] += split[k]
            sub_trajectories[k] += int(split[k] > 0)
        episode = {"id": entry["id"], "records": entry["steps"], "first_detection": first, "phase_records": list(split)}
        if first is not None:
            tail = np.asarray(scores[first:], dtype=float)
            persistence_hits += int((tail > threshold).sum())
            persistence_total += tail.size
        if states is not None and first is not None:
            state = np.asarray(states[entry["id"]])
            distance, bearing = float(state[first, 0]), float(state[first, 1])
            verdict = box_on_target(detections["episodes"][entry["id"]]["boxes"][first], bearing)
            transition_distances.append(distance)
            on_target["on_target" if verdict else "target_out_of_view" if verdict is None else "off_target"] += 1
            episode.update(first_detection_distance_m=distance, first_detection_on_target=verdict)
        elif states is not None:
            episode.update(first_detection_distance_m=None, first_detection_on_target=None)
        episodes.append(episode)
    p0 = phase_distribution(records)
    weights = resampling_weights(p0, alpha)
    result = {
        "format": FORMAT,
        "dataset": manifest["name"],
        "method": METHOD,
        "phases": list(PHASES),
        "threshold": threshold,
        "alpha": alpha,
        "detector": detections.get("detector"),
        "coverage": {"episodes": len(covered), "of": len(manifest["episodes"]), "partial": len(covered) < len(manifest["episodes"])},
        "counts": {"records": records, "sub_trajectories": sub_trajectories, "records_above_threshold": records_above,
                   "episodes_never_detected": sum(e["first_detection"] is None for e in episodes),
                   "episodes_detected_at_first_record": sum(e["first_detection"] == 0 for e in episodes)},
        "p0": list(p0),
        "kl_nats": kl_from_uniform(p0),
        "p_target": list(target_distribution(p0, alpha)),
        "weights": list(weights),
        "degenerate": any(w is None for w in weights),
        "resample_sizes": list(resample_sizes(sub_trajectories, weights)),
        # Of the records from each episode's first detection on, the share the detector still scores above the
        # threshold: near 1 means the transition is firm, low means the threshold sits where the detector flickers.
        "detection_persistence": (persistence_hits / persistence_total) if persistence_total else None,
        "episodes": episodes,
    }
    if states is not None:
        result["first_detection"] = {"distance_m_median": (float(np.median(transition_distances)) if transition_distances else None),
                                     **on_target}
    return result


def write_rebalance(path: Path, result: dict) -> None:
    """Write the document exclusively: an existing rebalance.json is never replaced (rerun into another --out)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
