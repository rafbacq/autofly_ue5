"""Normalisation statistics in OpenVLA's `dataset_statistics.json` schema (plan 6, revision 1, item 1).

The schema, read from openvla/openvla@c8f03f4 (`prismatic/vla/datasets/rlds/utils/data_utils.py`,
`get_dataset_statistics` and `save_dataset_statistics`; 2026-10-08):

    {"<dataset name>": {"action":  {"mean", "std", "max", "min", "q01", "q99": [float] * action dims, "mask": [bool] * dims},
                        "proprio": {the same over the state's dims},
                        "num_transitions": int, "num_trajectories": int}}

- Lists of Python floats; quantiles are numpy's default (linear) at 0.01 and 0.99, std is the population std (ddof 0).
- `mask` says which dimensions are normalised. OpenVLA's `make_dataset_from_rlds` adds it to the action block from its
  `action_normalization_mask`, and `normalize_action_and_proprio` and the HF model's un-normalisation read it with
  all-True as the default. Every dimension here is continuous, so it is all True, for proprio too.
- The dataset name keys the file, as in the `dataset_statistics.json` OpenVLA saves beside a checkpoint (the HF model's
  `norm_stats[unnorm_key]`); `make_dataset_from_rlds(dataset_statistics=...)` takes the inner dict.
- Train split only: the val split must stay unseen by anything fitted, normalisation included. (OpenVLA computes its
  own over split "all" when given none.)
- Accumulated in float64. OpenVLA sums float32 arrays along axis 0, which numpy does without pairwise summation, so its
  means and stds drift on long data; quantiles, min and max are the same either way.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

STATISTICS_FILE = "dataset_statistics.json"
QUANTILES = (0.01, 0.99)


def _block(values: np.ndarray) -> dict:
    return {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(), "max": values.max(0).tolist(),
            "min": values.min(0).tolist(), "q01": np.quantile(values, QUANTILES[0], axis=0).tolist(),
            "q99": np.quantile(values, QUANTILES[1], axis=0).tolist(), "mask": [True] * values.shape[1]}


def dataset_statistics(name: str, actions: Sequence[np.ndarray], states: Sequence[np.ndarray]) -> dict:
    """OpenVLA's statistics of the episodes whose per-record actions (T, 3) and states (T, 9) are given, one array per
    episode: the train split's."""
    if not actions or len(actions) != len(states):
        raise ValueError(f"{len(actions)} action arrays and {len(states)} state arrays; statistics need episodes")
    action = np.concatenate([np.asarray(a, dtype=np.float64) for a in actions])
    proprio = np.concatenate([np.asarray(s, dtype=np.float64) for s in states])
    if action.ndim != 2 or proprio.ndim != 2 or len(action) != len(proprio):
        raise ValueError(f"actions {action.shape} and states {proprio.shape} do not pair up record for record")
    return {name: {"action": _block(action), "proprio": _block(proprio), "num_transitions": int(len(action)),
                   "num_trajectories": len(actions)}}
