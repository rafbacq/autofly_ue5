"""Average several checkpoints' policy weights into one checkpoint: weight averaging along one training run.

Why (2026-10-06): run 6's neighbouring checkpoints flew very differently in deterministic flight. In its selection's
stage 1, step_500000 left the bounds in 1 of 40 episodes and step_520000 in 15. The mean of its 250k-550k checkpoints
flew 100 fresh validation episodes without a single exit: 0.94, against 0.83 for the stage-2 winner on the same
episodes (McNemar p = 0.035, docs/decisions/2026-10-06-s01d-r6-plan.md).

The network has no batch-norm or other running statistics (expert/features.py: Conv2d, Linear, ReLU), so the mean of
its parameters is the whole model. The output is an ordinary SB3 zip, which every evaluation script flies as it is.
Its other state (entropy coefficient, optimisers) is the first source's, and only matters to training. A record
beside it, <out>.json, names every source checkpoint and the result by sha256.

    env -u PYTHONPATH .venv/bin/python -m scripts.average_checkpoints runs/expert/s01d_r6/averaged/avg_250k_550k.zip \\
        runs/expert/s01d_r6/checkpoints/rl_model_{250000..550000..10000}_steps.zip
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap, as in m2_gate.py: a direct `python scripts/average_checkpoints.py` puts scripts/ at sys.path[0].
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hashlib  # noqa: E402
import json  # noqa: E402
from typing import Sequence  # noqa: E402

import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from autofly_ue5.durable import write_text_durably  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average(sources: Sequence[Path], out: Path) -> dict:
    """Write the mean of `sources`' policy weights to `out` (and its record to `out`.json); returns the record.
    Refuses, before writing anything, an empty source list, an existing `out`, and sources of different networks."""
    sources, out = [Path(p) for p in sources], Path(out)
    if not sources:
        raise ValueError("no checkpoints to average")
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite it")
    base = SAC.load(sources[0], device="cpu")
    reference = {k: v.shape for k, v in base.policy.state_dict().items()}
    sums = {k: torch.zeros(shape, dtype=torch.float64) for k, shape in reference.items()}
    for path in sources:
        state = SAC.load(path, device="cpu").policy.state_dict()
        if {k: v.shape for k, v in state.items()} != reference:
            raise ValueError(f"{path} holds another network than {sources[0]}")
        for key, value in state.items():
            sums[key] += value.double()
    base.policy.load_state_dict({k: (v / len(sources)).to(torch.float32) for k, v in sums.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    base.save(out)
    record = {"averaged": str(out), "sha256": _sha256(out), "n_sources": len(sources),
              "sources": [{"path": str(p), "sha256": _sha256(p)} for p in sources]}
    write_text_durably(out.with_suffix(".json"), json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        print("usage: average_checkpoints.py <out.zip> <checkpoint.zip> [<checkpoint.zip> ...]", file=sys.stderr)
        return 2
    try:
        record = average([Path(p) for p in args[1:]], Path(args[0]))
    except (FileExistsError, FileNotFoundError, ValueError) as err:
        print(f"refusing: {err}", file=sys.stderr)
        return 2
    print(f"{record['averaged']}: mean of {record['n_sources']} checkpoints, sha256 {record['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
