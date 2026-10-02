"""Where a run's evidence goes by default, and the rule that a committed record is never written over (CLAUDE.md).

Training, the gate and the throughput measurement used to write `docs/gates/m2_*.json` whatever scene they ran, so a
smoke run or a second scene without `--out` would have replaced M2's evidence. Each scene with a milestone now owns its
own prefix, any other scene must name its output, and an existing file under `docs/gates/` is refused before any work
starts. A superseded record is archived by hand (docs/gates/archive/), never overwritten by a run.
"""

from __future__ import annotations

from pathlib import Path

from autofly_ue5.paths import ROOT

GATES_DIR = ROOT / "docs" / "gates"
# The milestone whose evidence each scene's runs are (spec §12).
EVIDENCE_PREFIX = {"s01": "m2", "s01d": "m2d"}


def default_evidence_path(scene_id: str, kind: str) -> Path:
    """docs/gates/<milestone>_<kind>.json for a scene with a milestone (kind: train, gate, instances, ...)."""
    prefix = EVIDENCE_PREFIX.get(scene_id)
    if prefix is None:
        raise ValueError(f"scene {scene_id!r} has no milestone evidence file of its own; pass --out explicitly")
    return GATES_DIR / f"{prefix}_{kind}.json"


def refuse_existing_evidence(path: Path) -> None:
    """Raise if `path` is an existing file under docs/gates/: a committed record is never written over."""
    resolved = Path(path).resolve()
    if resolved.exists() and GATES_DIR.resolve() in resolved.parents:
        raise FileExistsError(f"{path} is a committed record and is never written over (CLAUDE.md, evidence rules); "
                              f"pass a new --out, or archive the old record first")
