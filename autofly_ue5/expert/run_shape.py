"""Which footprint shape scores a model: the one its training run trained under (2026-10-08, evening).

The footprint rule has two shapes (expert/footprint_contact.py): the outline, the task's rule since 2026-10-08
(afternoon), and the circle, s02_r1's. A model scored under the shape it did not train under is scored on another task,
and nothing in the result says so. s02_r1's pre-registered evaluation plan was written before the shape existed
(docs/decisions/2026-10-08-m5-s02-r1-evaluation-plan.md): with "outline" as every script's default, its commands would
have scored its circle-trained models under the outline without a word. So wherever the run is knowable, its shape is
read, not defaulted:

- A model's run is the training run whose sessions.json lies in the model's own directory or up to two above it:
  runs/expert/<run>/final.zip, .../checkpoints/rl_model_<n>_steps.zip, .../best/best_model.zip, .../averaged/<x>.zip.
  A run root is its own directory.
- Its shape is the last session's identity's footprint_contact.shape. An identity with a footprint_contact but no shape
  is from before shapes existed: the circle, the only shape there was (s02_r1; train.py resumes it under the circle on
  the same reading). An identity without a footprint_contact trained on a scene without mesh obstacles and names none.

On a scene with mesh obstacles the gate, the collector, the eval watch, the selection and the renders score that shape
(`scored_shape`). Models of runs that disagree are refused (score them separately), and so is a `--footprint-shape` that
contradicts a run's, unless `--against-trained-shape` says the contradiction is the point (the outline record of s02_r1's
pick, flown on the same 200 gate episodes, is one); the record then names the shape the run trained under. A model whose
shape cannot be read (no run record, or a run without a footprint rule) needs the flag. On a scene without mesh
obstacles the shape is irrelevant: it is never asked for, and no run record is read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from autofly_ue5.expert.footprint_contact import DEFAULT_SHAPE, SHAPES, check_shape, has_footprint_obstacles
from autofly_ue5.scenes.model import AssetRegistry, Layout

SESSIONS_FILE = "sessions.json"  # train.py's run record (train.py takes the name from here)
SEARCH_LEVELS = 2  # how many directories above a model's own its run record may lie
LEGACY_SHAPE = "circle"  # what a footprint_contact without a shape trained under
FLAG = "--footprint-shape"
AGAINST_FLAG = "--against-trained-shape"


def find_sessions(path: Path) -> Path | None:
    """The run record of the model at `path` (or of the run root `path`): the nearest sessions.json in its directory or
    up to SEARCH_LEVELS above it; None where there is none."""
    path = Path(path).absolute()
    start = path if path.is_dir() else path.parent
    for directory in (start, *list(start.parents)[:SEARCH_LEVELS]):
        candidate = directory / SESSIONS_FILE
        if candidate.is_file():
            return candidate
    return None


def identity_shape(identity: dict | None) -> str | None:
    """The footprint shape a run identity names: its footprint_contact's shape, the circle for one from before shapes,
    None for a run without a footprint rule (or without an identity)."""
    footprint = (identity or {}).get("footprint_contact")
    if not isinstance(footprint, dict):
        return None
    return check_shape(footprint.get("shape", LEGACY_SHAPE))


def trained_shape(path: Path) -> tuple[str | None, Path | None]:
    """(the shape the run of the model at `path` trained under, its sessions.json): (None, None) without a run record,
    (None, record) for a run that names no footprint rule. An unreadable record is refused (ValueError, naming it)."""
    sessions_path = find_sessions(path)
    if sessions_path is None:
        return None, None
    try:
        sessions = json.loads(sessions_path.read_text())["sessions"]
        shape = identity_shape(sessions[-1].get("identity")) if sessions else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        raise ValueError(f"{sessions_path}: unreadable run record ({type(err).__name__}: {err}), so the footprint shape "
                         f"its models trained under is unknown") from err
    return shape, sessions_path


def run_footprint_shape(layout: Layout, run_root: Path, requested: str | None,
                        registry: AssetRegistry | None = None) -> str | None:
    """The shape to hand the gate for a run's own checkpoints (the eval watch, the selection): None on a scene without
    mesh obstacles (nothing to hand), else `scored_shape`'s for the run root."""
    if not has_footprint_obstacles(layout, registry):
        return None
    return scored_shape(layout, {"the run": Path(run_root)}, requested, registry=registry)[0]


def scored_shape(layout: Layout, models: Mapping[str, Path], requested: str | None, *, against_trained: bool = False,
                 registry: AssetRegistry | None = None) -> tuple[str, str | None]:
    """(the shape to score `models` under on the scene of `layout`, the shape their runs trained under or None).

    Without mesh obstacles: (`requested` or the default, None), irrelevant, and no record is read. Otherwise the runs'
    shape (all must agree), checked against `requested`; `requested` alone where no run names one. A ValueError says
    what to do where neither decides, or where they contradict (see the module's doc)."""
    if requested is not None:
        check_shape(requested)
    if against_trained and requested is None:
        raise ValueError(f"{AGAINST_FLAG} needs {FLAG}: it scores the models under that shape, against their runs'")
    if not has_footprint_obstacles(layout, registry):
        return requested or DEFAULT_SHAPE, None
    known: dict[str, tuple[str, Path]] = {}
    unknown: dict[str, str] = {}
    for name, path in models.items():
        shape, sessions_path = trained_shape(path)
        if shape is not None:
            known[name] = (shape, sessions_path)
        elif sessions_path is None:
            unknown[name] = f"{path}: no {SESSIONS_FILE} in its directory or the {SEARCH_LEVELS} above it"
        else:
            unknown[name] = f"{path}: its run ({sessions_path}) trained without a footprint rule"
    trained = {shape for shape, _ in known.values()}
    if len(trained) > 1:
        runs = "; ".join(f"{name}: the {shape} ({sessions_path})" for name, (shape, sessions_path) in sorted(known.items()))
        raise ValueError(f"the models' runs trained under different footprint shapes ({runs}): score them in separate "
                         f"runs")
    run_shape = next(iter(trained), None)
    if requested is not None:
        if run_shape is not None and requested != run_shape and not against_trained:
            name, (_shape, sessions_path) = next(iter(sorted(known.items())))
            raise ValueError(f"{FLAG} {requested} contradicts the run {name} trained in, which trained under the "
                             f"{run_shape} ({sessions_path}): drop the flag to score it under its own rule, or add "
                             f"{AGAINST_FLAG} to score it under the {requested} on purpose (as a separately named record)")
        return requested, run_shape
    if unknown or run_shape is None:
        which = "; ".join(f"{name} ({why})" for name, why in sorted(unknown.items())) or "no model given"
        raise ValueError(f"the scene has mesh obstacles, and the footprint shape is unknown for {which}: pass "
                         f"{' or '.join(f'{FLAG} {shape}' for shape in SHAPES)}")
    return run_shape, run_shape
