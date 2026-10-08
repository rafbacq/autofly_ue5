"""Which footprint shape scores a model: the one its training run trained under (autofly_ue5/expert/run_shape.py).

s02_r1 trained under the footprint circle; its pre-registered gate's commands were written before the shape existed
(docs/decisions/2026-10-08-m5-s02-r1-evaluation-plan.md), and a default of "outline" would have scored them under
another rule without a word. So the shape is read from the run's own record wherever it can be: the sessions.json in
the model's directory or up to two above it, its last session's identity. These tests build run roots in tmp_path as
train.py writes them; the entry points' tests are in each script's own test file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_footprint_contact import tree_scene_and_layout

# s02_r1's identity entry, as runs/expert/s02_r1/sessions.json holds it (2026-10-08 02:49): before shapes existed.
LEGACY = {"contact_m": 0.9, "training_margin_m": 0.2}
CIRCLE = {"contact_m": 0.9, "training_margin_m": 0.2, "shape": "circle"}
OUTLINE = {"contact_m": 1.05, "training_margin_m": 0.0, "shape": "outline", "fields": {"quiver_tree_01": "a" * 64}}


def run_root(tmp_path: Path, name: str, *footprints, identity: bool = True) -> Path:
    """A training run's directory as train.py leaves it: sessions.json (one session per entry of `footprints`, each
    identity naming that footprint_contact, or none for a pillar run), final.zip, a checkpoint, best_model and a mean."""
    root = tmp_path / "runs" / "expert" / name
    sessions = []
    for i, footprint in enumerate(footprints or (None,)):
        entry = {"index": i, "reward_version": "3", "seed": 0}
        if identity:
            entry["identity"] = {"scene": "s99", **({"footprint_contact": footprint} if footprint is not None else {})}
        sessions.append(entry)
    for rel in ("final.zip", "checkpoints/rl_model_50000_steps.zip", "best/best_model.zip", "averaged/avg_1k_end.zip"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"not a model")
    (root / "sessions.json").write_text(json.dumps({"sessions": sessions}, indent=2))
    return root


MODELS = ("final.zip", "checkpoints/rl_model_50000_steps.zip", "best/best_model.zip", "averaged/avg_1k_end.zip")


@pytest.mark.parametrize("rel", MODELS)
def test_a_models_run_record_is_found_in_its_directory_or_the_two_above(tmp_path, rel):
    from autofly_ue5.expert.run_shape import find_sessions

    root = run_root(tmp_path, "s02_r1", LEGACY)
    assert find_sessions(root / rel) == root / "sessions.json"
    assert find_sessions(root) == root / "sessions.json", "a run root is its own directory"
    deep = root / "selection" / "a" / "b" / "model.zip"
    deep.parent.mkdir(parents=True)
    assert find_sessions(deep) is None, "three levels down is another run's business"
    two = root / "selection" / "a" / "model.zip"
    assert find_sessions(two) == root / "sessions.json"
    assert find_sessions(tmp_path / "elsewhere" / "model.zip") is None


@pytest.mark.parametrize("footprints, expected", [
    ((LEGACY,), "circle"),  # before shapes: the circle, the only shape there was
    ((CIRCLE,), "circle"),
    ((OUTLINE,), "outline"),
    ((None,), None),  # trained on a scene without mesh obstacles: no footprint rule
    ((LEGACY, CIRCLE), "circle"),  # the last session's identity
])
def test_the_shape_a_run_trained_under_is_its_last_sessions(tmp_path, footprints, expected):
    from autofly_ue5.expert.run_shape import trained_shape

    root = run_root(tmp_path, "r", *footprints)
    assert trained_shape(root / "final.zip") == (expected, root / "sessions.json")


def test_a_run_from_before_identities_names_no_shape_and_a_model_outside_any_run_none_either(tmp_path):
    from autofly_ue5.expert.run_shape import trained_shape

    root = run_root(tmp_path, "old", identity=False)
    assert trained_shape(root / "best" / "best_model.zip") == (None, root / "sessions.json")
    loose = tmp_path / "s01_r2_best_model" / "best_model.zip"
    loose.parent.mkdir()
    loose.write_bytes(b"x")
    assert trained_shape(loose) == (None, None)


def test_an_unreadable_run_record_is_refused_naming_it(tmp_path):
    from autofly_ue5.expert.run_shape import trained_shape

    root = run_root(tmp_path, "r", CIRCLE)
    (root / "sessions.json").write_text("{ half written")
    with pytest.raises(ValueError, match="sessions.json"):
        trained_shape(root / "final.zip")


# ----------------------------------------------------------------------------------------------------------------------
# The shape an evaluation scores: the runs', or the flag's where none is known; never a silent default.
# ----------------------------------------------------------------------------------------------------------------------
def _mesh():
    return tree_scene_and_layout()[1]


def _pillars():
    from tests.test_expert_episode import scene_and_layout

    return scene_and_layout()[1]


def test_without_the_flag_the_runs_own_shape_is_scored(tmp_path):
    from autofly_ue5.expert.run_shape import scored_shape

    s02_r1 = run_root(tmp_path, "s02_r1", LEGACY)
    models = {"avg_200k_end": s02_r1 / "averaged" / "avg_1k_end.zip", "final": s02_r1 / "final.zip"}
    assert scored_shape(_mesh(), models, None) == ("circle", "circle")
    r2 = run_root(tmp_path, "s02_r2", OUTLINE)
    assert scored_shape(_mesh(), {"final": r2 / "final.zip"}, None) == ("outline", "outline")
    assert scored_shape(_mesh(), {"final": r2 / "final.zip"}, "outline") == ("outline", "outline")


def test_models_of_runs_that_trained_under_different_shapes_are_refused(tmp_path):
    from autofly_ue5.expert.run_shape import scored_shape

    r1, r2 = run_root(tmp_path, "s02_r1", LEGACY), run_root(tmp_path, "s02_r2", OUTLINE)
    with pytest.raises(ValueError, match="separate"):
        scored_shape(_mesh(), {"r1": r1 / "final.zip", "r2": r2 / "final.zip"}, None)
    with pytest.raises(ValueError, match="separate"):
        scored_shape(_mesh(), {"r1": r1 / "final.zip", "r2": r2 / "final.zip"}, "outline")


def test_a_flag_that_contradicts_the_runs_shape_is_refused_unless_deliberately_against_it(tmp_path):
    from autofly_ue5.expert.run_shape import scored_shape

    r1 = run_root(tmp_path, "s02_r1", LEGACY)
    with pytest.raises(ValueError, match="--footprint-shape outline.*circle") as err:
        scored_shape(_mesh(), {"final": r1 / "final.zip"}, "outline")
    assert "--against-trained-shape" in str(err.value) and str(r1 / "sessions.json") in str(err.value)
    # the outline record of s02_r1's pick (the evaluation plan's addendum): scored under the outline on purpose
    assert scored_shape(_mesh(), {"final": r1 / "final.zip"}, "outline", against_trained=True) == ("outline", "circle")
    with pytest.raises(ValueError, match="--against-trained-shape needs --footprint-shape"):
        scored_shape(_mesh(), {"final": r1 / "final.zip"}, None, against_trained=True)


def test_a_model_whose_shape_cannot_be_read_needs_the_flag(tmp_path):
    from autofly_ue5.expert.run_shape import scored_shape

    loose = tmp_path / "model.zip"
    loose.write_bytes(b"x")
    pillar_run = run_root(tmp_path, "s01_r2", None)
    for model in (loose, pillar_run / "final.zip"):
        with pytest.raises(ValueError, match="pass --footprint-shape outline or --footprint-shape circle"):
            scored_shape(_mesh(), {"m": model}, None)
        assert scored_shape(_mesh(), {"m": model}, "circle") == ("circle", None)
    s02_r1 = run_root(tmp_path, "s02_r1", LEGACY)
    with pytest.raises(ValueError, match="--footprint-shape"):  # one unknown among known ones still needs the flag
        scored_shape(_mesh(), {"r1": s02_r1 / "final.zip", "m": loose}, None)
    assert scored_shape(_mesh(), {"r1": s02_r1 / "final.zip", "m": loose}, "circle") == ("circle", "circle")


def test_on_a_scene_without_mesh_obstacles_the_shape_is_never_asked_for(tmp_path):
    from autofly_ue5.expert.footprint_contact import DEFAULT_SHAPE
    from autofly_ue5.expert.run_shape import scored_shape

    s02_r1 = run_root(tmp_path, "s02_r1", LEGACY)
    (s02_r1 / "sessions.json").write_text("{ unreadable, and never read")
    loose = tmp_path / "model.zip"
    for models in ({"m": loose}, {"r1": s02_r1 / "final.zip"}):
        assert scored_shape(_pillars(), models, None) == (DEFAULT_SHAPE, None)
        assert scored_shape(_pillars(), models, "circle") == ("circle", None)


def test_without_any_model_the_shape_must_be_given():
    from autofly_ue5.expert.run_shape import scored_shape

    with pytest.raises(ValueError, match="no model given.*pass --footprint-shape"):
        scored_shape(_mesh(), {}, None)
    assert scored_shape(_mesh(), {}, "outline") == ("outline", None)


def test_an_unknown_flag_value_is_refused(tmp_path):
    from autofly_ue5.expert.run_shape import scored_shape

    with pytest.raises(ValueError, match="shape"):
        scored_shape(_mesh(), {"m": tmp_path / "m.zip"}, "square")
