"""The evaluation harness's frozen episode files (plan 6 item 6, revision 1 item 5): per condition, which (scene, seed,
target split) a policy is scored on, drawn from expert/seeds.py's harness range, pinned to each scene's file and layout,
written once and never over. The test_seen layout s06r is listed but pending until its gate passes."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tests.test_expert_episode import scene_and_layout

TRAINING = ("s01", "s01d", "s02", "s03", "s04", "s05", "s06", "s07", "s09")


def fake_resolver(calls: list | None = None):
    """Every scene id resolved to s01's layout under its own id and split (the stored layouts of the other scenes are
    build output this checkout may not have); each layout's hash is named after its scene."""
    from autofly_ue5.scenes.resolve import ResolvedScene

    scene, layout = scene_and_layout()

    def resolve(scene_id: str) -> ResolvedScene:
        if calls is not None:
            calls.append(scene_id)
        own = dataclasses.replace(scene, id=scene_id, split="test_seen" if scene_id.endswith("r") else "train")
        return ResolvedScene(scene=own, base_scene=own, layout=layout, layout_path=Path(f"/levels/{scene_id}.layout.json"),
                             layout_sha256=f"{scene_id}-layout", map_path=f"/Game/AutoFly/Maps/{scene_id.upper()}",
                             default_scene_config="scene_autofly_s01.jsonc")

    return resolve


def _gates(tmp_path: Path, **records) -> Path:
    gates = tmp_path / "gates"
    gates.mkdir(exist_ok=True)
    for name, passed in records.items():
        (gates / f"{name}.json").write_text(json.dumps({"pass": passed}))
    return gates


def test_the_conditions_name_their_scenes_and_target_splits():
    from autofly_ue5.evaluation.episodes import CONDITIONS
    from autofly_ue5.paths import SCENES_DIR

    assert CONDITIONS["seen"].scenes == TRAINING and CONDITIONS["seen"].target_split is None
    assert CONDITIONS["unseen_targets"].scenes == TRAINING and CONDITIONS["unseen_targets"].target_split == "unseen"
    assert CONDITIONS["test_seen"].scenes == ("s05r", "s06r") and CONDITIONS["test_seen"].target_split is None
    assert CONDITIONS["test_seen"].gated and not CONDITIONS["seen"].gated
    for spec in CONDITIONS.values():  # each scene file says the split the condition expects of it
        for scene_id in spec.scenes:
            (path,) = SCENES_DIR.glob(f"{scene_id}_*.json")
            data = json.loads(path.read_text())
            assert data["id"] == scene_id and data["split"] == spec.scene_split, path


def test_the_seed_slots_are_append_only():
    # A frozen file's seeds were drawn by these slots: reordering them would hand a new file another file's episodes.
    from autofly_ue5.evaluation.episodes import CONDITION_SLOTS, CONDITIONS, SCENE_SLOTS

    assert CONDITION_SLOTS[:3] == ("seen", "test_seen", "unseen_targets")
    assert SCENE_SLOTS[:11] == ("s01", "s01d", "s02", "s03", "s04", "s05", "s06", "s07", "s09", "s05r", "s06r")
    assert set(CONDITIONS) <= set(CONDITION_SLOTS) and len(set(SCENE_SLOTS)) == len(SCENE_SLOTS)


def test_each_scene_flies_the_first_seeds_of_its_own_slice(tmp_path):
    from autofly_ue5.evaluation.episodes import CONDITION_SLOTS, SCENE_SLOTS, condition_document
    from autofly_ue5.expert.seeds import eval_harness_seed_base

    for name, split in (("seen", None), ("unseen_targets", "unseen")):
        doc = condition_document(name, episodes_per_scene=20, resolve=fake_resolver(), gates_dir=_gates(tmp_path))
        assert doc["condition"] == name and doc["episodes_per_scene"] == 20 and doc["a0"] == "sector8"
        assert list(doc["scenes"]) == list(TRAINING)
        for scene_id, entry in doc["scenes"].items():
            base = eval_harness_seed_base(CONDITION_SLOTS.index(name), SCENE_SLOTS.index(scene_id))
            assert entry["seed_base"] == base and entry["slot"] == SCENE_SLOTS.index(scene_id)
            assert [e["seed"] for e in doc["episodes"] if e["scene"] == scene_id] == list(range(base, base + 20))
            assert entry["layout_sha256"] == f"{scene_id}-layout" and len(entry["scene_sha256"]) == 64
            assert entry["status_at_write"] == "ready" and entry["requires_gate"] is None and entry["skipped_seeds"] == []
        assert {e["target_split"] for e in doc["episodes"]} == {split}
        assert len(doc["episodes"]) == 20 * len(TRAINING)


def test_a_seed_the_sampler_cannot_place_is_skipped_and_named(tmp_path, monkeypatch):
    # The resilient wrapper retries an EpisodeSetupError as a reset fault, relaunching the simulator three times
    # before it gives up: such a seed must never reach a run.
    import autofly_ue5.evaluation.episodes as episodes
    from autofly_ue5.expert.episode import EpisodeSetupError
    from autofly_ue5.expert.seeds import eval_harness_seed_base

    base = eval_harness_seed_base(0, 0)
    real = episodes.sample_setup

    def refuses_one(scene, layout, rng):
        if int(rng.bit_generator.seed_seq.entropy) == base + 3:
            raise EpisodeSetupError("could not place the target in 200 tries")
        return real(scene, layout, rng)

    monkeypatch.setattr(episodes, "sample_setup", refuses_one)
    doc = episodes.condition_document("seen", scenes=("s01",), episodes_per_scene=5, resolve=fake_resolver(),
                                      gates_dir=_gates(tmp_path))
    assert [e["seed"] for e in doc["episodes"]] == [base, base + 1, base + 2, base + 4, base + 5]
    assert doc["scenes"]["s01"]["skipped_seeds"] == [{"seed": base + 3, "reason": "could not place the target in 200 tries"}]


def test_a_test_seen_scene_is_pending_until_a_gate_record_of_it_passes(tmp_path):
    from autofly_ue5.evaluation.episodes import condition_document, m4_gate_status

    gates = _gates(tmp_path, m4_s05r_gate=True)
    doc = condition_document("test_seen", episodes_per_scene=4, resolve=fake_resolver(), gates_dir=gates)
    s05r, s06r = doc["scenes"]["s05r"], doc["scenes"]["s06r"]
    assert s05r["status_at_write"] == "ready" and s05r["gate_at_write"] == "m4_s05r_gate.json"
    assert s06r["status_at_write"] == "pending" and "m4_s06r_gate.json" in s06r["pending_reason"]
    assert s05r["requires_gate"] == "docs/gates/m4_s05r_gate.json" and s06r["requires_gate"] == "docs/gates/m4_s06r_gate.json"
    assert sum(e["scene"] == "s06r" for e in doc["episodes"]) == 4, "a pending scene's seeds are frozen with the rest"
    _gates(tmp_path, m4_s06r_gate=False)
    assert m4_gate_status("s06r", gates) == {"ready": False, "gate": None,
                                             "reason": f"no passing m4_s06r_gate.json or m4_s06r_gate_run<N>.json in {gates}"}
    _gates(tmp_path, m4_s06r_gate_run2=True)  # a re-run's record, as s02's third gate was m4_s02_gate_run3.json
    assert m4_gate_status("s06r", gates) == {"ready": True, "gate": "m4_s06r_gate_run2.json", "reason": None}


def test_a_scene_whose_split_is_not_the_conditions_is_refused(tmp_path):
    from autofly_ue5.evaluation.episodes import condition_document

    resolve = fake_resolver()

    def wrong_split(scene_id):
        resolved = resolve(scene_id)
        return dataclasses.replace(resolved, scene=dataclasses.replace(resolved.scene, split="test_seen"))

    with pytest.raises(ValueError, match="split"):
        condition_document("seen", scenes=("s01",), episodes_per_scene=2, resolve=wrong_split, gates_dir=_gates(tmp_path))
    with pytest.raises(ValueError, match="s05r"):
        condition_document("seen", scenes=("s05r",), episodes_per_scene=2, resolve=resolve, gates_dir=_gates(tmp_path))


def test_a_written_file_loads_back_and_is_never_written_over(tmp_path):
    import hashlib

    from autofly_ue5.evaluation.episodes import EvalEpisode, load_condition, write_condition

    out = tmp_path / "eval_episodes"
    path = write_condition("seen", out, scenes=("s01", "s01d"), episodes_per_scene=3, resolve=fake_resolver(),
                           gates_dir=_gates(tmp_path))
    assert path == out / "seen.json"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_condition("seen", out, scenes=("s01",), episodes_per_scene=3, resolve=fake_resolver(),
                        gates_dir=_gates(tmp_path))
    assert path.read_bytes() == before
    loaded = load_condition(path)
    assert loaded.name == "seen" and loaded.scene_ids == ("s01", "s01d") and loaded.sha256 == hashlib.sha256(before).hexdigest()
    assert loaded.episodes_for("s01d")[0] == EvalEpisode(scene="s01d", seed=loaded.scenes["s01d"]["seed_base"],
                                                         target_split=None)
    assert len(loaded.episodes) == 6 and loaded.a0 == "sector8"


@pytest.mark.parametrize("tamper, match", [
    (lambda d: d["episodes"][0].update(seed=d["episodes"][0]["seed"] + 20_000), "slice"),
    (lambda d: d["episodes"][1].update(seed=d["episodes"][0]["seed"]), "twice"),
    (lambda d: d["episodes"][0].update(target_split="unseen"), "target_split"),
    (lambda d: d["episodes"][0].update(scene="s07"), "s07"),
    (lambda d: d.update(version=99), "version"),
    (lambda d: d.update(a0=None), "a0"),
    (lambda d: d["episodes"].pop(), "episodes_per_scene"),
])
def test_a_file_whose_episodes_left_their_slices_or_contradict_it_is_refused(tmp_path, tamper, match):
    from autofly_ue5.evaluation.episodes import load_condition, write_condition

    path = write_condition("seen", tmp_path / "e", scenes=("s01", "s01d"), episodes_per_scene=3, resolve=fake_resolver(),
                           gates_dir=_gates(tmp_path))
    data = json.loads(path.read_text())
    tamper(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_condition(path)
