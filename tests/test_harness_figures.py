"""scripts/harness_figures.py: one figure, one JSON of every number drawn and one markdown table from any list of
evaluation harness records (autofly_ue5/evaluation): per condition, scene and policy the success rate with its Wilson
interval, the collision and out-of-bounds rates and PER's interval; the expert's records, split by scene group, merged
into one policy per condition; and plan 6's done check (the expert's interval against its pilot's kept-rate interval,
the scripted baseline below it). Synthetic records in tmp_path, and one pair of real harness records flown on the fake."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PNG = b"\x89PNG\r\n\x1a\n"
EXPERT_S01 = {"name": "expert", "kind": "sac_expert", "checkpoint": "runs/expert/s01_r2/best/best_model.zip",
              "sha256": "a" * 64, "deterministic": False, "obs_config": {"depth_frames": 1, "depth_dtype": "float32"},
              "privileged_observation": True, "trained_footprint_shape": None}
EXPERT_S01D = {**EXPERT_S01, "checkpoint": "runs/expert/s01d_r6/averaged/avg_250k_550k.zip", "sha256": "b" * 64,
               "obs_config": {"depth_frames": 3, "depth_dtype": "float16", "mover_slots": 4}}
SCRIPTED = {"name": "scripted", "kind": "privileged_straight_line", "requires_privileged_state": True, "cruise_m_s": 2.0,
            "altitude_m": 2.0, "aligned_rad": 0.3, "turn_gain": 2.0, "climb_gain": 1.0}
HOVER = {"name": "hover", "kind": "floor", "action": [0.0, 0.0, 0.0]}
SEEN_SCENES = ["s01", "s01d", "s02", "s03", "s04", "s05", "s06", "s07", "s09"]
SCENE_SLOT = {s: i for i, s in enumerate(SEEN_SCENES + ["s05r", "s06r"])}


def _episode(scene: str, seed: int, outcome: str, *, per=(0.95, 1.0), steps: int = 150) -> dict:
    success = outcome == "success"
    return {"scene": scene, "seed": seed, "target_split": None, "outcome": outcome, "steps": steps,
            "instruction": "advance to red cube while avioding white pillars and other obstacles",
            "target": {"name": "red cube", "category": "geometry", "seen": True}, "distractors": [],
            "path_length_m": 66.0, "l_opt": {"straight_m": 62.0, "grid_m": 66.0},
            "per": {"lower": per[0], "upper": per[1]} if success else {"lower": None, "upper": None},
            "collision_source": ("footprint" if scene in ("s02", "s03") else "sim") if outcome == "collision" else None,
            "oob_kind": "altitude" if outcome == "out_of_bounds" else None,
            "policy_act_s": {"mean": 0.004, "max": 0.01}, "retries": 0, "fault_counts": {}}


def _outcomes(success=0, collision=0, out_of_bounds=0, timeout=0) -> list[str]:
    return ["success"] * success + ["collision"] * collision + ["out_of_bounds"] * out_of_bounds + ["timeout"] * timeout


def _record(directory: Path, name: str, *, policy: dict, scenes: dict, condition: str = "seen",
            privileged: bool = False, status: str = "ok", condition_sha: str = "c" * 64, max_steps: int = 300,
            episodes_per_scene: int = 20, condition_scenes=None, scene_status=None, seed_offset: int = 0,
            pool_sha: str = "p" * 64) -> Path:
    """A harness record as autofly_ue5/evaluation/harness.py writes it. `scenes`: {scene: outcomes} (a list of outcome
    names, or (outcome, PER pair) tuples); `scene_status`: {scene: (status, reason)} for scenes flown no episode of."""
    from autofly_ue5.evaluation.metrics import summarise

    condition_slot = {"seen": 0, "test_seen": 1, "unseen_targets": 2}[condition]
    episodes, entries = [], {}
    for scene, outcomes in scenes.items():
        base = 600_000_000 + condition_slot * 1_000_000 + SCENE_SLOT[scene] * 10_000 + seed_offset
        flown = []
        for i, outcome in enumerate(outcomes):
            outcome, per = outcome if isinstance(outcome, tuple) else (outcome, (0.95, 1.0))
            flown.append(_episode(scene, base + i, outcome, per=per))
        episodes += flown
        entries[scene] = {"status": "ok" if status == "ok" else "running", "reason": None, "error": None,
                          "requires_gate": None, "gate": None, "scene_sha256": "s" * 64, "layout_sha256": "l" * 64,
                          "target_split": None, "episodes_requested": episodes_per_scene, "metrics": summarise(flown)}
    for scene, (state, reason) in (scene_status or {}).items():
        entries[scene] = {"status": state, "reason": reason, "error": None, "requires_gate": None, "gate": None,
                          "scene_sha256": "s" * 64, "layout_sha256": "l" * 64, "target_split": None,
                          "episodes_requested": episodes_per_scene, "metrics": None}
    if condition_scenes is None:
        condition_scenes = {"seen": SEEN_SCENES, "unseen_targets": SEEN_SCENES, "test_seen": ["s05r", "s06r"]}[condition]
    record = {"harness": "autofly_ue5.evaluation", "record_version": 1, "description": "...",
              "condition": {"name": condition, "file": f"assets/eval_episodes/{condition}.json", "sha256": condition_sha,
                            "episodes_per_scene": episodes_per_scene, "scenes": list(condition_scenes)},
              "policy": policy, "privileged_state": privileged, "privileged_state_indices": [0, 1], "a0": "sector8",
              "max_steps": max_steps, "max_episodes_per_scene": None, "contact_rule_shape": "outline",
              "target_pool": {"path": "assets/targets.json", "sha256": pool_sha, "version": 1},
              "l_opt": {"inflate_m": 1.4, "resolution_m": 0.5}, "instance": 7, "provenance": None, "scenes": entries,
              "overall": summarise(episodes), "episodes": episodes, "status": status, "error": None,
              "run_started": "2026-10-08 22:07:17", "run_finished": None if status == "running" else "2026-10-08 23:00:00"}
    path = directory / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def _pilots(directory: Path) -> Path:
    def entry(pilot, sha, kept, flown, rate, interval):
        return {"pilot": pilot, "checkpoint": f"runs/expert/{pilot}.zip", "sha256": sha, "kept": kept, "flown": flown,
                "kept_rate": rate, "wilson_95": interval, "passes_bar": True}

    path = directory / "plan6_pilots.json"
    path.write_text(json.dumps({"description": "...", "scenes": {
        "s01": entry("s01_best", "a" * 64, 50, 52, 0.9615, [0.8702, 0.9894]),
        "s01d": entry("s01d_mean", "b" * 64, 50, 55, 0.9091, [0.8042, 0.9605]),
        "s07": entry("s01d_mean", "b" * 64, 50, 51, 0.9804, [0.897, 0.9965]),
        "s09": entry("s01d_mean", "b" * 64, 50, 50, 1.0, [0.9287, 1.0])}}))
    return path


def _world(tmp_path: Path) -> dict:
    """The expert flown in two scene groups (two pilots), the scripted baseline and hover on seen; hover on test_seen."""
    records = tmp_path / "records"
    return {
        "expert_s01": _record(records, "seen_expert_s01", policy=EXPERT_S01,
                              scenes={"s01": _outcomes(success=19, collision=1)}),
        "expert_group": _record(records, "seen_expert_s01d_s07", policy=EXPERT_S01D,
                                scenes={"s01d": _outcomes(success=17, collision=2, out_of_bounds=1),
                                        "s07": _outcomes(success=10, collision=10)}),
        "scripted": _record(records, "seen_scripted", policy=SCRIPTED, privileged=True,
                            scenes={"s01": _outcomes(success=2, collision=18), "s01d": _outcomes(collision=20),
                                    "s07": _outcomes(success=12, collision=8)}),
        "hover_test_seen": _record(records, "test_seen_hover", policy=HOVER, condition="test_seen",
                                   scenes={"s05r": _outcomes(timeout=20)},
                                   scene_status={"s06r": ("pending", "no passing m4_s06r_gate.json in docs/gates")}),
    }


def _build(tmp_path, paths, **kw):
    from scripts.harness_figures import build

    kw.setdefault("pilots_path", _pilots(tmp_path))
    return build(list(paths), tmp_path / "figures", **kw)


# ----------------------------------------------------------------------------------------------------------------------
# Merging and the numbers.
# ----------------------------------------------------------------------------------------------------------------------
def test_the_expert_s_records_split_by_scene_group_merge_into_one_policy_per_condition(tmp_path):
    from autofly_ue5.evaluation.metrics import summarise, wilson_interval

    world = _world(tmp_path)
    summary = _build(tmp_path, world.values())
    seen = summary["conditions"]["seen"]
    assert list(seen["policies"]) == ["expert", "scripted (privileged state)"], "the expert first, one entry for both files"
    expert = seen["policies"]["expert"]
    assert list(expert["scenes"]) == ["s01", "s01d", "s07"], "in the condition file's scene order"
    s01 = expert["scenes"]["s01"]
    assert s01["episodes"] == 20 and s01["outcomes"]["success"] == 19 and s01["complete"] is True
    lo, hi = wilson_interval(19, 20)
    assert s01["success_rate"]["value"] == 0.95 and s01["success_rate"]["wilson95"] == [round(lo, 4), round(hi, 4)]
    assert s01["checkpoint"] == EXPERT_S01["checkpoint"] and s01["sha256"] == "a" * 64
    assert expert["scenes"]["s07"]["sha256"] == "b" * 64, "each scene keeps the pilot that flew it"
    assert expert["scenes"]["s01d"]["out_of_bounds_rate"]["value"] == 0.05
    assert expert["scenes"]["s01d"]["records"] == [str(world["expert_group"])]
    episodes = [e for key in ("expert_s01", "expert_group") for e in json.loads(world[key].read_text())["episodes"]]
    overall = summarise(episodes)
    assert expert["all"]["episodes"] == 60 and expert["all"]["scenes"] == ["s01", "s01d", "s07"]
    assert expert["all"]["success_rate"]["value"] == round(overall["success_rate"]["value"], 4)
    assert expert["all"]["per"]["lower"] == round(overall["per"]["lower"], 4)
    scripted = seen["policies"]["scripted (privileged state)"]
    assert scripted["scenes"]["s01d"]["success_rate"]["value"] == 0.0 and scripted["scenes"]["s01d"]["per"]["lower"] is None
    assert summary["policies"]["expert"]["kind"] == "sac_expert" and summary["policies"]["scripted (privileged state)"][
        "privileged_state"] is True
    assert {r["path"] for r in summary["inputs"]["records"]} == {str(p) for p in world.values()}
    out = tmp_path / "figures"
    assert sorted(p.name for p in out.iterdir()) == ["harness_results.json", "harness_results.md", "harness_results.png"]
    assert (out / "harness_results.png").read_bytes()[:8] == PNG
    assert json.loads((out / "harness_results.json").read_text()) == json.loads(json.dumps(summary))


def test_a_scene_no_episode_was_flown_of_is_listed_with_its_reason_and_not_scored(tmp_path):
    world = _world(tmp_path)
    summary = _build(tmp_path, world.values())
    hover = summary["conditions"]["test_seen"]["policies"]["hover"]
    assert hover["scenes"]["s05r"]["timeout_rate"]["value"] == 1.0 and hover["scenes"]["s05r"]["episodes"] == 20
    s06r = hover["scenes"]["s06r"]
    assert s06r["status"] == "pending" and s06r["episodes"] == 0 and s06r["success_rate"] is None
    assert "m4_s06r_gate.json" in s06r["reason"]
    assert hover["all"]["scenes"] == ["s05r"], "a scene with no episode adds nothing to the condition's whole"
    md = (tmp_path / "figures" / "harness_results.md").read_text()
    assert "s06r" in md and "pending" in md and "m4_s06r_gate.json" in md


def test_a_running_record_s_scene_is_marked_incomplete(tmp_path):
    path = _record(tmp_path / "records", "seen_expert_s01", policy=EXPERT_S01, status="running",
                   scenes={"s01": _outcomes(success=4, collision=1)})
    summary = _build(tmp_path, [path])
    s01 = summary["conditions"]["seen"]["policies"]["expert"]["scenes"]["s01"]
    assert s01["status"] == "running" and s01["complete"] is False
    assert s01["episodes"] == 5 and s01["condition_episodes"] == 20
    assert summary["inputs"]["records"][0]["status"] == "running"
    md = (tmp_path / "figures" / "harness_results.md").read_text()
    assert "5 of 20" in md and "running" in md


# ----------------------------------------------------------------------------------------------------------------------
# Telling policies apart, and keeping each one's colour.
# ----------------------------------------------------------------------------------------------------------------------
def test_policies_are_told_apart_by_name_checkpoint_and_privileged_state(tmp_path):
    records = tmp_path / "records"
    vla_a = {"name": "openvla", "kind": "vla", "checkpoint": "ckpt/a", "sha256": "1" * 64}
    vla_b = {**vla_a, "checkpoint": "ckpt/b", "sha256": "2" * 64}
    paths = [_record(records, "vla_a", policy=vla_a, scenes={"s01": _outcomes(success=5)}),
             _record(records, "vla_a_priv", policy=vla_a, privileged=True, scenes={"s01": _outcomes(success=6)}),
             _record(records, "vla_b", policy=vla_b, scenes={"s01": _outcomes(success=7)}),
             _record(records, "expert_det", policy={**EXPERT_S01, "deterministic": True}, scenes={"s01": _outcomes(success=8)}),
             _record(records, "expert", policy=EXPERT_S01, scenes={"s01": _outcomes(success=9)})]
    summary = _build(tmp_path, paths)
    policies = summary["conditions"]["seen"]["policies"]
    assert len(policies) == 5, sorted(policies)
    assert list(policies)[:2] == ["expert", "expert (deterministic)"]
    successes = {label: p["scenes"]["s01"]["outcomes"]["success"] for label, p in policies.items()}
    assert sorted(successes.values()) == [5, 6, 7, 8, 9]
    vla_labels = [label for label in policies if label.startswith("openvla")]
    assert len(set(vla_labels)) == 3, "two checkpoints and the privileged run are three policies"
    assert any("privileged state" in label for label in vla_labels)
    colours = [summary["policies"][label]["colour"] for label in policies]
    assert len(set(colours)) == 5, "every policy drawn has a colour of its own"


def test_a_second_policy_of_a_reference_kind_takes_a_colour_of_its_own(tmp_path):
    from autofly_ue5.report.style import SLOTS

    records = tmp_path / "records"
    paths = [_record(records, "expert", policy=EXPERT_S01, scenes={"s01": _outcomes(success=9)}),
             _record(records, "expert_priv", policy=EXPERT_S01, privileged=True, scenes={"s01": _outcomes(success=9)})]
    policies = _build(tmp_path, paths)["policies"]
    assert policies["expert"]["colour"] == SLOTS[0]
    assert policies["expert (privileged state)"]["colour"] not in (SLOTS[0], SLOTS[1], SLOTS[2])


def test_a_policy_keeps_its_colour_whichever_others_are_drawn(tmp_path):
    from autofly_ue5.report.style import SLOTS

    world = _world(tmp_path)
    alone = _build(tmp_path, [world["hover_test_seen"]])
    assert alone["policies"]["hover"]["colour"] == SLOTS[2]
    summary = _build(tmp_path, world.values())
    assert summary["policies"]["expert"]["colour"] == SLOTS[0]
    assert summary["policies"]["scripted (privileged state)"]["colour"] == SLOTS[1]
    assert summary["policies"]["hover"]["colour"] == SLOTS[2]


# ----------------------------------------------------------------------------------------------------------------------
# What is refused.
# ----------------------------------------------------------------------------------------------------------------------
def test_one_policy_s_episodes_flown_twice_are_refused_until_one_copy_is_dropped(tmp_path):
    from scripts.harness_figures import HarnessFigureError

    records = tmp_path / "records"
    first = _record(records, "seen_expert_s01", policy=EXPERT_S01, scenes={"s01": _outcomes(success=19, collision=1)})
    again = _record(records, "seen_expert_s01_again", policy=EXPERT_S01, scenes={"s01": _outcomes(success=20)})
    with pytest.raises(HarnessFigureError, match="s01") as err:
        _build(tmp_path, [first, again])
    assert first.name in str(err.value) and again.name in str(err.value) and "--drop" in str(err.value)
    assert not (tmp_path / "figures").exists(), "a refused build writes nothing"
    summary = _build(tmp_path, [first, again], drops=[f"{again}:s01"])
    assert summary["conditions"]["seen"]["policies"]["expert"]["scenes"]["s01"]["outcomes"]["success"] == 19
    assert summary["inputs"]["dropped"] == [{"record": str(again), "scene": "s01", "episodes": 20}]
    with pytest.raises(HarnessFigureError, match="s07"):
        _build(tmp_path, [first], drops=[f"{first}:s07"])  # a drop that matches nothing is a typo, not a no-op


def test_two_checkpoints_flying_one_scene_as_one_policy_are_refused(tmp_path):
    from scripts.harness_figures import HarnessFigureError

    records = tmp_path / "records"
    one = _record(records, "a", policy=EXPERT_S01, scenes={"s07": _outcomes(success=10)})
    other = _record(records, "b", policy=EXPERT_S01D, scenes={"s07": _outcomes(success=10)}, seed_offset=10)
    with pytest.raises(HarnessFigureError, match="checkpoint"):
        _build(tmp_path, [one, other])


def test_disjoint_episodes_of_one_scene_and_checkpoint_are_merged(tmp_path):
    records = tmp_path / "records"
    one = _record(records, "a", policy=EXPERT_S01, scenes={"s01": _outcomes(success=10)})
    other = _record(records, "b", policy=EXPERT_S01, scenes={"s01": _outcomes(success=8, collision=2)}, seed_offset=10)
    s01 = _build(tmp_path, [one, other])["conditions"]["seen"]["policies"]["expert"]["scenes"]["s01"]
    assert s01["episodes"] == 20 and s01["outcomes"]["success"] == 18 and s01["records"] == [str(one), str(other)]


@pytest.mark.parametrize("change, says", [({"condition_sha": "d" * 64}, "condition file"),
                                          ({"max_steps": 200}, "max_steps")])
def test_records_of_one_condition_under_other_episodes_or_rules_are_refused(tmp_path, change, says):
    from scripts.harness_figures import HarnessFigureError

    records = tmp_path / "records"
    one = _record(records, "a", policy=EXPERT_S01, scenes={"s01": _outcomes(success=10)})
    other = _record(records, "b", policy=SCRIPTED, privileged=True, scenes={"s01": _outcomes(collision=10)}, **change)
    with pytest.raises(HarnessFigureError, match=says):
        _build(tmp_path, [one, other])


def test_a_file_that_is_not_a_harness_record_is_refused(tmp_path):
    from scripts.harness_figures import HarnessFigureError

    path = tmp_path / "other.json"
    path.write_text(json.dumps({"pass": True}))
    with pytest.raises(HarnessFigureError, match="harness record"):
        _build(tmp_path, [path])


def test_the_cli_writes_the_three_files_and_refuses_to_write_over_committed_evidence(tmp_path, capsys):
    from autofly_ue5 import evidence
    from scripts.harness_figures import main

    world = _world(tmp_path)
    out = tmp_path / "out"
    args = ["--records", *map(str, world.values()), "--pilots", str(_pilots(tmp_path))]
    assert main([*args, "--out-dir", str(out)]) == 0
    assert (out / "harness_results.png").is_file() and "harness_results.md" in capsys.readouterr().out
    gates_out = evidence.GATES_DIR / "figures"  # the suite's scratch stand-in for docs/gates/ (tests/conftest.py)
    gates_out.mkdir(parents=True)
    (gates_out / "harness_results.json").write_text("{}")
    assert main([*args, "--out-dir", str(gates_out)]) == 2
    assert (gates_out / "harness_results.json").read_text() == "{}" and not (gates_out / "harness_results.png").exists()
    assert main([*args, "--out-dir", str(tmp_path / "x"), "--drop", "no-colon"]) == 2


# ----------------------------------------------------------------------------------------------------------------------
# Plan 6's done check.
# ----------------------------------------------------------------------------------------------------------------------
def test_the_done_check_sets_the_expert_s_interval_against_its_pilot_s_kept_rate_and_the_scripted_baseline(tmp_path):
    summary = _build(tmp_path, _world(tmp_path).values())
    check = summary["done_check"]
    assert check["condition"] == "seen" and check["policy"] == "expert"
    s01, s01d, s07 = (check["scenes"][s] for s in ("s01", "s01d", "s07"))
    assert s01["intervals_overlap"] is True and s01["same_checkpoint"] is True and s01["expert_rate_in_pilot_interval"]
    assert s01["pilot"]["kept_rate"] == 0.9615 and s01["pilot"]["wilson95"] == [0.8702, 0.9894]
    assert s07["intervals_overlap"] is False, "10 of 20 [0.30, 0.70] against the pilot's [0.897, 0.997]"
    assert s07["same_checkpoint"] is True
    assert s01["scripted_below"] is True and s01["scripted_separated"] is True
    assert s07["scripted_below"] is False, "the scripted baseline's 12 of 20 is above the expert's 10 of 20"
    assert s01d["scripted_below"] is True
    assert check["overlap_all"] is False and check["scripted_below_all"] is False
    assert check["pilot_scenes_not_flown"] == ["s09"]
    comparisons = summary["conditions"]["seen"]["scripted_vs_expert"]
    assert comparisons["s07"]["below"] is False and comparisons["s01"]["separated"] is True
    md = (tmp_path / "figures" / "harness_results.md").read_text()
    assert "| s01 white pillars | 0.950 [0.764, 0.991] | 0.962 [0.870, 0.989] | yes | yes |" in md
    assert "| s07 stacked boxes | 0.500 [0.299, 0.701] | 0.980 [0.897, 0.997] | **no** |" in md


def test_a_pilot_whose_checkpoint_differs_from_the_expert_s_is_flagged(tmp_path):
    records = tmp_path / "records"
    path = _record(records, "seen_expert", policy={**EXPERT_S01, "sha256": "f" * 64}, scenes={"s01": _outcomes(success=19)})
    check = _build(tmp_path, [path])["done_check"]
    assert check["scenes"]["s01"]["same_checkpoint"] is False and check["scenes"]["s01"]["scripted_below"] is None
    assert check["scripted_below_all"] is None, "no scripted baseline was flown: the check cannot say"


def test_without_a_pilots_file_there_is_no_done_check_but_everything_else_is_drawn(tmp_path):
    summary = _build(tmp_path, _world(tmp_path).values(), pilots_path=None)
    assert summary["done_check"] is None and summary["conditions"]["seen"]["policies"]["expert"]["all"]["episodes"] == 60
    assert (tmp_path / "figures" / "harness_results.png").is_file()


# ----------------------------------------------------------------------------------------------------------------------
# Real records: the harness flown on the fake simulator.
# ----------------------------------------------------------------------------------------------------------------------
def test_records_the_harness_writes_are_read_as_written(tmp_path):
    from autofly_ue5.evaluation.policy import HoverPolicy, ScriptedPolicy
    from tests.test_eval_harness import _condition, _run

    condition = _condition(tmp_path, n=2)
    scripted = _run(tmp_path, ScriptedPolicy(), condition, out="scripted.json", privileged_state=True)
    hover = _run(tmp_path, HoverPolicy(), condition, out="hover.json", max_steps=300)
    summary = _build(tmp_path, [tmp_path / "out" / "scripted.json", tmp_path / "out" / "hover.json"])
    seen = summary["conditions"]["seen"]
    for label, record in (("scripted (privileged state)", scripted), ("hover", hover)):
        scene = seen["policies"][label]["scenes"]["s01"]
        metrics = record["scenes"]["s01"]["metrics"]
        assert scene["episodes"] == metrics["episodes"] == 2 and scene["status"] == "ok"
        assert scene["outcomes"] == metrics["outcomes"]
        assert scene["success_rate"]["value"] == round(metrics["success_rate"]["value"], 4)
    assert seen["scripted_vs_expert"] == {}, "no expert was flown"
