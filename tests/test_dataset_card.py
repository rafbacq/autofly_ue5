"""The merged dataset's card (plan 6, revision 1, items 1 and 6): counts by scene, target, category, seen and stand-in per
split; the instruction templates and every distinct instruction; the record definitions with state[0] and state[1]
marked privileged; the provenance; the known limits. Built from the export's merge.json and dataset_statistics.json."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from tests.test_merge import SYNTH_TARGETS, _export, write_pool, write_synthetic_store

STAND_INS = {"red sedan", "man in red shirt"}


@pytest.fixture
def export(tmp_path):
    a = write_synthetic_store(tmp_path / "data", "synth_a", per_target=4)
    b = write_synthetic_store(tmp_path / "data", "synth_b", scene_file="s07_stacked_boxes.json", per_target=2,
                              seed_base=497_000_000, pilot_sha256="9" * 64)
    legacy = write_synthetic_store(tmp_path / "data", "legacy", scene_file="s01_white_pillars.json", legacy_target=True,
                                   per_target=3, seed_base=496_000_000)
    _export(tmp_path, [a, b, legacy], name="autofly_ue5_test", pool=write_pool(tmp_path / "targets.json"))
    return tmp_path / "out" / "autofly_ue5_test"


def _episodes(export):
    return json.loads((export / "merge.json").read_text())["episodes"]


def test_the_card_counts_episodes_and_records_by_scene_target_category_seen_and_stand_in_per_split(export):
    from autofly_ue5.dataset.card import build_card

    card = build_card(export)
    episodes = _episodes(export)
    totals = card["totals"]
    assert totals["episodes"]["all"] == len(episodes) == 4 * 5 + 2 * 5 + 3
    assert totals["episodes"]["train"] + totals["episodes"]["val"] == totals["episodes"]["all"]
    assert totals["records"]["all"] == sum(e["steps"] for e in episodes)
    counts = card["counts"]
    assert {k: v["all"]["episodes"] for k, v in counts["scene"].items()} == {"s09": 20, "s07": 10, "s01": 3}
    assert {k: v["all"]["episodes"] for k, v in counts["target"].items()} == {
        **{t["name"]: 6 for t in SYNTH_TARGETS}, "orange cylinder": 3}
    assert {k: v["all"]["episodes"] for k, v in counts["category"].items()} == {
        "vehicle": 12, "geometry": 9, "human": 6, "furniture": 6}
    assert {k: v["all"]["episodes"] for k, v in counts["seen"].items()} == {"seen": 27, "unseen": 6}
    assert {k: v["all"]["episodes"] for k, v in counts["stand_in"].items()} == {"stand-in": 12, "not a stand-in": 21}
    for breakdown in counts.values():
        for value in breakdown.values():
            for measure in ("episodes", "records"):
                assert value["train"][measure] + value["val"][measure] == value["all"][measure]
    by_split = Counter(e["split"] for e in episodes if e["metadata"]["scene"] == "s09")
    assert counts["scene"]["s09"]["val"]["episodes"] == by_split["val"]
    assert sum(v["val"]["records"] for v in counts["scene"].values()) == totals["records"]["val"]


def test_the_card_holds_the_templates_verbatim_and_every_distinct_instruction_with_its_count(export):
    from autofly_ue5.dataset.card import build_card
    from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES

    card = build_card(export)
    instructions = card["instructions"]
    assert instructions["templates"] == list(INSTRUCTION_TEMPLATES)
    assert "avioding" in instructions["templates"][1], "the release's own spelling, kept"
    assert instructions["obstacle_phrases"] == {"s09": "coloured poles", "s07": "stacked boxes", "s01": "white pillars"}
    expected = Counter(e["instruction"] for e in _episodes(export))
    assert {d["instruction"]: d["episodes"] for d in instructions["distinct"]} == dict(expected)
    assert [d["episodes"] for d in instructions["distinct"]] == sorted(expected.values(), reverse=True)


def test_the_card_defines_the_records_and_marks_state_0_and_1_privileged(export):
    from autofly_ue5.dataset.card import build_card
    from autofly_ue5.dataset.state import STATE_FIELDS

    card = build_card(export)
    records = card["records"]
    assert [a["name"] for a in records["action"]] == ["forward_speed_m_s", "yaw_rate_rad_s", "vertical_speed_m_s"]
    assert [a["range"] for a in records["action"]] == [[0.0, 2.0], [-1.0, 1.0], [-1.0, 1.0]]
    state = records["state"]
    assert [s["name"] for s in state] == [f["name"] for f in STATE_FIELDS]
    assert [s["index"] for s in state] == list(range(9))
    assert [s["privileged"] for s in state] == [True, True] + [False] * 7
    for s in state[:2]:
        assert "true position" in s["privileged_reason"] and "without the image" in s["privileged_reason"]
    assert records["image_0"].startswith("RGB 256x256 PNG")
    assert {m["name"] for m in card["episode_metadata"]} == {
        "file_path", "episode_id", "store", "scene", "split_of_scene", "target_name", "target_category", "target_seen",
        "target_status", "target_stand_in", "seed", "pilot"}


def test_the_card_names_the_provenance_the_split_the_statistics_and_the_known_limits(export):
    from autofly_ue5.dataset.card import build_card

    card = build_card(export)
    provenance = card["provenance"]
    assert set(provenance["pilots"]) == {"1" * 64, "9" * 64}
    assert provenance["pilots"]["9" * 64]["stores"] == ["synth_b"]
    assert [s["name"] for s in provenance["stores"]] == ["synth_a", "synth_b", "legacy"]
    assert provenance["scene_configs"]["scene_autofly_s07_fast.jsonc"] == ["2" * 64]
    assert provenance["collector_git_heads"] == {"f" * 40: 33}
    (legacy,) = [s for s in provenance["stores"] if s["name"] == "legacy"]
    assert legacy["filled_from_pool"]["fields"] == ["target_category", "target_seen", "target_status"]
    assert card["split"]["val_fraction"] == 0.05 and card["split"]["stratify_by"] == ["scene", "target_name"]
    assert card["statistics"]["computed_on"] == "train" and set(card["statistics"]["action"]) >= {"q01", "q99"}
    limits = " ".join(card["known_limits"])
    for phrase in ("avioding", "Kenney", "0.95", "successful", "privileged", "not_autofly_verified", "legacy"):
        assert phrase in limits, phrase


def test_the_markdown_card_and_card_json_are_written_beside_the_export(export):
    from autofly_ue5.dataset.card import CARD_JSON, CARD_MD, write_card

    card = write_card(export)
    assert json.loads((export / CARD_JSON).read_text()) == card
    md = (export / CARD_MD).read_text()
    for text in ("# autofly_ue5_test", "PRIVILEGED", "avioding", "red sedan", "orange cylinder", "s07", "stand-in",
                 "q01", "9" * 64, "## Known limits"):
        assert text in md, text
    scene_rows = [line for line in md.splitlines() if line.startswith("| s09 |") and line.endswith(
        f"| 20 | {card['counts']['scene']['s09']['all']['records']} |")]
    assert len(scene_rows) == 1, "one row per scene in the scene counts"
    (bearing_row,) = [line for line in md.splitlines() if line.startswith("| 1 | bearing_to_target_rad")]
    assert bearing_row.replace("\\|", "").count("|") == 5, "a '|' inside a cell (the definitions' |r|) is escaped"


def test_the_card_script_writes_the_card_for_an_export(export, capsys):
    from autofly_ue5.dataset.card import CARD_JSON, CARD_MD
    from scripts.dataset_card import main

    assert main(["--export", str(export)]) == 0
    assert (export / CARD_MD).is_file() and (export / CARD_JSON).is_file()
    assert json.loads(capsys.readouterr().out)["episodes"] == 33
