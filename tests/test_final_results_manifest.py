"""docs/final_results_manifest.json: what final_results/ holds (plan 6 step 7), in scripts/build_final_results.py's format.
It must build (no destination written twice, none outside the five groups), name only committed sources that this
checkout has (the gate records, the status report's figures, the scene and pool files), and look for the plan 6 figure
tools' outputs under the names those tools write. Outputs of runs and host-only renders (runs/, data/, the status
report's git-ignored videos) are checked on the host when the folder is built, not here."""

from __future__ import annotations

import glob
import json
from pathlib import Path, PurePosixPath

import pytest

from autofly_ue5.paths import ROOT

MANIFEST = ROOT / "docs" / "final_results_manifest.json"
GROUPS = {"figures", "videos", "records", "dataset", "docs"}
# Committed in git, so present in every checkout (results/**/videos/ is git-ignored: the renders live on the host).
COMMITTED = ("docs/", "assets/", "scenes/", "results/2026-10-07-project-status/figures/",
             "results/2026-10-07-project-status/scenes/", "results/2026-10-07-project-status/maps/",
             "results/2026-10-07-project-status/records/")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _placeholders(root: Path, manifest: dict) -> None:
    """One file per non-optional source under `root`: the path itself, or for a glob one name it matches."""
    for entry in manifest["files"]:
        if entry.get("optional"):
            continue
        path = root / entry["source"].replace("*", "x")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry["source"].encode())


def test_the_manifest_builds_into_its_five_groups_and_the_readme(tmp_path):
    from scripts.build_final_results import build, plan

    manifest = _manifest()
    assert manifest["version"] == 1 and len(manifest["files"]) >= 100
    for entry in manifest["files"]:
        dest = PurePosixPath(entry["dest"])
        assert entry["dest"] == "README.md" or dest.parts[0] in GROUPS, entry
        assert entry["caption"].strip() and not entry["caption"].endswith("."), entry
    root = tmp_path / "repo"
    _placeholders(root, manifest)
    copies, skipped = plan(manifest, root)
    assert len(skipped) == sum(1 for e in manifest["files"] if e.get("optional")), "every optional source is absent here"
    assert len({c.dest for c in copies}) == len(copies)
    index = build(MANIFEST, tmp_path / "final_results", source_root=root)
    assert {f["status"] for f in index["files"]} == {"copied"} and not index["unlisted"]


def test_its_committed_sources_are_in_this_checkout():
    for entry in _manifest()["files"]:
        if entry.get("optional") or not entry["source"].startswith(COMMITTED):
            continue
        assert glob.glob(str(ROOT / entry["source"])), f"{entry['source']} matches nothing in this checkout"


@pytest.mark.parametrize("module, folder", [("scripts.harness_figures", "runs/eval/final/figures/"),
                                            ("scripts.rebalance_figures", "runs/rebalance/final/figures/")])
def test_it_looks_for_the_figure_tools_outputs_under_the_names_they_write(module, folder):
    import importlib

    sources = {e["source"] for e in _manifest()["files"]}
    for name in importlib.import_module(module).OUTPUTS:
        assert folder + name in sources, f"{module} writes {name}; the manifest does not take it from {folder}"
    assert "runs/report/pipeline/pipeline.png" in sources


def test_every_milestone_s_gate_record_is_taken_with_its_failures():
    sources = {e["source"] for e in _manifest()["files"]}
    for name in ("m0_gate", "m1_gate", "m2_gate", "m2d_gate", "m2d_r6_swa_gate", "m3_gate", "m4_s02_gate",
                 "m4_s02_gate_run3", "m4_s05r_gate", "m4_s06r_gate", "m5_s02_gate", "m5_s02_gate_outline",
                 "m5_pilot_screening", "plan6_pilots", "m4_targets_probe", "m4_targets_probe_run2"):
        assert f"docs/gates/{name}.json" in sources, name
