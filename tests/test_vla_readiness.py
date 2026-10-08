"""Plan 6 (revision 1, item 1): the merged export is VLA-ready when TFDS reads every episode back exactly and an
OpenVLA-style RLDS pipeline (dlimp, CPU) iterates both splits. Those run in their own venvs (TensorFlow cannot go into
the project venv), as subprocesses, and skip cleanly where the venv is absent; scripts/export_merged.py runs them."""

from __future__ import annotations

import json

import pytest

from tests.test_merge import HOST_ROOT, S01_PILOT, store_subset, write_pool, write_synthetic_store

TFDS_PYTHON = HOST_ROOT / "runs" / "tools" / "tfds_venv" / "bin" / "python"
VLA_PYTHON = HOST_ROOT / "runs" / "tools" / "vla_loader_venv" / "bin" / "python"
NAME = "autofly_ue5_smoke"


@pytest.fixture(scope="module")
def small_export(tmp_path_factory):
    from autofly_ue5.dataset.merge import export_merged

    root = tmp_path_factory.mktemp("vla_ready")
    stores = [write_synthetic_store(root / "data", "synth_a", per_target=3),
              write_synthetic_store(root / "data", "synth_b", scene_file="s05r_standing_stones.json", per_target=2,
                                    seed_base=497_000_000, pilot_sha256="9" * 64)]
    if (S01_PILOT / "manifest.json").is_file():  # one real M3 episode too, where this host has the pilot
        stores.append(store_subset(S01_PILOT, root / "view", 1))
    export_merged(stores, root / "out", name=NAME, split_file=root / "split.json",
                  pool=write_pool(root / "targets.json"), episodes_per_shard=4)
    return root / "out"


def test_the_export_script_writes_the_dataset_statistics_and_card_and_records_each_check(tmp_path, capsys):
    from scripts.export_merged import main

    store = write_synthetic_store(tmp_path / "data", "synth_a", per_target=2)
    fake = tmp_path / "fake_python"  # stands in for a venv's python: reports which check it was asked to run
    fake.write_text('#!/bin/bash\necho "{\\"pass\\": true, \\"script\\": \\"$(basename "$1")\\"}"\n')
    fake.chmod(0o755)
    args = ["--store", str(store), "--out", str(tmp_path / "out"), "--name", "autofly_ue5_test", "--split-file",
            str(tmp_path / "split.json"), "--targets", str(write_pool(tmp_path / "targets.json")),
            "--check-python", str(fake), "--vla-python", str(fake)]
    assert main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["splits"] == {"train": 9, "val": 1} and out["tfds_check"]["pass"] is True
    export = tmp_path / "out" / "autofly_ue5_test"
    for name in ("1.0.0/dataset_info.json", "1.0.0/features.json", "merge.json", "dataset_statistics.json",
                 "dataset_card.md", "card.json"):
        assert (export / name).is_file(), name
    assert json.loads((export / "tfds_check.json").read_text())["script"] == "check_rlds_with_tfds.py"
    assert json.loads((export / "vla_loader_check.json").read_text())["script"] == "check_vla_loader.py"
    assert main(args) == 2, "an existing export is refused"
    assert "refusing to export" in capsys.readouterr().err


def test_a_failed_check_fails_the_export_script(tmp_path, capsys):
    from scripts.export_merged import main

    store = write_synthetic_store(tmp_path / "data", "synth_a", per_target=2)
    fake = tmp_path / "fake_python"
    fake.write_text("#!/bin/bash\necho 'Traceback: no module named tensorflow' >&2\nexit 1\n")
    fake.chmod(0o755)
    assert main(["--store", str(store), "--out", str(tmp_path / "out"), "--split-file", str(tmp_path / "split.json"),
                 "--targets", str(write_pool(tmp_path / "targets.json")), "--check-python", str(fake)]) == 1
    report = json.loads((tmp_path / "out" / "autofly_ue5_proto" / "tfds_check.json").read_text())
    assert report["pass"] is False and "no module named tensorflow" in report["error"]


@pytest.mark.skipif(not TFDS_PYTHON.is_file(), reason="runs/tools/tfds_venv is git-ignored (docs/runbook-m3.md)")
def test_tfds_reads_every_episode_of_a_merged_export_back_exactly(small_export):
    from scripts.export_merged import run_check

    report = run_check(TFDS_PYTHON, "check_rlds_with_tfds.py",
                       ["--rlds", str(small_export / NAME / "1.0.0"), "--merged", str(small_export / NAME / "merge.json")],
                       small_export / NAME / "tfds_check.json")
    assert report["pass"] is True, report.get("mismatches") or report.get("error")
    assert set(report["splits"]) == {"train", "val"} and report["splits"] == report["expected_splits"]
    assert report["matched"] == sum(report["splits"].values())


@pytest.mark.skipif(not VLA_PYTHON.is_file(),
                    reason="runs/tools/vla_loader_venv is git-ignored (its recipe: scripts/check_vla_loader.py)")
def test_an_openvla_style_rlds_pipeline_reads_both_splits_of_a_merged_export(small_export):
    from scripts.export_merged import run_check

    report = run_check(VLA_PYTHON, "check_vla_loader.py",
                       ["--data-dir", str(small_export), "--name", NAME, "--frames", "200",
                        "--out", str(small_export / NAME / "vla_loader_check.json")],
                       small_export / NAME / "vla_loader_check.json")
    assert report["pass"] is True, report.get("failures") or report.get("error")
    versions = report["versions"]
    assert (versions["tensorflow"], versions["tensorflow_datasets"]) == ("2.15.0", "4.9.3")
    assert versions["dlimp"]["commit"] == "040105d256bd28866cc6620621a3d5f7b6b91b46"
    for split in ("train", "val"):
        assert report["splits"][split]["frames"] > 0 and report["splits"][split]["failure_count"] == 0
    assert report["builder"]["val_is_a_split"] is True and report["statistics_recomputed"]["pass"] is True
