import json
from pathlib import Path

from autofly_ue5.paths import ROOT


def test_manifest_records_versions_cuda_and_the_numpy_pin(tmp_path):
    from scripts.make_env_manifest import build_manifest

    # The live environment: pins and presence only, so this passes on any correctly set-up 3.12 venv (CPU-only
    # included). What the M2 numbers were measured on -- CUDA, the 4090 -- is asserted on the committed record below.
    m = build_manifest()
    assert m["packages"]["numpy"] == "1.26.4", "projectairsim 1.0.2 is built against numpy 1.26.4"
    for pkg in ("torch", "stable-baselines3", "gymnasium"):
        assert m["packages"][pkg], f"{pkg} missing from the manifest"
    assert m["python"].startswith("3.12")


def test_committed_manifest_records_the_gpu_m2_was_measured_on():
    committed = json.loads((ROOT / "docs" / "gates" / "m2_env_manifest.json").read_text())
    assert committed["cuda"]["available"] is True
    assert "4090" in committed["cuda"]["device_name"]
    assert committed["packages"]["numpy"] == "1.26.4"


def test_manifest_is_json_serialisable_and_committed():
    from scripts.make_env_manifest import build_manifest

    json.dumps(build_manifest())  # must not raise
    committed = ROOT / "docs" / "gates" / "m2_env_manifest.json"
    assert committed.exists(), "run scripts/make_env_manifest.py and commit its output"
