import json
from pathlib import Path

from autofly_ue5.paths import ROOT


def test_manifest_records_versions_cuda_and_the_numpy_pin(tmp_path):
    from scripts.make_env_manifest import build_manifest

    m = build_manifest()
    assert m["packages"]["numpy"] == "1.26.4", "projectairsim 1.0.2 is built against numpy 1.26.4"
    for pkg in ("torch", "stable-baselines3", "gymnasium"):
        assert m["packages"][pkg], f"{pkg} missing from the manifest"
    assert m["cuda"]["available"] is True
    assert "4090" in m["cuda"]["device_name"]
    assert m["python"].startswith("3.12")


def test_manifest_is_json_serialisable_and_committed():
    from scripts.make_env_manifest import build_manifest

    json.dumps(build_manifest())  # must not raise
    committed = ROOT / "docs" / "gates" / "m2_env_manifest.json"
    assert committed.exists(), "run scripts/make_env_manifest.py and commit its output"
