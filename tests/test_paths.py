from pathlib import Path

import pytest

from autofly_ue5 import paths


def test_root_is_project_root():
    # The checkout this test file lives in, wherever it was cloned -- not one machine's absolute path.
    assert paths.ROOT == Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not paths.ENGINE_DIR.is_dir(), reason="engine/ is git-ignored and only present on the GPU host")
def test_engine_binaries_exist():
    assert paths.UNREAL_EDITOR.is_file()
    assert paths.UNREAL_EDITOR_CMD.is_file()


def test_derived_paths():
    assert paths.UPROJECT == paths.ROOT / "ue_project" / "Blocks.uproject"
    assert paths.CONFIGS_DIR == paths.ROOT / "configs"
    assert paths.ASSET_REGISTRY == paths.ROOT / "assets" / "registry.json"
    assert paths.JOBS_DIR == paths.ROOT / "runs" / "jobs"
    assert paths.PACKAGED_BINARY.parts[-6:] == ("Development", "Linux", "Blocks", "Binaries", "Linux", "Blocks")


def test_ue_cache_env_stays_inside_root():
    assert paths.UE_CACHE_ENV == {
        "UE_ZenDataPath": str(paths.ROOT / "ue_project" / "DerivedDataCache" / "Zen"),
        "UE_LocalDataCachePath": str(paths.ROOT / "ue_project" / "DerivedDataCache"),
    }
    assert all(v.startswith(str(paths.ROOT)) for v in paths.UE_CACHE_ENV.values())
