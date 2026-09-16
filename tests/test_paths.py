from pathlib import Path

from autofly_ue5 import paths


def test_root_is_project_root():
    assert paths.ROOT == Path("/home/jk_edge/research_uav/autofly_ue5")


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
        "UE-ZenDataPath": "/home/jk_edge/research_uav/autofly_ue5/ue_project/DerivedDataCache/Zen",
        "UE-LocalDataCachePath": "/home/jk_edge/research_uav/autofly_ue5/ue_project/DerivedDataCache",
    }
