"""Absolute project paths; every other module takes its paths from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engine"
PLATFORM_DIR = ROOT / "platform"
UE_PROJECT_DIR = ROOT / "ue_project"
UPROJECT = UE_PROJECT_DIR / "Blocks.uproject"
CONFIGS_DIR = ROOT / "configs"
SCENES_DIR = ROOT / "scenes"
ASSET_REGISTRY = ROOT / "assets" / "registry.json"
RUNS_DIR = ROOT / "runs"
JOBS_DIR = RUNS_DIR / "jobs"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
UNREAL_EDITOR = ENGINE_DIR / "Engine" / "Binaries" / "Linux" / "UnrealEditor"
UNREAL_EDITOR_CMD = ENGINE_DIR / "Engine" / "Binaries" / "Linux" / "UnrealEditor-Cmd"
PACKAGED_BINARY = (
    UE_PROJECT_DIR / "Packaged" / "Development" / "Linux" / "Blocks" / "Binaries" / "Linux" / "Blocks"
)
# Derived-data cache inside ROOT (git-ignored) instead of ~/.config/Epic/UnrealEngine/Common/Zen/Data.
DDC_DIR = UE_PROJECT_DIR / "DerivedDataCache"
ZEN_DATA_DIR = DDC_DIR / "Zen"
UE_CACHE_ENV = {"UE-ZenDataPath": str(ZEN_DATA_DIR), "UE-LocalDataCachePath": str(DDC_DIR)}
