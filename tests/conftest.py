"""Suite-wide setup: the s01 layout several tests read from runs/levels/ is git-ignored build output, so a fresh
clone builds it once here (the same generator run as `scripts/build_scenes.py`; deterministic for the scene seed)."""

from autofly_ue5.paths import RUNS_DIR, SCENES_DIR


def pytest_sessionstart(session):
    levels = RUNS_DIR / "levels"
    if not (levels / "s01.layout.json").is_file() or not (levels / "s01.level.json").is_file():
        from autofly_ue5.scenes.build import build_scene

        build_scene(SCENES_DIR / "s01_white_pillars.json", levels)
