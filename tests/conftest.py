"""Suite-wide setup: the s01 layout several tests read from runs/levels/ is git-ignored build output, so a fresh
clone builds it once here (the same generator run as `scripts/build_scenes.py`; deterministic for the scene seed)."""

from autofly_ue5.paths import RUNS_DIR, SCENES_DIR


def pytest_sessionstart(session):
    levels = RUNS_DIR / "levels"
    if not (levels / "s01.layout.json").is_file() or not (levels / "s01.level.json").is_file():
        from autofly_ue5.scenes.build import build_scene

        build_scene(SCENES_DIR / "s01_white_pillars.json", levels)


# --------------------------------------------------------------------------------------------------------
# docs/gates/ holds committed evidence, and no test may change it (CLAUDE.md). On 2026-10-02 the RED run of a test of
# m2_gate.main()'s refusal -- written before the refusal existed -- ran the gate with its old default --out and wrote
# over docs/gates/m2_gate.json (restored from git). Every test is now checked: any file under docs/gates/ that a test
# changes, adds or deletes is put back from the session's snapshot, and the test fails.
# --------------------------------------------------------------------------------------------------------
import pytest  # noqa: E402

from autofly_ue5.paths import ROOT  # noqa: E402

GATES = ROOT / "docs" / "gates"


def _gate_files() -> dict:
    return {p: p.stat() for p in GATES.rglob("*") if p.is_file()}


@pytest.fixture(scope="session")
def _gates_snapshot():
    return {p: p.read_bytes() for p in GATES.rglob("*") if p.is_file()}


@pytest.fixture(autouse=True)
def _evidence_is_never_touched(_gates_snapshot):
    before = {p: (s.st_size, s.st_mtime_ns) for p, s in _gate_files().items()}
    yield
    after = {p: (s.st_size, s.st_mtime_ns) for p, s in _gate_files().items()}
    if after == before:
        return
    damaged = sorted(str(p.relative_to(ROOT)) for p in set(before) | set(after) if before.get(p) != after.get(p))
    for p in set(after) - set(_gates_snapshot):
        p.unlink()
    for p, blob in _gates_snapshot.items():
        if not p.is_file() or p.read_bytes() != blob:
            p.write_bytes(blob)
    pytest.fail(f"this test changed committed evidence {damaged} (restored from the session snapshot)")
