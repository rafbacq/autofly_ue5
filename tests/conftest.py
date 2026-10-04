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
# over docs/gates/m2_gate.json (restored from git). Two defences:
#   1. every test's evidence directory (autofly_ue5.evidence.GATES_DIR) is a scratch directory, so a default evidence
#      path can never reach the real one;
#   2. a test during which a file under the real docs/gates/ changes fails. Nothing is deleted or rewritten: a live run
#      may legitimately write its record while the suite runs (a guard that restored a snapshot would destroy it), so
#      the failure says what changed and leaves the decision to whoever reads it.
# --------------------------------------------------------------------------------------------------------
import pytest  # noqa: E402

from autofly_ue5 import evidence  # noqa: E402

COMMITTED = evidence.COMMITTED_GATES_DIR


def _gate_files() -> dict:
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in COMMITTED.rglob("*") if p.is_file()}


@pytest.fixture(autouse=True)
def _evidence_is_never_touched(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(evidence, "GATES_DIR", tmp_path_factory.mktemp("gates"))
    monkeypatch.setattr(evidence, "NOT_STARTED_DIR", tmp_path_factory.mktemp("not_started"))
    before = _gate_files()
    yield
    after = _gate_files()
    if after != before:
        changed = sorted(str(p.relative_to(COMMITTED.parent.parent)) for p in set(before) | set(after)
                         if before.get(p) != after.get(p))
        pytest.fail(f"{changed} changed under docs/gates/ while this test ran. If no live run wrote them, this test "
                    f"touched committed evidence: restore it with `git checkout -- docs/gates` and fix the test")


# --------------------------------------------------------------------------------------------------------
# Durable records (autofly_ue5/durable.py): the order in which a write reached the disk. ("fsync", path) names the file
# or directory a synced descriptor points at; ("replace", src, dst) is a rename. Real paths, so tmp_path symlinks match.
# --------------------------------------------------------------------------------------------------------
import os  # noqa: E402


class SyncEvents(list):
    def wrote_durably(self, path) -> bool:
        """`path` was renamed into place after its data was synced, and its directory was synced after the rename."""
        final = os.path.realpath(path)
        renames = [i for i, e in enumerate(self) if e[0] == "replace" and e[2] == final]
        if not renames:
            return False
        at = renames[-1]
        return ("fsync", self[at][1]) in self[:at] and ("fsync", os.path.dirname(final)) in self[at + 1:]


@pytest.fixture
def sync_events(monkeypatch) -> SyncEvents:
    events = SyncEvents()
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", os.path.realpath(os.readlink(f"/proc/self/fd/{fd}"))))
        real_fsync(fd)

    def replace(src, dst):
        events.append(("replace", os.path.realpath(src), os.path.realpath(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    return events
