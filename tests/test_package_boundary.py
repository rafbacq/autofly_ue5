"""The installable package must not depend on the repo-only `scripts/` directory (it is not packaged, and only
resolves when a process happens to start from the repo root)."""

import ast
import json

import pytest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "autofly_ue5"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_no_package_module_imports_the_scripts_directory():
    offenders = {
        str(p.relative_to(PACKAGE.parent)): sorted(m for m in _imported_modules(p) if m == "scripts" or m.startswith("scripts."))
        for p in PACKAGE.rglob("*.py")
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_build_scene_writes_the_layout_and_level_spec(tmp_path):
    from autofly_ue5.paths import SCENES_DIR
    from autofly_ue5.scenes.build import build_scene
    from autofly_ue5.scenes.generate import generate_layout
    from autofly_ue5.scenes.model import load_registry, load_scene_file

    summary = build_scene(SCENES_DIR / "s01_white_pillars.json", tmp_path)
    assert summary["reachability"]["ok"] is True
    layout = json.loads((tmp_path / "s01.layout.json").read_text())["layout"]
    expected = generate_layout(load_scene_file(SCENES_DIR / "s01_white_pillars.json"), load_registry())
    assert layout == json.loads(json.dumps(expected.to_json()))
    assert (tmp_path / "s01.level.json").is_file()


ENTRY_POINTS_HOLDING_A_SIMULATOR = ["autofly_ue5/expert/train.py", "autofly_ue5/validate/live_m1.py",
                                    "scripts/m2_gate.py", "scripts/measure_instances.py", "scripts/probe_crash_reset.py"]


@pytest.mark.parametrize("path", ENTRY_POINTS_HOLDING_A_SIMULATOR)
def test_entry_points_that_hold_a_simulator_force_their_exit(path):
    # projectairsim leaves a non-daemon thread alive; after an abandoned close() it blocks interpreter shutdown and
    # run_job.sh sees the job "still running" forever. sys.exit() is not enough -- os._exit() is.
    tree = ast.parse((PACKAGE.parent / path).read_text())
    main_blocks = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
    assert main_blocks, f"{path} has no __main__ block"
    calls = {ast.unparse(c.func) for c in ast.walk(main_blocks[0]) if isinstance(c, ast.Call)}
    assert "os._exit" in calls, f"{path}'s __main__ block must end with os._exit(...), found {sorted(calls)}"
