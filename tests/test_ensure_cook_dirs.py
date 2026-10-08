"""scripts/ensure_cook_dirs.py: nothing references a target or a runtime colour from a level, so the cook takes them only
from ue_project/Config/DefaultGame.ini's +DirectoriesToAlwaysCook (a packaged build without them spawns nothing: the
spawn table holds only cooked assets). Nothing in the repo owns that file (ue_project/ is git-ignored; init copies Blocks'
Config once), so the script adds the entries idempotently, and scripts/package_sim.sh refuses to package while an
imported target's or a runtime material's directory is not cooked."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autofly_ue5.paths import ROOT

INI = """[/Script/EngineSettings.GeneralProjectSettings]
ProjectName="Blocks"

[/Script/UnrealEd.ProjectPackagingSettings]
Build=IfProjectHasCode
bCookAll=False
+DirectoriesToAlwaysCook=(Path="/Game/Geometry")
+DirectoriesToAlwaysCook=(Path="/Game/AutoFly/Materials")
+DirectoriesToAlwaysCook=(Path="/ProjectAirSim")
bCompressed=True
+MapsToCook=(FilePath="/Game/AutoFly/Maps/S01")
"""


def _registry(tmp_path: Path, *, targets: bool, material_path: str = "/Game/AutoFly/Materials/Runtime/M_Red") -> Path:
    registry = {"version": 1, "assets": {"cylinder": {"ue_path": "/Engine/BasicShapes/Cylinder", "role": "obstacle"}},
                "materials": {"runtime_red": {"kind": "base_material", "ue_path": material_path},
                              "white": {"kind": "color_instance", "ue_path": "/Game/AutoFly/Materials/MI_White"}}}
    if targets:
        registry["assets"]["tgt_sedan"] = {"ue_path": "/Game/AutoFly/Targets/tgt_sedan/tgt_sedan", "role": "target"}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    return path


def _ini(tmp_path: Path, text: str = INI) -> Path:
    path = tmp_path / "DefaultGame.ini"
    path.write_text(text)
    return path


def test_ensure_adds_the_target_directory_once_beside_the_other_cook_entries(tmp_path):
    from scripts.ensure_cook_dirs import cooked_directories, main

    ini = _ini(tmp_path)
    registry = _registry(tmp_path, targets=True)
    assert main(["--ini", str(ini), "--registry", str(registry)]) == 0
    text = ini.read_text()
    lines = text.splitlines()
    added = lines.index('+DirectoriesToAlwaysCook=(Path="/Game/AutoFly/Targets")')
    assert lines[added - 1] == '+DirectoriesToAlwaysCook=(Path="/ProjectAirSim")', "after the last existing entry"
    assert [ln for ln in lines if ln != lines[added]] == INI.splitlines(), "nothing else changes"
    assert "/Game/AutoFly/Targets" in cooked_directories(text)
    assert main(["--ini", str(ini), "--registry", str(registry)]) == 0
    assert ini.read_text() == text, "a second run changes nothing"


def test_ensure_adds_the_target_root_before_any_target_is_imported(tmp_path):
    from scripts.ensure_cook_dirs import main

    ini = _ini(tmp_path)
    assert main(["--ini", str(ini), "--registry", str(_registry(tmp_path, targets=False))]) == 0
    assert '+DirectoriesToAlwaysCook=(Path="/Game/AutoFly/Targets")' in ini.read_text()


def test_check_refuses_an_imported_target_whose_directory_is_not_cooked(tmp_path, capsys):
    from scripts.ensure_cook_dirs import main

    ini = _ini(tmp_path)
    before = ini.read_text()
    assert main(["--ini", str(ini), "--registry", str(_registry(tmp_path, targets=True)), "--check"]) == 1
    assert "/Game/AutoFly/Targets/tgt_sedan" in capsys.readouterr().err
    assert ini.read_text() == before, "--check never writes"
    assert main(["--ini", str(ini), "--registry", str(_registry(tmp_path, targets=True))]) == 0
    assert main(["--ini", str(ini), "--registry", str(_registry(tmp_path, targets=True)), "--check"]) == 0


def test_check_passes_before_targets_exist_and_covers_runtime_materials_by_their_parent(tmp_path):
    """Packaging for a scene before the targets' window must not be blocked: only what the registry spawns at runtime is
    required, and /Game/AutoFly/Materials already covers /Game/AutoFly/Materials/Runtime."""
    from scripts.ensure_cook_dirs import main

    ini = _ini(tmp_path)
    assert main(["--ini", str(ini), "--registry", str(_registry(tmp_path, targets=False)), "--check"]) == 0
    elsewhere = _registry(tmp_path, targets=False, material_path="/Game/Elsewhere/M_Red")
    assert main(["--ini", str(ini), "--registry", str(elsewhere), "--check"]) == 1


def test_a_removed_entry_is_not_cooked_and_a_missing_section_is_added(tmp_path):
    from scripts.ensure_cook_dirs import cooked_directories, main

    text = INI + '[/Script/UnrealEd.ProjectPackagingSettings]\n-DirectoriesToAlwaysCook=(Path="/Game/AutoFly/Materials")\n'
    assert "/Game/AutoFly/Materials" not in cooked_directories(text)
    bare = _ini(tmp_path, "[/Script/EngineSettings.GeneralProjectSettings]\nProjectName=\"Blocks\"\n")
    assert main(["--ini", str(bare), "--registry", str(_registry(tmp_path, targets=True))]) == 0
    got = bare.read_text()
    assert "[/Script/UnrealEd.ProjectPackagingSettings]" in got and "/Game/AutoFly/Targets" in cooked_directories(got)
    assert "/Game/AutoFly/Materials" in cooked_directories(got)


def test_package_sim_checks_the_cook_list_before_it_cooks():
    text = (ROOT / "scripts" / "package_sim.sh").read_text()
    assert "scripts/ensure_cook_dirs.py\" --check" in text
    assert text.index("ensure_cook_dirs.py") < text.index("RunUAT.sh")


def test_the_script_runs_without_the_project_venv(tmp_path):
    """package_sim.sh calls it with the system python3, like its own pid checks: stdlib only."""
    ini = _ini(tmp_path)
    result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "ensure_cook_dirs.py"), "--ini", str(ini),
                             "--registry", str(_registry(tmp_path, targets=True)), "--check"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "ensure_cook_dirs.py" in result.stderr
