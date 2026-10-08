"""Merge the runtime colours' registry entries once the editor has made them (plan 5 C6; spec §7.1).

    env -u PYTHONPATH .venv/bin/python scripts/ue_editor_run.py autofly_ue5/scenes/ue/create_runtime_materials.py \\
        --out runs/assets/runtime_materials.json --arg patch=$PWD/assets/pending/runtime_materials.registry.json
    env -u PYTHONPATH .venv/bin/python scripts/apply_runtime_materials.py --report runs/assets/runtime_materials.json

Project AirSim's setMaterial loads only a base UMaterial (WorldSimApi.cpp: StaticLoadObject(UMaterial::StaticClass())),
and the scene palette's colours are material instances (MI_Red ...), so a "red cube" target would spawn grey. The editor
script makes base UMaterials of the same colours under /Game/AutoFly/Materials/Runtime. Their registry entries (kind
"base_material", which expert/episode.py's RUNTIME_SPAWNABLE_MATERIAL_KINDS admits) wait in
assets/pending/runtime_materials.registry.json until the editor's report shows each one made: in the registry earlier, the
pool would draw "red cube" and the runtime would fail to load M_Red (or, worse, a stale package would paint something
else). Checked before anything is written: the report passed, every patch entry was made at its path as a base Material
(not an instance) with the patch's colour wired into BaseColor and an opaque blend, and the editor log holds no
"Failed to compile Material" for it (a material that fails to compile is only a warning to the editor, which renders the
default grid in its place: six grounds reached a packaged build that way on 2026-10-07). An entry that differs from one
already in the registry is refused unless --replace.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402

from scripts.import_assets import check_editor_log, update_registry  # noqa: E402

PATCH_FORMAT = "autofly_ue5_registry_patch/1"
EDITOR_SCRIPT = "autofly_ue5/scenes/ue/create_runtime_materials.py"
COLOUR_TOLERANCE = 1e-4  # a LinearColor read back from the saved asset: float32 of the patch's value


def load_patch(path: Path) -> dict:
    patch = json.loads(Path(path).read_text())
    if patch.get("format") != PATCH_FORMAT:
        raise ValueError(f"{path}: format {patch.get('format')!r}, expected {PATCH_FORMAT!r}")
    for key, entry in patch["materials"].items():
        if entry.get("kind") != "base_material" or not str(entry.get("ue_path", "")).startswith("/Game/"):
            raise ValueError(f"{path}: {key} must be a base_material with a /Game/ path, got {entry}")
    return patch


def check_material_report(patch: dict, report: dict) -> list[str]:
    """Everything the registry entries rely on, from the editor's report (the editor log is checked separately)."""
    problems: list[str] = []
    if not report.get("pass"):
        problems.append(f"the editor reported pass={report.get('pass')}: {report.get('error')}")
    roughness = ((report.get("shading") or {}).get("roughness") or {}).get("value")
    if not isinstance(roughness, (int, float)) or not 0.0 <= roughness <= 1.0:
        problems.append(f"the report gives no roughness in [0, 1]: {report.get('shading')}")
    reported = {m.get("key"): m for m in report.get("materials", [])}
    for key, entry in patch["materials"].items():
        got = reported.get(key)
        if got is None:
            problems.append(f"{key}: not in the report")
            continue
        if got.get("path") != entry["ue_path"]:
            problems.append(f"{key}: made at path {got.get('path')}, the patch says {entry['ue_path']}")
        if got.get("class") != "Material" or not got.get("is_base_material"):
            problems.append(f"{key}: a {got.get('class')}, not a base UMaterial: the runtime's setMaterial cannot load it")
        colour = got.get("base_color_readback")
        if not colour or any(abs(a - b) > COLOUR_TOLERANCE for a, b in zip(colour[:3], entry["rgba"][:3])):
            problems.append(f"{key}: base colour reads back {colour}, the palette's colour is {entry['rgba']}")
        if "MP_BASE_COLOR" not in got.get("connected", []):
            problems.append(f"{key}: nothing is wired into MP_BASE_COLOR ({got.get('connected')})")
        if got.get("blend_mode") != "BLEND_OPAQUE":
            problems.append(f"{key}: blend mode {got.get('blend_mode')}, expected BLEND_OPAQUE")
    return problems


def material_entries(patch: dict, report: dict) -> dict[str, dict]:
    """The registry entries: the patch's, with the roughness the editor gave them and the script that made them."""
    roughness = report["shading"]["roughness"]["value"]
    return {key: {**entry, "roughness": round(float(roughness), 6), "made_by": EDITOR_SCRIPT}
            for key, entry in patch["materials"].items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", type=Path, required=True, help="the editor script's report (scripts/ue_editor_run.py --out)")
    p.add_argument("--patch", type=Path, default=_ROOT / "assets" / "pending" / "runtime_materials.registry.json")
    p.add_argument("--registry", type=Path, default=_ROOT / "assets" / "registry.json")
    p.add_argument("--log", type=Path, default=None, help="the editor log; default: the report's path with .log "
                                                           "(where scripts/ue_editor_run.py writes it)")
    p.add_argument("--replace", action="store_true", help="overwrite registry entries that differ (a deliberate re-make)")
    args = p.parse_args(argv)
    patch = load_patch(args.patch)
    report = json.loads(args.report.read_text())
    log = args.log if args.log is not None else args.report.with_suffix(".log")
    problems = check_material_report(patch, report) + check_editor_log(log)
    if problems:
        print("the runtime materials do not pass; the registry is untouched:", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 2
    entries = material_entries(patch, report)
    try:
        update_registry(args.registry, {}, entries, replace=args.replace)
    except ValueError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    print(f"registry updated: {len(entries)} base materials ({', '.join(sorted(entries))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
