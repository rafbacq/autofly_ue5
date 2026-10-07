"""Import the fetched CC0 assets into the UE project and write their registry entries (plan 5, task C3b).

    env -u PYTHONPATH .venv/bin/python scripts/import_assets.py plan      # writes runs/assets/import_spec.json
    bash scripts/run_job.sh start import_assets -- env -u PYTHONPATH .venv/bin/python scripts/import_assets.py run
    env -u PYTHONPATH .venv/bin/python scripts/import_assets.py apply     # checks the report, updates assets/registry.json

Three stages, because the editor is the slow and opaque one:

1. `plan`: from `assets/sources.json` and `assets/measurements.json`, the job the editor runs: every glTF model with the
   nodes it must yield as static meshes, every ambientCG set with its four texture maps (Color, NormalDX, Roughness,
   AmbientOcclusion) and the tiling that gives the 200 m ground slab tiles of the material's real size.
2. `run`: the full editor with `-ExecCmds="py autofly_ue5/scenes/ue/import_assets.py"` (the commandlet crashes on editor
   subsystems, scripts/build_level.sh), which imports, enables Nanite, sets complex-as-simple collision, builds one
   material per ground and writes a report.
3. `apply`: the report is checked against the measurements (every node present, UE bounds against glTF extents, Nanite
   on, collision set, every material input connected) before `assets/registry.json` gets one entry per static mesh
   (base pivot, circle footprint, the radius-by-height profile) and one per ground material. An entry that would
   change an existing one under the same name is refused.

Collision is complex-as-simple: the drone collides with the same triangles the depth camera sees, which is the agreement
the live checks measure; with Nanite on, UE uses the fallback mesh for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

from autofly_ue5.paths import ENGINE_DIR, RUNS_DIR, UE_CACHE_ENV, UE_PROJECT_DIR, UPROJECT, ZEN_DATA_DIR  # noqa: E402

ASSETS_ROOT = "/Game/AutoFly/Assets"
MATERIALS_ROOT = "/Game/AutoFly/Materials/Ground"
GROUND_SIZE_M = 200.0  # scenes/level_spec.py's slab; one UV tile spans it, so tiling = slab / material size
EXTENT_TOLERANCE = 0.02  # 2 % of the extent, or 2 cm, whichever is larger: float32 geometry through two converters
EDITOR_SCRIPT = _ROOT / "autofly_ue5" / "scenes" / "ue" / "import_assets.py"
WORK_DIR = RUNS_DIR / "assets"
# The registry key a ground material is referenced by in scene files (spec §6.3's grounds), by ambientCG set.
GROUND_KEYS = {"PavingStones070": "paving", "Ground054": "sand", "Grass004": "grass", "Gravel022": "gravel", "Snow006": "snow",
               "Ground037": "earth"}
TEXTURE_MAPS = {"color": "_Color.jpg", "normal": "_NormalDX.jpg", "roughness": "_Roughness.jpg", "ao": "_AmbientOcclusion.jpg"}
REQUIRED_INPUTS = ("MP_BASE_COLOR", "MP_NORMAL", "MP_ROUGHNESS", "MP_AMBIENT_OCCLUSION")


def material_key(set_id: str) -> str:
    return GROUND_KEYS.get(set_id, set_id.lower())


def sanitize_name(name: str) -> str:
    """What UE makes of a glTF mesh name as an asset name (ObjectTools::SanitizeObjectName): "Cube.070" -> "Cube_070"."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _node_key(asset_id: str, node: str, n_nodes: int) -> str:
    """The registry key of a mesh: the asset's id for a single-node model, the node's name (without its _LOD0 suffix) for a
    variant of a set."""
    name = node[:-5] if node.endswith("_LOD0") else node
    return asset_id if n_nodes == 1 else name


def plan_import(sources: dict, measurements: dict, *, assets_root: str = ASSETS_ROOT, materials_root: str = MATERIALS_ROOT) -> dict:
    downloads = Path(sources["downloads_dir"])
    by_asset: dict[str, list[dict]] = {}
    for m in measurements["meshes"]:
        by_asset.setdefault(m["asset"], []).append(m)
    models, materials = [], []
    for asset in sources["assets"]:
        if asset["kind"] == "model":
            gltf = next(f["path"] for f in asset["files"] if f["path"].endswith(".gltf"))
            nodes = by_asset.get(asset["id"], [])
            if not nodes:
                raise ValueError(f"{asset['id']} has no measurements: run scripts/measure_assets.py first")
            models.append({"id": asset["id"], "gltf": str(downloads / asset["source"] / asset["id"] / gltf),
                           "destination": f"{assets_root}/{asset['id']}",
                           # key: the registry name the editor renames the mesh to; mesh: the glTF mesh name Interchange
                           # names the asset after (sanitised), which is how the two are matched
                           "nodes": [{"name": n["node"], "key": _node_key(asset["id"], n["node"], len(nodes)),
                                      "mesh": sanitize_name(n.get("mesh", n["node"])), "extent_m": n["extent_m"],
                                      "triangles": n["triangles"]} for n in nodes]})
        else:
            folder = downloads / asset["source"] / asset["id"]
            prefix = f"{asset['id']}_{asset['resolution']}"
            textures = {kind: str(folder / f"{prefix}{suffix}") for kind, suffix in TEXTURE_MAPS.items()}
            missing = [kind for kind, path in textures.items() if not Path(path).is_file()]
            if missing:
                raise ValueError(f"{asset['id']}: texture maps missing in {folder}: {missing}")
            dims = asset.get("dimensions_cm") or [200.0, 200.0]
            tile_m = round(float(dims[0]) / 100.0, 4)
            materials.append({"id": asset["id"], "name": f"M_{asset['id']}", "destination": materials_root, "textures": textures,
                              "tile_m": tile_m, "tiling": GROUND_SIZE_M / tile_m})
    return {"format": "autofly_ue5_import_spec/1", "planned": time.strftime("%Y-%m-%d %H:%M:%S"), "nanite": True,
            "collision": "complex_as_simple", "models": models, "materials": materials}


COMPILE_FAILURE = "Failed to compile Material"


def check_editor_log(log_path: Path) -> list[str]:
    """A material that fails to compile is not an error to the editor: it logs a warning and renders the default grid
    material in its place, which is how six grey grounds reached a packaged build (2026-10-07). The log is the only
    place it shows."""
    if not Path(log_path).is_file():
        return [f"editor log {log_path} missing"]
    hits = set()
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if COMPILE_FAILURE not in line or "AutoFly" not in line:
            continue
        path = line.split("[AssetLog] ")[-1].split(":")[0].strip()  # .../M_Grass004.uasset or /Game/.../M_Grass004.M_Grass004
        hits.add(path.rsplit("/", 1)[-1].split(".")[0])
    return [f"material failed to compile in the editor (it would render as the default grid): {name}" for name in sorted(hits)]


def check_report(spec: dict, measurements: dict, report: dict) -> list[str]:
    """Everything the registry would rely on, checked before it is written."""
    problems: list[str] = []
    if not report.get("pass"):
        problems.append(f"the editor reported pass={report.get('pass')}: {report.get('error')}")
    measured = {(m["asset"], m["node"]): m for m in measurements["meshes"]}
    reported = _reported_meshes(spec, report)
    for model in spec["models"]:
        for node in model["nodes"]:
            key = (model["id"], node["name"])
            mesh = reported.get(key)
            if mesh is None:
                problems.append(f"{model['id']}: node {node['name']} produced no static mesh")
                continue
            ext = [2.0 * e / 100.0 for e in mesh["box_extent_cm"]]  # UE half-extents in cm -> full extents in m
            gx, gy, gz = measured[key]["extent_m"]  # glTF: x, y up, z
            tol = lambda a: max(EXTENT_TOLERANCE * a, 0.02)  # noqa: E731
            if abs(ext[2] - gy) > tol(gy):
                problems.append(f"{model['id']}/{node['name']}: height {ext[2]:.3f} m in UE, {gy:.3f} m in the glTF")
            if not (abs(ext[0] - gx) <= tol(gx) and abs(ext[1] - gz) <= tol(gz)) and \
                    not (abs(ext[0] - gz) <= tol(gz) and abs(ext[1] - gx) <= tol(gx)):
                problems.append(f"{model['id']}/{node['name']}: horizontal extents {ext[:2]} in UE, ({gx:.3f}, {gz:.3f}) in the glTF")
            if spec.get("nanite") and not mesh.get("nanite_enabled"):
                problems.append(f"{model['id']}/{node['name']}: Nanite is not enabled")
            if spec.get("collision") == "complex_as_simple" and mesh.get("collision_trace_flag") != "CTF_USE_COMPLEX_AS_SIMPLE":
                problems.append(f"{model['id']}/{node['name']}: collision is {mesh.get('collision_trace_flag')}")
    reported_materials = {m["id"]: m for m in report.get("materials", [])}
    for material in spec["materials"]:
        got = reported_materials.get(material["id"])
        if got is None:
            problems.append(f"{material['name']}: not in the report")
            continue
        lacking = [i for i in REQUIRED_INPUTS if i not in got.get("connected", [])]
        if lacking:
            problems.append(f"{material['name']}: material inputs not connected: {lacking}")
    return problems


def _reported_meshes(spec: dict, report: dict) -> dict:
    """(model id, node name) -> the static mesh the editor reported for it: by its registry key (the editor renames to
    it), by the node name, or by the glTF mesh name Interchange used; a one-node model's one mesh matches regardless."""
    out = {}
    for model in spec["models"]:
        reported = next((m for m in report.get("models", []) if m["id"] == model["id"]), None)
        if reported is None:
            continue
        meshes = reported.get("meshes", [])
        if len(model["nodes"]) == 1 and len(meshes) == 1:
            out[(model["id"], model["nodes"][0]["name"])] = meshes[0]
            continue
        by_name = {m["name"]: m for m in meshes}
        by_name.update({m.get("imported_name"): m for m in meshes if m.get("imported_name")})
        for node in model["nodes"]:
            mesh = by_name.get(node["key"]) or by_name.get(node["name"]) or by_name.get(node["mesh"])
            if mesh is not None:
                out[(model["id"], node["name"])] = mesh
    return out


def registry_entries(spec: dict, sources: dict, measurements: dict, report: dict) -> tuple[dict, dict]:
    by_id = {a["id"]: a for a in sources["assets"]}
    measured = {(m["asset"], m["node"]): m for m in measurements["meshes"]}
    reported = _reported_meshes(spec, report)
    assets: dict[str, dict] = {}
    for model in spec["models"]:
        source = by_id[model["id"]]
        for node in model["nodes"]:
            mesh = reported[(model["id"], node["name"])]
            m = measured[(model["id"], node["name"])]
            key = _node_key(model["id"], node["name"], len(model["nodes"]))
            half = mesh["box_extent_cm"]
            assets[key] = {
                "ue_path": mesh["path"],
                "base_size_m": [round(2.0 * h / 100.0, 4) for h in half],
                "pivot": "base",
                "footprint": "circle",
                "category": "nature",
                "role": "obstacle",
                "seen": None,
                "measured_extent_cm_at_unit_scale": [round(h, 4) for h in half],
                "bounds_origin_cm": [round(o, 4) for o in mesh["origin_cm"]],
                "profile_step_m": m["profile_step_m"],
                "radius_profile_m": m["radius_profile_m"],
                "ground_radius_m": m["ground_radius_m"],
                "band_m": measurements["band_m"],
                "band_radius_m": m["band_radius_m"],
                "band_centre_offset_m": m["band_centre_offset_m"],
                "triangles": m["triangles"],  # the glTF's, from the measurements: UE reports the Nanite fallback's
                "nanite": mesh["nanite_enabled"],
                "nanite_fallback_triangles": mesh.get("fallback_triangles"),
                "collision": mesh["collision_trace_flag"],
                "source": {"source": source["source"], "id": source["id"], "node": node["name"], "licence": source.get("licence"),
                           "authors": source.get("authors")},
                "measured_by": "scripts/measure_assets.py (glTF geometry) and scripts/import_assets.py (UE bounds)",
            }
    materials: dict[str, dict] = {}
    for material in report["materials"]:
        source = by_id[material["id"]]
        tile = next(m["tile_m"] for m in spec["materials"] if m["id"] == material["id"])
        materials[material_key(material["id"])] = {"kind": "textured", "ue_path": material["path"], "tile_m": tile,
                                                   "source": {"source": source["source"], "id": source["id"], "licence": source.get("licence")}}
    return assets, materials


def update_registry(path: Path, assets: dict, materials: dict, *, replace: bool = False) -> None:
    """Merge new entries into the registry; an entry that differs under an existing name is refused unless `replace`
    (a re-import of the same sources, on purpose)."""
    path = Path(path)
    registry = json.loads(path.read_text())
    for section, entries in (("assets", assets), ("materials", materials)):
        for key, entry in entries.items():
            existing = registry[section].get(key)
            if existing is not None and existing != entry and not replace:
                raise ValueError(f"{section} entry {key!r} exists with different content; pass --replace to overwrite it")
            registry[section][key] = entry
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(registry, indent=2) + "\n")
    tmp.replace(path)


def run_editor(spec_path: Path, report_path: Path, log_path: Path) -> int:
    """The full editor on the import script, like scripts/build_level.sh; the report decides, not the exit code."""
    editor = ENGINE_DIR / "Engine" / "Binaries" / "Linux" / "UnrealEditor"
    ZEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(DISPLAY=":1", SDL_VIDEODRIVER="x11", AUTOFLY_IMPORT_SPEC=str(spec_path), AUTOFLY_IMPORT_OUT=str(report_path), **UE_CACHE_ENV)
    cmd = [str(editor), str(UPROJECT), f"-ExecCmds=py {EDITOR_SCRIPT}", "-unattended", "-nop4", "-nosplash", "-notraceserver",
           "-RenderOffscreen", f"-ZenDataPath={ZEN_DATA_DIR}", "-stdout", "-FullStdOutLogOutput", f"-abslog={log_path}"]
    with open(str(log_path) + ".stdout", "w") as out:
        return subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, cwd=str(UE_PROJECT_DIR)).returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("plan", "run", "apply"))
    p.add_argument("--sources", type=Path, default=_ROOT / "assets" / "sources.json")
    p.add_argument("--measurements", type=Path, default=_ROOT / "assets" / "measurements.json")
    p.add_argument("--registry", type=Path, default=_ROOT / "assets" / "registry.json")
    p.add_argument("--work", type=Path, default=WORK_DIR)
    p.add_argument("--replace", action="store_true", help="apply: overwrite registry entries that differ (a deliberate re-import)")
    args = p.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    spec_path, report_path = args.work / "import_spec.json", args.work / "import_report.json"
    sources = json.loads(args.sources.read_text())
    measurements = json.loads(args.measurements.read_text())
    if args.stage == "plan":
        spec = plan_import(sources, measurements)
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"{len(spec['models'])} models ({sum(len(m['nodes']) for m in spec['models'])} meshes) and {len(spec['materials'])} "
              f"materials planned in {spec_path}")
        return 0
    if args.stage == "run":
        if not spec_path.is_file():
            p.error(f"{spec_path} missing: run `plan` first")
        if report_path.exists():
            report_path.unlink()
        code = run_editor(spec_path, report_path, args.work / "import_editor.log")
        print(f"editor exit code {code} (information); report {'written' if report_path.is_file() else 'MISSING'}: {report_path}")
        return 0 if report_path.is_file() else 1
    spec = json.loads(spec_path.read_text())
    report = json.loads(report_path.read_text())
    problems = check_report(spec, measurements, report) + check_editor_log(args.work / "import_editor.log")
    if problems:
        print("the report does not pass; the registry is untouched:", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 2
    assets, materials = registry_entries(spec, sources, measurements, report)
    update_registry(args.registry, assets, materials, replace=args.replace)
    print(f"registry updated: {len(assets)} assets ({', '.join(sorted(assets))}) and {len(materials)} materials ({', '.join(sorted(materials))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
