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

Targets (plan 5 C6; `--targets`, work dir runs/assets/targets/, docs/runbook-targets.md):

    env -u PYTHONPATH .venv/bin/python scripts/import_assets.py plan --targets   # merged glTFs + runs/assets/targets/import_spec.json
    bash scripts/run_job.sh start import_targets -- env -u PYTHONPATH .venv/bin/python scripts/import_assets.py run --targets
    env -u PYTHONPATH .venv/bin/python scripts/import_assets.py apply --targets [--use-placeholders]

A target is spawned at runtime as ONE static mesh (Project AirSim's SpawnObject places one asset by its short name), so
`plan --targets` first writes every model target of the pool (assets/targets.json: Poly Haven and Kenney entries, status
candidate, placeholder or imported) as a single-node glTF with its node transforms baked in and its pivot at its base
centre (scenes/gltf.py merge_to_single_mesh; only the nodes the entry names, when it names some). The same editor script
imports it under /Game/AutoFly/Targets/<key>/<key> with the obstacles' settings, except that the Nanite fallback keeps
every triangle: complex collision runs on the fallback, and a 72-triangle Kenney character at 10 % would collide as a
handful of triangles. Keys are "tgt_" + the model's name: Project AirSim's spawn table is a case-insensitive map of short
names, in which a Kenney "cone" would replace the engine's Cone (two assets of that name exist already). `apply --targets`
checks the report (one mesh per target at its path, the merged glTF's size, the base pivot, Nanite with a full fallback,
complex collision, no translucent slot, no failed material compile) before the registry gets one role "target" entry
per model and the pool entry its asset (status candidate -> imported; a Kenney stand-in stays "placeholder", out of the
draw, unless --use-placeholders: whether stand-ins are drawn is the user's call).
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

import hashlib  # noqa: E402

from autofly_ue5.paths import ENGINE_DIR, RUNS_DIR, UE_CACHE_ENV, UE_PROJECT_DIR, UPROJECT, ZEN_DATA_DIR  # noqa: E402
from autofly_ue5.scenes.gltf import load_meshes, measure, merge_to_single_mesh, split_by_opacity  # noqa: E402

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


def plan_import(sources: dict, measurements: dict, *, assets_root: str = ASSETS_ROOT, materials_root: str = MATERIALS_ROOT,
                split_dir: Path | None = None) -> dict:
    """A model whose measurements show alpha cards (leaves, twigs: `parts.cards`) is split into two glTF files
    (`gltf.split_by_opacity`, under `split_dir`): the editor imports the solid part under the registry key, with
    collision, and the cards part as `<key>_cards`, without. The drone's collision mesh is then wood and rock only,
    while the depth camera sees both (2026-10-07: a crash into a canopy wedged the actor among the cards).

    Only scene assets are planned: obstacle models and ground materials. The target pool's downloads (plan 5, C5: `use`
    only ["targets"], and Kenney kits, `kind` "model_pack") are imported by the targets' own step, since this one
    writes obstacle entries (role "obstacle", a footprint circle) and would take a kit for a ground material."""
    downloads = Path(sources["downloads_dir"])
    split_dir = Path(split_dir) if split_dir is not None else WORK_DIR / "split"
    by_asset: dict[str, list[dict]] = {}
    for m in measurements["meshes"]:
        by_asset.setdefault(m["asset"], []).append(m)
    models, materials = [], []
    for asset in sources["assets"]:
        if not set(asset.get("use", [])) - {"targets"}:
            continue
        if asset["kind"] == "model":
            gltf = next(f["path"] for f in asset["files"] if f["path"].endswith(".gltf"))
            nodes = by_asset.get(asset["id"], [])
            if not nodes:
                raise ValueError(f"{asset['id']} has no measurements: run scripts/measure_assets.py first")
            gltf_path = downloads / asset["source"] / asset["id"] / gltf
            model = {"id": asset["id"], "gltf": str(gltf_path), "gltf_cards": None,
                     "destination": f"{assets_root}/{asset['id']}",
                     # key: the registry name the editor renames the mesh to; mesh: the glTF mesh name Interchange
                     # names the asset after (sanitised), which is how the two are matched
                     "nodes": [{"name": n["node"], "key": _node_key(asset["id"], n["node"], len(nodes)),
                                "mesh": sanitize_name(n.get("mesh", n["node"])), "extent_m": n["extent_m"],
                                "triangles": n["triangles"], "parts": n.get("parts")} for n in nodes]}
            if any((n.get("parts") or {}).get("cards") for n in nodes):
                parts = split_by_opacity(gltf_path, split_dir / asset["id"])
                model["gltf"], model["gltf_cards"] = str(parts["solid"]), str(parts["cards"])
            models.append(model)
        elif asset["kind"] == "material":
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
    reported_cards = _reported_meshes(spec, report, part="cards_meshes", suffix="_cards")
    for model in spec["models"]:
        for node in model["nodes"]:
            key = (model["id"], node["name"])
            mesh = reported.get(key)
            if mesh is None:
                problems.append(f"{model['id']}: node {node['name']} produced no static mesh")
                continue
            parts = measured[key].get("parts") or {}
            solid_extent = (parts.get("solid") or {}).get("extent_m", measured[key]["extent_m"])
            problems += _extent_problems(f"{model['id']}/{node['name']}", mesh, solid_extent)
            cards_measured = parts.get("cards")
            cards_mesh = reported_cards.get(key)
            if cards_measured and cards_mesh is None:
                problems.append(f"{model['id']}/{node['name']}: its alpha cards produced no static mesh")
            elif cards_measured:
                problems += _extent_problems(f"{model['id']}/{node['name']} (cards)", cards_mesh, cards_measured["extent_m"])
                if cards_mesh.get("collision_trace_flag") != "CTF_USE_SIMPLE_AS_COMPLEX" or cards_mesh.get("simple_collision_primitives", 1) != 0:
                    problems.append(f"{model['id']}/{node['name']} (cards): must have no collision (simple-as-complex with no primitives), "
                                    f"got {cards_mesh.get('collision_trace_flag')} with {cards_mesh.get('simple_collision_primitives')} primitives")
            elif cards_mesh is not None:
                problems.append(f"{model['id']}/{node['name']}: a cards mesh was imported but the measurements show no cards")
            if spec.get("nanite") and not mesh.get("nanite_enabled"):
                problems.append(f"{model['id']}/{node['name']}: Nanite is not enabled")
            if spec.get("collision") == "complex_as_simple" and mesh.get("collision_trace_flag") != "CTF_USE_COMPLEX_AS_SIMPLE":
                problems.append(f"{model['id']}/{node['name']}: collision is {mesh.get('collision_trace_flag')}")
            # a translucent slot writes no depth: the depth camera would see through the leaves (s02, 2026-10-07)
            translucent = [s["material"] for s in mesh.get("foliage", {}).get("slots", []) if s["blend_mode"] == "BLEND_TRANSLUCENT"]
            if translucent or "foliage" not in mesh:
                problems.append(f"{model['id']}/{node['name']}: translucent material slots {translucent or '(not reported)'}")
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


def _extent_problems(label: str, mesh: dict, gltf_extent: list) -> list[str]:
    """UE's box of an imported mesh against the glTF's (x, y up, z): the height must match, and the horizontal pair
    either way round (UE's y is glTF's z)."""
    ext = [2.0 * e / 100.0 for e in mesh["box_extent_cm"]]  # UE half-extents in cm -> full extents in m
    gx, gy, gz = gltf_extent
    tol = lambda a: max(EXTENT_TOLERANCE * a, 0.02)  # noqa: E731
    problems = []
    if abs(ext[2] - gy) > tol(gy):
        problems.append(f"{label}: height {ext[2]:.3f} m in UE, {gy:.3f} m in the glTF")
    if not (abs(ext[0] - gx) <= tol(gx) and abs(ext[1] - gz) <= tol(gz)) and \
            not (abs(ext[0] - gz) <= tol(gz) and abs(ext[1] - gx) <= tol(gx)):
        problems.append(f"{label}: horizontal extents {ext[:2]} in UE, ({gx:.3f}, {gz:.3f}) in the glTF")
    return problems


def _reported_meshes(spec: dict, report: dict, *, part: str = "meshes", suffix: str = "") -> dict:
    """(model id, node name) -> the static mesh the editor reported for it: by its registry key (the editor renames to
    it, plus `suffix` for a cards part), by the node name, or by the glTF mesh name Interchange used; a one-node
    model's one mesh matches regardless."""
    out = {}
    for model in spec["models"]:
        reported = next((m for m in report.get("models", []) if m["id"] == model["id"]), None)
        if reported is None:
            continue
        meshes = reported.get(part, [])
        if len(model["nodes"]) == 1 and len(meshes) == 1:
            out[(model["id"], model["nodes"][0]["name"])] = meshes[0]
            continue
        by_name = {m["name"]: m for m in meshes}
        by_name.update({m.get("imported_name"): m for m in meshes if m.get("imported_name")})
        for node in model["nodes"]:
            mesh = by_name.get(node["key"] + suffix) or by_name.get(node["name"]) or by_name.get(node["mesh"])
            if mesh is not None:
                out[(model["id"], node["name"])] = mesh
    return out


def _union_box(meshes: list[dict]) -> tuple[list[float], list[float]]:
    """Centre and half-extents (UE cm) of the box containing every mesh's local box."""
    lo = [min(m["origin_cm"][i] - m["box_extent_cm"][i] for m in meshes) for i in range(3)]
    hi = [max(m["origin_cm"][i] + m["box_extent_cm"][i] for m in meshes) for i in range(3)]
    return [(a + b) / 2.0 for a, b in zip(lo, hi)], [(b - a) / 2.0 for a, b in zip(lo, hi)]


def _part_entry(mesh: dict) -> dict:
    return {"ue_path": mesh["path"], "base_size_m": [round(2.0 * h / 100.0, 4) for h in mesh["box_extent_cm"]],
            "bounds_origin_cm": [round(o, 4) for o in mesh["origin_cm"]], "collision": mesh.get("collision_trace_flag"),
            "fallback_triangles": mesh.get("fallback_triangles")}


def registry_entries(spec: dict, sources: dict, measurements: dict, report: dict) -> tuple[dict, dict]:
    by_id = {a["id"]: a for a in sources["assets"]}
    measured = {(m["asset"], m["node"]): m for m in measurements["meshes"]}
    reported = _reported_meshes(spec, report)
    reported_cards = _reported_meshes(spec, report, part="cards_meshes", suffix="_cards")
    assets: dict[str, dict] = {}
    for model in spec["models"]:
        source = by_id[model["id"]]
        for node in model["nodes"]:
            mesh = reported[(model["id"], node["name"])]
            cards = reported_cards.get((model["id"], node["name"]))
            m = measured[(model["id"], node["name"])]
            key = _node_key(model["id"], node["name"], len(model["nodes"]))
            # the whole model's box is the union of its parts' (what UE reported for the one mesh before the split)
            origin, half = _union_box([mesh] + ([cards] if cards else []))
            parts = {"solid": _part_entry(mesh)}
            if cards:
                parts["cards"] = _part_entry(cards)
            assets[key] = {
                "ue_path": mesh["path"],
                "base_size_m": [round(2.0 * h / 100.0, 4) for h in half],
                "pivot": "base",
                "footprint": "circle",
                "category": "nature",
                "role": "obstacle",
                "seen": None,
                "measured_extent_cm_at_unit_scale": [round(h, 4) for h in half],
                "bounds_origin_cm": [round(o, 4) for o in origin],
                # the actors the level builder spawns for one instance: the solid part (collision) under the tag, the
                # alpha cards (no collision) as <tag>_cards (plan 5, 2026-10-07)
                "parts": parts,
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


# ---------------------------------------------------------------------------------------------------------------------
# Targets (plan 5 C6)
# ---------------------------------------------------------------------------------------------------------------------

TARGETS_ROOT = "/Game/AutoFly/Targets"
TARGET_KEY_PREFIX = "tgt_"
# Complex collision runs on the Nanite fallback (C3b): 10 % suits a 2 M-triangle tree, not a 72-triangle character. The
# targets' fallback keeps every triangle, so the drone collides with what the camera sees.
TARGET_NANITE_FALLBACK_PERCENT = 1.0
TARGET_FALLBACK_SHARE = 0.9  # apply: the fallback must keep at least this share of the imported triangles
# A target is one mesh; a model whose non-opaque triangles exceed this share is foliage-like and would need its cards
# split off without collision (the obstacles' import), which a one-asset spawn cannot carry. The pool's glass and labels:
# 0.2-3.6 %.
TARGET_NON_OPAQUE_LIMIT = 0.10
TARGET_STATUSES = ("candidate", "placeholder", "imported")  # entries with a fetched model to import ("manual": none yet)


class TargetPlanError(ValueError):
    """A pool entry that cannot be planned for import as it stands."""


def target_work_dir(work: Path) -> Path:
    return Path(work) / "targets"


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", text.lower())).strip("_")


def target_key(source: dict) -> str:
    """The registry key and the UE short name of a target's mesh: tgt_ + the kit model's name, the Poly Haven id, or, for
    an entry that takes some nodes of a file (the suitcase file holds two suitcases), their common prefix."""
    if source.get("nodes"):
        names = list(source["nodes"])
        prefix = names[0]
        for n in names[1:]:
            while not n.startswith(prefix):
                prefix = prefix[:-1]
        base = prefix if _slug(prefix) else names[0]
    else:
        base = source.get("model") or source["id"]
    return TARGET_KEY_PREFIX + _slug(base)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_file(entry: dict, sources: dict) -> tuple[Path, str, dict]:
    """The downloaded file of a pool entry, checked against its fetch record (assets/sources.json): the record exists for
    the target pool, names the file, and every file it records a sha256 for still has it."""
    src, name = entry["source"], entry["name"]
    record = next((a for a in sources["assets"] if a["id"] == src.get("id")), None)
    if record is None or "targets" not in record.get("use", []):
        raise TargetPlanError(f"{name!r}: {src.get('id')!r} has no fetch record for the target pool in assets/sources.json "
                              f"(scripts/fetch_assets.py)")
    folder = Path(sources["downloads_dir"]) / record["source"] / record["id"]
    relative = Path(src["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise TargetPlanError(f"{name!r}: file {src['file']!r} leaves the download folder")
    if record["kind"] == "model_pack":
        model = next((m for m in record.get("models", []) if m["name"] == src.get("model")), None)
        if model is None or model["path"] != src["file"]:
            raise TargetPlanError(f"{name!r}: the kit {record['id']} has no model {src.get('model')!r} at {src['file']!r}")
        recorded = {src["file"]: model.get("sha256")}
    elif record["kind"] == "model":
        recorded = {f["path"]: f.get("sha256") for f in record["files"] if not f["path"].endswith(".zip")}
        if src["file"] not in recorded:
            raise TargetPlanError(f"{name!r}: {record['id']}'s fetch record has no file {src['file']!r}")
    else:
        raise TargetPlanError(f"{name!r}: {record['id']} is a {record['kind']}, not a model")
    for rel, expected in recorded.items():
        path = folder / rel
        if not path.is_file():
            raise TargetPlanError(f"{name!r}: {path} is missing (scripts/fetch_assets.py)")
        if expected and _sha256(path) != expected:
            raise TargetPlanError(f"{name!r}: {path} has sha256 {_sha256(path)[:12]}..., the fetch record says "
                                  f"{expected[:12]}...: not the file that was fetched")
    path = folder / relative
    return path, _sha256(path), record


def content_short_names(content_roots: list[Path]) -> dict[str, list[Path]]:
    """Lower-cased asset name -> its .uasset files, under every content root that exists."""
    names: dict[str, list[Path]] = {}
    for root in content_roots:
        if Path(root).is_dir():
            for path in Path(root).rglob("*.uasset"):
                names.setdefault(path.stem.lower(), []).append(path)
    return names


def default_content_roots() -> list[Path]:
    return [UE_PROJECT_DIR / "Content", UE_PROJECT_DIR / "Plugins", ENGINE_DIR / "Engine" / "Content", ENGINE_DIR / "Engine" / "Plugins"]


def _within(value: float, expected: float) -> bool:
    return abs(value - expected) <= max(EXTENT_TOLERANCE * abs(expected), 0.02)


def plan_targets(pool: dict, sources: dict, measurements: dict, *, work_dir: Path, registry: dict,
                 content_roots: list[Path], targets_root: str = TARGETS_ROOT) -> dict:
    """The editor job for every model target of the pool, each merged first into one single-node glTF under
    <work_dir>/merged/<key>/ (see the module docstring). Refuses (TargetPlanError) a download that is not the fetched
    file, a merge whose size is not the measured one (measurements.json's whole-model record, or the entry's own
    extent_m when it takes some nodes), a foliage-like model, and a key that another asset already has, in any case."""
    if not any(Path(r).is_dir() for r in content_roots):
        raise TargetPlanError(f"none of the content roots exists ({[str(r) for r in content_roots]}): cannot check that the "
                              f"keys are unique short names")
    names = content_short_names(content_roots)
    registry_names = {a["ue_path"].rsplit("/", 1)[-1].lower(): k for k, a in registry["assets"].items()}
    models_measured = {(m["asset"], m["model"]): m for m in measurements.get("models", [])}
    band = tuple(measurements.get("band_m", (1.0, 3.0)))
    keys: dict[str, str] = {}
    models = []
    for entry in pool["targets"]:
        src = entry.get("source") or {}
        if entry["status"] not in TARGET_STATUSES or src.get("source") in (None, "engine"):
            continue
        name = entry["name"]
        key = target_key(src)
        if key.lower() in keys:
            raise TargetPlanError(f"{name!r} and {keys[key.lower()]!r} would both be {key}; give one of them its own nodes")
        keys[key.lower()] = name
        own = f"/autofly/targets/{key.lower()}/"
        clashes = [p for p in names.get(key.lower(), []) if own not in p.as_posix().lower()]
        if clashes:
            raise TargetPlanError(f"{name!r}: {key} is already the short name of {[str(p) for p in clashes]}; Project AirSim's "
                                  f"spawn table would keep one of them (a case-insensitive map of short names)")
        if registry_names.get(key.lower(), key) != key:
            raise TargetPlanError(f"{name!r}: {key} is already the short name of registry asset {registry_names[key.lower()]}")
        path, sha, record = _target_file(entry, sources)
        merge = merge_to_single_mesh(path, Path(work_dir) / "merged" / key, name=key, nodes=src.get("nodes"))
        if merge["non_opaque_triangle_share"] > TARGET_NON_OPAQUE_LIMIT:
            raise TargetPlanError(f"{name!r}: {merge['non_opaque_triangle_share']:.0%} of its triangles are not opaque "
                                  f"(alpha cards?): a target is one mesh and cannot carry a no-collision cards part")
        gx, gy, gz = merge["extent_m"]
        if src.get("nodes"):
            expected = entry.get("extent_m")
            scale = entry["scale"]
            got = [gx * scale[0], gz * scale[1], gy * scale[2]]
            if expected is None or not (_within(got[2], expected[2]) and (
                    (_within(got[0], expected[0]) and _within(got[1], expected[1]))
                    or (_within(got[0], expected[1]) and _within(got[1], expected[0])))):
                raise TargetPlanError(f"{name!r}: the merged nodes' extent at scale is {[round(v, 3) for v in got]}, the "
                                      f"entry's extent_m is {expected}")
        else:
            measured = models_measured.get((record["id"], src.get("model") or record["id"]))
            if measured is None:
                raise TargetPlanError(f"{name!r}: no whole-model record in the measurements (scripts/measure_assets.py)")
            if not all(_within(a, b) for a, b in zip(merge["extent_m"], measured["extent_m"])):
                raise TargetPlanError(f"{name!r}: the merged extent {merge['extent_m']} is not the measured extent "
                                      f"{measured['extent_m']}")
        (mesh,) = load_meshes(Path(merge["path"]))
        measurement = measure(mesh, band_m=band)
        models.append({
            "id": key, "gltf": merge["path"], "gltf_cards": None, "destination": f"{targets_root}/{key}",
            "nodes": [{"name": key, "key": key, "mesh": key, "extent_m": measurement["extent_m"],
                       "triangles": measurement["triangles"], "parts": None}],
            "target": {"name": name, "category": entry["category"], "seen": entry["seen"], "status": entry["status"],
                       "scale": entry["scale"], "source": src, "file": str(path), "sha256": sha,
                       "licence": record.get("licence"), "authors": record.get("authors"), "page": src.get("page") or record.get("page")},
            "merge": {k: v for k, v in merge.items() if k != "path"},
            "measurement": measurement})
    return {"format": "autofly_ue5_import_spec/1", "kind": "targets", "planned": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nanite": True, "collision": "complex_as_simple", "nanite_fallback_percent": TARGET_NANITE_FALLBACK_PERCENT,
            "models": models, "materials": []}


def check_target_report(spec: dict, report: dict) -> list[str]:
    """What the registry and a runtime spawn rely on, from the editor's report (the editor log is checked separately)."""
    problems: list[str] = []
    if not report.get("pass"):
        problems.append(f"the editor reported pass={report.get('pass')}: {report.get('error')}")
    reported = {m["id"]: m for m in report.get("models", [])}
    for model in spec["models"]:
        key = model["id"]
        got = reported.get(key)
        if got is None:
            problems.append(f"{key}: not in the report")
            continue
        meshes = got.get("meshes", [])
        if len(meshes) != 1:
            problems.append(f"{key}: {len(meshes)} meshes imported; a target must be one static mesh "
                            f"({[m.get('name') for m in meshes]})")
            continue
        if got.get("cards_meshes"):
            problems.append(f"{key}: a cards part was imported; a target spawns as one mesh")
        mesh = meshes[0]
        (node,) = model["nodes"]
        # Interchange lays a merged single-mesh import out as <destination>/<key>/StaticMeshes/<key> (beside Materials/
        # and Textures/), measured live on 2026-10-08; the obstacles' import is flat. Either is spawnable: the spawn
        # table keys meshes by short name (the key) and +DirectoriesToAlwaysCook covers the destination recursively.
        # The registry records the path the mesh actually has.
        accepted = (f"{model['destination']}/{key}", f"{model['destination']}/{key}/StaticMeshes/{key}")
        if mesh.get("name") != key or mesh.get("path") not in accepted:
            problems.append(f"{key}: imported as {mesh.get('name')} at path {mesh.get('path')}, expected "
                            f"{' or '.join(accepted)}")
        problems += _extent_problems(key, mesh, node["extent_m"])
        gx, gy, gz = node["extent_m"]
        ox, oy, oz = mesh["origin_cm"]
        tol = lambda a: max(EXTENT_TOLERANCE * a, 0.02) * 100.0  # noqa: E731 -- cm
        if abs(ox) > tol(max(gx, gz)) or abs(oy) > tol(max(gx, gz)) or abs(oz - 50.0 * gy) > tol(gy):
            problems.append(f"{key}: bounds centre at {mesh['origin_cm']} cm; the base pivot puts it at (0, 0, "
                            f"{50.0 * gy:.1f}): the merge's pivot did not survive the import")
        if spec.get("nanite") and not mesh.get("nanite_enabled"):
            problems.append(f"{key}: Nanite is not enabled")
        source_tris, fallback = mesh.get("source_triangles") or 0, mesh.get("fallback_triangles") or 0
        if fallback < TARGET_FALLBACK_SHARE * source_tris:
            problems.append(f"{key}: complex collision would run on {fallback} of {source_tris} triangles (the Nanite "
                            f"fallback); the target's spec asks for {spec.get('nanite_fallback_percent')}")
        if mesh.get("collision_trace_flag") != "CTF_USE_COMPLEX_AS_SIMPLE":
            problems.append(f"{key}: collision is {mesh.get('collision_trace_flag')}")
        translucent = [s["material"] for s in mesh.get("foliage", {}).get("slots", []) if s["blend_mode"] == "BLEND_TRANSLUCENT"]
        if translucent or "foliage" not in mesh:
            problems.append(f"{key}: translucent material slots {translucent or '(not reported)'}")
        if not mesh.get("materials") or any(m is None for m in mesh["materials"]):
            problems.append(f"{key}: a material slot is empty ({mesh.get('materials')}): it would render the default material")
    return problems


def target_registry_entries(spec: dict, report: dict, sources: dict) -> dict[str, dict]:
    reported = {m["id"]: m["meshes"][0] for m in report["models"] if m.get("meshes")}
    entries: dict[str, dict] = {}
    for model in spec["models"]:
        key, mesh, target, measured = model["id"], reported[model["id"]], model["target"], model["measurement"]
        src = target["source"]
        source = {"source": src["source"], "id": src["id"], "file": src["file"]}
        for optional in ("model", "nodes"):
            if src.get(optional):
                source[optional] = src[optional]
        source.update(licence=target["licence"], authors=target["authors"], page=target["page"], sha256=target["sha256"])
        entries[key] = {
            "ue_path": mesh["path"],
            "base_size_m": [round(2.0 * h / 100.0, 4) for h in mesh["box_extent_cm"]],
            "pivot": "base",
            "footprint": "circle",
            "category": target["category"],
            "role": "target",
            "seen": target["seen"],
            "measured_extent_cm_at_unit_scale": [round(h, 4) for h in mesh["box_extent_cm"]],
            "bounds_origin_cm": [round(o, 4) for o in mesh["origin_cm"]],
            "profile_step_m": measured["profile_step_m"],
            "radius_profile_m": measured["radius_profile_m"],
            "ground_radius_m": measured["ground_radius_m"],
            "band_m": measured["band_m"],
            "band_radius_m": measured["band_radius_m"],
            "triangles": measured["triangles"],
            "nanite": mesh["nanite_enabled"],
            "nanite_fallback_triangles": mesh.get("fallback_triangles"),
            "collision": mesh["collision_trace_flag"],
            "target_name": target["name"],
            "merged": {"nodes": model["merge"]["nodes"], "translation_m": model["merge"]["translation_m"],
                       "rewrites": model["merge"]["rewrites"], "skinned_nodes": model["merge"]["skinned_nodes"]},
            "source": source,
            "measured_by": "scenes/gltf.py merge_to_single_mesh + measure (the merged glTF) and scripts/import_assets.py "
                           "(UE bounds)",
        }
    return entries


def update_pool_file(path: Path, spec: dict, *, use_placeholders: bool = False) -> list[str]:
    """Each imported entry gets its registry key; a candidate becomes imported (drawn), a placeholder only with
    `use_placeholders`, and then keeps `"stand_in": true`, which every episode drawing it records (plan 6 step 2b).
    Returns the names now imported."""
    path = Path(path)
    pool = json.loads(path.read_text())
    by_name = {t["name"]: t for t in pool["targets"]}
    imported = []
    for model in spec["models"]:
        entry = by_name[model["target"]["name"]]
        entry["asset"] = model["id"]
        if entry["status"] == "placeholder" and use_placeholders:
            entry["status"], entry["stand_in"] = "imported", True
        if entry["status"] == "candidate":
            entry["status"] = "imported"
        if entry["status"] == "imported":
            imported.append(entry["name"])
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(pool, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return imported


def _main_targets(args) -> int:
    spec_path, report_path, log_path = args.work / "import_spec.json", args.work / "import_report.json", args.work / "import_editor.log"
    if args.stage == "plan":
        try:
            spec = plan_targets(json.loads(args.pool.read_text()), json.loads(args.sources.read_text()),
                                json.loads(args.measurements.read_text()), work_dir=args.work,
                                registry=json.loads(args.registry.read_text()),
                                content_roots=args.content_root or default_content_roots())
        except TargetPlanError as err:
            print(f"refused: {err}", file=sys.stderr)
            return 2
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        for model in spec["models"]:
            merge = model["merge"]
            print(f"{model['id']:34s} {model['target']['name']:26s} {merge['triangles']:6d} tris  extent {merge['extent_m']}  "
                  f"{[r['change'] for r in merge['rewrites']] or ''}")
        print(f"{len(spec['models'])} targets planned in {spec_path}")
        return 0
    if args.stage == "run":
        if not spec_path.is_file():
            print(f"{spec_path} missing: run `plan --targets` first", file=sys.stderr)
            return 2
        if report_path.exists():
            report_path.unlink()
        code = run_editor(spec_path, report_path, log_path)
        print(f"editor exit code {code} (information); report {'written' if report_path.is_file() else 'MISSING'}: {report_path}")
        return 0 if report_path.is_file() else 1
    spec = json.loads(spec_path.read_text())
    report = json.loads(report_path.read_text())
    problems = check_target_report(spec, report) + check_editor_log(log_path)
    if problems:
        print("the report does not pass; the registry and the pool are untouched:", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 2
    entries = target_registry_entries(spec, report, json.loads(args.sources.read_text()))
    try:
        update_registry(args.registry, entries, {}, replace=args.replace)
    except ValueError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    imported = update_pool_file(args.pool, spec, use_placeholders=args.use_placeholders)
    print(f"registry updated: {len(entries)} targets; {len(imported)} pool entries imported (drawn): {', '.join(imported)}")
    return 0


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
    p.add_argument("--work", type=Path, default=None, help=f"default {WORK_DIR} ({target_work_dir(WORK_DIR)} with --targets)")
    p.add_argument("--replace", action="store_true", help="apply: overwrite registry entries that differ (a deliberate re-import)")
    p.add_argument("--targets", action="store_true", help="the target pool's models instead of the scene assets")
    p.add_argument("--pool", type=Path, default=_ROOT / "assets" / "targets.json", help="--targets: the pool file")
    p.add_argument("--content-root", type=Path, action="append", default=None,
                   help="--targets plan: content folders whose asset names a key must not repeat (default: the project's and "
                        "the engine's Content and Plugins)")
    p.add_argument("--use-placeholders", action="store_true",
                   help="--targets apply: mark Kenney stand-ins (status placeholder) imported, so the pool draws them")
    args = p.parse_args(argv)
    if args.work is None:
        args.work = target_work_dir(WORK_DIR) if args.targets else WORK_DIR
    args.work.mkdir(parents=True, exist_ok=True)
    if args.targets:
        return _main_targets(args)
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
