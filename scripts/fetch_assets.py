"""Fetch the CC0 assets M4's scenes are built from (plan 5, task C2; docs/decisions/2026-10-02-m4-asset-survey.md).

    env -u PYTHONPATH .venv/bin/python scripts/fetch_assets.py [--only id ...] [--downloads downloads] [--record assets/sources.json]

Poly Haven models (glTF at 2k with their textures; every file checked against the API's md5) and ambientCG materials
(the 2K JPG zip, unpacked beside itself) go to `downloads/<source>/<id>/` (git-ignored), and `assets/sources.json`
records each asset's source, licence, authors, dimensions, files and sha256, so the registry built in the editor
(task C3) can cite exactly what it imported. A file whose sha256 already matches is not fetched again.

The list is curated here, not configured: which asset stands in for which of spec §6.3's scenes is a design choice
with its reasons (sizes from the API on 2026-10-07; a tree or rock must have mass in the 1-3 m flight band, so rocks
under 0.3 m and 20 m firs are left out). Vehicles and buildings are not here: the survey found no CC0 source that
is not low-poly cartoon, so they come from Fab under its Standard License, through the user's account (decision U1).

The target pool (plan 5, C5; `assets/targets.json`) adds Poly Haven furniture and equipment, photoreal, at 1k: a target
fills at most ~100 of the camera's 256 px at the 5 m success radius, and a 1k texture on a 1-2 m object gives 500-1000
texels per metre, while every imported texture stays resident (never streamed). Vehicles, people and the paper's
unseen animals and crane have no photoreal CC0 source, so Kenney's low-poly kits (CC0, a zip per kit linked from its
asset page, GLB models) stand in for them as placeholders until a Fab session with the user (survey 2026-10-08).
Quaternius (also CC0) was surveyed too: its packs are Google Drive folders, not direct downloads, and its vehicles and
posed humans come as FBX/OBJ/Blend only, so it is not used.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402

USER_AGENT = "autofly_ue5 asset fetch (research; CC0 assets)"
SCENE_IDS = {f"s{i:02d}" for i in range(1, 13)} | {"s05r", "s06r", "targets"}
LICENCE = "CC0-1.0"

# id, source, kind, resolution, the scenes it serves (spec §6.3), and why. Poly Haven dimensions (x, y, z) from its API.
ASSETS: list[dict] = [
    # Trees: s02 "sparse young trees" (grass), s06 "bushy tree clusters" (grass). Dimensions are Poly Haven's (x, y, z):
    # these stand 2.3-4.7 m tall and spread up to 8.5 m, so trunk and canopy are in the flight band; 20 m firs and the
    # 24 m jacaranda would be all trunk at 1-3 m and cost 230-980 MB each.
    {"id": "island_tree_02", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s02", "s06"],
     "why": "3.4 m tall, 8.5 m wide leaning broadleaf, 59 MB"},
    {"id": "island_tree_03", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s02", "s06"],
     "why": "2.6 m tall, 8.5 m wide leaning broadleaf, 98 MB"},
    {"id": "searsia_lucida", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s02", "s06"],
     "why": "2.3 m tall, 4.8 m wide bush, 25 MB: a cluster member with mass at flight altitude"},
    {"id": "searsia_burchellii", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s06"],
     "why": "3.2 m tall, 8.4 m wide bushy tree, 38 MB"},
    {"id": "tree_small_02", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s02"],
     "why": "4.7 m tall, 4.3 m wide small tree, 115 MB"},
    {"id": "quiver_tree_01", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s02"],
     "why": "2.7 m quiver tree: a short, thick trunk entirely in the flight band, 22 MB"},
    # Rocks: s03 "scattered rocks" (grass), s04 "rocks and coloured stones" (gravel), s05 "large standing stones, boulders"
    # (snow / grey stone). Boulders of 0.7-3 m; the generator's scale range makes s05's standing stones of them.
    {"id": "boulder_01", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s03", "s04", "s05"],
     "why": "1.3 x 1.8 x 1.0 m boulder, 14 MB"},
    {"id": "namaqualand_boulder_02", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s03", "s04", "s05"],
     "why": "2.5 x 1.2 x 0.9 m, 13 MB"},
    {"id": "namaqualand_boulder_03", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s03", "s05"],
     "why": "2.4 x 3.1 x 1.5 m, 10 MB"},
    {"id": "namaqualand_boulder_04", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s03", "s05"],
     "why": "2.5 x 2.5 x 1.9 m, 10 MB: the tallest, s05's standing stone"},
    {"id": "namaqualand_boulder_05", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s04"],
     "why": "1.4 x 0.7 x 0.5 m, 12 MB: a stone of the stone field"},
    {"id": "namaqualand_boulder_06", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s04"],
     "why": "0.7 x 1.2 x 0.7 m, 13 MB"},
    {"id": "sand_rocks_small_01", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s04", "s08"],
     "why": "4.6 x 3.9 x 0.5 m flat rock group, 31 MB: low mass, a stone field's floor pieces"},
    # Grounds (ambientCG, 2K JPG): one per ground spec §6.3 names. The engine grid stays s01's.
    {"id": "Grass004", "source": "ambientcg", "kind": "material", "resolution": "2K-JPG", "use": ["s02", "s03", "s06", "s06r"],
     "why": "dense garden grass"},
    {"id": "Gravel022", "source": "ambientcg", "kind": "material", "resolution": "2K-JPG", "use": ["s04"],
     "why": "grey gravel with pebbles: the stone field's floor"},
    {"id": "Snow006", "source": "ambientcg", "kind": "material", "resolution": "2K-JPG", "use": ["s05", "s05r", "s10", "s12"],
     "why": "flat snow: standing stones, the snowy yard, the desert village's snow variant"},
    {"id": "Ground054", "source": "ambientcg", "kind": "material", "resolution": "2K-JPG", "use": ["s07", "s08", "s12"],
     "why": "beach sand: stacked boxes' sand tiles, the ruins' sand, the village"},
    {"id": "PavingStones070", "source": "ambientcg", "kind": "material", "resolution": "2K-JPG", "use": ["s09"],
     "why": "old cobbles: the coloured poles' paving"},
    {"id": "Ground037", "source": "ambientcg", "kind": "material", "resolution": "2K-JPG", "use": ["s11"],
     "why": "damp earth: a light ground for the city blocks"},
    # Target pool (assets/targets.json; spec §6.4, Fig. 9 and 3c). Poly Haven dimensions (x, y, z; z up) and the 1k glTF's
    # size from its API on 2026-10-08. Furniture: the paper's sofas and chairs, then tables, a bench, cabinets.
    {"id": "Sofa_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "beige sofa, 1.57 x 0.66 x 0.80 m, 0.5 MB"},
    {"id": "sofa_02", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "black leather sofa, 1.81 x 0.82 x 0.71 m, 0.4 MB"},
    {"id": "ArmChair_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "grey upholstered armchair, 0.85 x 0.77 x 1.07 m, 0.8 MB"},
    {"id": "GreenChair_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "green upholstered armchair, 0.67 x 0.66 x 1.06 m, 0.6 MB"},
    {"id": "plastic_monobloc_chair_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "white plastic garden chair, 0.64 x 0.63 x 0.88 m, 2.0 MB"},
    {"id": "SchoolChair_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "blue school chair, 0.57 x 0.68 x 1.01 m, 0.5 MB"},
    {"id": "painted_wooden_bench", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "red painted wooden bench, 1.16 x 0.50 x 0.89 m, 2.0 MB"},
    {"id": "round_wooden_table_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "round wooden table, 1.40 x 1.40 x 1.01 m, 0.6 MB"},
    {"id": "painted_wooden_cabinet", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "white painted cabinet, 1.19 x 0.62 x 1.18 m, 1.9 MB"},
    {"id": "vintage_grandfather_clock_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "grandfather clock, 1.00 x 0.80 x 2.19 m, 2.4 MB: an unseen furniture target (no CC0 playground slide exists)"},
    {"id": "wooden_picnic_table", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "wooden picnic table, 2.24 x 3.02 x 0.75 m, 4.3 MB: an unseen furniture target"},
    # Equipment: the paper's barrels, suitcases, TV, ladder; things a yard or a street holds.
    {"id": "Barrel_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "red oil drum, 0.56 x 0.56 x 0.88 m, 0.7 MB"},
    {"id": "Barrel_02", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "blue plastic drum, 0.49 x 0.48 x 0.88 m, 0.4 MB"},
    {"id": "wine_barrel_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "wooden barrel, 0.74 x 0.76 x 0.87 m, 0.9 MB"},
    {"id": "vintage_suitcase", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "green leather suitcase, 1.61 x 0.24 x 0.57 m for the set of variants, 4.3 MB"},
    {"id": "Television_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "old CRT television, 0.60 x 0.47 x 0.46 m, 0.5 MB"},
    {"id": "korean_fire_extinguisher_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "red fire extinguisher on its stand, 0.28 x 0.37 x 0.66 m, 1.6 MB"},
    {"id": "portable_generator", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "yellow portable generator, 0.82 x 0.56 x 0.58 m, 3.4 MB"},
    {"id": "metal_tool_chest", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "red tool chest, 0.69 x 0.41 x 0.65 m, 2.4 MB"},
    {"id": "WetFloorSign_01", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "yellow wet floor sign, 0.30 x 0.36 x 0.63 m, 0.3 MB"},
    {"id": "wooden_ladder_02", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "wooden step ladder, 0.99 x 0.62 x 1.72 m, 2.3 MB: the paper's unseen ladder"},
    # Vehicle and other: the only photoreal CC0 car (under a cover), and a statue for the 5 % "others" (ocean_buoy, the
    # first choice, hangs 0.81 m of ballast below its pivot: spawned on its pivot it would stand half buried).
    {"id": "covered_car", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "car under a dark fabric cover, 1.79 x 4.38 x 1.41 m, 2.3 MB: Poly Haven's only car, scaled to 4 m long"},
    {"id": "bronze_whale_statue", "source": "polyhaven", "kind": "model", "resolution": "1k", "use": ["targets"],
     "why": "bronze whale statue on a plinth, 0.93 x 1.37 x 1.07 m, 2.2 MB"},
    # Low-poly placeholders (Kenney, CC0): vehicles, people, the unseen tiger, elephant, crane and traffic cone. Sizes in
    # Kenney units are scaled to real ones in targets.json; a Fab session with the user may replace them.
    {"id": "kenney_car-kit", "source": "kenney", "kind": "model_pack", "kit": "car-kit", "resolution": "GLB", "use": ["targets"],
     "models": ["sedan", "hatchback-sports", "van", "taxi", "police", "ambulance", "firetruck", "garbage-truck", "suv-luxury",
                "truck-flat", "tractor-shovel", "cone"],
     "why": "Car Kit, 4.8 MB: ten road vehicles in one style, a front loader and a traffic cone (both unseen)"},
    {"id": "kenney_factory-kit", "source": "kenney", "kind": "model_pack", "kit": "factory-kit", "resolution": "GLB", "use": ["targets"],
     "models": ["crane"], "why": "Factory Kit, 4.5 MB: a jib crane, the paper's unseen crane (the only CC0 crane found)"},
    {"id": "kenney_blocky-characters", "source": "kenney", "kind": "model_pack", "kit": "blocky-characters", "resolution": "GLB",
     "use": ["targets"], "models": ["character-a", "character-b", "character-c", "character-e", "character-i", "character-j",
                                    "character-p", "character-q"],
     "why": "Blocky Characters, 2.1 MB: adult-proportioned people in distinct clothes (no robots or zombies)"},
    {"id": "kenney_mini-characters", "source": "kenney", "kind": "model_pack", "kit": "mini-characters", "resolution": "GLB",
     "use": ["targets"], "models": ["character-female-b", "character-female-d"],
     "why": "Mini Characters, 2.4 MB: two people in a second style, one of them the unseen person"},
    {"id": "kenney_cube-pets", "source": "kenney", "kind": "model_pack", "kit": "cube-pets", "resolution": "GLB", "use": ["targets"],
     "models": ["animal-tiger", "animal-elephant"], "why": "Cube Pets, 2.8 MB: the paper's unseen tiger and elephant"},
]


class Web:
    """The network, as the script uses it; tests pass a fake with the same three methods."""

    def json(self, url: str) -> dict:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as r:
            return json.loads(r.read())

    def text(self, url: str) -> str:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as r:
            return r.read().decode("utf-8", "replace")

    def download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=120) as r, open(tmp, "wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
        tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _fetch_file(web, url: str, dest: Path, *, md5: str | None, known_sha256: str | None, log) -> str:
    """Download `url` to `dest` unless it is already there with `known_sha256`; check the source's md5; return the sha256."""
    if dest.is_file() and known_sha256 and _sha256(dest) == known_sha256:
        return known_sha256
    log(f"  fetching {dest.name}")
    web.download(url, dest)
    if md5 is not None and _md5(dest) != md5:
        dest.unlink()
        raise ValueError(f"{url}: md5 {md5} expected by the source's API, the download differs; removed")
    return _sha256(dest)


def _polyhaven(entry: dict, downloads: Path, web, previous: dict | None, log) -> dict:
    files_api = web.json(f"https://api.polyhaven.com/files/{entry['id']}")
    info = web.json(f"https://api.polyhaven.com/info/{entry['id']}")
    gltf = files_api["gltf"][entry["resolution"]]["gltf"]
    root = downloads / "polyhaven" / entry["id"]
    known = {f["path"]: f["sha256"] for f in (previous or {}).get("files", [])}
    files = []
    main_name = gltf["url"].rsplit("/", 1)[-1]
    for rel, item in [(main_name, gltf)] + sorted(gltf.get("include", {}).items()):
        sha = _fetch_file(web, item["url"], root / rel, md5=item.get("md5"), known_sha256=known.get(rel), log=log)
        files.append({"path": rel, "url": item["url"], "size": item.get("size"), "md5": item.get("md5"), "sha256": sha})
    dims = info.get("dimensions")
    return {**entry, "licence": LICENCE, "licence_url": "https://polyhaven.com/license", "name": info.get("name"),
            "authors": info.get("authors"), "categories": info.get("categories"),
            "dimensions_m": [round(d / 1000.0, 3) for d in dims] if dims else None,
            "page": f"https://polyhaven.com/a/{entry['id']}", "files": files}


def _ambientcg(entry: dict, downloads: Path, web, previous: dict | None, log) -> dict:
    api = web.json(f"https://ambientcg.com/api/v2/full_json?id={entry['id']}&include=downloadData,displayData,tagData,dimensionsData")
    found = api.get("foundAssets") or []
    if not found:
        raise ValueError(f"ambientCG has no asset {entry['id']!r}")
    asset = found[0]
    zips = asset["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
    match = [z for z in zips if z["attribute"] == entry["resolution"]]
    if not match:
        raise ValueError(f"ambientCG {entry['id']} has no {entry['resolution']} download (has {[z['attribute'] for z in zips]})")
    item = match[0]
    root = downloads / "ambientcg" / entry["id"]
    name = f"{entry['id']}_{entry['resolution']}.zip"
    known = {f["path"]: f["sha256"] for f in (previous or {}).get("files", [])}
    sha = _fetch_file(web, item["downloadLink"], root / name, md5=None, known_sha256=known.get(name), log=log)
    with zipfile.ZipFile(root / name) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        if not all((root / m).is_file() for m in members):
            z.extractall(root)
    dims = [asset.get("dimensionX"), asset.get("dimensionY")]  # the material's real size in cm, for the ground's tiling
    return {**entry, "licence": LICENCE, "licence_url": "https://docs.ambientcg.com/license/", "name": asset.get("displayName"),
            "tags": asset.get("tags"), "page": f"https://ambientcg.com/a/{entry['id']}",
            "dimensions_cm": dims if all(d is not None for d in dims) else None,
            "files": [{"path": name, "url": item["downloadLink"], "size": item.get("size"), "md5": None, "sha256": sha,
                       "unpacked": sorted(members)}]}


KENNEY_LICENCE_MARK = "Creative Commons Zero, CC0"


def _safe_members(zip_path: Path) -> list[str]:
    """The zip's file members, refusing any that would land outside the folder it is unpacked into."""
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
    for m in members:
        parts = Path(m).parts
        if m.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
            raise ValueError(f"{zip_path}: member {m!r} would unpack outside its folder; refused")
    return members


def _kenney(entry: dict, downloads: Path, web, previous: dict | None, log) -> dict:
    """A Kenney kit: the zip linked from its asset page (the link carries a hash that changes with each kit version, so
    it is read from the page, not pinned), unpacked beside itself. Its own License.txt must say CC0, and every model the
    entry names must be in it as `Models/GLB format/<name>.glb`; each is recorded with its sha256."""
    page = f"https://kenney.nl/assets/{entry['kit']}"
    root = downloads / "kenney" / entry["id"]
    prev_zip = ((previous or {}).get("files") or [None])[0]
    if prev_zip and (root / prev_zip["path"]).is_file() and _sha256(root / prev_zip["path"]) == prev_zip["sha256"]:
        url, name, sha = prev_zip["url"], prev_zip["path"], prev_zip["sha256"]
    else:
        links = re.findall(r"https://kenney\.nl/media/pages/assets/" + re.escape(entry["kit"]) + r"/[^\"'\s<>]+?\.zip", web.text(page))
        if not links:
            raise ValueError(f"{page}: no download link for the kit's zip on its page; fetch it in a browser and record "
                             f"the entry as status 'manual'")
        url = links[0]
        name = url.rsplit("/", 1)[-1]
        sha = _fetch_file(web, url, root / name, md5=None, known_sha256=None, log=log)
    members = _safe_members(root / name)
    with zipfile.ZipFile(root / name) as z:
        licence_files = [m for m in members if m.rsplit("/", 1)[-1].lower() == "license.txt"]
        licence_text = z.read(licence_files[0]).decode("utf-8", "replace") if licence_files else ""
        if KENNEY_LICENCE_MARK not in licence_text:
            raise ValueError(f"{url}: the kit's License.txt does not say {KENNEY_LICENCE_MARK!r} (CC0); refused")
        wanted = {m: f"Models/GLB format/{m}.glb" for m in entry["models"]}
        missing = [m for m, path in wanted.items() if path not in members]
        if missing:
            raise ValueError(f"{url}: no model {missing} in the kit (expected Models/GLB format/<name>.glb)")
        if not all((root / m).is_file() for m in members):
            z.extractall(root)
    version = next((line.strip() for line in licence_text.splitlines() if line.strip()), None)
    return {**entry, "licence": LICENCE, "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "licence_file": licence_files[0], "kit_version": version, "authors": {"Kenney": "All"}, "page": page,
            "files": [{"path": name, "url": url, "size": (root / name).stat().st_size, "md5": None, "sha256": sha,
                       "unpacked_files": len(members)}],
            "models": [{"name": m, "path": path, "sha256": _sha256(root / path)} for m, path in wanted.items()]}


FETCHERS = {"polyhaven": _polyhaven, "ambientcg": _ambientcg, "kenney": _kenney}


def fetch_all(entries: list[dict], downloads: Path, record_path: Path, *, web=None, log=print) -> dict:
    """Fetch every entry, check it, and write the record (merging with an existing one, so a partial run resumes)."""
    web = web or Web()
    downloads, record_path = Path(downloads), Path(record_path)
    previous = {}
    if record_path.is_file():
        previous = {a["id"]: a for a in json.loads(record_path.read_text()).get("assets", [])}
    results = []
    for entry in entries:
        log(f"{entry['source']}/{entry['id']} ({entry['kind']}, {entry['resolution']}) for {', '.join(entry['use'])}")
        results.append(FETCHERS[entry["source"]](entry, downloads, web, previous.get(entry["id"]), log))
    merged = {**previous, **{a["id"]: a for a in results}}
    record = {"format": "autofly_ue5_asset_sources/1", "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
              "downloads_dir": str(downloads), "assets": [merged[k] for k in sorted(merged)]}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = record_path.with_name(record_path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(record_path)
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=None, help="asset ids to fetch (default: the whole curated list)")
    p.add_argument("--downloads", type=Path, default=_ROOT / "downloads")
    p.add_argument("--record", type=Path, default=_ROOT / "assets" / "sources.json")
    args = p.parse_args(argv)
    entries = [a for a in ASSETS if args.only is None or a["id"] in args.only]
    unknown = set(args.only or []) - {a["id"] for a in entries}
    if unknown:
        p.error(f"not in the curated list: {sorted(unknown)}")
    record = fetch_all(entries, args.downloads, args.record)
    total = sum(f.get("size") or 0 for a in record["assets"] for f in a["files"])
    print(f"{len(record['assets'])} assets recorded in {args.record} ({total / 1e6:.0f} MB of sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
