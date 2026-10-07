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
]


class Web:
    """The network, as the script uses it; tests pass a fake with the same two methods."""

    def json(self, url: str) -> dict:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as r:
            return json.loads(r.read())

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
        fetch = _polyhaven if entry["source"] == "polyhaven" else _ambientcg
        results.append(fetch(entry, downloads, web, previous.get(entry["id"]), log))
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
