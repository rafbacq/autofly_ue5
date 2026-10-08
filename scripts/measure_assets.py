"""Measure every downloaded model (assets/sources.json) for the registry: height, extents, flight-band footprint.

    env -u PYTHONPATH .venv/bin/python scripts/measure_assets.py [--sources assets/sources.json] [--out assets/measurements.json]

One entry per glTF node (a Poly Haven set of variants is several), from `autofly_ue5/scenes/gltf.py`, in the mesh's own
frame (glTF: y up, metres). The editor import (plan 5, C3) checks each static mesh's bounds against `extent_m` and takes
`band_radius_m` -- the farthest the surface reaches from the pivot between 1 and 3 m -- as the registry's footprint
circle, with `band_centre_offset_m` saying how far off the pivot that mass sits.

Targets (plan 5, C5) are measured whole as well, under `models`: one record per kit model (a Kenney GLB, whose body and
wheels or torso and limbs are nodes of one object) and per Poly Haven model whose `use` names the target pool, all
nodes together in the file's frame. That is the size the pool's scale and spawn height come from. `meshes` keeps one
record per node of every Poly Haven model, unchanged: the import makes a static mesh of each.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

from autofly_ue5.scenes.gltf import combine, load_meshes, measure, part_extents  # noqa: E402

DEFAULT_BAND_M = (1.0, 3.0)  # the altitude band of spec §6.1


def measure_sources(sources_path: Path, *, band_m: tuple[float, float] = DEFAULT_BAND_M) -> dict:
    sources = json.loads(Path(sources_path).read_text())
    downloads = Path(sources["downloads_dir"])
    meshes, models = [], []

    def whole(asset: dict, name: str, file: str, nodes: list) -> None:
        m = measure(combine(nodes, name=name), band_m=band_m)
        models.append({"asset": asset["id"], "source": asset["source"], "model": name, "file": file,
                       "nodes": [n.name for n in nodes], **m})

    for asset in sources["assets"]:
        if asset["kind"] == "model_pack":
            for model in asset["models"]:
                whole(asset, model["name"], model["path"], load_meshes(downloads / asset["source"] / asset["id"] / model["path"]))
            continue
        if asset["kind"] != "model":
            continue
        gltf_files = [f for f in asset["files"] if f["path"].endswith(".gltf")]
        for f in gltf_files:
            path = downloads / asset["source"] / asset["id"] / f["path"]
            nodes = load_meshes(path)
            for mesh in nodes:
                m = measure(mesh, band_m=band_m)
                # the two static meshes the import makes of a model with alpha cards (gltf.split_by_opacity): the box of
                # each is what the import's bounds check compares against
                parts = {"solid": part_extents(mesh, opaque=True), "cards": part_extents(mesh, opaque=False)}
                meshes.append({"asset": asset["id"], "source": asset["source"], "file": f["path"], "node": mesh.name,
                               "mesh": mesh.mesh_name, "node_transformed": mesh.transformed, **m, "parts": parts})
            if "targets" in asset.get("use", []):
                whole(asset, asset["id"], f["path"], nodes)
    return {"format": "autofly_ue5_asset_measurements/1", "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": str(sources_path), "band_m": list(band_m), "frame": "glTF: x right, y up, z towards the viewer; metres",
            "meshes": meshes, "models": models}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", type=Path, default=_ROOT / "assets" / "sources.json")
    p.add_argument("--out", type=Path, default=_ROOT / "assets" / "measurements.json")
    p.add_argument("--band", type=float, nargs=2, default=list(DEFAULT_BAND_M), metavar=("LO", "HI"))
    args = p.parse_args(argv)
    report = measure_sources(args.sources, band_m=(args.band[0], args.band[1]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for m in report["meshes"]:
        print(f"{m['asset']:24s} {m['node']:28s} h {m['height_m']:5.2f} m  extent {m['extent_m']}  band r {m['band_radius_m']}  "
              f"offset {m['band_centre_offset_m']}  tris {m['triangles']}")
    for m in report["models"]:
        print(f"{m['asset']:24s} {m['model']:28s} h {m['height_m']:5.2f} m  extent {m['extent_m']}  (whole, {len(m['nodes'])} nodes)")
    print(f"{len(report['meshes'])} meshes and {len(report['models'])} whole models measured into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
