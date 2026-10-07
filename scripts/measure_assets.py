"""Measure every downloaded model (assets/sources.json) for the registry: height, extents, flight-band footprint.

    env -u PYTHONPATH .venv/bin/python scripts/measure_assets.py [--sources assets/sources.json] [--out assets/measurements.json]

One entry per glTF node (a Poly Haven set of variants is several), from `autofly_ue5/scenes/gltf.py`, in the mesh's own
frame (glTF: y up, metres). The editor import (plan 5, C3) checks each static mesh's bounds against `extent_m` and takes
`band_radius_m` -- the farthest the surface reaches from the pivot between 1 and 3 m -- as the registry's footprint
circle, with `band_centre_offset_m` saying how far off the pivot that mass sits.
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

from autofly_ue5.scenes.gltf import load_meshes, measure  # noqa: E402

DEFAULT_BAND_M = (1.0, 3.0)  # the altitude band of spec §6.1


def measure_sources(sources_path: Path, *, band_m: tuple[float, float] = DEFAULT_BAND_M) -> dict:
    sources = json.loads(Path(sources_path).read_text())
    downloads = Path(sources["downloads_dir"])
    meshes = []
    for asset in sources["assets"]:
        if asset["kind"] != "model":
            continue
        gltf_files = [f for f in asset["files"] if f["path"].endswith(".gltf")]
        for f in gltf_files:
            path = downloads / asset["source"] / asset["id"] / f["path"]
            for mesh in load_meshes(path):
                m = measure(mesh, band_m=band_m)
                meshes.append({"asset": asset["id"], "source": asset["source"], "file": f["path"], "node": mesh.name,
                               "mesh": mesh.mesh_name, "node_transformed": mesh.transformed, **m})
    return {"format": "autofly_ue5_asset_measurements/1", "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": str(sources_path), "band_m": list(band_m), "frame": "glTF: x right, y up, z towards the viewer; metres",
            "meshes": meshes}


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
    print(f"{len(report['meshes'])} meshes measured into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
