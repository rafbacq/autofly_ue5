"""Build each mesh obstacle's outline field (autofly_ue5/scenes/footprint_field.py) into assets/footprints/.

    env -u PYTHONPATH nice -n 19 .venv/bin/python scripts/footprint_fields.py           # every mesh obstacle
    env -u PYTHONPATH nice -n 19 .venv/bin/python scripts/footprint_fields.py --only quiver_tree_01 boulder_01
    env -u PYTHONPATH .venv/bin/python scripts/footprint_fields.py --check               # manifest vs files, sources

A mesh obstacle is a registry asset with a radius profile and the obstacle role: the models the footprint rule covers
(expert/footprint_contact.py). Each gets <key>.npz, built from the same glTF triangles the live depth checks are cast
against (cut at the ground: footprint_field.clip_above_ground) (validate/obstacles.MeshLibrary: the registry names the file and node, sources.json the download, its sha256),
and an entry in manifest.json: the file's sha256, its content's (what a run's identity names), the source glTF's and its
buffers' sha256 (as assets/sources.json recorded them at fetch; a download that no longer matches is refused), the
triangle counts (opaque wood or rock, and leaf cards) and how far the outline reaches from the pivot. Building is
deterministic: the same inputs write the same bytes. --only rebuilds the named keys and keeps every other entry.

--check reads everything back: every mesh obstacle has an entry of this field version; each file's bytes and content
are the manifest's; each source is the one sources.json recorded (and, where the download is on disk, its bytes); and
each outline reaches no farther from the pivot than the registry's widest profile slice, the footprint circle every
layout was generated with, so the outline lies inside the circle and the layouts stay conservative. Exit 1 on any
problem, each printed.
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

import numpy as np  # noqa: E402

from autofly_ue5.expert.footprint_contact import is_measured_obstacle  # noqa: E402
from autofly_ue5.paths import ASSET_FOOTPRINTS, ASSET_MEASUREMENTS, ASSET_REGISTRY, ASSET_SOURCES  # noqa: E402
from autofly_ue5.scenes.footprint_field import (  # noqa: E402
    CELL_M,
    FIELD_VERSION,
    FRAME,
    MANIFEST,
    MANIFEST_FORMAT,
    MARGIN_M,
    SUPERSAMPLE,
    FootprintField,
    build_field,
    file_sha256,
    outline_triangles,
    triangle_corners,
)

REACH_TOLERANCE_M = 0.002  # the registry's profile is rounded to 0.1 mm; the outline may not pass the circle by more


def mesh_obstacles(registry: dict) -> list[str]:
    """The registry keys the footprint rule covers and a field is built for, by the rule's own definition
    (footprint_contact.is_measured_obstacle): measured models (a radius profile) in the obstacle role. Targets and engine
    primitives are not."""
    return sorted(key for key, entry in registry["assets"].items()
                  if is_measured_obstacle(entry.get("role"), entry.get("radius_profile_m")))


class SourceError(Exception):
    """A model's source is not the one assets/sources.json recorded."""


def source_record(key: str, registry: dict, sources: dict, measurements: dict, *, verify_bytes: bool) -> dict:
    """Where `key`'s triangles come from: the source, id, node and glTF file, with the sha256 sources.json recorded for
    the glTF and each buffer it names. `verify_bytes`: the files must be on disk with those bytes."""
    source = registry["assets"][key].get("source")
    if not isinstance(source, dict):
        raise SourceError(f"{key}: the registry names no glTF source")
    files = {(m["asset"], m["node"]): m["file"] for m in measurements["meshes"]}
    gltf_name = files.get((source["id"], source["node"]))
    if gltf_name is None:
        raise SourceError(f"{key}: assets/measurements.json has no node {source['node']!r} of {source['id']!r}")
    recorded = {f["path"]: f["sha256"] for a in sources.get("assets", []) if a["id"] == source["id"]
                and a["source"] == source["source"] for f in a["files"]}
    folder = Path(sources["downloads_dir"]) / source["source"] / source["id"]
    gltf_path = folder / gltf_name
    if gltf_name not in recorded:
        raise SourceError(f"{key}: assets/sources.json records no sha256 for {gltf_name}")
    buffers = {}
    if gltf_path.is_file():
        for buffer in json.loads(gltf_path.read_text()).get("buffers", []):
            uri = buffer.get("uri")
            if uri is not None and not uri.startswith("data:"):
                if uri not in recorded:
                    raise SourceError(f"{key}: assets/sources.json records no sha256 for {uri}")
                buffers[uri] = recorded[uri]
    elif verify_bytes:
        raise SourceError(f"{key}: {gltf_path} is not on disk")
    if verify_bytes:
        for name, expected in ((gltf_name, recorded[gltf_name]), *buffers.items()):
            actual = file_sha256(folder / name)
            if actual != expected:
                raise SourceError(f"{key}: {folder / name} has sha256 {actual}, assets/sources.json recorded {expected}")
    return {"source": source["source"], "id": source["id"], "node": source["node"], "file": gltf_name,
            "gltf_sha256": recorded[gltf_name], "buffers": buffers}


def build_one(key: str, out_dir: Path, *, registry_path: Path, sources_path: Path, measurements_path: Path,
              registry: dict, sources: dict, measurements: dict) -> dict:
    """Build `key`'s field into out_dir/<key>.npz and return its manifest entry."""
    from autofly_ue5.validate.obstacles import MeshLibrary

    source = source_record(key, registry, sources, measurements, verify_bytes=True)
    t0 = time.monotonic()
    triangles = MeshLibrary.from_files({key}, registry_path=registry_path, sources_path=sources_path,
                                       measurements_path=measurements_path)[key]
    corners = triangle_corners(triangles)
    projected = outline_triangles(triangles)
    field = build_field(projected)
    path = out_dir / f"{key}.npz"
    field.save(path)
    flat = projected.reshape(-1, 2)
    opaque = int(np.count_nonzero(triangles.opaque))
    entry = {
        "file": path.name,
        "npz_sha256": file_sha256(path),
        "field_sha256": field.content_sha256(),
        "field_version": FIELD_VERSION,
        "grid": list(field.values.shape),
        "origin": list(field.origin),
        "source": source,
        "triangles": {"all": int(triangles.count), "opaque": opaque, "cards": int(triangles.count) - opaque,
                      "below_ground": int(np.count_nonzero(corners[:, :, 1].max(axis=1) < 0.0)),
                      "cut_at_ground": int(np.count_nonzero((corners[:, :, 1].min(axis=1) < 0.0)
                                                            & (corners[:, :, 1].max(axis=1) >= 0.0)))},
        "projection_m": {"x": [round(float(flat[:, 0].min()), 6), round(float(flat[:, 0].max()), 6)],
                         "z": [round(float(flat[:, 1].min()), 6), round(float(flat[:, 1].max()), 6)]},
        "pivot_reach_m": round(float(np.hypot(flat[:, 0], flat[:, 1]).max()), 6),
    }
    print(f"{key}: {triangles.count} triangles ({opaque} opaque), grid {field.values.shape[1]} x "
          f"{field.values.shape[0]}, {path.stat().st_size / 1e3:.0f} kB, {time.monotonic() - t0:.1f} s", flush=True)
    return entry


def write_manifest(out_dir: Path, fields: dict) -> None:
    manifest = {"format": MANIFEST_FORMAT, "field_version": FIELD_VERSION, "frame": FRAME, "cell_m": CELL_M,
                "margin_m": MARGIN_M, "supersample": SUPERSAMPLE,
                "fields": {key: fields[key] for key in sorted(fields)}}
    path = out_dir / MANIFEST
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n")
    tmp.replace(path)


def check(out_dir: Path, *, registry: dict, sources: dict, measurements: dict) -> list[str]:
    """Every way the manifest, the files, the sources and the registry no longer agree (see the module's doc)."""
    problems = []
    try:
        manifest = json.loads((out_dir / MANIFEST).read_text())
    except (OSError, ValueError) as err:
        return [f"{out_dir / MANIFEST}: unreadable ({err})"]
    if manifest.get("format") != MANIFEST_FORMAT:
        problems.append(f"{MANIFEST}: format {manifest.get('format')!r}, expected {MANIFEST_FORMAT!r}")
    entries = manifest.get("fields", {})
    for key in mesh_obstacles(registry):
        if key not in entries:
            problems.append(f"{key}: a mesh obstacle without a field")
    for key, entry in sorted(entries.items()):
        if entry.get("field_version") != FIELD_VERSION:
            problems.append(f"{key}: field version {entry.get('field_version')}, this code builds {FIELD_VERSION}")
        path = out_dir / entry.get("file", f"{key}.npz")
        if not path.is_file():
            problems.append(f"{key}: {path.name} is missing")
            continue
        if file_sha256(path) != entry.get("npz_sha256"):
            problems.append(f"{key}: {path.name}'s bytes are not the manifest's (npz_sha256)")
        try:
            field = FootprintField.load(path)
        except Exception as err:  # a damaged archive reads as a problem, not a crash of the check
            problems.append(f"{key}: {path.name} does not load ({type(err).__name__}: {err})")
            continue
        if field.content_sha256() != entry.get("field_sha256"):
            problems.append(f"{key}: {path.name}'s content is not the manifest's (field_sha256)")
        if key not in registry["assets"]:
            problems.append(f"{key}: not in the registry")
            continue
        try:
            disk = (Path(sources["downloads_dir"]) / entry["source"]["source"] / entry["source"]["id"]
                    / entry["source"]["file"]).is_file()
            expected = source_record(key, registry, sources, measurements, verify_bytes=disk)
        except (SourceError, KeyError) as err:
            problems.append(f"{key}: {err}")
        else:
            if expected != entry.get("source"):
                problems.append(f"{key}: its source is not the one the registry and sources.json name now")
        profile = [r for r in registry["assets"][key].get("radius_profile_m") or [] if r is not None]
        if profile and entry.get("pivot_reach_m", 0.0) > max(profile) + REACH_TOLERANCE_M:
            problems.append(f"{key}: the outline reaches {entry['pivot_reach_m']:.4f} m from the pivot, past the "
                            f"registry's footprint circle ({max(profile):.4f} m): the layouts' circles would not hold it")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, metavar="KEY", help="rebuild these keys; keep every other entry")
    ap.add_argument("--check", action="store_true", help="verify the manifest against the files, sources and registry")
    ap.add_argument("--out-dir", type=Path, default=ASSET_FOOTPRINTS)
    ap.add_argument("--registry", type=Path, default=ASSET_REGISTRY)
    ap.add_argument("--sources", type=Path, default=ASSET_SOURCES)
    ap.add_argument("--measurements", type=Path, default=ASSET_MEASUREMENTS)
    args = ap.parse_args(argv)
    registry = json.loads(args.registry.read_text())
    sources = json.loads(args.sources.read_text())
    measurements = json.loads(args.measurements.read_text())
    if args.check:
        problems = check(args.out_dir, registry=registry, sources=sources, measurements=measurements)
        for problem in problems:
            print(f"PROBLEM {problem}")
        print(f"{len(problems)} problem(s) in {args.out_dir / MANIFEST}" if problems else
              f"{args.out_dir / MANIFEST}: every field matches its file, source and registry entry")
        return 1 if problems else 0
    covered = mesh_obstacles(registry)
    keys = covered if args.only is None else sorted(set(args.only))
    unknown = [key for key in keys if key not in covered]
    if unknown:
        print(f"refusing: {', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not a mesh obstacle of "
              f"{args.registry} (a radius profile and the obstacle role)", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = {}
    if args.only is not None and (args.out_dir / MANIFEST).is_file():
        fields = dict(json.loads((args.out_dir / MANIFEST).read_text()).get("fields", {}))
    t0 = time.monotonic()
    for key in keys:
        try:
            fields[key] = build_one(key, args.out_dir, registry_path=args.registry, sources_path=args.sources,
                                    measurements_path=args.measurements, registry=registry, sources=sources,
                                    measurements=measurements)
        except SourceError as err:
            print(f"refusing: {err}", file=sys.stderr)
            return 2
    write_manifest(args.out_dir, fields)
    total = sum((args.out_dir / fields[key]["file"]).stat().st_size for key in fields)
    print(f"{len(keys)} field(s) built in {time.monotonic() - t0:.1f} s; {len(fields)} in {args.out_dir / MANIFEST}, "
          f"{total / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
