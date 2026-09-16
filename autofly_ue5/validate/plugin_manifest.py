"""Verify an unzipped Project AirSim plugin release against its build-manifest.json.

Usage: env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.plugin_manifest downloads/plugin_ue57_1.0.1
"""

import hashlib
import json
import sys
from pathlib import Path

REQUIRED_DIRS = (
    "Plugins/ProjectAirSim/Source",
    "Plugins/ProjectAirSim/SimLibs",
    "Plugins/Drone",
    "Plugins/Rover",
)
EXPECTED_SOURCE_PREFIX = "0975545"
EXPECTED_UNREAL_VERSION = "5.7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict:
    manifest = json.loads((root / "build-manifest.json").read_text())
    missing: list[str] = []
    mismatched: list[str] = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
        elif _sha256(path) != entry["sha256"]:
            mismatched.append(entry["path"])
    missing_dirs = [d for d in REQUIRED_DIRS if not (root / d).is_dir()]
    plugins = root / "Plugins"
    binaries = sorted(str(p.relative_to(root)) for p in plugins.rglob("Binaries") if p.is_dir()) if plugins.is_dir() else []
    source_sha = str(manifest.get("source_sha", ""))
    unreal_version = str(manifest.get("unreal_version", ""))
    return {
        "files": len(manifest["files"]),
        "missing": missing[:50],
        "mismatched": mismatched[:50],
        "missing_dirs": missing_dirs,
        "binaries_dirs": binaries,
        "source_sha": source_sha,
        "unreal_version": unreal_version,
        "pass": not missing
        and not mismatched
        and not missing_dirs
        and source_sha.startswith(EXPECTED_SOURCE_PREFIX)
        and unreal_version == EXPECTED_UNREAL_VERSION,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    report = verify_manifest(Path(args[0]))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
