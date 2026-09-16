"""Fingerprint a packaged Development Linux binary and its content archive (R17: manifest provenance).

The manifest describes the TREE that produced the binary, not a commit -- packaging always happens
before the commit that records its manifest, so `built_from_git_head` is expected to differ from HEAD
by the time this file is read; that is not a bug (see the `description` field). Comparing `git_head`
to HEAD is circular and is deliberately not attempted here.

env -u PYTHONPATH .venv/bin/python scripts/make_package_manifest.py --level-spec runs/levels/s01.level.json --level-verify runs/levels/s01.verify.json --out runs/package/package_manifest.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from autofly_ue5.paths import PACKAGED_BINARY, ROOT, UE_PROJECT_DIR

DESCRIPTION = (
    "Fingerprints the packaged Development Linux binary and content archive (pak) actually produced by "
    "scripts/package_sim.sh, plus the git tree that produced them. built_from_git_head is the repo's HEAD "
    "at package time, which routinely precedes the commit that later records this file -- packaging "
    "happens before committing, not after. working_tree_dirty says whether uncommitted changes were "
    "present in that tree (e.g. an in-progress fix not yet committed). Do not compare built_from_git_head "
    "to the current HEAD to judge freshness; that comparison is circular. Gate on binary.sha256 instead."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(cwd: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def working_tree_dirty(cwd: Path) -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True).stdout
    return bool(out.strip())


def build_manifest(binary: Path, pak: Path, level_spec: Path, level_verify: Path, repo_root: Path) -> dict:
    verify = json.loads(level_verify.read_text())
    engine_build = json.loads((repo_root / "engine" / "Engine" / "Build" / "Build.version").read_text())
    return {
        "description": DESCRIPTION,
        "binary": {"path": str(binary.relative_to(repo_root)), "sha256": sha256_file(binary)},
        "paks": {str(pak.relative_to(repo_root)): sha256_file(pak)},
        "built_from_git_head": git_head(repo_root),
        "working_tree_dirty": working_tree_dirty(repo_root),
        "level_spec_sha256": sha256_file(level_spec),
        "level_verify": {"map_path": verify["map_path"], "pass": verify["pass"], "obstacles": verify["obstacles"],
                         "worst_location_error_cm": verify["worst_location_error_cm"], "game_mode": verify["game_mode"]},
        "engine_build_version": engine_build,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=PACKAGED_BINARY)
    parser.add_argument("--pak", type=Path, default=UE_PROJECT_DIR / "Packaged" / "Development" / "Linux" / "Blocks" / "Content" / "Paks" / "Blocks-Linux.pak")
    parser.add_argument("--level-spec", type=Path, required=True)
    parser.add_argument("--level-verify", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(args.binary, args.pak, args.level_spec, args.level_verify, ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "description"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
