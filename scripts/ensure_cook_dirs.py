"""Keep the runtime-spawned content in the cook (plan 5 C6): ue_project/Config/DefaultGame.ini's +DirectoriesToAlwaysCook.

    env -u PYTHONPATH .venv/bin/python scripts/ensure_cook_dirs.py           # add what is missing (idempotent)
    env -u PYTHONPATH python3 scripts/ensure_cook_dirs.py --check            # scripts/package_sim.sh: refuse if missing

A packaged build cooks the maps it is given (package_sim.sh's -map=) and whatever they reference, plus the directories
listed as always cooked. Targets and the runtime colours are referenced by no level: they are spawned by short name and
painted by path at runtime, so a build that does not cook their directories spawns nothing ("There were no objects with
name ... found in the Registry") or leaves the material unloaded. Nothing in the repository owns DefaultGame.ini
(ue_project/ is git-ignored; scripts/create_ue_project.sh copies Blocks' Config once, at init; the MapsToCook lines were
added by hand), so this script adds the entries itself: /Game/AutoFly/Targets (every target imports under it,
scripts/import_assets.py --targets) and /Game/AutoFly/Materials (the runtime colours live in its Runtime folder), plus the
folder of any registry target or base material outside them. A second run changes nothing.

--check writes nothing: it exits 1 when a directory the registry spawns from at runtime (a role "target" asset, a
"base_material") is not cooked, so packaging is refused before a 30-minute cook produces a binary without them. Before
any target is imported it requires nothing new. Stdlib only, so the packaging script can run it with the system python3.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTION = "[/Script/UnrealEd.ProjectPackagingSettings]"
ENTRY = re.compile(r'^([+\-.!]?)DirectoriesToAlwaysCook=\(Path="([^"]*)"\)\s*$')
# What every packaged build should cook for runtime spawns, whether or not anything is imported there yet.
RUNTIME_ROOTS = ("/Game/AutoFly/Targets", "/Game/AutoFly/Materials")


def cooked_directories(text: str) -> list[str]:
    """The always-cooked directories the packaging settings section lists, after UE's array operators: "+" (or none, or
    ".") adds, "-" removes, "!" clears."""
    dirs: list[str] = []
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == SECTION
            continue
        match = ENTRY.match(stripped) if in_section else None
        if match is None:
            continue
        op, path = match.groups()
        if op == "!":
            dirs = []
        elif op == "-":
            dirs = [d for d in dirs if d != path]
        elif path not in dirs:
            dirs.append(path)
    return dirs


def covered(directory: str, cooked: list[str]) -> bool:
    return any(directory == c or directory.startswith(c.rstrip("/") + "/") for c in cooked)


def required_directories(registry: dict) -> list[str]:
    """The folders the registry spawns from at runtime: imported targets and base materials under /Game."""
    out = set()
    for entry in registry.get("assets", {}).values():
        if entry.get("role") == "target" and entry.get("ue_path", "").startswith("/Game/"):
            out.add(entry["ue_path"].rsplit("/", 1)[0])
    for entry in registry.get("materials", {}).values():
        if entry.get("kind") == "base_material" and entry.get("ue_path", "").startswith("/Game/"):
            out.add(entry["ue_path"].rsplit("/", 1)[0])
    return sorted(out)


def ensure(text: str, wanted: list[str]) -> tuple[str, list[str]]:
    """`text` with an entry added for each wanted directory not yet covered (the runtime root it falls under, if any),
    after the section's last DirectoriesToAlwaysCook line; and what was added."""
    cooked = cooked_directories(text)
    added: list[str] = []
    for directory in wanted:
        if covered(directory, cooked + added):
            continue
        root = next((r for r in RUNTIME_ROOTS if covered(directory, [r])), directory)
        if root not in added:
            added.append(root)
    if not added:
        return text, []
    new = [f'+DirectoriesToAlwaysCook=(Path="{d}")\n' for d in added]
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    start = next((i for i, ln in enumerate(lines) if ln.strip() == SECTION), None)
    if start is None:
        return "".join(lines + (["\n"] if lines and lines[-1].strip() else []) + [SECTION + "\n"] + new), added
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
    entries = [i for i in range(start + 1, end) if ENTRY.match(lines[i].strip())]
    if entries:
        at = entries[-1] + 1
    else:
        at = end
        while at > start + 1 and not lines[at - 1].strip():
            at -= 1
    return "".join(lines[:at] + new + lines[at:]), added


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ini", type=Path, default=ROOT / "ue_project" / "Config" / "DefaultGame.ini")
    p.add_argument("--registry", type=Path, default=ROOT / "assets" / "registry.json")
    p.add_argument("--check", action="store_true", help="write nothing; exit 1 if a runtime-spawned directory is not cooked")
    args = p.parse_args(argv)
    text = args.ini.read_text()
    required = required_directories(json.loads(args.registry.read_text()))
    if args.check:
        cooked = cooked_directories(text)
        missing = [d for d in required if not covered(d, cooked)]
        if missing:
            print(f"{args.ini}: not cooked, though the registry spawns from them at runtime: {missing}. Run "
                  f"`env -u PYTHONPATH .venv/bin/python scripts/ensure_cook_dirs.py` before packaging.", file=sys.stderr)
            return 1
        print(f"cook list covers the {len(required)} runtime-spawn folders of the registry")
        return 0
    new, added = ensure(text, list(RUNTIME_ROOTS) + required)
    if not added:
        print(f"{args.ini}: already cooks {list(RUNTIME_ROOTS)} and every runtime-spawn folder; unchanged")
        return 0
    tmp = args.ini.with_name(args.ini.name + ".tmp")
    tmp.write_text(new)
    tmp.replace(args.ini)
    print(f"{args.ini}: added +DirectoriesToAlwaysCook for {added}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
