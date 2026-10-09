"""Assemble final_results/ (plan 6 step 7) from a manifest, and check that its README's every link resolves.

    env -u PYTHONPATH .venv/bin/python scripts/build_final_results.py build --manifest <manifest.json> \\
        [--out final_results] [--source-root .] [--force]
    env -u PYTHONPATH .venv/bin/python scripts/build_final_results.py check [--dir final_results] [--allow-outside]

The manifest says what goes where and why, so the folder is rebuilt the same way after every new figure or record:

    {"version": 1, "files": [
      {"source": "runs/report/figures/composition.png", "dest": "figures/composition.png",
       "caption": "Kept episodes per scene, category and target"},
      {"source": "runs/report/figures/trajectories_*.png", "dest": "figures/", "caption": "Every kept trajectory"},
      {"source": "docs/gates/m3_gate.json", "dest": "records/", "caption": "M3's gate"},
      {"source": "runs/report/videos/s02_montage.mp4", "dest": "videos/", "caption": "...", "optional": true}]}

- `source`: a file, or a glob of files, relative to --source-root (the repository by default) or absolute. A directory
  is refused: name its files, so the index lists exactly what was meant.
- `dest`: a path inside the folder; ending in "/" it is a directory and each file keeps its name (a glob needs one).
- `caption`: what the file shows, carried into the index for whoever writes the README.
- `optional`: a source that does not exist yet is skipped and listed as skipped, instead of refusing the build.

Every entry is checked before anything is copied (a source missing, a destination outside the folder or written twice,
an unknown key), so a refused build leaves the folder as it was. A destination that already holds the same bytes is left
alone ("unchanged"); one that holds different bytes is refused unless --force ("replaced"): the folder is what a reader
is sent, and a figure is never replaced by accident. index.json, rewritten by each build, lists every file with its
source, caption, size and sha256, what was skipped, and any file in the folder that no manifest entry wrote.

`check` is plan 6's done check for final_results ("every link in its README resolves"): it reads README.md as Markdown
(inline links and images, reference definitions, HTML src/href; not inside code spans or fenced blocks), and every
relative target must exist, an image must be a file, a #fragment into a Markdown file must name one of its headings
(GitHub's anchors), and nothing may point outside the folder (it is sent on its own) unless --allow-outside. External
URLs are counted, not fetched. When index.json exists, every file it lists must still have its sha256: a figure changed
after the build is reported, so the index never describes files that are not there.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import PurePosixPath  # noqa: E402
from urllib.parse import unquote  # noqa: E402

from autofly_ue5.paths import ROOT  # noqa: E402

VERSION = 1
INDEX = "index.json"
README = "README.md"
ENTRY_KEYS = {"source", "dest", "caption", "optional"}
GLOB_CHARS = set("*?[")


class ManifestError(ValueError):
    """A manifest that cannot be built as written."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Copy:
    source: Path
    source_text: str  # as the index records it: relative to the source root where it lies under it
    dest: str  # POSIX, relative to the folder
    caption: str


def _dest_path(dest: str, where: str) -> PurePosixPath:
    if not isinstance(dest, str) or not dest.strip():
        raise ManifestError(f"{where}: dest must be a non-empty relative path")
    path = PurePosixPath(dest)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", dest):
        raise ManifestError(f"{where}: dest {dest!r} must be a relative path inside the folder")
    if ".." in path.parts:
        raise ManifestError(f"{where}: dest {dest!r} must stay inside the folder (no '..')")
    return path


def plan(manifest: dict, source_root: Path) -> tuple[list[Copy], list[dict]]:
    """Every copy the manifest asks for, checked; and the optional entries whose sources do not exist yet."""
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise ManifestError(f"the manifest must be an object with version {VERSION}")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("the manifest's 'files' must be a non-empty list")
    copies: list[Copy] = []
    skipped: list[dict] = []
    for i, entry in enumerate(entries):
        where = f"files[{i}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"{where}: an entry must be an object")
        unknown = set(entry) - ENTRY_KEYS
        if unknown:
            raise ManifestError(f"{where}: unknown key(s) {sorted(unknown)}; an entry holds {sorted(ENTRY_KEYS)}")
        source, caption, optional = entry.get("source"), entry.get("caption"), entry.get("optional", False)
        if not isinstance(source, str) or not source:
            raise ManifestError(f"{where}: source must be a path")
        if not isinstance(caption, str) or not caption.strip():
            raise ManifestError(f"{where} ({source}): caption must say what the file shows")
        if not isinstance(optional, bool):
            raise ManifestError(f"{where} ({source}): optional must be true or false")
        dest = _dest_path(entry.get("dest"), f"{where} ({source})")
        is_dir_dest = str(entry["dest"]).endswith("/")
        pattern = Path(source) if Path(source).is_absolute() else Path(source_root) / source
        is_glob = bool(GLOB_CHARS & set(source))
        if is_glob and not is_dir_dest:
            raise ManifestError(f"{where} ({source}): a glob's dest must be a directory (end with '/')")
        matches = sorted(Path(p) for p in glob.glob(str(pattern))) if is_glob else ([pattern] if pattern.exists() else [])
        if not matches:
            if optional:
                skipped.append({"source": source, "dest": entry["dest"], "caption": caption, "why": "matches no file"})
                continue
            raise ManifestError(f"{where}: {source} matches no file under {source_root}")
        for match in matches:
            if match.is_dir():
                raise ManifestError(f"{where}: {match} is a directory: list its files, or a glob of them")
            target = dest / match.name if is_dir_dest else dest
            if target.as_posix() == INDEX:
                raise ManifestError(f"{where}: {INDEX} is the build's own index; choose another dest")
            try:
                shown = match.resolve().relative_to(Path(source_root).resolve()).as_posix()
            except ValueError:
                shown = str(match)
            copies.append(Copy(source=match, source_text=shown, dest=target.as_posix(), caption=caption.strip()))
    seen: dict[str, Copy] = {}
    for c in copies:
        if c.dest in seen and seen[c.dest].source.resolve() != c.source.resolve():
            raise ManifestError(f"{c.dest} is written twice: from {seen[c.dest].source_text} and {c.source_text}")
        seen[c.dest] = c
    return list(seen.values()), skipped


def build(manifest_path: Path, out: Path, *, source_root: Path = ROOT, force: bool = False) -> dict:
    blob = Path(manifest_path).read_bytes()
    copies, skipped = plan(json.loads(blob), source_root)
    out = Path(out)
    statuses = {}
    conflicts = []
    for c in copies:
        target = out / c.dest
        if target.is_dir():
            raise ManifestError(f"{target} is a directory; {c.source_text} cannot be copied over it")
        if target.exists():
            same = sha256_of(target) == sha256_of(c.source)
            statuses[c.dest] = "unchanged" if same else "replaced"
            if not same and not force:
                conflicts.append(c.dest)
        else:
            statuses[c.dest] = "copied"
    if conflicts:
        raise FileExistsError(f"{len(conflicts)} file(s) in {out} differ from their sources and would be replaced: "
                              f"{conflicts}; nothing was copied (pass --force to replace them)")
    files = []
    for c in copies:
        target = out / c.dest
        if statuses[c.dest] != "unchanged":
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.tmp")
            shutil.copyfile(c.source, tmp)
            os.replace(tmp, target)
        files.append({"dest": c.dest, "source": c.source_text, "caption": c.caption, "bytes": target.stat().st_size,
                      "sha256": sha256_of(target), "status": statuses[c.dest]})
    written = {f["dest"] for f in files} | {INDEX}
    unlisted = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
                      and p.relative_to(out).as_posix() not in written)
    index = {"version": VERSION, "tool": "scripts/build_final_results.py", "built": time.strftime("%Y-%m-%d %H:%M:%S"),
             "manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(blob).hexdigest()},
             "source_root": str(source_root), "files": sorted(files, key=lambda f: f["dest"]), "skipped": skipped,
             "unlisted": unlisted}
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{INDEX}.tmp"
    tmp.write_text(json.dumps(index, indent=2) + "\n")
    os.replace(tmp, out / INDEX)
    return index


# ----------------------------------------------------------------------------------------------------------------------
# check: every link of the README.
# ----------------------------------------------------------------------------------------------------------------------
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
REFERENCE = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*(<[^>]*>|\S+)")
HTML_ATTRIBUTE = re.compile(r"""<[^>]*?\b(src|href)\s*=\s*(["'])(.*?)\2""", re.IGNORECASE)
HTML_ID = re.compile(r"""<[^>]*?\b(?:id|name)\s*=\s*(["'])(.*?)\1""", re.IGNORECASE)
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def _blank_code_spans(line: str) -> str:
    """The line with every inline code span blanked (columns kept), so a link written as code is no link."""
    out, i = list(line), 0
    while i < len(line):
        if line[i] == "`":
            run = len(line[i:]) - len(line[i:].lstrip("`"))
            end = line.find("`" * run, i + run)
            if end == -1:
                i += run
                continue
            for k in range(i, end + run):
                out[k] = " "
            i = end + run
        else:
            i += 1
    return "".join(out)


def _opening_bracket(line: str, close: int) -> int:
    depth = 0
    for k in range(close, -1, -1):
        if line[k] == "]":
            depth += 1
        elif line[k] == "[":
            depth -= 1
            if depth == 0:
                return k
    return -1


def _destination(line: str, start: int) -> str | None:
    """An inline link's destination beginning at `start` (just after '](': '<...>' or a run with balanced parentheses),
    without its title."""
    k = start
    while k < len(line) and line[k] in " \t":
        k += 1
    if k < len(line) and line[k] == "<":
        end = line.find(">", k)
        return None if end == -1 else line[k + 1:end]
    depth, begin = 0, k
    while k < len(line):
        ch = line[k]
        if ch == "\\":
            k += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        elif ch in " \t" and depth == 0:
            break
        k += 1
    return line[begin:k]


def links(text: str) -> list[dict]:
    """Every link target in Markdown text: {"line", "kind" (image, link, definition, html), "target"}."""
    found = []
    fence = None
    for number, raw in enumerate(text.splitlines(), start=1):
        opened = FENCE.match(raw)
        if fence is not None:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence):
                fence = None
            continue
        if opened:
            fence = opened.group(1)
            continue
        line = _blank_code_spans(raw)
        definition = REFERENCE.match(line)
        if definition:
            found.append({"line": number, "kind": "definition", "target": definition.group(2).strip("<>")})
            continue
        i = 0
        while (j := line.find("](", i)) != -1:
            target = _destination(line, j + 2)
            if target is not None:
                bracket = _opening_bracket(line, j)
                kind = "image" if bracket > 0 and line[bracket - 1] == "!" else "link"
                found.append({"line": number, "kind": kind, "target": target})
            i = j + 2
        for match in HTML_ATTRIBUTE.finditer(line):
            found.append({"line": number, "kind": "image" if match.group(1).lower() == "src" else "html",
                          "target": match.group(3)})
    return found


def github_anchor(heading: str) -> str:
    """GitHub's anchor for a heading's text: markup dropped, lower case, punctuation removed (letters, digits, '_' and
    '-' stay), each space a hyphen."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]+>", "", text).replace("`", "")
    text = re.sub(r"(\*\*|\*)(\S(?:.*?\S)?)\1", r"\2", text)
    text = re.sub(r"(?<!\w)(__|_)(\S(?:.*?\S)?)\1(?!\w)", r"\2", text)
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """The anchors a Markdown file offers: its headings' (a repeated one gets -1, -2, ... as on GitHub) and HTML ids."""
    found: set[str] = set()
    counts: dict[str, int] = {}
    fence = None
    for raw in text.splitlines():
        opened = FENCE.match(raw)
        if fence is not None:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence):
                fence = None
            continue
        if opened:
            fence = opened.group(1)
            continue
        heading = HEADING.match(raw)
        if heading:
            slug = github_anchor(heading.group(2))
            n = counts.get(slug, 0)
            counts[slug] = n + 1
            found.add(slug if n == 0 else f"{slug}-{n}")
        found |= {m.group(2) for m in HTML_ID.finditer(raw)}
    return found


def check(folder: Path, *, allow_outside: bool = False) -> dict:
    folder = Path(folder)
    readme = folder / README
    if not readme.is_file():
        return {"readme": str(readme), "ok": False, "checked": 0, "external": 0,
                "broken": [{"line": None, "target": README, "why": "missing"}], "index": None}
    text = readme.read_text()
    root = folder.resolve()
    broken, checked, external = [], 0, 0
    anchor_cache: dict[Path, set[str]] = {}
    for link in links(text):
        target = link["target"]
        why = None
        if not target:
            why = "an empty link"
        elif SCHEME.match(target) or target.startswith("//"):
            external += 1
            continue
        elif target.startswith("/"):
            why = "an absolute path: it resolves nowhere once the folder is moved"
        else:
            checked += 1
            path_part, _, fragment = target.partition("#")
            path_part = unquote(path_part.split("?", 1)[0])
            resolved = (readme.parent / path_part).resolve() if path_part else readme.resolve()
            if not resolved.exists():
                why = "no such file or directory"
            elif not allow_outside and resolved != root and root not in resolved.parents:
                why = "outside the folder, which is sent on its own"
            elif link["kind"] == "image" and not resolved.is_file():
                why = "an image that is not a file"
            elif fragment and resolved.is_file() and resolved.suffix.lower() == ".md":
                if resolved not in anchor_cache:
                    anchor_cache[resolved] = anchors(resolved.read_text())
                if unquote(fragment) not in anchor_cache[resolved]:
                    why = f"no heading or id #{fragment} in {resolved.name}"
        if why is not None:
            broken.append({"line": link["line"], "kind": link["kind"], "target": target, "why": why})
    index_report = None
    if (folder / INDEX).is_file():
        index = json.loads((folder / INDEX).read_text())
        missing = [f["dest"] for f in index.get("files", []) if not (folder / f["dest"]).is_file()]
        changed = [f["dest"] for f in index.get("files", []) if (folder / f["dest"]).is_file()
                   and sha256_of(folder / f["dest"]) != f.get("sha256")]
        index_report = {"files": len(index.get("files", [])), "missing": missing, "changed": changed}
    ok = not broken and (index_report is None or not (index_report["missing"] or index_report["changed"]))
    return {"readme": str(readme), "ok": ok, "checked": checked, "external": external, "broken": broken,
            "index": index_report}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="copy the manifest's files into the folder and index them")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--out", type=Path, default=ROOT / "final_results")
    b.add_argument("--source-root", type=Path, default=ROOT, help="what relative sources are relative to")
    b.add_argument("--force", action="store_true", help="replace files whose bytes differ from their sources")
    c = sub.add_parser("check", help="every link and image of the folder's README.md must resolve")
    c.add_argument("--dir", type=Path, default=ROOT / "final_results")
    c.add_argument("--allow-outside", action="store_true", help="accept links that leave the folder")
    args = p.parse_args(argv)
    if args.command == "check":
        report = check(args.dir, allow_outside=args.allow_outside)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    try:
        index = build(args.manifest, args.out, source_root=args.source_root, force=args.force)
    except (ManifestError, FileExistsError, json.JSONDecodeError) as err:
        print(f"refusing to build: {err}", file=sys.stderr)
        return 2
    counts: dict[str, int] = {}
    for f in index["files"]:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    print(json.dumps({"out": str(args.out), "files": counts, "skipped": index["skipped"],
                      "unlisted": index["unlisted"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
