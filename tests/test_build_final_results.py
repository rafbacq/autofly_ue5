"""scripts/build_final_results.py assembles final_results/ from a manifest (source -> destination, with a caption),
never replacing a different file without --force, indexes what it copied with sha256; `check` fails on any README
link or image that does not resolve (plan 6's done check for final_results). Every test builds in tmp_path."""

from __future__ import annotations

import hashlib
import json

import pytest


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(tmp_path):
    src = tmp_path / "src"
    for rel, body in {"figures/composition.png": b"png-a", "figures/trajectories_s01.png": b"png-s01",
                      "figures/trajectories_s02.png": b"png-s02", "docs/final_readme.md": b"# Results\n",
                      "records/m3_gate.json": b'{"pass": true}\n', "videos/montage.mp4": b"mp4"}.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(body)
    return src


def _manifest(tmp_path, files, version=1):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": version, "files": files}))
    return path


FILES = [
    {"source": "docs/final_readme.md", "dest": "README.md", "caption": "the report"},
    {"source": "figures/composition.png", "dest": "figures/composition.png", "caption": "what the dataset holds"},
    {"source": "figures/trajectories_*.png", "dest": "figures/", "caption": "every kept trajectory, per scene"},
    {"source": "records/m3_gate.json", "dest": "records/", "caption": "M3's gate"},
    {"source": "videos/missing.mp4", "dest": "videos/missing.mp4", "caption": "not made yet", "optional": True},
]


def _build(tmp_path, files=FILES, *extra):
    from scripts import build_final_results

    out = tmp_path / "final_results"
    code = build_final_results.main(["build", "--manifest", str(_manifest(tmp_path, files)), "--out", str(out),
                                     "--source-root", str(tmp_path / "src"), *extra])
    return code, out


def test_a_manifest_s_files_and_globs_are_copied_and_indexed_with_their_sha256(tmp_path):
    src = _sources(tmp_path)
    code, out = _build(tmp_path)
    assert code == 0
    index = json.loads((out / "index.json").read_text())
    by_dest = {f["dest"]: f for f in index["files"]}
    assert sorted(by_dest) == ["README.md", "figures/composition.png", "figures/trajectories_s01.png",
                               "figures/trajectories_s02.png", "records/m3_gate.json"]
    for dest, entry in by_dest.items():
        assert (out / dest).read_bytes() == (src / entry["source"]).read_bytes()
        assert entry["sha256"] == _sha(out / dest) and entry["bytes"] == (out / dest).stat().st_size
        assert entry["status"] == "copied" and entry["caption"]
    assert by_dest["figures/trajectories_s02.png"]["caption"] == "every kept trajectory, per scene"
    assert [s["source"] for s in index["skipped"]] == ["videos/missing.mp4"]
    assert index["manifest"]["sha256"] == _sha(tmp_path / "manifest.json")
    code, out = _build(tmp_path)
    assert code == 0 and {f["status"] for f in json.loads((out / "index.json").read_text())["files"]} == {"unchanged"}


def test_a_different_existing_file_is_refused_before_anything_is_copied_unless_forced(tmp_path):
    _sources(tmp_path)
    out = tmp_path / "final_results"
    (out / "figures").mkdir(parents=True)
    (out / "figures" / "composition.png").write_bytes(b"someone else's figure")
    code, out = _build(tmp_path)
    assert code == 2
    assert not (out / "README.md").exists() and not (out / "index.json").exists(), "a refused build copies nothing"
    assert (out / "figures" / "composition.png").read_bytes() == b"someone else's figure"
    code, out = _build(tmp_path, FILES, "--force")
    assert code == 0 and (out / "figures" / "composition.png").read_bytes() == b"png-a"
    statuses = {f["dest"]: f["status"] for f in json.loads((out / "index.json").read_text())["files"]}
    assert statuses["figures/composition.png"] == "replaced" and statuses["README.md"] == "copied"


@pytest.mark.parametrize("entry, why", [
    ({"source": "figures/composition.png", "dest": "../escape.png", "caption": "c"}, "inside"),
    ({"source": "figures/composition.png", "dest": "/abs/escape.png", "caption": "c"}, "relative"),
    ({"source": "figures/trajectories_*.png", "dest": "figures/one.png", "caption": "c"}, "directory"),
    ({"source": "videos/missing.mp4", "dest": "videos/m.mp4", "caption": "c"}, "no file"),
    ({"source": "figures", "dest": "figures/", "caption": "c"}, "directory"),
    ({"source": "figures/composition.png", "dest": "index.json", "caption": "c"}, "index.json"),
    ({"source": "figures/composition.png", "dest": "x.png", "caption": ""}, "caption"),
    ({"source": "figures/composition.png", "dest": "x.png", "caption": "c", "captoin": "typo"}, "captoin"),
    ({"source": "records/m3_gate.json", "dest": "figures/composition.png", "caption": "c"}, "twice"),
])
def test_a_manifest_that_cannot_be_built_as_written_is_refused_and_nothing_is_copied(tmp_path, capsys, entry, why):
    _sources(tmp_path)
    code, out = _build(tmp_path, [FILES[1], entry])
    assert code == 2 and why in capsys.readouterr().err
    assert not out.exists() or not any(out.rglob("*")), "nothing copied"


GOOD_README = """# Results

![composition](figures/composition.png)
See [the records](records/m3_gate.json "M3's gate") and [the videos](videos/), [a note](docs/notes.md#a-heading),
[numbers](#numbers-worth-knowing), [the paper](https://arxiv.org/abs/2602.09657) and <img src="figures/composition.png">.
Not links: `[x](missing.png)` in code, and

```
![broken](nowhere.png)
```

[ref]: figures/composition.png

## Numbers worth knowing
"""


def _folder(tmp_path, readme: str):
    out = tmp_path / "final_results"
    for rel in ("figures/composition.png", "records/m3_gate.json", "videos/montage.mp4"):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(b"x")
    (out / "docs").mkdir(exist_ok=True)
    (out / "docs" / "notes.md").write_text("# Notes\n\n## A heading\n")
    (out / "README.md").write_text(readme)
    (tmp_path / "outside.md").write_text("# Outside\n")
    return out


def test_check_passes_a_readme_whose_every_link_and_image_resolves(tmp_path, capsys):
    from scripts import build_final_results

    out = _folder(tmp_path, GOOD_README)
    assert build_final_results.main(["check", "--dir", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True and report["broken"] == [] and report["external"] == 1
    assert report["checked"] == 7, "the image, three links, the README's own anchor, the html image, the definition"


def test_check_fails_on_each_link_that_does_not_resolve_and_names_its_line(tmp_path, capsys):
    from scripts import build_final_results

    readme = GOOD_README + ("![gone](figures/gone.png)\n[missing](records/none.json)\n[out](../outside.md)\n"
                            "[bad anchor](docs/notes.md#no-such-heading)\n[abs](/etc/hosts)\n![dir](figures)\n"
                            "[ref2]: videos/none.mp4\n")
    out = _folder(tmp_path, readme)
    assert build_final_results.main(["check", "--dir", str(out)]) == 1
    report = json.loads(capsys.readouterr().out)
    broken = {b["target"]: b for b in report["broken"]}
    assert set(broken) == {"figures/gone.png", "records/none.json", "../outside.md", "docs/notes.md#no-such-heading",
                           "/etc/hosts", "figures", "videos/none.mp4"}
    first = GOOD_README.count("\n") + 1
    assert broken["figures/gone.png"]["line"] == first and broken["/etc/hosts"]["line"] == first + 4
    assert build_final_results.main(["check", "--dir", str(out), "--allow-outside"]) == 1
    assert "../outside.md" not in {b["target"] for b in json.loads(capsys.readouterr().out)["broken"]}


def test_check_fails_when_an_indexed_file_changed_after_the_build(tmp_path, capsys):
    from scripts import build_final_results

    _sources(tmp_path)
    code, out = _build(tmp_path)
    assert code == 0
    capsys.readouterr()
    (out / "README.md").write_text("# Results\n\n![composition](figures/composition.png)\n")
    assert build_final_results.main(["check", "--dir", str(out)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["index"]["changed"] == ["README.md"] and report["broken"] == []
