"""scripts/footprint_fields.py builds every mesh obstacle's outline field into assets/footprints/ (one .npz per registry
key and a manifest naming each file's sha256, its content's, its source glTF's and the triangle counts), and --check holds
the manifest to the files, the sources and the registry. Run here on synthetic models in tmp_path, never on assets/."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_footprint_field import l_prism, leaning_tree_parts, synthetic_library


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, Path]:
    """Two mesh obstacles (a leaning tree with cards, an L prism) and an engine primitive, as the registry, the sources
    and the measurements files describe them."""
    paths = synthetic_library(tmp_path, {"lean": leaning_tree_parts(), "l_prism": [(*l_prism(), True)]})
    registry = json.loads(paths["registry_path"].read_text())
    registry["assets"]["lean"].update(role="obstacle", radius_profile_m=[0.07, 0.07, 1.58])
    registry["assets"]["l_prism"].update(role="obstacle", radius_profile_m=[2.07, 2.07])
    registry["assets"]["cylinder"] = {"role": "obstacle", "radius_profile_m": None}
    paths["registry_path"].write_text(json.dumps(registry))
    downloads = Path(json.loads(paths["sources_path"].read_text())["downloads_dir"])
    sources = {"downloads_dir": str(downloads), "assets": [
        {"id": key, "source": "synthetic",
         "files": [{"path": p.name, "sha256": _sha(p)} for p in sorted((downloads / "synthetic" / key).iterdir())]}
        for key in ("lean", "l_prism")]}
    paths["sources_path"].write_text(json.dumps(sources))
    return paths


def _argv(paths: dict[str, Path], out: Path, *extra: str) -> list[str]:
    return ["--registry", str(paths["registry_path"]), "--sources", str(paths["sources_path"]),
            "--measurements", str(paths["measurements_path"]), "--out-dir", str(out), *extra]


def test_it_builds_a_field_per_mesh_obstacle_and_a_manifest_that_names_every_input(tmp_path):
    from autofly_ue5.scenes.footprint_field import FIELD_VERSION, FootprintFields
    from scripts import footprint_fields

    paths, out = _inputs(tmp_path), tmp_path / "footprints"
    assert footprint_fields.main(_argv(paths, out)) == 0
    assert sorted(p.name for p in out.iterdir()) == ["l_prism.npz", "lean.npz", "manifest.json"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["field_version"] == FIELD_VERSION and set(manifest["fields"]) == {"lean", "l_prism"}
    lean = manifest["fields"]["lean"]
    assert lean["file"] == "lean.npz" and lean["npz_sha256"] == _sha(out / "lean.npz")
    assert lean["triangles"] == {"all": 24, "opaque": 12, "cards": 12, "below_ground": 0, "cut_at_ground": 0}
    gltf = Path(json.loads(paths["sources_path"].read_text())["downloads_dir"]) / "synthetic" / "lean" / "lean.gltf"
    assert lean["source"]["gltf_sha256"] == _sha(gltf) and lean["source"]["node"] == "lean"
    assert lean["source"]["buffers"] == {"lean.bin": _sha(gltf.with_suffix(".bin"))}
    assert lean["pivot_reach_m"] == pytest.approx(np.hypot(1.5, 0.5), abs=1e-6)
    fields = FootprintFields.load({"lean", "l_prism"}, out)
    assert fields.digest() == {"l_prism": manifest["fields"]["l_prism"]["field_sha256"], "lean": lean["field_sha256"]}
    assert footprint_fields.main(_argv(paths, out, "--check")) == 0


def test_a_rebuild_writes_the_same_bytes_and_only_rebuilds_what_it_is_asked_to(tmp_path):
    from scripts import footprint_fields

    paths, out = _inputs(tmp_path), tmp_path / "footprints"
    assert footprint_fields.main(_argv(paths, out)) == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    (out / "l_prism.npz").unlink()
    assert footprint_fields.main(_argv(paths, out, "--only", "l_prism")) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before, "deterministic, and --only keeps the rest"


@pytest.mark.parametrize("damage, why", [
    ("npz", "lean.npz"),
    ("source", "lean.gltf"),
    ("missing", "l_prism"),
    ("version", "version"),
])
def test_check_finds_what_no_longer_matches(tmp_path, capsys, damage, why):
    from scripts import footprint_fields

    paths, out = _inputs(tmp_path), tmp_path / "footprints"
    assert footprint_fields.main(_argv(paths, out)) == 0
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "npz":
        data = bytearray((out / "lean.npz").read_bytes())
        data[-30] ^= 0xFF
        (out / "lean.npz").write_bytes(bytes(data))
    elif damage == "source":
        gltf = Path(json.loads(paths["sources_path"].read_text())["downloads_dir"]) / "synthetic" / "lean" / "lean.gltf"
        gltf.write_text(gltf.read_text() + " ")
    elif damage == "missing":
        del manifest["fields"]["l_prism"]
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest["fields"]["lean"]["field_version"] = 0
        manifest_path.write_text(json.dumps(manifest))
    capsys.readouterr()
    assert footprint_fields.main(_argv(paths, out, "--check")) == 1
    assert why in capsys.readouterr().out


def test_a_source_that_changed_since_it_was_fetched_is_refused_at_build(tmp_path, capsys):
    from scripts import footprint_fields

    paths, out = _inputs(tmp_path), tmp_path / "footprints"
    gltf = Path(json.loads(paths["sources_path"].read_text())["downloads_dir"]) / "synthetic" / "lean" / "lean.gltf"
    gltf.write_text(gltf.read_text() + " ")
    assert footprint_fields.main(_argv(paths, out, "--only", "lean")) == 2
    assert "lean.gltf" in capsys.readouterr().err and not (out / "lean.npz").exists()


def test_an_asset_that_is_not_a_mesh_obstacle_is_refused_by_only(tmp_path, capsys):
    from scripts import footprint_fields

    paths, out = _inputs(tmp_path), tmp_path / "footprints"
    assert footprint_fields.main(_argv(paths, out, "--only", "cylinder")) == 2
    assert "cylinder" in capsys.readouterr().err


def test_loading_refuses_a_missing_asset_a_changed_file_or_another_version(tmp_path):
    from autofly_ue5.scenes.footprint_field import FootprintFields
    from scripts import footprint_fields

    paths, out = _inputs(tmp_path), tmp_path / "footprints"
    assert footprint_fields.main(_argv(paths, out, "--only", "lean")) == 0
    with pytest.raises(ValueError, match="no footprint field for l_prism"):
        FootprintFields.load({"lean", "l_prism"}, out)
    with pytest.raises(ValueError, match="manifest.json is missing"):
        FootprintFields.load({"lean"}, tmp_path / "nowhere")
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["fields"]["lean"]["field_sha256"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="content"):
        FootprintFields.load({"lean"}, out)
    manifest["fields"]["lean"]["field_version"] = 99
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version"):
        FootprintFields.load({"lean"}, out)


def test_the_outline_never_reaches_past_the_footprint_circle_the_registry_measured(tmp_path, capsys):
    # The layouts, the sampler and the levels keep their circles because the outline lies inside each: --check holds
    # every field's reach from the pivot to the registry's widest profile slice.
    from scripts import footprint_fields

    paths, out = _inputs(tmp_path), tmp_path / "footprints"
    registry = json.loads(paths["registry_path"].read_text())
    registry["assets"]["lean"]["radius_profile_m"] = [0.07, 0.07, 1.2]  # narrower than the crown reaches
    paths["registry_path"].write_text(json.dumps(registry))
    assert footprint_fields.main(_argv(paths, out)) == 0
    capsys.readouterr()
    assert footprint_fields.main(_argv(paths, out, "--check")) == 1
    assert "lean" in capsys.readouterr().out


def test_the_scripts_mesh_obstacles_are_the_footprint_rules_own_definition():
    # One definition (expert/footprint_contact.is_measured_obstacle): the script builds a field for exactly the assets
    # the rule covers, and a measured model of another role (a target) gets none, as it may never be a layout instance.
    from autofly_ue5.expert.footprint_contact import is_measured_obstacle
    from scripts.footprint_fields import mesh_obstacles

    entries = {"tree": {"role": "obstacle", "radius_profile_m": [0.5, 1.0]},
               "tgt_car": {"role": "target", "radius_profile_m": [2.0, 2.2]},
               "cylinder": {"role": "obstacle", "radius_profile_m": None},
               "orange": {"role": "target", "radius_profile_m": None}}
    assert mesh_obstacles({"assets": entries}) == ["tree"]
    assert [k for k, e in sorted(entries.items()) if is_measured_obstacle(e["role"], e["radius_profile_m"])] == ["tree"]
