import hashlib
import json

from autofly_ue5.validate.plugin_manifest import verify_manifest


def _make_plugin_tree(root):
    files = {
        "Plugins/ProjectAirSim/Source/ProjectAirSim/ProjectAirSim.Build.cs": b"build rules",
        "Plugins/ProjectAirSim/SimLibs/core_sim/Release/libcore_sim.a": b"archive",
        "Plugins/Drone/Drone.uplugin": b"{}",
        "Plugins/Rover/Rover.uplugin": b"{}",
    }
    entries = []
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"source_sha": "09755454b8d8", "unreal_version": "5.7", "files": entries}
    (root / "build-manifest.json").write_text(json.dumps(manifest))
    return files


def test_intact_tree_passes(tmp_path):
    _make_plugin_tree(tmp_path)
    report = verify_manifest(tmp_path)
    assert report["pass"] is True
    assert report["files"] == 4
    assert report["mismatched"] == [] and report["missing"] == [] and report["missing_dirs"] == []


def test_corrupted_file_fails(tmp_path):
    _make_plugin_tree(tmp_path)
    (tmp_path / "Plugins/Drone/Drone.uplugin").write_bytes(b"tampered")
    report = verify_manifest(tmp_path)
    assert report["pass"] is False
    assert report["mismatched"] == ["Plugins/Drone/Drone.uplugin"]


def test_wrong_engine_version_fails(tmp_path):
    _make_plugin_tree(tmp_path)
    manifest = json.loads((tmp_path / "build-manifest.json").read_text())
    manifest["unreal_version"] = "5.2"
    (tmp_path / "build-manifest.json").write_text(json.dumps(manifest))
    assert verify_manifest(tmp_path)["pass"] is False
