"""scripts/fetch_assets.py: the CC0 asset list is curated in code, every download is checked against the source's own
checksum and recorded with its sha256 and licence, and a second run downloads nothing it already has."""

from __future__ import annotations

import hashlib
import json
import zipfile

import pytest


class _FakeWeb:
    """Stands in for the network: Poly Haven's files/info APIs and ambientCG's full_json, plus the file bodies."""

    def __init__(self) -> None:
        self.bodies = {
            "https://dl/fir_tree_01_2k.gltf": b"gltf-body",
            "https://dl/textures/fir_diff_2k.jpg": b"jpg-body",
        }
        self.zip_bytes = self._zip({"PavingStones070_2K_Color.jpg": b"color", "PavingStones070_2K_NormalGL.jpg": b"normal"})
        self.bodies["https://ambientcg.com/get?file=PavingStones070_2K-JPG.zip"] = self.zip_bytes
        self.fetched: list[str] = []

    @staticmethod
    def _zip(files: dict[str, bytes]) -> bytes:
        import io

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for name, data in files.items():
                z.writestr(name, data)
        return buf.getvalue()

    def json(self, url: str) -> dict:
        self.fetched.append(url)
        if url.endswith("/files/fir_tree_01"):
            return {"gltf": {"2k": {"gltf": {"url": "https://dl/fir_tree_01_2k.gltf", "md5": hashlib.md5(b"gltf-body").hexdigest(), "size": 9,
                                            "include": {"textures/fir_diff_2k.jpg": {"url": "https://dl/textures/fir_diff_2k.jpg",
                                                                                    "md5": hashlib.md5(b"jpg-body").hexdigest(), "size": 8}}}}}}
        if url.endswith("/info/fir_tree_01"):
            return {"name": "Fir Tree 01", "authors": {"Rob Tuytel": "photography"}, "categories": ["trees"], "dimensions": [2673.0, 654.0, 1926.0]}
        if "ambientcg.com/api" in url:
            return {"foundAssets": [{"assetId": "PavingStones070", "displayName": "Paving Stones 070", "tags": ["paving"],
                                     "dimensionX": 115, "dimensionY": 115,
                                     "downloadFolders": {"default": {"downloadFiletypeCategories": {"zip": {"downloads": [
                                         {"attribute": "1K-JPG", "downloadLink": "https://ambientcg.com/get?file=PavingStones070_1K-JPG.zip", "size": 1},
                                         {"attribute": "2K-JPG", "downloadLink": "https://ambientcg.com/get?file=PavingStones070_2K-JPG.zip",
                                          "size": len(self.zip_bytes)}]}}}}}]}
        raise KeyError(url)

    def download(self, url: str, dest) -> None:
        self.fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.bodies[url])


def test_models_and_materials_are_fetched_checked_recorded_and_not_fetched_twice(tmp_path):
    from scripts.fetch_assets import fetch_all

    entries = [{"id": "fir_tree_01", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s02"]},
               {"id": "PavingStones070", "source": "ambientcg", "kind": "material", "resolution": "2K-JPG", "use": ["s09"]}]
    web = _FakeWeb()
    record = fetch_all(entries, tmp_path / "downloads", tmp_path / "sources.json", web=web)
    saved = json.loads((tmp_path / "sources.json").read_text())
    assert saved == record and {a["id"] for a in saved["assets"]} == {"fir_tree_01", "PavingStones070"}
    fir = next(a for a in saved["assets"] if a["id"] == "fir_tree_01")
    assert fir["licence"] == "CC0-1.0" and fir["authors"] == {"Rob Tuytel": "photography"} and fir["dimensions_m"] == [2.673, 0.654, 1.926]
    assert {f["path"] for f in fir["files"]} == {"fir_tree_01_2k.gltf", "textures/fir_diff_2k.jpg"}
    assert all(f["sha256"] == hashlib.sha256(web.bodies[f["url"]]).hexdigest() for f in fir["files"])
    assert (tmp_path / "downloads" / "polyhaven" / "fir_tree_01" / "textures" / "fir_diff_2k.jpg").read_bytes() == b"jpg-body"
    paving = next(a for a in saved["assets"] if a["id"] == "PavingStones070")
    assert paving["files"][0]["path"] == "PavingStones070_2K-JPG.zip" and paving["files"][0]["sha256"] == hashlib.sha256(web.zip_bytes).hexdigest()
    assert paving["dimensions_cm"] == [115, 115]
    assert sorted(p.name for p in (tmp_path / "downloads" / "ambientcg" / "PavingStones070").iterdir()) == [
        "PavingStones070_2K-JPG.zip", "PavingStones070_2K_Color.jpg", "PavingStones070_2K_NormalGL.jpg"], "the zip is unpacked beside itself"
    downloads_first = [u for u in web.fetched if u.startswith("https://dl/") or "get?file" in u]
    assert len(downloads_first) == 3
    fetch_all(entries, tmp_path / "downloads", tmp_path / "sources.json", web=web)
    downloads_second = [u for u in web.fetched if u.startswith("https://dl/") or "get?file" in u]
    assert len(downloads_second) == 3, "files whose sha256 already matches are not downloaded again"


def test_a_checksum_mismatch_is_refused_and_the_bad_file_removed(tmp_path):
    from scripts.fetch_assets import fetch_all

    web = _FakeWeb()
    web.bodies["https://dl/fir_tree_01_2k.gltf"] = b"corrupted"
    entries = [{"id": "fir_tree_01", "source": "polyhaven", "kind": "model", "resolution": "2k", "use": ["s02"]}]
    with pytest.raises(ValueError, match="md5"):
        fetch_all(entries, tmp_path / "downloads", tmp_path / "sources.json", web=web)
    assert not (tmp_path / "downloads" / "polyhaven" / "fir_tree_01" / "fir_tree_01_2k.gltf").exists()
    assert not (tmp_path / "sources.json").exists()


def test_the_curated_list_is_well_formed():
    from scripts.fetch_assets import ASSETS, SCENE_IDS

    ids = [a["id"] for a in ASSETS]
    assert len(ids) == len(set(ids)) and all(a["source"] in ("polyhaven", "ambientcg", "kenney") for a in ASSETS)
    assert all(a["kind"] in ("model", "material", "model_pack") for a in ASSETS)
    assert all(a["kind"] == "model_pack" and a["models"] and a["kit"] for a in ASSETS if a["source"] == "kenney")
    assert all(a.get("why") for a in ASSETS if "targets" in a["use"]), "every target asset says why it was chosen"
    assert all(set(a["use"]) <= SCENE_IDS and a["use"] for a in ASSETS), "every asset is for at least one scene"
    assert {s for a in ASSETS for s in a["use"]} >= {"s02", "s03", "s04", "s05", "s06", "s07", "s09"}, "the first scenes are covered"


_KENNEY_LICENCE = b"""Car Kit (3.1)
Created/distributed by Kenney (www.kenney.nl)
License: (Creative Commons Zero, CC0)
http://creativecommons.org/publicdomain/zero/1.0/
"""


class _FakeKenney(_FakeWeb):
    """Kenney: an asset page whose download link names the kit's zip, and the zip (License.txt, GLB models, textures)."""

    PAGE = "https://kenney.nl/assets/demo-kit"
    ZIP = "https://kenney.nl/media/pages/assets/demo-kit/abc123-1775131960/kenney_demo-kit.zip"

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        super().__init__()
        self.kit = self._zip(files if files is not None else {
            "License.txt": _KENNEY_LICENCE, "Models/GLB format/sedan.glb": b"glb-sedan", "Models/GLB format/van.glb": b"glb-van",
            "Models/GLB format/Textures/colormap.png": b"png", "Models/FBX format/sedan.fbx": b"fbx"})
        self.bodies[self.ZIP] = self.kit
        self.pages = {self.PAGE: f'<a href="{self.ZIP}" class="button">Download</a>'}

    def text(self, url: str) -> str:
        self.fetched.append(url)
        return self.pages[url]


def _kit_entry(**over) -> dict:
    return {"id": "kenney_demo-kit", "source": "kenney", "kind": "model_pack", "kit": "demo-kit", "resolution": "GLB",
            "models": ["sedan", "van"], "use": ["targets"], "why": "test", **over}


def test_a_kenney_kit_is_fetched_from_its_page_unpacked_and_its_models_recorded(tmp_path):
    from scripts.fetch_assets import fetch_all

    web = _FakeKenney()
    record = fetch_all([_kit_entry()], tmp_path / "downloads", tmp_path / "sources.json", web=web)
    (kit,) = record["assets"]
    assert kit["licence"] == "CC0-1.0" and kit["page"] == "https://kenney.nl/assets/demo-kit" and kit["authors"] == {"Kenney": "All"}
    assert kit["kit_version"] == "Car Kit (3.1)", "the version line of the kit's own License.txt"
    (zip_file,) = kit["files"]
    assert zip_file["path"] == "kenney_demo-kit.zip" and zip_file["url"] == _FakeKenney.ZIP
    assert zip_file["sha256"] == hashlib.sha256(web.kit).hexdigest() and zip_file["unpacked_files"] == 5
    root = tmp_path / "downloads" / "kenney" / "kenney_demo-kit"
    assert (root / "Models" / "GLB format" / "Textures" / "colormap.png").read_bytes() == b"png", "unpacked beside itself"
    assert kit["models"] == [{"name": "sedan", "path": "Models/GLB format/sedan.glb", "sha256": hashlib.sha256(b"glb-sedan").hexdigest()},
                             {"name": "van", "path": "Models/GLB format/van.glb", "sha256": hashlib.sha256(b"glb-van").hexdigest()}]
    first = list(web.fetched)
    fetch_all([_kit_entry()], tmp_path / "downloads", tmp_path / "sources.json", web=web)
    assert web.fetched == first, "a kit whose zip is already here with its sha256 is neither looked up nor downloaded again"


def test_a_kenney_kit_without_the_named_model_or_a_cc0_licence_is_refused(tmp_path):
    from scripts.fetch_assets import fetch_all

    with pytest.raises(ValueError, match="truck"):
        fetch_all([_kit_entry(models=["sedan", "truck"])], tmp_path / "d1", tmp_path / "s1.json", web=_FakeKenney())
    not_cc0 = _FakeKenney({"License.txt": b"All rights reserved", "Models/GLB format/sedan.glb": b"x"})
    with pytest.raises(ValueError, match="CC0"):
        fetch_all([_kit_entry(models=["sedan"])], tmp_path / "d2", tmp_path / "s2.json", web=not_cc0)
    escaping = _FakeKenney({"License.txt": _KENNEY_LICENCE, "../outside.glb": b"x", "Models/GLB format/sedan.glb": b"x"})
    with pytest.raises(ValueError, match="outside"):
        fetch_all([_kit_entry(models=["sedan"])], tmp_path / "d3", tmp_path / "s3.json", web=escaping)
    assert not (tmp_path / "d3" / "kenney" / "outside.glb").exists() and not any(p.exists() for p in (tmp_path / "s1.json", tmp_path / "s2.json", tmp_path / "s3.json"))


def test_a_kenney_page_without_a_download_link_says_to_fetch_it_by_hand(tmp_path):
    from scripts.fetch_assets import fetch_all

    web = _FakeKenney()
    web.pages[_FakeKenney.PAGE] = "<html>a page that needs a browser</html>"
    with pytest.raises(ValueError, match="manual"):
        fetch_all([_kit_entry()], tmp_path / "downloads", tmp_path / "sources.json", web=web)
