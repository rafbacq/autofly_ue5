"""scripts/dataset_montage_video.py tiles one kept episode per scene into one video at the dataset's 5 Hz, each tile
captioned with its scene and instruction, a shorter episode holding its last frame; and delegates a single episode to
scripts/dataset_episode_video.py."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from tests.synthetic_stores import Flight, frame_level, pool_target, write_store

SOFA = pool_target("beige sofa", "furniture")
RED, GREEN, BLUE = (220, 40, 40), (40, 200, 60), (40, 60, 220)


def _stores(tmp_path):
    data = tmp_path / "data"
    return [write_store(data, "s01_c1", "s01", [Flight(1, 3, SOFA, colour=RED), Flight(2, 5, SOFA, colour=RED),
                                                Flight(3, 7, SOFA, colour=RED)]),
            write_store(data, "s02_c1", "s02", [Flight(1, 4, SOFA, colour=GREEN)]),
            write_store(data, "s07_c1", "s07", [Flight(1, 6, SOFA, colour=BLUE)])]


def _frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    ok, image = cap.read()
    while ok:
        frames.append(image)
        ok, image = cap.read()
    cap.release()
    return frames


def _tile_mean(frame, index, cols, tile_px):
    from scripts.dataset_montage_video import tile_origin

    x, y = tile_origin(index, cols, tile_px)
    m = tile_px // 4  # the tile's middle, away from its overlay text
    return frame[y + m:y + tile_px - m, x + m:x + tile_px - m].reshape(-1, 3).mean(axis=0)


def _expected_bgr(colour, i, n):
    return np.array([round(c * frame_level(i, n)) for c in colour][::-1], dtype=float)


def test_one_episode_per_scene_plays_at_5_hz_and_a_shorter_one_holds_its_last_frame(tmp_path):
    from autofly_ue5.report.stores import load_stores
    from scripts.dataset_montage_video import pick_per_scene, render_montage

    picked = pick_per_scene(load_stores(_stores(tmp_path)))
    assert [(e.scene, e.id) for e in picked] == [("s01", "s01_2"), ("s02", "s02_1"), ("s07", "s07_1")], \
        "per scene the kept episode of median length"
    out = tmp_path / "videos" / "montage.mp4"
    report = render_montage(picked, out, fps=5.0, cols=3, tile_px=128, hold_s=1.0)
    frames = _frames(out)
    assert len(frames) == report["frames"] == 6 + 5, "the longest picked episode's records, then a 1 s hold"
    height, width = frames[0].shape[:2]
    assert (width, height) == tuple(report["size"]) and width % 2 == 0 and height % 2 == 0
    assert [t["episode"] for t in report["tiles"]] == ["s01_2", "s02_1", "s07_1"]
    for t, (colour, n) in enumerate([(RED, 5), (GREEN, 4), (BLUE, 6)]):
        for f in (0, 2, 5, 10):  # record f, or the last one once the episode has ended
            assert _tile_mean(frames[f], t, 3, 128) == pytest.approx(_expected_bgr(colour, min(f, n - 1), n), abs=14), \
                (t, f)


def test_a_pick_replaces_its_scene_s_episode_and_fewer_scenes_make_fewer_columns(tmp_path):
    from scripts import dataset_montage_video

    stores = _stores(tmp_path)
    out = tmp_path / "m.mp4"
    record = tmp_path / "m.json"
    assert dataset_montage_video.main(["montage", "--stores", str(stores[0]), str(stores[1]), "--out", str(out),
                                       "--pick", "s01_c1:s01_3", "--tile", "128", "--hold-s", "0",
                                       "--json", str(record)]) == 0
    report = json.loads(record.read_text())
    assert [t["episode"] for t in report["tiles"]] == ["s01_3", "s02_1"] and report["cols"] == 2
    assert len(_frames(out)) == 7
    with pytest.raises(SystemExit):
        dataset_montage_video.main(["montage", "--stores", str(stores[0]), "--out", str(out), "--pick", "s01_c1:nope"])


def test_several_episodes_per_scene_take_its_targets_in_turn_each_at_its_most_typical_length(tmp_path):
    from autofly_ue5.report.stores import load_stores
    from scripts.dataset_montage_video import pick_per_scene

    chair = pool_target("grey armchair", "furniture")
    store = write_store(tmp_path / "data", "s01_c1", "s01", [
        Flight(1, 3, SOFA), Flight(2, 5, SOFA), Flight(3, 9, chair), Flight(4, 6, chair), Flight(5, 4, SOFA)])
    stores = load_stores([store])
    assert [e.id for e in pick_per_scene(stores, 2)] == ["s01_2", "s01_4"], "median 5: sofa at 5, chair at 6"
    assert [e.id for e in pick_per_scene(stores, 3)] == ["s01_2", "s01_4", "s01_5"]
    assert [e.id for e in pick_per_scene(stores)] == ["s01_2"]


def test_a_frame_that_cannot_be_read_leaves_no_partial_video(tmp_path):
    from autofly_ue5.report.stores import load_stores
    from scripts.dataset_montage_video import pick_per_scene, render_montage

    stores = _stores(tmp_path)
    (stores[1] / "episodes" / "s02_1" / "frames" / "000002.png").write_bytes(b"not a png")
    out = tmp_path / "broken.mp4"
    with pytest.raises(ValueError, match="000002.png"):
        render_montage(pick_per_scene(load_stores(stores)), out, tile_px=64, hold_s=0.0)
    assert not out.exists(), "an aborted montage is no video"


def test_the_episode_mode_is_scripts_dataset_episode_video(tmp_path):
    from scripts import dataset_montage_video

    stores = _stores(tmp_path)
    out = tmp_path / "ep.mp4"
    assert dataset_montage_video.main(["episode", "--raw", str(stores[1]), "--episode", "s02_1", "--out", str(out)]) == 0
    frames = _frames(out)
    assert len(frames) == 4 + 7, "every record, then dataset_episode_video's 1.5 s hold at 5 fps"
    assert frames[0].shape[:2] == (512, 512 + 420)
