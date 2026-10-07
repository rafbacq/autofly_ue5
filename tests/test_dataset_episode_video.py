"""scripts/dataset_episode_video.py plays a stored episode back, one video frame per record, with or without detections."""

from __future__ import annotations

import json

import cv2
import numpy as np

from tests.test_collect import _collect


def test_a_stored_episode_becomes_a_video_with_one_frame_per_record_plus_the_hold(tmp_path):
    from scripts.dataset_episode_video import compose, render

    _summary, root = _collect(tmp_path, n_keep=1)
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    out = tmp_path / "videos" / "ep.mp4"
    report = render(root, entry["id"], out, fps=5.0, hold_s=1.0)
    assert out.is_file() and report["records"] == entry["steps"] and report["first_detection"] is None
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == entry["steps"] + 5, "every record once, then a 1 s hold"
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 512 + 420 and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 512
    cap.release()
    frame = compose(np.zeros((256, 256, 3), np.uint8), instruction="reach the thing", step=3, n_steps=10,
                    state=np.zeros(9, np.float32), action=np.zeros(3, np.float32), score=0.8, box=[0.5, 0.5, 0.1, 0.2],
                    threshold=0.7, first_detection=2)
    assert frame.shape == (512, 932, 3) and frame.dtype == np.uint8
