"""Play a recorded dataset episode back as a video: what the dataset holds, frame by frame.

    env -u PYTHONPATH .venv/bin/python scripts/dataset_episode_video.py --raw data/<name> --episode <id> --out <file>.mp4

Each video frame shows the record's RGB image (scaled up), the instruction, the step and simulated time, state[9]'s
distance, bearing, speed and altitude, the action the expert commanded, and -- when the store has detections.json and
rebalance.json -- Grounding DINO's confidence for the target with its box, and the phase the rebalancing assigned
(obstacle avoidance until the first confident detection, target seeking from then on). Real time at 5 Hz by default.
H.264 through ffmpeg when installed (as scripts/render_episodes.py does), else OpenCV's mp4v.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SCALE = 2  # 256 px frames are small on a screen
PANEL_W = 420
STEP_S = 0.2
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, text, xy, scale=0.5, colour=(235, 235, 235), thick=1):
    cv2.putText(img, text, xy, FONT, scale, colour, thick, cv2.LINE_AA)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return lines + [line]


def compose(rgb: np.ndarray, *, instruction: str, step: int, n_steps: int, state: np.ndarray, action: np.ndarray,
            score: float | None, box: list[float] | None, threshold: float | None, first_detection: int | None) -> np.ndarray:
    """One video frame (BGR) from one record."""
    image = cv2.resize(rgb[:, :, ::-1].copy(), (256 * SCALE, 256 * SCALE), interpolation=cv2.INTER_NEAREST)
    if box is not None and score is not None and threshold is not None:
        cx, cy, w, h = box
        x0, y0 = int((cx - w / 2) * 256 * SCALE), int((cy - h / 2) * 256 * SCALE)
        x1, y1 = int((cx + w / 2) * 256 * SCALE), int((cy + h / 2) * 256 * SCALE)
        colour = (80, 220, 80) if score > threshold else (90, 90, 230)
        cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2)
        _text(image, f"{score:.2f}", (max(x0, 2), max(y0 - 6, 14)), 0.5, colour, 1)
    panel = np.full((256 * SCALE, PANEL_W, 3), 28, dtype=np.uint8)
    y = 28
    for line in _wrap(f'"{instruction}"', 44):
        _text(panel, line, (12, y), 0.52, (255, 230, 150))
        y += 22
    y += 10
    _text(panel, f"record {step + 1}/{n_steps}   t = {step * STEP_S:5.1f} s", (12, y)); y += 26
    _text(panel, f"distance to target  {state[0]:6.1f} m", (12, y)); y += 22
    _text(panel, f"bearing to target   {math.degrees(state[1]):+6.1f} deg", (12, y)); y += 22
    _text(panel, f"speed               {state[3]:6.2f} m/s", (12, y)); y += 22
    _text(panel, f"altitude            {state[8]:6.2f} m", (12, y)); y += 30
    _text(panel, "action (expert, stochastic):", (12, y), 0.5, (180, 200, 255)); y += 22
    _text(panel, f"forward {action[0]:5.2f} m/s  yaw {action[1]:+5.2f} rad/s  vz {action[2]:+5.2f} m/s", (12, y), 0.45); y += 32
    if score is not None and threshold is not None:
        phase = "target seeking" if first_detection is not None and step >= first_detection else "obstacle avoidance"
        colour = (80, 220, 80) if phase == "target seeking" else (90, 160, 230)
        _text(panel, "Grounding DINO confidence for the target:", (12, y), 0.5, (180, 200, 255)); y += 20
        bar_w = PANEL_W - 24
        cv2.rectangle(panel, (12, y), (12 + bar_w, y + 14), (70, 70, 70), -1)
        cv2.rectangle(panel, (12, y), (12 + int(bar_w * min(score, 1.0)), y + 14), colour, -1)
        tx = 12 + int(bar_w * threshold)
        cv2.line(panel, (tx, y - 3), (tx, y + 17), (255, 255, 255), 1)
        _text(panel, f"{score:.3f}  (threshold {threshold})", (12, y + 32), 0.45); y += 52
        _text(panel, f"phase: {phase}", (12, y), 0.6, colour, 2); y += 26
        if first_detection is not None:
            _text(panel, f"transition at record {first_detection + 1}", (12, y), 0.45, (200, 200, 200))
    return np.concatenate([image, panel], axis=1)


def render(raw: Path, episode_id: str, out: Path, *, fps: float = 1.0 / STEP_S, hold_s: float = 1.5) -> dict:
    manifest = json.loads((raw / "manifest.json").read_text())
    entry = next(e for e in manifest["episodes"] if e["id"] == episode_id)
    steps = np.load(raw / entry["path"] / "steps.npz")
    frames = sorted((raw / entry["path"] / "frames").glob("*.png"))
    scores = boxes = None
    threshold = first = None
    if (raw / "detections.json").is_file():
        det = json.loads((raw / "detections.json").read_text())["episodes"].get(episode_id)
        if det is not None:
            scores, boxes = det["scores"], det["boxes"]
    if (raw / "rebalance.json").is_file():
        reb = json.loads((raw / "rebalance.json").read_text())
        threshold = reb["threshold"]
        first = next((e["first_detection"] for e in reb["episodes"] if e["id"] == episode_id), None)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="episode_video_"))
    raw_mp4 = tmpdir / "raw.mp4"
    writer = None
    n = len(frames)
    for i, path in enumerate(frames):
        rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1]
        frame = compose(rgb, instruction=entry["instruction"], step=i, n_steps=n, state=steps["state"][i], action=steps["action"][i],
                        score=scores[i] if scores else None, box=boxes[i] if boxes else None, threshold=threshold, first_detection=first)
        if writer is None:
            writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if i == n - 1:
            for _ in range(int(hold_s * fps)):
                writer.write(frame)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_mp4), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                               "-crf", "20", str(out)], capture_output=True, text=True)
        if done.returncode != 0:
            shutil.copy(raw_mp4, out)
    else:
        shutil.copy(raw_mp4, out)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {"episode": episode_id, "records": n, "video": str(out), "instruction": entry["instruction"],
            "first_detection": first, "threshold": threshold}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=float, default=1.0 / STEP_S)
    args = p.parse_args(argv)
    print(json.dumps(render(args.raw, args.episode, args.out, fps=args.fps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
