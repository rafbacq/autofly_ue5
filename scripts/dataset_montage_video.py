"""Dataset playback videos for final_results/ (plan 6 step 7): a montage of one kept episode per scene, or one episode.

    # one MP4 tiling one kept episode per scene (3 x 3 for the nine training scenes)
    env -u PYTHONPATH .venv/bin/python scripts/dataset_montage_video.py montage --stores data/s01_c1 data/s02_c1 ... \\
        --out runs/report/videos/dataset_montage.mp4 [--fps 5] [--pick <store>:<episode> ...] [--json <file>]

    # one episode, its instruction, state and action beside the frame (scripts/dataset_episode_video.py, unchanged)
    env -u PYTHONPATH .venv/bin/python scripts/dataset_montage_video.py episode --raw data/<store> --episode <id> \\
        --out runs/report/videos/<id>.mp4 [--fps 5]

The montage: each tile is an episode's recorded RGB frames, scaled up and captioned with its scene and instruction (the
frames the dataset holds, nothing re-rendered). Records advance at the dataset's 5 Hz and play at --fps (5 = real time,
10 = twice as fast). An episode shorter than the longest holds its last frame, stamped "reached", and the whole holds
for --hold-s at the end. Per scene the tile is the kept episode of median length (the store order breaks ties), so no
scene is shown by its quickest or its most tortuous flight, unless --pick names the episode.

Encoding: H.264 (yuv420p, CRF 20) with raw frames piped into ffmpeg when it is installed, so the frames are compressed
once (scripts/dataset_episode_video.py encodes mp4v first and re-encodes that); OpenCV's mp4v otherwise, or when ffmpeg
fails. Writes --out (and --json when given) and nothing else.
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
from collections import defaultdict  # noqa: E402
from collections.abc import Callable, Iterable, Sequence  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autofly_ue5.report.stores import Episode, Store, load_stores  # noqa: E402
from scripts.dataset_episode_video import FONT, STEP_S  # noqa: E402
from scripts.dataset_figures import pick_first_frames, scene_label  # noqa: E402

DEFAULT_TILE = 384  # px: 1.5 x the 256 px frames; 3 x 3 tiles make a 1176 x 1404 video
DEFAULT_COLS = 3
GUTTER = 6
CAPTION_PX = 66  # the scene's name and up to three lines of instruction under each tile
BACKGROUND = (28, 28, 28)  # BGR, as scripts/dataset_episode_video.py's panel
TEXT = (235, 235, 235)
SCENE_TEXT = (150, 230, 255)  # the per-episode video's instruction colour
REACHED_TEXT = (150, 230, 150)


def pick_per_scene(stores: Sequence[Store], per_scene: int = 1) -> list[Episode]:
    """Per scene, in scene order: one episode, the kept one of median length (the lower median); or `per_scene` of them
    with the scene's targets taken in turn (scripts/dataset_figures.pick_first_frames), each target's most typical length
    first, so a per-scene montage shows its object diversity. A stable sort keeps the store order among equals."""
    by_scene: dict[str, list[Episode]] = defaultdict(list)
    for store in stores:
        for e in store.episodes:
            by_scene[e.scene].append(e)
    picked = []
    for scene in sorted(by_scene):
        episodes = by_scene[scene]
        median = sorted(e.steps for e in episodes)[(len(episodes) - 1) // 2]
        typical = sorted(episodes, key=lambda e: abs(e.steps - median))
        picked += [typical[0]] if per_scene == 1 else pick_first_frames(typical, per_scene)
    return picked


def tile_origin(index: int, cols: int, tile_px: int) -> tuple[int, int]:
    """The top-left pixel (x, y) of tile `index`'s image."""
    r, c = divmod(index, cols)
    return GUTTER + c * (tile_px + GUTTER), GUTTER + r * (tile_px + CAPTION_PX + GUTTER)


def canvas_size(n: int, cols: int, tile_px: int) -> tuple[int, int]:
    """(width, height) of a montage of n tiles, rounded up to even numbers (H.264 with yuv420p needs them)."""
    rows = math.ceil(n / cols)
    width = GUTTER + cols * (tile_px + GUTTER)
    height = GUTTER + rows * (tile_px + CAPTION_PX + GUTTER)
    return width + width % 2, height + height % 2


def fit_lines(text: str, width_px: int, scale: float, max_lines: int) -> list[str]:
    """`text` broken into lines no wider than width_px in the overlay font, cut to max_lines with '...'."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if not line or cv2.getTextSize(trial, FONT, scale, 1)[0][0] <= width_px:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and cv2.getTextSize(last + "...", FONT, scale, 1)[0][0] > width_px:
            last = last[:-1]
        lines[-1] = last.rstrip() + "..."
    return lines


class Tile:
    """One episode's frames at tile size, read as the playback reaches them (one decode per record)."""

    def __init__(self, episode: Episode, tile_px: int) -> None:
        self.episode = episode
        self.tile_px = tile_px
        self.scale = max(0.33, 0.5 * tile_px / DEFAULT_TILE)
        self.label = scene_label(episode.scene)
        self.caption = fit_lines(f'"{episode.instruction}"', tile_px - 10, 0.9 * self.scale, 3)
        self._index: int | None = None
        self._image: np.ndarray | None = None

    def image(self, t: int) -> np.ndarray:
        i = min(t, self.episode.steps - 1)
        if i != self._index:
            bgr = cv2.imread(str(self.episode.frame(i)), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"{self.episode.frame(i)} does not decode as an image")
            exact = self.tile_px % bgr.shape[1] == 0  # an integer scale keeps the pixels square
            self._image = cv2.resize(bgr, (self.tile_px, self.tile_px),
                                     interpolation=cv2.INTER_NEAREST if exact else cv2.INTER_LINEAR)
            self._index = i
        return self._image


def _stamp(image: np.ndarray, text: str, colour, scale: float) -> None:
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    cv2.rectangle(image, (0, 0), (w + 8, h + base + 8), (0, 0, 0), -1)
    cv2.putText(image, text, (4, h + 4), FONT, scale, colour, 1, cv2.LINE_AA)


def compose(tiles: Sequence[Tile], t: int, cols: int, size: tuple[int, int]) -> np.ndarray:
    """Montage frame t (BGR): every tile at record min(t, its last), captioned."""
    canvas = np.full((size[1], size[0], 3), BACKGROUND, dtype=np.uint8)
    for k, tile in enumerate(tiles):
        x, y = tile_origin(k, cols, tile.tile_px)
        image = tile.image(t).copy()
        n = tile.episode.steps
        if t >= n:
            _stamp(image, f"reached ({n} records)", REACHED_TEXT, tile.scale)
        else:
            _stamp(image, f"{t * STEP_S:4.1f} s", TEXT, tile.scale)
        canvas[y:y + tile.tile_px, x:x + tile.tile_px] = image
        line = int(round(30 * tile.scale))
        cv2.putText(canvas, tile.label, (x + 3, y + tile.tile_px + line), FONT, tile.scale, SCENE_TEXT, 1, cv2.LINE_AA)
        for j, text in enumerate(tile.caption):
            cv2.putText(canvas, text, (x + 3, y + tile.tile_px + line * (j + 2)), FONT, 0.9 * tile.scale, TEXT, 1,
                        cv2.LINE_AA)
    return canvas


def write_video(make_frames: Callable[[], Iterable[np.ndarray]], out: Path, fps: float, size: tuple[int, int]) -> dict:
    """H.264 through ffmpeg's stdin when ffmpeg is installed and succeeds, else OpenCV's mp4v. `make_frames` is called
    again for the fallback, so a failed ffmpeg costs a second pass, never a broken file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                   "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                   "-movflags", "+faststart", str(out)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        count = 0
        try:
            for frame in make_frames():
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                count += 1
            process.stdin.close()
        except BrokenPipeError:
            pass
        except BaseException:  # a frame that cannot be made: no truncated video is left under the name asked for
            process.kill()
            process.wait()
            out.unlink(missing_ok=True)
            raise
        error = process.stderr.read().decode(errors="replace")
        if process.wait() == 0:
            return {"codec": "h264", "frames": count}
        print(f"ffmpeg failed ({error.strip()[:300]}); writing mp4v with OpenCV instead", file=sys.stderr)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write {out}")
    count = 0
    try:
        for frame in make_frames():
            writer.write(frame)
            count += 1
    except BaseException:
        writer.release()
        out.unlink(missing_ok=True)
        raise
    writer.release()
    return {"codec": "mp4v", "frames": count}


def render_montage(episodes: Sequence[Episode], out: Path, *, fps: float = 1.0 / STEP_S, cols: int = DEFAULT_COLS,
                   tile_px: int = DEFAULT_TILE, hold_s: float = 2.0) -> dict:
    if not episodes:
        raise ValueError("no episode to show")
    cols = max(1, min(cols, len(episodes)))
    tiles = [Tile(e, tile_px) for e in episodes]
    size = canvas_size(len(tiles), cols, tile_px)
    total = max(e.steps for e in episodes) + int(round(hold_s * fps))
    encoded = write_video(lambda: (compose(tiles, t, cols, size) for t in range(total)), Path(out), fps, size)
    return {"video": str(out), **encoded, "fps": fps, "records_per_second": 1.0 / STEP_S, "hold_s": hold_s,
            "size": list(size), "cols": cols,
            "tiles": [{"scene": e.scene, "label": scene_label(e.scene), "store": e.store, "episode": e.id,
                       "records": e.steps, "target": e.target.name, "instruction": e.instruction} for e in episodes]}


def _picks(stores: Sequence[Store], picks: Sequence[str], parser: argparse.ArgumentParser) -> list[Episode]:
    chosen: dict[str, Episode] = {}
    for pick in picks:
        store_name, sep, episode_id = pick.partition(":")
        store = next((s for s in stores if s.name == store_name), None)
        episode = next((e for e in store.episodes if e.id == episode_id), None) if store and sep else None
        if episode is None:
            parser.error(f"--pick {pick!r}: no kept episode {episode_id!r} in a listed store {store_name!r}")
        if episode.scene in chosen:
            parser.error(f"--pick names two episodes of scene {episode.scene}")
        chosen[episode.scene] = episode
    return [chosen.get(e.scene, e) for e in pick_per_scene(stores)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("montage", help="one kept episode per scene, tiled")
    m.add_argument("--stores", type=Path, nargs="+", required=True, help="raw stores (data/<name>)")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--fps", type=float, default=1.0 / STEP_S, help="playback rate; the records are 5 Hz (default: real time)")
    m.add_argument("--cols", type=int, default=DEFAULT_COLS)
    m.add_argument("--tile", type=int, default=DEFAULT_TILE, help="tile image size in pixels")
    m.add_argument("--hold-s", type=float, default=2.0, help="seconds the last frame holds at the end")
    m.add_argument("--pick", action="append", default=[], metavar="STORE:EPISODE",
                   help="the episode to show for its scene (default: the scene's kept episode of median length)")
    m.add_argument("--per-scene", type=int, default=1,
                   help="episodes per scene, its targets taken in turn (e.g. 9 for one scene's 3 x 3 montage)")
    m.add_argument("--json", type=Path, default=None, help="also write the montage's description here")
    e = sub.add_parser("episode", help="one episode with its record panel (scripts/dataset_episode_video.py)")
    e.add_argument("--raw", type=Path, required=True)
    e.add_argument("--episode", required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--fps", type=float, default=1.0 / STEP_S)
    args = p.parse_args(argv)
    if args.mode == "episode":
        from scripts.dataset_episode_video import render

        report = render(args.raw, args.episode, args.out, fps=args.fps)
    else:
        if args.fps <= 0 or args.tile < 32 or args.cols < 1 or args.hold_s < 0 or args.per_scene < 1:
            m.error("--fps must be positive, --tile at least 32, --cols and --per-scene at least 1, --hold-s not "
                    "negative")
        if args.pick and args.per_scene != 1:
            m.error("--pick names a scene's one episode; it cannot be combined with --per-scene")
        stores = load_stores(args.stores, read_rejects_dirs=False)
        episodes = _picks(stores, args.pick, m) if args.pick else pick_per_scene(stores, args.per_scene)
        report = render_montage(episodes, args.out, fps=args.fps, cols=args.cols, tile_px=args.tile,
                                hold_s=args.hold_s)
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
