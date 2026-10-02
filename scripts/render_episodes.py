"""Replay chosen M2 gate episodes with a trained expert and save what it flew: an MP4 and a top-down map each.

Why: every simulator runs headless (-RenderOffScreen, sim/process.py), and the gate record keeps only each episode's
outcome and end pose, so nothing showed *how* the expert crosses the pillar field. The drone's front camera already
renders RGB and depth at every step (the depth is what the policy flies on), so this keeps those frames and the pose,
and draws the path over the layout's pillars.

It replays the gate's own seeds (seed = --seed-base + i for each `--episodes i`), so every video is an episode the
recorded gate scored, and compares each replay's outcome and step count with that record (--gate). A mismatch is
written down, not raised: nothing guarantees a bit-exact replay (the physics runs in the Unreal process), and a
difference is worth seeing in itself. Only a deterministic replay can be expected to match; a stochastic one draws
fresh action noise. Frames are kept per attempt and rendered only once the attempt ends with a real outcome, so an
attempt cut short by a backend fault (replayed on the same seed, as evaluate.py does) never reaches a video.

Writes into --out-dir: ep<i>_seed<seed>.mp4 (camera RGB | depth | top-down map, over a status line; H.264 when
ffmpeg is installed, else OpenCV's mp4v), ep<i>_seed<seed>_map.png, overview.png, and summary.json (rewritten after
every episode). It refuses an --out-dir that already holds a summary.json. main() ends with os._exit, like
m2_gate.py: the projectairsim client leaves a non-daemon thread that blocks interpreter shutdown.

    bash scripts/run_job.sh start render_episodes -- env -u PYTHONPATH .venv/bin/python -m scripts.render_episodes \
        --run-root runs/expert/s01_r2 --checkpoint best_model --scene-config scene_autofly_s01_fast.jsonc \
        --episodes 61 97 175 179 --out-dir runs/viz/s01_r2_best_model
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap, as in m2_gate.py: a direct `python scripts/render_episodes.py` puts scripts/ at sys.path[0].
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Callable, Iterable  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autofly_ue5.expert.evaluate import DEFAULT_MAX_FAULT_RETRIES_PER_EPISODE, DEFAULT_MAX_STEPS_PER_EPISODE  # noqa: E402
from autofly_ue5.expert.obs import DEPTH_CLIP_M  # noqa: E402
from autofly_ue5.expert.reward import RewardConfig  # noqa: E402
from autofly_ue5.expert.seeds import EVAL_SEED_BASE  # noqa: E402
from autofly_ue5.expert.train import scene_and_layout, sha256_of  # noqa: E402
from autofly_ue5.expert.vec import teardown  # noqa: E402
from autofly_ue5.paths import ROOT  # noqa: E402
from autofly_ue5.scenes.model import Bounds, Layout  # noqa: E402
from autofly_ue5.sim.airsim_backend import scene_config_factory, scene_config_record  # noqa: E402
from autofly_ue5.sim.process import SIM_RUN_DIR, stop_instances, sweep_orphaned_instances  # noqa: E402
from autofly_ue5.sim.types import CONTROL_DT_S  # noqa: E402
from scripts.m2_gate import build_eval_env, default_sac_loader, parse_model_args  # noqa: E402

CAMERA_PX = 256  # FrontCamera's capture size (configs/robot_autofly_quadrotor.jsonc); frames are resized to it
HUD_PX = 66  # status-line height per unit of --scale: three lines of text
SUCCESS_RADIUS_M = RewardConfig().success_radius_m
DEFAULT_GATE = ROOT / "docs" / "gates" / "m2_gate.json"
OUTCOME_BGR = {"success": (90, 200, 90), "collision": (70, 70, 235), "out_of_bounds": (220, 90, 220),
               "timeout": (0, 190, 255)}
OUTCOME_MARKER = {"success": ("*", "tab:green"), "collision": ("X", "tab:red"), "out_of_bounds": ("P", "tab:purple"),
                  "timeout": ("s", "tab:olive")}


# --------------------------------------------------------------------------------------------------------
# Recording: what the camera saw at every step, and what the episode spawned.
# --------------------------------------------------------------------------------------------------------
class FrameRecorder:
    """The latest observation any wrapped simulator returned, what the current episode spawned, and where every moved
    scene object (a dynamic scene's pillar, spec §6.5) was last put."""

    def __init__(self) -> None:
        self.last = None
        self.spawned: list[tuple[str, tuple[float, float, float]]] = []
        self.object_poses: dict[str, tuple[float, float, float]] = {}


class RecordingSimulator:
    """Forwards every call to `inner` and notes each observation and spawn in `recorder`.

    It wraps the simulator, not the env: AutoFlyEnv hands the policy an 84x84 depth and a vector, never the camera
    frames. Each env step calls observe() exactly once, so the latest observation is that step's frame
    (`capture_frame` checks this against the pose the env reports). A relaunch builds a fresh simulator through the
    same factory, so it records into the same recorder."""

    def __init__(self, inner, recorder: FrameRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def reset(self, pose):
        self._recorder.spawned.clear()  # AutoFlyEnv.reset() spawns the episode's objects after this call
        obs = self._inner.reset(pose)
        self._recorder.last = obs
        return obs

    def observe(self):
        obs = self._inner.observe()
        self._recorder.last = obs
        return obs

    def spawn(self, name, asset, pose, scale, material=None):
        actual = self._inner.spawn(name, asset, pose, scale, material)
        self._recorder.spawned.append((name, (float(pose.x), float(pose.y), float(pose.z))))
        return actual

    def set_object_poses(self, poses):
        # Explicit, not left to __getattr__: that would forward the call and the recorder would never see a mover.
        self._inner.set_object_poses(poses)
        self._recorder.object_poses.update({name: (float(p.x), float(p.y), float(p.z)) for name, p in poses.items()})


@dataclass
class Frame:
    step: int
    rgb: np.ndarray | None  # (H, W, 3) uint8 in RGB order (decode.py flips the wire's BGR); dropped once rendered
    depth: np.ndarray | None  # (H, W) float32 metres, +inf = no hit
    pose: tuple[float, float, float, float]  # x, y, z (NED, metres), yaw (rad)
    distance_m: float
    bearing_deg: float
    command: tuple[float, float, float] | None  # what was flown into this frame; None for the reset frame
    reward: float


@dataclass
class EpisodeRecording:
    index: int
    seed: int
    outcome: str
    steps: int
    final_distance_m: float
    is_success: bool
    episode_return: float
    oob_kind: str | None
    retries: int
    target: tuple[float, float, float]
    distractors: list[tuple[float, float, float]]
    frames: list[Frame]


def capture_frame(recorder: FrameRecorder, info: dict, *, step: int, action, reward: float) -> Frame:
    obs = recorder.last
    if obs is None:
        raise RuntimeError(f"step {step}: no observation has been recorded yet")
    pose = tuple(float(v) for v in info["pose"])
    seen = (obs.pose.x, obs.pose.y, obs.pose.z, obs.pose.yaw)
    if any(abs(a - b) > 1e-6 for a, b in zip(seen, pose)):
        raise RuntimeError(f"step {step}: the last observation is at {seen} but the env reported {pose}; "
                           f"the video would show the wrong frame")
    return Frame(step=step, rgb=np.array(obs.rgb, copy=True), depth=np.array(obs.depth, dtype=np.float32, copy=True),
                 pose=pose, distance_m=float(info["final_distance_m"]), bearing_deg=float(info["bearing_deg"]),
                 command=None if action is None else tuple(float(a) for a in action), reward=float(reward))


def _spawned_objects(recorder: FrameRecorder) -> tuple[tuple[float, float, float], list[tuple[float, float, float]]]:
    targets = [xyz for name, xyz in recorder.spawned if name == "target"]
    if len(targets) != 1:
        raise RuntimeError(f"expected exactly one spawned target this episode, found {recorder.spawned}")
    return targets[0], [xyz for name, xyz in recorder.spawned if name != "target"]


def record_episode(model, env, recorder: FrameRecorder, *, index: int, seed: int, deterministic: bool,
                   max_steps: int, max_fault_retries: int) -> EpisodeRecording:
    """Fly one gate seed and keep every frame of the attempt that reaches a real outcome."""
    low, high = env.action_space.low, env.action_space.high
    retries = 0
    for _attempt in range(max_fault_retries + 1):
        obs, info = env.reset(seed=seed)
        target, distractors = _spawned_objects(recorder)
        measured = math.hypot(target[0] - info["pose"][0], target[1] - info["pose"][1])
        if abs(measured - info["final_distance_m"]) > 1e-3:
            raise RuntimeError(f"episode {index}: the spawned target is {measured:.3f} m away but the env measures "
                               f"{info['final_distance_m']:.3f} m; the map would show the wrong target")
        frames = [capture_frame(recorder, info, step=0, action=None, reward=0.0)]
        episode_return = 0.0
        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=deterministic)
            command = np.clip(np.asarray(action, dtype=np.float32).ravel(), low, high)  # what AutoFlyEnv.step() flies
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("sim_fault"):
                retries += 1
                print(f"episode {index} (seed {seed}): {info['sim_fault']} at step {step}; dropping this attempt's "
                      f"frames and flying the seed again", file=sys.stderr)
                break
            episode_return += float(reward)
            frames.append(capture_frame(recorder, info, step=step, action=command, reward=reward))
            if terminated or truncated:
                if int(info.get("steps", step)) != step:
                    raise RuntimeError(f"episode {index}: the env counted {info.get('steps')} steps, this loop {step}")
                return EpisodeRecording(
                    index=index, seed=seed, outcome=str(info.get("outcome", "unknown")), steps=step,
                    final_distance_m=float(info["final_distance_m"]), is_success=bool(info.get("is_success", False)),
                    episode_return=episode_return, oob_kind=info.get("oob_kind"), retries=retries, target=target,
                    distractors=distractors, frames=frames)
        else:
            # AutoFlyEnv's own step limit (300) always ends an episode first; as in evaluate.py, report rather than spin.
            return EpisodeRecording(
                index=index, seed=seed, outcome="exceeded_max_steps", steps=max_steps,
                final_distance_m=float(info["final_distance_m"]), is_success=False, episode_return=episode_return,
                oob_kind=None, retries=retries, target=target, distractors=distractors, frames=frames)
    raise RuntimeError(f"episode {index} (seed {seed}): no real outcome after {max_fault_retries} backend-fault retries")


# --------------------------------------------------------------------------------------------------------
# The gate record this replays.
# --------------------------------------------------------------------------------------------------------
def load_gate(path: Path | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path is not None and Path(path).is_file() else None


def gate_record(gate: dict | None, checkpoint_name: str, condition: str) -> tuple[str | None, dict[int, dict]]:
    """(the gate's sha256 for this checkpoint, {seed: its per-episode record}); (None, {}) without one."""
    checkpoint = (gate or {}).get("checkpoints", {}).get(checkpoint_name)
    if checkpoint is None:
        return None, {}
    per_episode = (checkpoint.get(condition) or {}).get("per_episode") or []
    return checkpoint.get("sha256"), {int(e["seed"]): e for e in per_episode}


def compare_with_gate(rec: EpisodeRecording, gate_episode: dict | None) -> dict:
    if gate_episode is None:
        return {"gate": None, "matches_gate": None, "final_pose_offset_m": None}
    gate_pose = gate_episode.get("final_pose")
    end = rec.frames[-1].pose
    return {
        "gate": {k: gate_episode.get(k) for k in ("outcome", "steps", "final_distance_m", "final_pose")},
        "matches_gate": gate_episode.get("outcome") == rec.outcome and gate_episode.get("steps") == rec.steps,
        "final_pose_offset_m": math.dist(end[:3], gate_pose[:3]) if gate_pose else None,
    }


def _gate_note(comparison: dict) -> str:
    gate = comparison["gate"]
    if gate is None:
        return "not in the gate record"
    if comparison["matches_gate"]:
        return "same outcome and step count as the gate"
    return f"the gate recorded {gate['outcome']} after {gate['steps']} steps"


# --------------------------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------------------------
def world_to_px(x: float, y: float, bounds: Bounds, size: int, margin: int) -> tuple[float, float]:
    """NED (x north, y east) to (col, row) on a square north-up map: north is up, east is right."""
    span = max(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min)
    s = (size - 2 * margin) / span
    return margin + (y - bounds.y_min) * s, margin + (bounds.x_max - x) * s


def _pt(col_row: tuple[float, float]) -> tuple[int, int]:
    return int(round(col_row[0])), int(round(col_row[1]))


def depth_to_bgr(depth: np.ndarray) -> np.ndarray:
    """Metres to colour: near is red, and DEPTH_CLIP_M (the policy's own clip, expert/obs.py) and beyond is blue.
    +inf (no hit, e.g. sky) reads as far."""
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=DEPTH_CLIP_M, posinf=DEPTH_CLIP_M, neginf=0.0)
    near = np.round(255.0 * (1.0 - np.clip(d, 0.0, DEPTH_CLIP_M) / DEPTH_CLIP_M)).astype(np.uint8)
    return cv2.applyColorMap(near, cv2.COLORMAP_TURBO)


def frame_size(scale: int) -> tuple[int, int]:
    """(width, height) of a video frame: three square panels over the status lines. Both even, as H.264 needs."""
    return 3 * CAMERA_PX * scale, (CAMERA_PX + HUD_PX) * scale


def _map_margin(size: int) -> int:
    return max(4, size // 64)


def _map_base(layout: Layout, rec: EpisodeRecording, size: int) -> np.ndarray:
    """The episode's static top-down picture: bounds, pillars, distractors, target (with its success radius), start."""
    b, margin = layout.bounds, _map_margin(size)
    px_per_m = (size - 2 * margin) / max(b.x_max - b.x_min, b.y_max - b.y_min)

    def to_px(x, y):
        return _pt(world_to_px(x, y, b, size, margin))

    img = np.full((size, size, 3), 28, np.uint8)
    cv2.rectangle(img, to_px(b.x_max, b.y_min), to_px(b.x_min, b.y_max), (95, 95, 95), 1)
    for inst in layout.instances:
        cv2.circle(img, to_px(inst.x, inst.y), max(2, round(inst.radius_m * px_per_m)), (205, 205, 205), -1, cv2.LINE_AA)
    half = max(2, round(0.5 * px_per_m))
    for dx, dy, _dz in rec.distractors:
        c = to_px(dx, dy)
        cv2.rectangle(img, (c[0] - half, c[1] - half), (c[0] + half, c[1] + half), (150, 120, 90), -1)
    tx, ty, _tz = rec.target
    cv2.circle(img, to_px(tx, ty), round(SUCCESS_RADIUS_M * px_per_m), (0, 140, 255), 1, cv2.LINE_AA)
    cv2.circle(img, to_px(tx, ty), max(3, round(0.7 * px_per_m)), (0, 140, 255), -1, cv2.LINE_AA)
    sx, sy = rec.frames[0].pose[:2]
    cv2.circle(img, to_px(sx, sy), max(3, round(0.7 * px_per_m)), (90, 200, 90), -1, cv2.LINE_AA)
    return img


def _fitting_size(text: str, size: float, thickness: int, max_width: int) -> float:
    """The font size, at most `size`, at which `text` fits `max_width` pixels: OpenCV clips silently."""
    while size > 0.2 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0][0] > max_width:
        size *= 0.9
    return size


def _label(img: np.ndarray, text: str, scale: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = _fitting_size(text, 0.42 * scale, 1, img.shape[1] - 8 * scale)
    (w, h), base = cv2.getTextSize(text, font, size, 1)
    cv2.rectangle(img, (0, 0), (w + 8 * scale, h + base + 6 * scale), (0, 0, 0), -1)
    cv2.putText(img, text, (4 * scale, h + 3 * scale), font, size, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(frame: Frame, rec: EpisodeRecording, base_map: np.ndarray, path_px: np.ndarray, *, scale: int,
                 label: str) -> np.ndarray:
    p = CAMERA_PX * scale
    rgb = cv2.resize(np.ascontiguousarray(frame.rgb[:, :, ::-1]), (p, p), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth_to_bgr(frame.depth), (p, p), interpolation=cv2.INTER_NEAREST)
    top = base_map.copy()
    if frame.step > 0:
        cv2.polylines(top, [path_px[: frame.step + 1]], False, (255, 220, 0), max(1, scale), cv2.LINE_AA)
    col, row = path_px[frame.step]
    yaw = frame.pose[3]
    heading = (int(round(col + 10 * scale * math.sin(yaw))), int(round(row - 10 * scale * math.cos(yaw))))
    cv2.line(top, (int(col), int(row)), heading, (255, 255, 255), max(1, scale), cv2.LINE_AA)
    cv2.circle(top, (int(col), int(row)), 3 * scale, (255, 255, 255), -1, cv2.LINE_AA)
    _label(rgb, "front camera (RGB)", scale)
    _label(depth, "depth (policy input), red = near", scale)
    _label(top, "top-down, north up", scale)
    final = frame.step == rec.steps
    if final:
        colour = OUTCOME_BGR.get(rec.outcome, (255, 255, 255))
        text = rec.outcome.replace("_", " ").upper()
        size = _fitting_size(text, 0.9 * scale, 2 * scale, p - 16 * scale)
        (w, _h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2 * scale)
        cv2.putText(top, text, (p - w - 8 * scale, p - 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, size, colour,
                    2 * scale, cv2.LINE_AA)

    hud = np.zeros((HUD_PX * scale, 3 * p, 3), np.uint8)
    lines = [
        f"{label}   gate episode {rec.index} (seed {rec.seed})   step {frame.step}/{rec.steps}   "
        f"t = {frame.step * CONTROL_DT_S:.1f} s",
        f"to target {frame.distance_m:5.1f} m   bearing {frame.bearing_deg:+4.0f} deg   "
        f"altitude {-frame.pose[2]:.1f} m",
        "reset: not moving yet" if frame.command is None else
        f"command: forward {frame.command[0]:.2f} m/s   yaw rate {frame.command[1]:+.2f} rad/s   "
        f"climb {frame.command[2]:+.2f} m/s",
    ]
    if final:
        lines[1] += f"   -> {rec.outcome.replace('_', ' ')}" + (f" ({rec.oob_kind})" if rec.oob_kind else "")
    for i, text in enumerate(lines):
        size = _fitting_size(text, 0.42 * scale, 1, 3 * p - 16 * scale)
        cv2.putText(hud, text, (8 * scale, (19 + 20 * i) * scale), cv2.FONT_HERSHEY_SIMPLEX, size,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([np.hstack([rgb, depth, top]), hud])


def episode_video_frames(rec: EpisodeRecording, layout: Layout, *, scale: int, label: str,
                         hold_frames: int) -> Iterable[np.ndarray]:
    size = CAMERA_PX * scale
    margin = _map_margin(size)
    base_map = _map_base(layout, rec, size)
    path_px = np.array([_pt(world_to_px(f.pose[0], f.pose[1], layout.bounds, size, margin)) for f in rec.frames],
                       dtype=np.int32)
    img = None
    for frame in rec.frames:
        img = render_frame(frame, rec, base_map, path_px, scale=scale, label=label)
        yield img
    for _ in range(hold_frames):  # hold the outcome on screen
        yield img


def write_video(frames: Iterable[np.ndarray], path: Path, fps: float, size: tuple[int, int]) -> str:
    """Stream frames into `path`; returns the codec used. OpenCV's own writer offers only mp4v, which QuickTime and
    browsers will not play, so the file is re-encoded to H.264 by ffmpeg (CPU libx264, leaving the shared GPU alone)
    when ffmpeg is installed."""
    raw = path.with_name(path.stem + ".mp4v.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {raw}")
    try:
        for img in frames:
            if (img.shape[1], img.shape[0]) != size:
                raise ValueError(f"frame is {img.shape[1]}x{img.shape[0]}, the video {size[0]}x{size[1]}")
            writer.write(img)
    finally:
        writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt",
                               "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)],
                              capture_output=True, text=True, timeout=600)
        if done.returncode == 0:
            raw.unlink()
            return "h264"
        print(f"WARNING: ffmpeg could not re-encode {raw.name} ({done.stderr.strip()[:300]}); keeping mp4v",
              file=sys.stderr)
    raw.replace(path)
    return "mp4v"


def draw_map(ax, layout: Layout, rec: EpisodeRecording, *, compact: bool = False) -> None:
    """The episode on matplotlib axes, east to the right and north up, like the video's map panel."""
    from matplotlib.patches import Circle, Rectangle

    b = layout.bounds
    ax.add_patch(Rectangle((b.y_min, b.x_min), b.y_max - b.y_min, b.x_max - b.x_min, fill=False, ec="0.5", lw=0.8))
    for inst in layout.instances:
        ax.add_patch(Circle((inst.y, inst.x), inst.radius_m, color="0.6", lw=0))
    ms = 4 if compact else 7
    for i, (dx, dy, _dz) in enumerate(rec.distractors):
        ax.plot(dy, dx, "s", color="0.25", ms=ms, label="distractors" if i == 0 else None)
    tx, ty, _tz = rec.target
    ax.add_patch(Circle((ty, tx), SUCCESS_RADIUS_M, fill=False, ec="tab:orange", ls="--", lw=0.9))
    ax.plot(ty, tx, "o", color="tab:orange", ms=ms + 1, label=f"target ({SUCCESS_RADIUS_M:.0f} m success radius)")
    xs = [f.pose[0] for f in rec.frames]
    ys = [f.pose[1] for f in rec.frames]
    ax.plot(ys, xs, "-", color="tab:blue", lw=1.0 if compact else 1.4, label="flown path")
    sx, sy, _sz, yaw = rec.frames[0].pose
    ax.plot(sy, sx, "^", color="tab:green", ms=ms, label="start")
    ax.annotate("", xy=(sy + 4.0 * math.sin(yaw), sx + 4.0 * math.cos(yaw)), xytext=(sy, sx),
                arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 1.2})
    marker, colour = OUTCOME_MARKER.get(rec.outcome, ("D", "black"))
    ax.plot(ys[-1], xs[-1], marker, color=colour, ms=ms + 4, mec="black", mew=0.6,
            label=f"end: {rec.outcome.replace('_', ' ')}")
    ax.set_xlim(b.y_min - 1.0, b.y_max + 1.0)
    ax.set_ylim(b.x_min - 1.0, b.x_max + 1.0)
    ax.set_aspect("equal")
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("east, y (m)")
        ax.set_ylabel("north, x (m)")


def save_map(path: Path, layout: Layout, rec: EpisodeRecording, *, label: str, note: str) -> None:
    from matplotlib.figure import Figure  # no pyplot: no global backend or window

    fig = Figure(figsize=(7.0, 7.6))
    ax = fig.add_subplot()
    draw_map(ax, layout, rec)
    ax.set_title(f"{label}: gate episode {rec.index} (seed {rec.seed})\n{rec.outcome.replace('_', ' ')} after "
                 f"{rec.steps} steps ({rec.steps * CONTROL_DT_S:.0f} s), ending {rec.final_distance_m:.1f} m from "
                 f"the target; {note}", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def save_overview(path: Path, layout: Layout, recs: list[EpisodeRecording], *, label: str) -> None:
    from matplotlib.figure import Figure

    ncols = min(4, len(recs))
    nrows = math.ceil(len(recs) / ncols)
    fig = Figure(figsize=(3.6 * ncols, 3.8 * nrows + 0.6), layout="constrained")  # keeps the title by the maps
    for i, rec in enumerate(recs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        draw_map(ax, layout, rec, compact=True)
        ax.set_title(f"ep {rec.index}: {rec.outcome.replace('_', ' ')} ({rec.steps} steps)", fontsize=9)
    fig.suptitle(f"{label}: top-down paths, north up (grey: pillars, orange: target)", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")


# --------------------------------------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------------------------------------
def run(*, scene: str, checkpoint_name: str, checkpoint_path: Path, episodes: list[int], condition: str,
        seed_base: int, instance: int, out_dir: Path, gate_path: Path | None, sim_factory: Callable[[], Any],
        load_model: Callable[[Path], Any], scene_config: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS_PER_EPISODE, max_fault_retries: int = DEFAULT_MAX_FAULT_RETRIES_PER_EPISODE,
        fps: float = 1.0 / CONTROL_DT_S, scale: int = 2, hold_s: float = 2.0, sim_root: Path = SIM_RUN_DIR) -> dict:
    out_dir, checkpoint_path = Path(out_dir), Path(checkpoint_path)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} already exists: use a fresh --out-dir rather than overwrite a render")
    if len(set(episodes)) != len(episodes) or any(i < 0 for i in episodes):
        raise ValueError(f"episode indices must be distinct and non-negative, got {episodes}")
    if condition not in ("deterministic", "stochastic"):
        raise ValueError(f"condition must be deterministic or stochastic, got {condition!r}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    _scene_file, layout = scene_and_layout(scene)
    gate_sha, gate_episodes = gate_record(load_gate(gate_path), checkpoint_name, condition)
    sha = sha256_of(checkpoint_path)
    label = f"{checkpoint_name} ({condition})"
    hold_frames = int(round(hold_s * fps))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "description": "Replayed M2 gate episodes rendered to video and top-down maps (scripts/render_episodes.py).",
        "scene": scene,
        "scene_config": scene_config_record(scene_config) if scene_config else None,
        "checkpoint": {"name": checkpoint_name, "path": str(checkpoint_path), "sha256": sha,
                       "matches_gate_sha256": None if gate_sha is None else gate_sha == sha},
        "condition": condition,
        "gate": str(gate_path) if gate_path is not None and Path(gate_path).is_file() else None,
        "seed_base": seed_base,
        "episodes_requested": list(episodes),
        "episodes": [],
        "n_compared": 0,
        "n_matching_gate": 0,
        "overview": None,
        "backend_faults": None,
        "status": "running",
        "error": None,
        "run_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_finished": None,
    }

    def _write() -> dict:
        summary["run_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    _write()
    swept = sweep_orphaned_instances(sim_root)  # a crashed earlier run's simulators; never a live run's
    if swept:
        print(f"swept orphaned instances before starting: {swept}", file=sys.stderr)
    recorder = FrameRecorder()
    recordings: list[EpisodeRecording] = []
    env = None
    try:
        model = load_model(checkpoint_path)
        env = build_eval_env(scene, instance=instance, sim_factory=lambda: RecordingSimulator(sim_factory(), recorder),
                             seed_base=seed_base, sim_root=sim_root)
        for index in episodes:
            seed = seed_base + index
            print(f"=== episode {index} (seed {seed}) ===", file=sys.stderr)
            rec = record_episode(model, env, recorder, index=index, seed=seed,
                                 deterministic=(condition == "deterministic"), max_steps=max_steps,
                                 max_fault_retries=max_fault_retries)
            comparison = compare_with_gate(rec, gate_episodes.get(seed))
            stem = f"ep{index:03d}_seed{seed}"
            codec = write_video(episode_video_frames(rec, layout, scale=scale, label=label, hold_frames=hold_frames),
                                out_dir / f"{stem}.mp4", fps, frame_size(scale))
            save_map(out_dir / f"{stem}_map.png", layout, rec, label=label, note=_gate_note(comparison))
            for frame in rec.frames:  # rendered: keep only what the maps and the summary need
                frame.rgb = frame.depth = None
            recordings.append(rec)
            summary["episodes"].append({
                "index": index, "seed": seed, "outcome": rec.outcome, "steps": rec.steps,
                "final_distance_m": rec.final_distance_m, "is_success": rec.is_success, "return": rec.episode_return,
                "oob_kind": rec.oob_kind, "retries": rec.retries, "target_xy_z": list(rec.target),
                "distractors": [list(d) for d in rec.distractors], "video": f"{stem}.mp4", "video_codec": codec,
                "map": f"{stem}_map.png", **comparison,
                "trajectory": [list(f.pose) for f in rec.frames],
                "commands": [None if f.command is None else list(f.command) for f in rec.frames],
            })
            summary["n_compared"] = sum(e["matches_gate"] is not None for e in summary["episodes"])
            summary["n_matching_gate"] = sum(e["matches_gate"] is True for e in summary["episodes"])
            print(f"episode {index}: {rec.outcome} after {rec.steps} steps ({_gate_note(comparison)})", file=sys.stderr)
            _write()
        save_overview(out_dir / "overview.png", layout, recordings, label=label)
        summary["overview"] = "overview.png"
        summary["status"] = "ok"
    except Exception as err:
        summary["status"] = "failed"
        summary["error"] = f"{type(err).__name__}: {err}"
        print(f"render_episodes failed: {summary['error']}", file=sys.stderr)
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                summary["backend_faults"] = env.get_fault_summary()
            except Exception as err:
                print(f"WARNING: get_fault_summary() raised {type(err).__name__}: {err}", file=sys.stderr)
            try:
                teardown(env, [instance], sim_root)  # bounded close + stop this run's own slot, nothing else
            except Exception as err:
                print(f"WARNING: teardown() raised {type(err).__name__}: {err}", file=sys.stderr)
                traceback.print_exc()
        else:
            try:
                stop_instances([instance], sim_root)
            except Exception as err:
                print(f"WARNING: stop_instances() raised {type(err).__name__}: {err}", file=sys.stderr)
    return _write()


# --------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="s01")
    p.add_argument("--run-root", type=Path, default=None,
                   help="the training run whose checkpoint to fly (default runs/expert/<scene>)")
    p.add_argument("--checkpoint", choices=("best_model", "final"), default="best_model")
    p.add_argument("--condition", choices=("deterministic", "stochastic"), default="deterministic")
    p.add_argument("--episodes", type=int, nargs="+", required=True,
                   help="gate episode indices i; each flies seed --seed-base + i")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--gate", type=Path, default=DEFAULT_GATE,
                   help="the gate record to compare with (a missing file skips the comparison)")
    p.add_argument("--scene-config", default=None,
                   help="Project AirSim scene config in configs/ (default scene_autofly_<scene>.jsonc): fly on the "
                        "clock the expert was trained and gated on")
    p.add_argument("--instance", type=int, default=0)
    p.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    p.add_argument("--device", default="auto")
    p.add_argument("--fps", type=float, default=1.0 / CONTROL_DT_S, help="default: real time (one step is 0.2 s)")
    p.add_argument("--scale", type=int, default=2, help="video panels are 256 x scale pixels")
    p.add_argument("--hold-s", type=float, default=2.0, help="seconds the last frame stays on screen")
    return p


def checkpoint_path(args: argparse.Namespace) -> Path:
    return parse_model_args(None, args.scene, run_root=args.run_root)[args.checkpoint]


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scene_config = args.scene_config or f"scene_autofly_{args.scene}.jsonc"
    summary = run(scene=args.scene, checkpoint_name=args.checkpoint, checkpoint_path=checkpoint_path(args),
                  episodes=list(args.episodes), condition=args.condition, seed_base=args.seed_base,
                  instance=args.instance, out_dir=args.out_dir, gate_path=args.gate,
                  sim_factory=scene_config_factory(scene_config),
                  load_model=lambda path: default_sac_loader(path, device=args.device), scene_config=scene_config,
                  fps=args.fps, scale=args.scale, hold_s=args.hold_s)
    print(json.dumps({"status": summary["status"], "error": summary["error"], "out_dir": str(args.out_dir),
                      "n_compared": summary["n_compared"], "n_matching_gate": summary["n_matching_gate"]}, indent=2))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    _code = main()
    # Not sys.exit(): as in m2_gate.py, the projectairsim client leaves a non-daemon thread that blocks interpreter
    # shutdown forever. Everything durable (summary.json, videos, maps) is already written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
