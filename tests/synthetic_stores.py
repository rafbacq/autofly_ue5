"""Synthetic raw stores and layouts for the report tools' tests (autofly_ue5/report, scripts/dataset_figures.py,
scripts/dataset_montage_video.py).

Stores are written by the real RawDatasetWriter (dataset/raw.py), so every file sits where a collected store has it,
and each flight is a straight line from its start to 4 m short of its target, so every summary a test expects is known
exactly. A frame is one solid colour whose brightness encodes its record index (`frame_level`), so a video test can
tell which record a tile shows. Layouts are written for the real scene files (resolve_scene checks their sha256)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autofly_ue5.dataset.raw import RawDatasetWriter
from autofly_ue5.dataset.state import STATE2_OFFSET_M
from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES
from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.model import SceneFile, load_scene_file

STEP_NS = 200_000_000
ALTITUDE_M = 2.0
SPEED_M_S = 2.0
END_SHORT_M = 4.0  # the last record is this far from the target: inside the 5 m success radius
COMMON_PROVENANCE = {"expert": {"path": "best_model.zip", "sha256": "0" * 64, "deterministic": False},
                     "platform": {"projectairsim_client": "test"}, "collector_git_head": "test"}


def scene_file(scene_id: str) -> SceneFile:
    (path,) = sorted(SCENES_DIR.glob(f"{scene_id}_*.json"))
    return load_scene_file(path)


def pool_target(name: str, category: str, *, seen: bool = True, status: str = "imported",
                stand_in: bool | None = None) -> dict:
    """A target as a pool-drawn episode's provenance records it (dataset/targets.py TargetEntry.to_json, plus the
    collector's additions); `stand_in` only where given, as before plan 6 step 2b's mark."""
    record = {"name": name, "category": category, "seen": seen, "asset": f"tgt_{name.replace(' ', '_')}",
              "scale": [1.0, 1.0, 1.0], "material": None, "status": status, "material_applied": "mesh_default",
              "spawn_z_ned": 0.0, "name_status": "pool"}
    if stand_in is not None:
        record["stand_in"] = stand_in
    return record


def legacy_target(name: str = "orange cylinder") -> dict:
    """M3's target record: a placeholder name, no category, seen or status (data/s01_pilot's provenance)."""
    return {"name": name, "name_status": "placeholder_until_M4", "asset": "cylinder", "material": "orange",
            "scale": [1.0, 1.0, 2.0]}


@dataclass
class Flight:
    seed: int
    steps: int
    target: dict
    start: tuple[float, float] = (-30.0, 0.0)
    target_xy: tuple[float, float] = (30.0, 0.0)
    bearing_end_rad: float = 0.0  # state[1] at the last record
    colour: tuple[int, int, int] = (200, 120, 40)
    movers: list = field(default_factory=list)  # provenance "movers": route records, each with a "tag"
    template: int = 0
    extra: dict = field(default_factory=dict)  # further provenance keys (prewarm_steps, first_frame_visibility, ...)


def frame_level(i: int, n: int) -> float:
    """The brightness factor of record i of n: 0.25 at the first record, 1.0 at the last."""
    return 0.25 + 0.75 * (i / (n - 1) if n > 1 else 1.0)


def frame(colour: tuple[int, int, int], i: int, n: int) -> np.ndarray:
    level = frame_level(i, n)
    return np.full((256, 256, 3), [round(c * level) for c in colour], dtype=np.uint8)


def flight_records(f: Flight) -> dict:
    """states, actions, times, poses and the final pose of a straight flight at its target."""
    sx, sy = f.start
    tx, ty = f.target_xy
    length = math.hypot(tx - sx, ty - sy)
    ux, uy = (tx - sx) / length, (ty - sy) / length
    yaw = math.atan2(uy, ux)
    run = length - END_SHORT_M
    poses, states = [], []
    for i in range(f.steps):
        d = run * (i / (f.steps - 1) if f.steps > 1 else 1.0)
        x, y = sx + ux * d, sy + uy * d
        poses.append([x, y, -ALTITUDE_M, yaw])
        bearing = f.bearing_end_rad if i == f.steps - 1 else 0.0
        states.append([math.hypot(tx - x, ty - y), bearing, ALTITUDE_M - STATE2_OFFSET_M, 0.0 if i == 0 else SPEED_M_S,
                       0.0, 0.0, x - sx, y - sy, ALTITUDE_M])
    last = poses[-1]
    return {"states": np.asarray(states, dtype=np.float32),
            "actions": np.tile(np.asarray([SPEED_M_S, 0.0, 0.0], dtype=np.float32), (f.steps, 1)),
            "times": np.asarray([1_000_000_000 + i * STEP_NS for i in range(f.steps)], dtype=np.int64),
            "poses": poses, "final_pose": [last[0] + ux * 0.4, last[1] + uy * 0.4, -ALTITUDE_M, yaw]}


def _provenance(scene: SceneFile, f: Flight, records: dict, termination: str, collision_source: str | None = None) -> dict:
    return {"scene": {"id": scene.id, "file": scene.path, "sha256": scene.sha256}, "layout_sha256": "test-layout",
            "seed": f.seed, "target": {**f.target, "spawned_as": "target", "xyz": [f.target_xy[0], f.target_xy[1], -1.0]},
            "distractors": [], "start_pose": [f.start[0], f.start[1], -ALTITUDE_M, records["poses"][0][3]],
            "a0": {"mode": "sector8", "start_yaw_rad": records["poses"][0][3]}, "movers": list(f.movers),
            "mover_positions_per_record": [[] for _ in range(f.steps)], "poses_per_record": records["poses"],
            "sim_time_ns": [int(t) for t in records["times"]], "final_pose": records["final_pose"],
            "termination": termination, "collision_source": collision_source, "oob_kind": None,
            "final_distance_m": END_SHORT_M - 0.4, "deterministic": False, "pilot": "sac_expert", **f.extra}


def write_store(data_root: Path, name: str, scene_id: str, flights: list[Flight], *,
                rejects: list[tuple[Flight, str, str | None]] = ()) -> Path:
    """A store of `flights` (all kept) and `rejects` ((flight, reason, collision_source)), in that order."""
    scene = scene_file(scene_id)
    writer = RawDatasetWriter(Path(data_root), name, scene=scene, provenance={**COMMON_PROVENANCE, "dataset": name})
    for f in flights:
        records = flight_records(f)
        target_name = f.target["name"]
        writer.write_episode(
            episode_id=f"{scene_id}_{f.seed}", scene_id=scene_id, split=scene.split, seed=f.seed,
            instruction=INSTRUCTION_TEMPLATES[f.template].format(target=target_name, obstacle=scene.instruction_obstacle),
            target_name=target_name, frames=[frame(f.colour, i, f.steps) for i in range(f.steps)],
            states=records["states"], actions=records["actions"], sim_time_ns=records["times"],
            provenance=_provenance(scene, f, records, "success"))
    for f, reason, source in rejects:
        records = flight_records(f)
        writer.write_reject(episode_id=f"{scene_id}_{f.seed}", reason=reason,
                            provenance=_provenance(scene, f, records, reason, source))
    return Path(data_root) / name


def write_layout(levels_dir: Path, scene_id: str, instances: list[dict]) -> Path:
    """`runs/levels/<id>.layout.json` for a real scene file, holding `instances` (each a layout instance's fields)."""
    scene = scene_file(scene_id)
    levels_dir = Path(levels_dir)
    levels_dir.mkdir(parents=True, exist_ok=True)
    b = scene.bounds
    path = levels_dir / f"{scene_id}.layout.json"
    path.write_text(json.dumps({
        "scene_path": scene.path, "scene_sha256": scene.sha256,
        "layout": {"scene_id": scene_id, "seed": scene.seed,
                   "bounds": {"x_min": b.x_min, "x_max": b.x_max, "y_min": b.y_min, "y_max": b.y_max},
                   "instances": instances}}, indent=1))
    return path


def instance(tag: str, asset: str, x: float, y: float, *, scale=(1.0, 1.0, 1.0), yaw: float = 0.0,
             radius_m: float = 0.5, height_m: float = 2.0, z_center: float = -1.0, material: str = "white") -> dict:
    return {"tag": tag, "asset": asset, "x": x, "y": y, "z_center": z_center, "yaw": yaw, "scale": list(scale),
            "material": material, "radius_m": radius_m, "height_m": height_m}
