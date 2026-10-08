"""Crash into every mesh obstacle of a scene and reset: can the backend free the drone from foliage every time?

    bash scripts/run_job.sh start canopy_probe -- env -u PYTHONPATH .venv/bin/python scripts/probe_canopy_reset.py \\
        --scene s02 --instance 5 --out runs/m4/s02/canopy_reset_probe.json

Why: a reset after a crash into a tree's canopy left the swept Unreal actor among the leaf cards while the kinematics
moved on (s02's live checks, 2026-10-07); the backend's reset now climbs out by an escape ladder
(sim/airsim_backend.RESET_ESCAPES). One crash is one sample of a non-deterministic contact; this flies into each tree
from a clear bearing at the altitudes given, records which rung freed the actor, how many steps the reset cost, and
every failure. Ends with os._exit like the other live scripts (the projectairsim client's non-daemon thread).
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
import os  # noqa: E402
import time  # noqa: E402

from autofly_ue5.scenes.model import Bounds, load_registry  # noqa: E402
from autofly_ue5.scenes.resolve import resolve_scene  # noqa: E402
from autofly_ue5.sim.types import Pose  # noqa: E402
from autofly_ue5.validate.geometry import CAMERA_OFFSET_M  # noqa: E402
from autofly_ue5.validate.live_m1 import scene_poses  # noqa: E402
from autofly_ue5.validate.obstacles import MeshLibrary, mesh_keys, obstacles_from_layout_json  # noqa: E402

APPROACH_M = 6.0
MAX_STEPS = 40
DT = 0.2


def approach_pose(obstacle, others, bounds: Bounds, altitude_m: float) -> Pose | None:
    for k in range(8):
        yaw = k * math.pi / 4
        ux, uy = math.cos(yaw), math.sin(yaw)
        standoff = obstacle.radius_m + APPROACH_M + CAMERA_OFFSET_M
        x, y = obstacle.x - standoff * ux, obstacle.y - standoff * uy
        if not (bounds.x_min + 2 <= x <= bounds.x_max - 2 and bounds.y_min + 2 <= y <= bounds.y_max - 2):
            continue
        if any(math.hypot(x - q.x, y - q.y) < q.radius_m + 1.5 for q in others):
            continue
        return Pose(x, y, -altitude_m, yaw)
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="s02")
    p.add_argument("--instance", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--altitudes", type=float, nargs="+", default=[2.0, 2.5])
    p.add_argument("--scene-config", default=None)
    args = p.parse_args(argv)
    from autofly_ue5.sim.airsim_backend import ProjectAirSimSimulator

    resolved = resolve_scene(args.scene)
    config = args.scene_config or resolved.default_scene_config.replace(".jsonc", "_fast.jsonc")
    data = json.loads(resolved.layout_path.read_text())
    registry = load_registry()
    obstacles = obstacles_from_layout_json(data, registry, MeshLibrary.from_files(mesh_keys(data, registry)))
    meshes = [o for o in obstacles if o.kind == "mesh"]
    b = data["layout"]["bounds"]
    bounds = Bounds(b["x_min"], b["x_max"], b["y_min"], b["y_max"])
    home = scene_poses(bounds).field_view
    report = {"scene": args.scene, "scene_config": config, "altitudes_m": args.altitudes, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "trials": []}
    sim = ProjectAirSimSimulator(scene_config=config)
    try:
        sim.launch(resolved.map_path, args.instance)
        sim.reset(home)
        for altitude in args.altitudes:
            for o in meshes:
                pose = approach_pose(o, [q for q in obstacles if q.tag != o.tag], bounds, altitude)
                trial = {"tag": o.tag, "altitude_m": altitude, "approach": None if pose is None else [pose.x, pose.y, pose.z, pose.yaw]}
                report["trials"].append(trial)
                if pose is None:
                    trial["outcome"] = "no clear approach"
                    continue
                try:
                    sim.reset(pose)
                    collided = None
                    for step in range(MAX_STEPS):
                        sim.command_velocity(2.0, 0.0, 0.0)
                        sim.step(DT)
                        obs = sim.observe()
                        if obs.step_collisions:
                            ev = obs.step_collisions[0]
                            collided = {"step": step + 1, "object": ev.object_name, "impact": list(ev.impact_point), "normal": list(ev.normal),
                                        "pose": [obs.pose.x, obs.pose.y, obs.pose.z, obs.pose.yaw]}
                            break
                    trial["collision"] = collided
                    if collided is None:
                        trial["outcome"] = "no collision"
                        continue
                    steps_before, t0 = sim.steps_taken, time.monotonic()
                    after = sim.reset(home)
                    trial["reset"] = {"escape": list(sim.last_reset_escape), "steps": sim.steps_taken - steps_before,
                                      "seconds": round(time.monotonic() - t0, 2), "camera_pose_error_m": after.camera_pose_error_m,
                                      "pose_error_m": math.dist((after.pose.x, after.pose.y, after.pose.z), (home.x, home.y, home.z))}
                    trial["outcome"] = "reset ok"
                except Exception as err:  # noqa: BLE001 -- every failure is the result
                    trial["outcome"] = f"{type(err).__name__}: {err}"
                    # a second reset is what the training wrapper would do next; record whether it works
                    try:
                        sim.reset(home)
                        trial["retry"] = {"escape": list(sim.last_reset_escape), "ok": True}
                    except Exception as err2:  # noqa: BLE001
                        trial["retry"] = {"ok": False, "error": f"{type(err2).__name__}: {err2}"}
                        break
                print(json.dumps(trial), flush=True)
    finally:
        try:
            sim.close()
        finally:
            outcomes = {}
            for t in report["trials"]:
                key = t.get("outcome", "?").split(":")[0]
                outcomes[key] = outcomes.get(key, 0) + 1
            escapes = {}
            for t in report["trials"]:
                if "reset" in t:
                    k = "%s %.1f" % tuple(t["reset"]["escape"])
                    escapes[k] = escapes.get(k, 0) + 1
            report["summary"] = {"outcomes": outcomes, "escapes": escapes, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
