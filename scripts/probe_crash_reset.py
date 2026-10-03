"""Live probe: does a reset straight after a crash put the drone where it was asked (C9, 2026-09-24 review)?

The recorded M2 gate had 15 one-step "collisions" whose drone was >= 4 m from its start -- every one right after a
collision episode (15 of 73 such resets, 0 of 723 others). This flies the drone into a pillar, then resets it to an
episode start pose (drawn exactly as training does), exactly as `AutoFlyEnv.reset()` follows a crashed episode, and
records how far off the reset landed, whether the next zero-velocity step (the env's render step) reports a
collision or a jump, and whether a second reset fixes a bad one. The backend's own C9 checks are disabled here so the
raw behaviour is measured rather than raised.

    bash scripts/run_job.sh start crash_reset_probe -- env -u PYTHONPATH .venv/bin/python scripts/probe_crash_reset.py \
        --trials 30 --out runs/m1/crash_reset_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from autofly_ue5.expert.episode import sample_setup  # noqa: E402
from autofly_ue5.expert.seeds import PROBE_SEED_BASE  # noqa: E402  (disjoint from every other range)
from autofly_ue5.frames import wrap_pi  # noqa: E402
from autofly_ue5.paths import RUNS_DIR  # noqa: E402
from autofly_ue5.scenes.model import Bounds  # noqa: E402
from autofly_ue5.sim.types import Pose  # noqa: E402
from autofly_ue5.validate.geometry import choose_depth_probes, pillars_from_layout_json  # noqa: E402

DT = 0.2
POSITION_TOLERANCE_M = 0.3  # the backend's ResetPoseError thresholds
YAW_TOLERANCE_RAD = 0.1
JUMP_M = 2.0  # 10 m/s x 0.2 s: the backend's KinematicsJumpError threshold


def _attempt_reset(sim, pose: Pose) -> dict:
    try:
        obs = sim.reset(pose)
    except Exception as err:  # CameraPoseError and friends: recorded, not fatal to the probe
        return {"raised": f"{type(err).__name__}: {err}", "bad": True}
    got = obs.pose
    position_error = math.dist((got.x, got.y, got.z), (pose.x, pose.y, pose.z))
    yaw_error = abs(wrap_pi(got.yaw - pose.yaw))
    return {"position_error_m": position_error, "yaw_error_rad": yaw_error, "pose": [got.x, got.y, got.z, got.yaw],
            "bad": position_error > POSITION_TOLERANCE_M or yaw_error > YAW_TOLERANCE_RAD}


def _render_step(sim) -> dict:
    before = sim.observe().pose
    sim.command_velocity(0.0, 0.0, 0.0)
    sim.step(DT)
    obs = sim.observe()
    moved = math.hypot(obs.pose.x - before.x, obs.pose.y - before.y)
    return {"collided": bool(obs.collided), "moved_m": moved, "jump": moved > JUMP_M}


def crash_reset_trial(sim, probe, start: Pose, max_steps: int = 25) -> dict:
    trial: dict = {"start": [start.x, start.y, start.z, start.yaw]}
    approach = _attempt_reset(sim, probe.pose)
    if approach.get("raised"):
        return {**trial, "approach_reset": approach, "crashed": False}
    steps_to_collision = None
    for step in range(max_steps):
        sim.command_velocity(2.0, 0.0, 0.0)
        sim.step(DT)
        if sim.observe().collided:
            steps_to_collision = step + 1
            break
    crash = sim.observe().pose
    trial.update(crashed=steps_to_collision is not None, steps_to_collision=steps_to_collision,
                 crash_pose=[crash.x, crash.y, crash.z, crash.yaw])
    first = _attempt_reset(sim, start)
    trial["first_reset"] = first
    if not first.get("raised"):
        trial["render_step"] = _render_step(sim)
    if first["bad"]:
        second = _attempt_reset(sim, start)
        trial["second_reset"] = second
    return trial


def run_probe(sim, probe, scene, layout, trials: int) -> dict:
    rows = []
    for i in range(trials):
        start = sample_setup(scene, layout, np.random.default_rng(PROBE_SEED_BASE + i)).start
        try:
            rows.append(crash_reset_trial(sim, probe, start))
        except Exception as err:  # e.g. a CameraPoseError mid-flight: record it, keep the other trials
            rows.append({"start": [start.x, start.y, start.z, start.yaw], "error": f"{type(err).__name__}: {err}"})
    crashed = [t for t in rows if t.get("crashed")]
    bad = [t for t in crashed if t["first_reset"]["bad"]]
    summary = {
        "trials": trials,
        "errors": sum(1 for t in rows if "error" in t),
        "crashed": len(crashed),
        "bad_first_reset": len(bad),
        "fixed_by_second_reset": sum(1 for t in bad if not t.get("second_reset", {}).get("bad", True)),
        "render_step_collided": sum(1 for t in crashed if t.get("render_step", {}).get("collided")),
        "render_step_jump": sum(1 for t in crashed if t.get("render_step", {}).get("jump")),
        "worst_first_reset_error_m": max((t["first_reset"].get("position_error_m", math.inf) for t in crashed), default=None),
    }
    return {"summary": summary, "trials": rows}


def main(argv: list[str] | None = None) -> int:
    from autofly_ue5.scenes.resolve import resolve_scene
    from autofly_ue5.sim.airsim_backend import ProjectAirSimSimulator, scene_config_record
    from autofly_ue5.sim.process import instance_dir, route_client_log, sweep_orphaned_instances

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--scene", default="s01")
    parser.add_argument("--scene-config", default="scene_autofly_s01.jsonc")
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "m1" / "crash_reset_probe.json")
    args = parser.parse_args(argv)

    resolved = resolve_scene(args.scene)
    scene, layout = resolved.scene, resolved.layout
    layout_json = {"layout": layout.to_json()}
    b = layout_json["layout"]["bounds"]
    probe = choose_depth_probes(pillars_from_layout_json(layout_json), Bounds(b["x_min"], b["x_max"], b["y_min"], b["y_max"]))[1]
    swept = sweep_orphaned_instances()
    if swept:
        print(f"swept orphaned instances before starting: {swept}", file=sys.stderr)
    route_client_log(instance_dir(args.instance) / "client.log")
    # The backend's own C9 checks off: this measures what they would react to.
    sim = ProjectAirSimSimulator(scene_config=args.scene_config, reset_position_tolerance_m=math.inf,
                                 reset_yaw_tolerance_rad=math.inf, max_speed_m_s=math.inf)
    report: dict = {"description": __doc__.split("\n\n")[0], "scene_config": scene_config_record(args.scene_config),
                    "instance": args.instance,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        sim.launch(resolved.map_path, args.instance)
        report.update(run_probe(sim, probe, scene, layout, args.trials))
        status = 0
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
        status = 1
    finally:
        try:
            sim.close()
        except Exception as err:
            print(f"WARNING: close() raised {type(err).__name__}: {err}", file=sys.stderr)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("summary", {"error": report.get("error")}), indent=2))
    return status


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)  # the projectairsim client leaves a non-daemon thread that blocks normal interpreter exit
