"""Live go/no-go for moving pillars (spec §6.5, milestone M2d): can a *baked* s01 pillar be teleported at runtime,
and does the simulator then render it and collide with it where it now stands?

M0 moved a spawned cube to within 5e-7 m and M1 found the baked pillars by tag, but nothing has ever moved a baked
actor. Scene s01d moves 8-12 of them every step, so before any s01d code is trusted this measures, on the packaged S01
map, every assumption it rests on (docs/decisions/2026-10-02-dynamic-obstacles.md):

  a. a tagged pillar moves to within 0.05 m of where it was sent, and its neighbours do not move;
  b. sending it home restores it to within 0.05 m;
  c. the first frame after a move shows the pillar where it now is: centre depth within 0.3 m of the camera geometry;
  d. flying into a moved pillar collides, also on the very first step after the move;
  e. flying through a vacated home spot gives no collision and no CameraPoseError;
  f. parking 50 m underground (how s01d clears displaced pillars before a reset) and restoring both work.

It also times a 12-pose batch (one request per name, as the backend sends it) against the same batch sent
concurrently with projectairsim's request_async, and the wall time of a step with and without 10 moves before it:
the throughput cost of s01d. request_async disconnects the whole client on any error (client.py:check_for_exception),
so it is timed last.

The output is the M2d go/no-go: copy it to docs/gates/m2d_mover_probe.json whether or not it passes (docs/runbook-m2d.md).

    bash scripts/run_job.sh start probe_movers -- env -u PYTHONPATH .venv/bin/python scripts/probe_movers.py \\
        --scene-config scene_autofly_s01_fast.jsonc --out runs/m2d/mover_probe.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from autofly_ue5.paths import RUNS_DIR  # noqa: E402
from autofly_ue5.scenes.model import Instance, Layout  # noqa: E402
from autofly_ue5.sim.types import CONTROL_DT_S, Pose  # noqa: E402
from autofly_ue5.validate.geometry import CAMERA_OFFSET_M, choose_depth_probes, pillars_from_layout_json  # noqa: E402

MOVE_TOLERANCE_M = 0.05
NEIGHBOUR_DRIFT_M = 0.01
DEPTH_TOLERANCE_M = 0.3
PARK_DEPTH_M = 50.0  # the same parking depth expert/movers.py uses
# How far the rotors reach straight ahead: props at x = +0.253 m with radius 0.1143 m (configs/robot_autofly_quadrotor.jsonc;
# 0.472 m is the diagonal, at 45 degrees).
FRONT_TIP_M = 0.253 + 0.1143
AHEAD_M = 4.0  # check c: the pillar's near surface, this far in front of the camera
# Check d2. From hover the drone is slow to speed up: M0 flew 2.33 m in its first 10 steps at a 2 m/s command
# (docs/gates/m0_smoke_inst0.json), so one step from hover covers only centimetres. d2 first flies a clear runway
# for RUNWAY_STEPS, then puts the pillar's surface FIRST_STEP_FRACTION of the last step's travel beyond the rotor tips:
# the next step can only collide if the moved pillar is solid at once.
RUNWAY_STEPS = 10
FIRST_STEP_FRACTION = 0.5
MIN_RUNWAY_STEP_M = 0.2  # the drone must be up to speed (M1 measured 1.81 m/s, 0.36 m per step)
MAX_FLY_STEPS = 40
LATENCY_BATCH = 12
LATENCY_REPEATS = 20
STEP_TIMING_STEPS = 20
STEP_TIMING_MOVERS = 10
ALTITUDE_M = 2.0


def _home(inst: Instance) -> Pose:
    return Pose(inst.x, inst.y, inst.z_center, 0.0)


def _offset(pose: Pose, dx: float, dy: float, dz: float = 0.0) -> Pose:
    return Pose(pose.x + dx, pose.y + dy, pose.z + dz, pose.yaw)


def _error(a: Pose | None, b: Pose) -> float:
    return math.inf if a is None else math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def _centre_depth(obs) -> float:
    """Median planar depth of the image's central 5 x 5 pixels (+inf for no hit)."""
    d = np.asarray(obs.depth, dtype=np.float32)
    cy, cx = d.shape[0] // 2, d.shape[1] // 2
    return float(np.median(d[cy - 2:cy + 3, cx - 2:cx + 3]))


def _hold(sim) -> object:
    sim.command_velocity(0.0, 0.0, 0.0)
    sim.step(CONTROL_DT_S)
    return sim.observe()


def _fly(sim, steps: int, v: float = 2.0) -> tuple[int | None, object]:
    """Fly straight ahead; the 1-based step of the first collision (None if none) and the last observation."""
    obs = sim.observe()
    for i in range(steps):
        sim.command_velocity(v, 0.0, 0.0)
        sim.step(CONTROL_DT_S)
        obs = sim.observe()
        if obs.collided:
            return i + 1, obs
    return None, obs


def _surface_gap(pose: Pose, centre: Pose, radius: float) -> float:
    return math.hypot(pose.x - centre.x, pose.y - centre.y) - radius


def _check(name: str, fn) -> dict:
    """Run one check; an exception is recorded as that check failing, never as the probe dying."""
    try:
        result = fn()
    except Exception as err:
        traceback.print_exc()
        return {"pass": False, "error": f"{type(err).__name__}: {err}"}
    print(f"check {name}: {'pass' if result['pass'] else 'FAIL'} {json.dumps(result)}", file=sys.stderr)
    return result


def _clear_line(layout: Layout, sx: float, sy: float, ex: float, ey: float, clearance_m: float, skip: str | None = None) -> bool:
    b = layout.bounds
    if not all(b.x_min + 2 <= x <= b.x_max - 2 and b.y_min + 2 <= y <= b.y_max - 2 for x, y in ((sx, sy), (ex, ey))):
        return False
    vx, vy = ex - sx, ey - sy
    for q in layout.instances:
        if q.tag == skip:
            continue
        t = max(0.0, min(1.0, ((q.x - sx) * vx + (q.y - sy) * vy) / (vx * vx + vy * vy)))
        if math.hypot(q.x - (sx + t * vx), q.y - (sy + t * vy)) < q.radius_m + clearance_m:
            return False
    return True


def _runway(layout: Layout, length_m: float = 14.0, clearance_m: float = 2.0) -> Pose:
    """A start pose with `length_m` of open air ahead (every pillar >= clearance_m from the line): in s01, the free ring
    around the pillar field."""
    b = layout.bounds
    for x in np.linspace(b.x_min + 4.0, b.x_max - 4.0, 15):
        for y in np.linspace(b.y_min + 4.0, b.y_max - 4.0, 15):
            for k in range(8):
                yaw = k * math.pi / 4
                ex, ey = x + length_m * math.cos(yaw), y + length_m * math.sin(yaw)
                if _clear_line(layout, float(x), float(y), ex, ey, clearance_m):
                    return Pose(float(x), float(y), -ALTITUDE_M, yaw)
    raise ValueError(f"no {length_m} m runway with {clearance_m} m clearance")


def _through_line(layout: Layout, approach_m: float = 3.0, beyond_m: float = 3.0, clearance_m: float = 1.0):
    """A pillar and a straight line that runs through its home spot with every other pillar >= clearance_m away from
    the line (check e flies it with the pillar parked)."""
    for inst in sorted(layout.instances, key=lambda i: i.tag):
        for k in range(8):
            yaw = k * math.pi / 4
            ux, uy = math.cos(yaw), math.sin(yaw)
            sx, sy = inst.x - (approach_m + inst.radius_m) * ux, inst.y - (approach_m + inst.radius_m) * uy
            ex, ey = inst.x + beyond_m * ux, inst.y + beyond_m * uy
            if _clear_line(layout, sx, sy, ex, ey, clearance_m, skip=inst.tag):
                return inst, Pose(sx, sy, -ALTITUDE_M, yaw)
    raise ValueError("no pillar has a clear line through its home spot")


def run_probe(sim, layout: Layout, *, async_batch=None) -> dict:
    """Every check against a launched simulator that can move the layout's tags (`sim.get_object_poses` reads back).
    `async_batch(poses)`: sends one batch concurrently (the live backend's request_async); None skips that timing."""
    by_tag = {inst.tag: inst for inst in layout.instances}
    probe = choose_depth_probes(pillars_from_layout_json({"layout": layout.to_json()}), layout.bounds, distances=(6.0,))[0]
    victim = by_tag[probe.tag]
    home = _home(victim)
    view = probe.pose
    ux, uy = math.cos(view.yaw), math.sin(view.yaw)
    neighbours = sorted((i for i in layout.instances if i.tag != victim.tag),
                        key=lambda i: math.hypot(i.x - victim.x, i.y - victim.y))[:3]
    report: dict = {"victim": victim.tag, "victim_home": [home.x, home.y, home.z], "view_pose": list(map(float, (
        view.x, view.y, view.z, view.yaw))), "neighbours": [n.tag for n in neighbours], "checks": {}}
    checks = report["checks"]

    def restore_victim() -> None:
        sim.set_object_poses({victim.tag: home})

    def a_move() -> dict:
        before = sim.get_object_poses([victim.tag] + [n.tag for n in neighbours])
        home_error = _error(before[victim.tag], home)
        target = _offset(home, 1.0, 0.5)
        sim.set_object_poses({victim.tag: target})
        _hold(sim)
        after = sim.get_object_poses([victim.tag] + [n.tag for n in neighbours])
        moved_error = _error(after[victim.tag], target)
        drift = max(_error(after[n.tag], before[n.tag]) if before[n.tag] is not None else math.inf for n in neighbours)
        return {"pass": home_error <= MOVE_TOLERANCE_M and moved_error <= MOVE_TOLERANCE_M and drift <= NEIGHBOUR_DRIFT_M,
                "home_readback_error_m": home_error, "moved_error_m": moved_error, "neighbour_drift_m": drift}

    def b_restore() -> dict:
        restore_victim()
        _hold(sim)
        error = _error(sim.get_object_poses([victim.tag])[victim.tag], home)
        return {"pass": error <= MOVE_TOLERANCE_M, "restored_error_m": error}

    def c_depth() -> dict:
        sim.reset(view)
        sim.set_object_poses({victim.tag: _offset(home, 0.0, 0.0, PARK_DEPTH_M)})
        before = _centre_depth(_hold(sim))
        cam = (view.x + CAMERA_OFFSET_M * ux, view.y + CAMERA_OFFSET_M * uy)
        centre = Pose(cam[0] + (AHEAD_M + victim.radius_m) * ux, cam[1] + (AHEAD_M + victim.radius_m) * uy, home.z, 0.0)
        sim.set_object_poses({victim.tag: centre})
        after = _centre_depth(_hold(sim))  # the first frame rendered after the move
        return {"pass": abs(after - AHEAD_M) <= DEPTH_TOLERANCE_M and before > AHEAD_M + 2.0,
                "expected_m": AHEAD_M, "depth_after_move_m": after, "depth_before_move_m": before,
                "pillar_at": [centre.x, centre.y]}

    def d_collide() -> dict:
        # d1: the pillar c left 4 m ahead; fly into it from hover.
        centre = Pose(view.x + (CAMERA_OFFSET_M + AHEAD_M + victim.radius_m) * ux,
                      view.y + (CAMERA_OFFSET_M + AHEAD_M + victim.radius_m) * uy, home.z, 0.0)
        sim.reset(view)
        sim.set_object_poses({victim.tag: centre})
        step, obs = _fly(sim, MAX_FLY_STEPS)
        gap = _surface_gap(obs.pose, centre, victim.radius_m)
        d1 = {"collided_at_step": step, "surface_gap_at_collision_m": gap}
        # d2: up to speed on an open runway, then the pillar appears just beyond the rotor tips, inside the next
        # step's travel. Only a pillar that is solid the moment it is moved stops that step.
        start = _runway(layout)
        sim.reset(start)
        runway_step, obs = _fly(sim, RUNWAY_STEPS - 1)
        last_step, after = (None, obs) if runway_step is not None else _fly(sim, 1)
        if runway_step is not None or last_step is not None:
            return {"pass": False, "d1": d1, "d2": {"error": "collided on the open runway", "runway_start": [start.x, start.y]}}
        travel = math.hypot(after.pose.x - obs.pose.x, after.pose.y - obs.pose.y)  # the last runway step
        pose = after.pose
        gap_ahead = FRONT_TIP_M + FIRST_STEP_FRACTION * travel
        wx, wy = math.cos(pose.yaw), math.sin(pose.yaw)
        near = Pose(pose.x + (gap_ahead + victim.radius_m) * wx, pose.y + (gap_ahead + victim.radius_m) * wy, home.z, 0.0)
        sim.set_object_poses({victim.tag: near})
        step2, _obs2 = _fly(sim, 1)
        d2 = {"runway_start": [start.x, start.y, start.yaw], "last_runway_step_m": travel,
              "rotor_gap_at_move_m": gap_ahead - FRONT_TIP_M, "collided_on_first_step": step2 == 1}
        up_to_speed = travel >= MIN_RUNWAY_STEP_M
        return {"pass": step is not None and gap < 1.0 and up_to_speed and step2 == 1, "d1": d1, "d2": d2}

    def e_vacated() -> dict:
        inst, start = _through_line(layout)
        sim.set_object_poses({inst.tag: _offset(_home(inst), 0.0, 0.0, PARK_DEPTH_M)})
        sim.reset(start)
        past_home = math.hypot(inst.x - start.x, inst.y - start.y) + 1.0  # through the spot and 1 m beyond it
        step, raised = None, None
        try:
            for i in range(MAX_FLY_STEPS):  # however slowly the drone speeds up from hover
                step, obs = _fly(sim, 1)
                if step is not None:
                    step = i + 1
                    break
                if math.hypot(obs.pose.x - start.x, obs.pose.y - start.y) >= past_home:
                    break
        except Exception as err:  # a CameraPoseError here is exactly what this check looks for
            raised = f"{type(err).__name__}: {err}"
        obs = sim.observe()
        flown = math.hypot(obs.pose.x - start.x, obs.pose.y - start.y)
        through = flown >= past_home
        sim.reset(start)  # out of the home spot before the pillar returns
        sim.set_object_poses({inst.tag: _home(inst)})
        return {"pass": step is None and raised is None and through, "pillar": inst.tag, "collided_at_step": step,
                "raised": raised, "flown_m": flown, "passed_home_spot": through}

    def f_park() -> dict:
        parked = _offset(home, 0.0, 0.0, PARK_DEPTH_M)
        sim.set_object_poses({victim.tag: parked})
        _hold(sim)
        park_error = _error(sim.get_object_poses([victim.tag])[victim.tag], parked)
        restore_victim()
        _hold(sim)
        restore_error = _error(sim.get_object_poses([victim.tag])[victim.tag], home)
        return {"pass": park_error <= MOVE_TOLERANCE_M and restore_error <= MOVE_TOLERANCE_M,
                "park_error_m": park_error, "restore_error_m": restore_error}

    sim.reset(view)
    for name, fn in (("a_move", a_move), ("b_restore", b_restore), ("c_depth_same_step", c_depth),
                     ("d_collision", d_collide), ("e_vacated_home", e_vacated), ("f_park_and_restore", f_park)):
        checks[name] = _check(name, fn)
        try:  # every check starts from the level as built
            sim.reset(view)
            restore_victim()
        except Exception as err:
            checks[name].setdefault("cleanup_error", f"{type(err).__name__}: {err}")
    report["latency"] = _latency(sim, layout, victim.tag, view, async_batch)
    report["pass"] = all(c.get("pass") for c in checks.values())
    return report


def _latency(sim, layout: Layout, skip: str, view: Pose, async_batch) -> dict:
    movers = [i for i in sorted(layout.instances, key=lambda i: i.tag) if i.tag != skip][:LATENCY_BATCH]
    homes = {i.tag: _home(i) for i in movers}
    shifted = {tag: _offset(p, 0.3, 0.0) for tag, p in homes.items()}
    out: dict = {"batch_size": len(movers)}
    try:
        times = []
        for r in range(LATENCY_REPEATS):
            t0 = time.perf_counter()
            sim.set_object_poses(shifted if r % 2 == 0 else homes)
            times.append(time.perf_counter() - t0)
        out["sequential_batch_ms"] = _ms_stats(times)
        sim.set_object_poses(homes)
        sim.reset(view)
        few = dict(list(shifted.items())[:STEP_TIMING_MOVERS])
        few_home = {tag: homes[tag] for tag in few}
        plain, moving = [], []
        for _ in range(STEP_TIMING_STEPS):
            t0 = time.perf_counter()
            _hold(sim)
            plain.append(time.perf_counter() - t0)
        for i in range(STEP_TIMING_STEPS):
            t0 = time.perf_counter()
            sim.set_object_poses(few if i % 2 == 0 else few_home)
            _hold(sim)
            moving.append(time.perf_counter() - t0)
        sim.set_object_poses(homes)
        out["step_ms_without_moves"] = _ms_stats(plain)
        out[f"step_ms_with_{len(few)}_moves"] = _ms_stats(moving)
    except Exception as err:
        traceback.print_exc()
        out["error"] = f"{type(err).__name__}: {err}"
    if async_batch is not None:
        try:
            times = []
            for r in range(LATENCY_REPEATS):
                t0 = time.perf_counter()
                async_batch(shifted if r % 2 == 0 else homes)
                times.append(time.perf_counter() - t0)
            out["async_batch_ms"] = _ms_stats(times)
        except Exception as err:
            traceback.print_exc()
            out["async_error"] = f"{type(err).__name__}: {err}"
    return out


def _ms_stats(seconds: list[float]) -> dict:
    ms = sorted(1000.0 * s for s in seconds)
    return {"median": statistics.median(ms), "p95": ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))],
            "max": ms[-1], "n": len(ms)}


def _live_async_batch(sim):
    """One batch of SetObjectPose requests in flight at once through projectairsim's request_async (measurement only:
    the backend sends them one by one). Reaches into the backend's client, world and event loop on purpose."""
    def send(poses: dict) -> None:
        async def _all():
            tasks = []
            for name, pose in poses.items():
                request = {"method": f"{sim._world.parent_topic}/SetObjectPose",
                           "params": {"object_name": name, "pose": sim._pas_pose(pose), "teleport": True},
                           "version": 1.0}
                tasks.append(await sim._client.request_async(request))
            return await asyncio.gather(*tasks)

        results = sim._loop.run_until_complete(_all())
        if not all(results):
            raise RuntimeError(f"an async SetObjectPose returned {results}")

    return send


def main(argv: list[str] | None = None) -> int:
    from autofly_ue5.scenes.resolve import resolve_scene
    from autofly_ue5.sim.airsim_backend import ProjectAirSimSimulator, scene_config_record
    from autofly_ue5.sim.process import instance_dir, route_client_log, sweep_orphaned_instances

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="s01", help="the scene whose built level's pillars are moved")
    parser.add_argument("--scene-config", default=None, help="default: the scene's own (scene_autofly_<level>.jsonc)")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "m2d" / "mover_probe.json")
    parser.add_argument("--no-async", action="store_true", help="skip the request_async timing")
    args = parser.parse_args(argv)

    resolved = resolve_scene(args.scene)
    scene_config = args.scene_config or resolved.default_scene_config
    layout = resolved.layout
    report: dict = {"description": __doc__.split("\n\n")[0], "scene": args.scene, "map": resolved.map_path,
                    "scene_config": scene_config_record(scene_config), "instance": args.instance, "pass": False,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    swept = sweep_orphaned_instances()
    if swept:
        print(f"swept orphaned instances before starting: {swept}", file=sys.stderr)
    route_client_log(instance_dir(args.instance) / "client.log")
    sim = ProjectAirSimSimulator(scene_config=scene_config, movable_objects=[i.tag for i in layout.instances])
    status = 1
    try:
        sim.launch(resolved.map_path, args.instance)
        report.update(run_probe(sim, layout, async_batch=None if args.no_async else _live_async_batch(sim)))
        status = 0 if report["pass"] else 1
    except Exception as err:
        traceback.print_exc()
        report["error"] = f"{type(err).__name__}: {err}"
    finally:
        try:
            sim.close()
        except Exception as err:
            print(f"WARNING: close() raised {type(err).__name__}: {err}", file=sys.stderr)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "error": report.get("error"),
                      "checks": {k: v.get("pass") for k, v in report.get("checks", {}).items()},
                      "latency": report.get("latency")}, indent=2))
    return status


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)  # the projectairsim client leaves a non-daemon thread that blocks normal interpreter exit
