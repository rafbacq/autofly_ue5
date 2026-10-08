"""Live check of the target pool on the packaged simulator (plan 5 C6; docs/runbook-targets.md): does every target the
pool can draw exist in the cooked spawn table, spawn where the collector would stand it, have the size the registry says,
show in the depth camera, and wear its runtime colour?

    bash scripts/run_job.sh start probe_targets -- env -u PYTHONPATH .venv/bin/python scripts/probe_targets.py \\
        --scene s01 --instance 5 --targets assets/targets.json --out runs/m4/targets_probe

Probed: every pool entry with an asset in the registry (the drawn ones, primitive or imported, and an imported stand-in
the pool keeps out of the draw). One simulator on the slot given (the launch guard refuses it beside a foreign GPU job;
`--instance` must be a slot no live run owns). From one view on the scene's open ground (a clear line straight ahead, at
2 m, heading along a world axis so a yaw-0 object shows a face squarely), each target in turn:

  listed  its short name is in world.list_assets() (Project AirSim spawns only what the cook put in its table);
  spawned spawn() at a point straight ahead, standing on the ground as expert/episode.py's apply_setup stands it
          (standing_z_ned, yaw 0), at its pool scale, with its material only when the runtime can apply it (a base
          UMaterial; the backend raises if set_object_material fails);
  bounds  object_bounds() (the world-axis box, NED metres) against the registry: base_size_m x scale, centred at the
          spawn point plus the scaled bounds origin; 2 % or 2 cm per axis, 5 cm or 2 % at the centre;
  depth   after one hover step and `--settle-steps` more (default the env's PREWARM_HOLD_STEPS, 5: what the first
          recorded frame of an episode shows, since the reset holds that long for an object new to the session; plan 6
          step 2a), against the frame before the spawn: inside the target's projected box the depth changed to within
          its near and far faces on at least 3 % of the pixels (4 at least), and nothing there is nearer than its near
          face less 0.3 m. The same check on the first step's frame is recorded as `first_step_visible`, for information
          (live, a target new to the session drew nothing there; runs/m4/targets_settle_diag/settle.json);
  colour  for a runtime colour with an RGBA in the registry: the target's pixels keep the colour's channel order (red over
          green and blue ...), which the mesh's own grey does not;

then destroys it. Writes summary.json (per target: each check's numbers and pass), <k>_<key>_rgb.png and _depth.png per
target, and contact_sheet.png: every target as the drone's camera saw it, cropped to its box, labelled with the
instruction name and its failed checks. Exit 0 when every probed target passes, 1 when one fails, 2 when the pool does
not load (nothing is launched then).

Ends with os._exit, like scripts/m2_gate.py: the projectairsim client leaves a non-daemon thread behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autofly_ue5.dataset.targets import TargetPool, TargetPoolError  # noqa: E402
from autofly_ue5.expert.env import PREWARM_HOLD_STEPS  # noqa: E402
from autofly_ue5.expert.episode import _runtime_material_path, _spawn_asset_name, standing_z_ned  # noqa: E402
from autofly_ue5.scenes.model import AssetRegistry, Layout, load_registry  # noqa: E402
from autofly_ue5.sim.types import CONTROL_DT_S, ObjectNotFoundError, Pose  # noqa: E402
from autofly_ue5.validate.geometry import CAMERA_OFFSET_M  # noqa: E402

ALTITUDE_M = 2.0
REACH_M = 13.0  # the farthest target's far face from the body: 0.4 m mount + 10 m + half a 4 m car
VIEW_CLEARANCE_M = 3.0  # every pillar's surface this far from the line of view (a 4 m car's half-diagonal is 2.2 m)
MIN_DISTANCE_M, MAX_DISTANCE_M = 4.0, 10.0  # camera to the target's centre: 1.5 x its largest side + 3 m, clamped
SPAWN_NAME = "autofly_probe_target"
SIZE_TOLERANCE = 0.02  # of the axis, or SIZE_FLOOR_M: float32 bounds through the cook
SIZE_FLOOR_M = 0.02
CENTRE_FLOOR_M = 0.05
CHANGE_M = 0.05  # a depth pixel changed by the spawn (depth arrives as float16: 4 mm at 4 m, 8 mm at 10 m)
DEPTH_TOLERANCE_M = 0.3  # around the box's near and far faces (pixel footprints at grazing rows, float16)
MIN_HIT_SHARE = 0.03  # of the projected box: a ladder or a chair fills little of its box
MIN_TARGET_PX = 4
COLOUR_ORDER_MARGIN = 0.15  # check the order of two channels only where the colour separates them by this much
TILE_PX = 160
SHEET_COLUMNS = 8


def choose_view(layout: Layout, *, reach_m: float = REACH_M, clearance_m: float = VIEW_CLEARANCE_M) -> Pose:
    """The pose, at ALTITUDE_M and heading along a world axis, whose line of view (reach_m ahead of the body) keeps the
    most room to every obstacle (at least clearance_m to its surface), both ends 2 m inside the layout's bounds."""
    b = layout.bounds
    best, best_gap = None, -math.inf
    for x in np.linspace(b.x_min + 2.0, b.x_max - 2.0, 29):
        for y in np.linspace(b.y_min + 2.0, b.y_max - 2.0, 29):
            for yaw in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0):
                ux, uy = round(math.cos(yaw)), round(math.sin(yaw))
                ex, ey = x + reach_m * ux, y + reach_m * uy
                if not (b.x_min + 2.0 <= ex <= b.x_max - 2.0 and b.y_min + 2.0 <= ey <= b.y_max - 2.0):
                    continue
                gap = math.inf
                for inst in layout.instances:
                    t = max(0.0, min(reach_m, (inst.x - x) * ux + (inst.y - y) * uy))
                    gap = min(gap, math.hypot(inst.x - (x + t * ux), inst.y - (y + t * uy)) - inst.radius_m)
                if gap >= clearance_m and gap > best_gap:
                    best, best_gap = Pose(float(x), float(y), -ALTITUDE_M, yaw), gap
    if best is None:
        raise ValueError(f"no {reach_m} m line of view with {clearance_m} m clearance in {layout.scene_id}")
    return best


def probe_entries(pool: TargetPool, registry: AssetRegistry):
    """What the runtime could spawn: drawn entries, and entries that carry an imported asset but are not drawn."""
    return [e for e in pool.entries if e.usable or (e.asset is not None and e.asset in registry.assets)]


def _camera(view: Pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwd = np.array([math.cos(view.yaw), math.sin(view.yaw), 0.0])
    right = np.array([-math.sin(view.yaw), math.cos(view.yaw), 0.0])
    return np.array([view.x, view.y, view.z]) + CAMERA_OFFSET_M * fwd, fwd, right


def _project(points: np.ndarray, view: Pose, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates (u right, v down; pixel k spans [k, k+1)) and planar depth of NED points, for the 90 deg pinhole
    camera CAMERA_OFFSET_M ahead of the body (sim/fake.py renders the same camera)."""
    cam, fwd, right = _camera(view)
    rel = points - cam
    f = rel @ fwd
    return n / 2.0 + (rel @ right) / f * n / 2.0, n / 2.0 + rel[:, 2] / f * n / 2.0, f


def _expected_box(asset, scale, position) -> tuple[list[float], list[float]]:
    o = asset.bounds_origin_cm or (0.0, 0.0, 0.0)
    size = [b * s for b, s in zip(asset.base_size_m, scale)]
    centre = [position[0] + o[0] * scale[0] / 100.0, position[1] + o[1] * scale[1] / 100.0, position[2] - o[2] * scale[2] / 100.0]
    return centre, size


def _region(centre, size, view: Pose, n: int) -> tuple[np.ndarray, dict]:
    corners = np.array([[centre[0] + sx * size[0] / 2, centre[1] + sy * size[1] / 2, centre[2] + sz * size[2] / 2]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    u, v, f = _project(corners, view, n)
    c0, c1 = int(np.clip(math.floor(u.min()), 0, n - 1)), int(np.clip(math.ceil(u.max()) - 1, 0, n - 1))
    r0, r1 = int(np.clip(math.floor(v.min()), 0, n - 1)), int(np.clip(math.ceil(v.max()) - 1, 0, n - 1))
    mask = np.zeros((n, n), bool)
    mask[r0:r1 + 1, c0:c1 + 1] = True
    cu, cv, _ = _project(np.array([centre]), view, n)
    return mask, {"rows": [r0, r1], "cols": [c0, c1], "near_m": float(f.min()), "far_m": float(f.max()),
                  "centre_px": [float(cv[0]), float(cu[0])]}


def _depth_check(before: np.ndarray, after: np.ndarray, region: np.ndarray, box: dict) -> tuple[dict, np.ndarray]:
    fin_b, fin_a = np.isfinite(before), np.isfinite(after)
    with np.errstate(invalid="ignore"):
        changed = (fin_a != fin_b) | (fin_a & fin_b & (np.abs(after - before) > CHANGE_M))
        in_range = fin_a & (after >= box["near_m"] - DEPTH_TOLERANCE_M) & (after <= box["far_m"] + DEPTH_TOLERANCE_M)
    target = region & changed & in_range
    n_region = int(region.sum())
    seen = after[region & fin_a]
    r, c = (int(round(x - 0.5)) for x in box["centre_px"])
    window = after[max(r - 2, 0): r + 3, max(c - 2, 0): c + 3]
    window = window[np.isfinite(window)]
    region_min = float(seen.min()) if seen.size else math.inf
    hit_share = target.sum() / max(n_region, 1)
    ok = bool(target.sum() >= MIN_TARGET_PX and hit_share >= MIN_HIT_SHARE and region_min >= box["near_m"] - DEPTH_TOLERANCE_M)
    return {"expected_near_m": round(box["near_m"], 3), "expected_far_m": round(box["far_m"], 3),
            "centre_depth_m": round(float(np.median(window)), 3) if window.size else None,
            "region_min_m": round(region_min, 3) if math.isfinite(region_min) else None,
            "region_px": n_region, "target_px": int(target.sum()), "hit_share": round(float(hit_share), 4),
            "changed_share": round(float((region & changed).sum() / max(n_region, 1)), 4), "ok": ok}, target


def _colour_check(rgba, rgb: np.ndarray, target: np.ndarray) -> dict:
    pairs = [(i, j) for i in range(3) for j in range(3) if rgba[i] - rgba[j] > COLOUR_ORDER_MARGIN]
    observed = rgb[target].reshape(-1, 3).mean(axis=0) if target.any() else None
    ok = observed is not None and all(observed[i] > observed[j] for i, j in pairs)
    channels = "RGB"
    return {"expected_rgba": list(rgba), "observed_mean_rgb": None if observed is None else [round(float(v), 1) for v in observed],
            "checked_order": [f"{channels[i]}>{channels[j]}" for i, j in pairs], "ok": bool(ok)}


def _bounds_check(got, centre, size) -> dict:
    out = {"expected_centre": [round(v, 4) for v in centre], "expected_size": [round(v, 4) for v in size],
           "got_centre": None, "got_size": None, "ok": False}
    if got is None:
        return out
    (gc, gs) = got
    size_err = [abs(a - b) for a, b in zip(gs, size)]
    centre_err = math.dist(gc, centre)
    ok = all(e <= max(SIZE_TOLERANCE * s, SIZE_FLOOR_M) for e, s in zip(size_err, size)) and \
        centre_err <= max(SIZE_TOLERANCE * max(size), CENTRE_FLOOR_M)
    out.update(got_centre=[round(float(v), 4) for v in gc], got_size=[round(float(v), 4) for v in gs],
               size_error_m=[round(float(e), 4) for e in size_err], centre_error_m=round(float(centre_err), 4), ok=bool(ok))
    return out


def _depth_png(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    scaled = np.zeros(depth.shape, np.uint8)
    scaled[finite] = np.clip(255.0 * np.log1p(depth[finite]) / math.log1p(60.0), 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
    img[~finite] = (20, 20, 20)
    return img


def _tile(rgb: np.ndarray, rows, cols, label: str, failed: list[str]) -> np.ndarray:
    n = rgb.shape[0]
    if rows is None:
        crop = rgb
    else:
        side = max(rows[1] - rows[0], cols[1] - cols[0], 16) * 1.4
        cr, cc = (rows[0] + rows[1]) / 2.0, (cols[0] + cols[1]) / 2.0
        r0, c0 = int(max(cr - side / 2, 0)), int(max(cc - side / 2, 0))
        crop = rgb[r0: min(int(r0 + side) + 1, n), c0: min(int(c0 + side) + 1, n)]
    tile = cv2.resize(np.ascontiguousarray(crop[:, :, ::-1]), (TILE_PX, TILE_PX), interpolation=cv2.INTER_AREA)
    strip = np.full((30, TILE_PX, 3), 255, np.uint8)
    cv2.putText(strip, label[:26], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(strip, "ok" if not failed else "FAIL: " + ",".join(failed), (3, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (0, 140, 0) if not failed else (0, 0, 220), 1, cv2.LINE_AA)
    return np.concatenate([tile, strip], axis=0)


def contact_sheet(tiles: list[np.ndarray]) -> np.ndarray:
    rows = []
    for k in range(0, len(tiles), SHEET_COLUMNS):
        row = tiles[k: k + SHEET_COLUMNS]
        row += [np.full_like(tiles[0], 255)] * (SHEET_COLUMNS - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def _hold(sim):
    sim.command_velocity(0.0, 0.0, 0.0)
    sim.step(CONTROL_DT_S)
    return sim.observe()


def run_probe(sim, pool: TargetPool, registry: AssetRegistry, layout: Layout, *, out_dir: Path,
              settle_steps: int = PREWARM_HOLD_STEPS) -> dict:
    """Every check against a launched simulator (`sim.list_assets` lists the spawn table), each target judged after
    one hover step and `settle_steps` more (the env's first-use hold; 0: on the first step, as before plan 6). Returns
    the summary's body; the contact sheet's tiles are written by main()."""
    if settle_steps < 0:
        raise ValueError(f"settle_steps must be 0 or more, got {settle_steps}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = probe_entries(pool, registry)
    view = choose_view(layout)
    table = set(sim.list_assets(".*"))
    rows, tiles = [], []
    for k, entry in enumerate(entries):
        asset = registry.assets[entry.asset]
        short = _spawn_asset_name(asset.ue_path)
        material = _runtime_material_path(registry, entry.material)
        size = [b * s for b, s in zip(asset.base_size_m, entry.scale)]
        distance = float(np.clip(1.5 * max(size) + 3.0, MIN_DISTANCE_M, MAX_DISTANCE_M))
        cam, fwd, _right = _camera(view)
        position = [float(cam[0] + distance * fwd[0]), float(cam[1] + distance * fwd[1]), standing_z_ned(asset, entry.scale)]
        centre, size = _expected_box(asset, entry.scale, position)
        row = {"index": k, "name": entry.name, "status": entry.status, "drawn": entry.usable, "asset": entry.asset,
               "short_name": short, "material_key": entry.material, "material": material, "scale": list(entry.scale),
               "distance_m": round(distance, 3), "position": [round(v, 4) for v in position], "listed": short in table,
               "spawned": False, "spawn_error": None, "collided": None, "bounds": _bounds_check(None, centre, size),
               "depth": {"ok": False}, "colour": None, "rgb_png": None, "depth_png": None, "settle_steps": settle_steps,
               "first_step_visible": None, "first_step_depth": None}
        rows.append(row)
        region_box = None
        if row["listed"]:
            spawned = None
            try:
                obs0 = sim.reset(view)  # the frame without the target; a reset fault fails this row only
                spawned = sim.spawn(SPAWN_NAME, short, Pose(position[0], position[1], position[2], 0.0), entry.scale, material)
                row["spawned"] = True
                obs1 = _hold(sim)
                n = obs1.depth.shape[0]
                region, region_box = _region(centre, size, view, n)
                row["first_step_depth"], _target = _depth_check(obs0.depth, obs1.depth, region, region_box)
                row["first_step_visible"] = row["first_step_depth"]["ok"]  # information: the settled frame decides
                settled = obs1
                for _ in range(settle_steps):
                    settled = _hold(sim)
                row["collided"] = bool(settled.collided)
                row["bounds"] = _bounds_check(sim.object_bounds(spawned), centre, size)
                row["depth"], target = _depth_check(obs0.depth, settled.depth, region, region_box)
                row["projected_box_px"] = {"rows": region_box["rows"], "cols": region_box["cols"]}
                rgba = registry.materials[entry.material].rgba if material is not None else None
                if rgba is not None:
                    row["colour"] = _colour_check(rgba, settled.rgb, target)
                key = entry.asset if entry.asset else "target"
                row["rgb_png"], row["depth_png"] = f"{k:02d}_{key}_rgb.png", f"{k:02d}_{key}_depth.png"
                cv2.imwrite(str(out_dir / row["rgb_png"]), np.ascontiguousarray(settled.rgb[:, :, ::-1]))
                cv2.imwrite(str(out_dir / row["depth_png"]), _depth_png(settled.depth))
            except Exception as err:  # a failing target is recorded, never the end of the probe
                traceback.print_exc()
                row["spawn_error"] = f"{type(err).__name__}: {err}"
            finally:
                if spawned is not None:
                    try:
                        sim.destroy(spawned)
                    except ObjectNotFoundError:
                        pass
        failed = [name for name, ok in (("listed", row["listed"]), ("spawned", row["spawned"]),
                                        ("bounds", row["bounds"]["ok"]), ("depth", row["depth"].get("ok")),
                                        ("colour", row["colour"] is None or row["colour"]["ok"]),
                                        ("collided", not row["collided"])) if not ok]
        row["failed"], row["pass"] = failed, not failed
        print(json.dumps({"name": entry.name, "pass": row["pass"], "failed": failed}), flush=True)
        if row["rgb_png"] is not None:
            rgb = cv2.imread(str(out_dir / row["rgb_png"]))[:, :, ::-1]
            tiles.append(_tile(rgb, region_box["rows"], region_box["cols"], entry.name, failed))
        else:
            tiles.append(_tile(np.zeros((TILE_PX, TILE_PX, 3), np.uint8), None, None, entry.name, failed))
    missing = sorted({r["short_name"] for r in rows if not r["listed"]})
    if tiles:
        cv2.imwrite(str(out_dir / "contact_sheet.png"), contact_sheet(tiles))
    return {"view_pose": [view.x, view.y, view.z, view.yaw], "spawn_table": {"size": len(table), "missing": missing},
            "settle_steps": settle_steps, "targets": rows, "failed": [r["name"] for r in rows if not r["pass"]],
            "first_step_invisible": [r["name"] for r in rows if r["first_step_visible"] is False],
            "pass": bool(rows) and all(r["pass"] for r in rows)}


def main(argv: list[str] | None = None, *, sim_factory=None) -> int:
    from autofly_ue5.scenes.resolve import resolve_scene

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="s01", help="the built scene whose open ground the targets stand on")
    p.add_argument("--instance", type=int, required=True, help="a slot no live run owns (training owns 0..N)")
    p.add_argument("--targets", type=Path, default=_ROOT / "assets" / "targets.json")
    p.add_argument("--registry", type=Path, default=_ROOT / "assets" / "registry.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scene-config", default=None, help="default: scene_autofly_<base>_fast.jsonc")
    p.add_argument("--settle-steps", type=int, default=PREWARM_HOLD_STEPS,
                   help="hover steps after the first one before a target is judged: the env's first-use hold "
                        f"(default {PREWARM_HOLD_STEPS}); 0 judges the first step's frame. The first step's frame is "
                        "recorded as first_step_visible either way")
    args = p.parse_args(argv)
    if args.settle_steps < 0:
        p.error(f"--settle-steps must be 0 or more, got {args.settle_steps}")
    resolved = resolve_scene(args.scene)
    config = args.scene_config or resolved.default_scene_config.replace(".jsonc", "_fast.jsonc")
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"scene": args.scene, "map_path": resolved.map_path, "scene_config": config, "instance": args.instance,
                     "pool": {"path": str(args.targets), "sha256": hashlib.sha256(args.targets.read_bytes()).hexdigest()},
                     "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
                     "started": time.strftime("%Y-%m-%d %H:%M:%S"), "pass": False, "error": None}
    registry = load_registry(args.registry)
    try:
        pool = TargetPool.load(args.targets, registry=registry)
    except TargetPoolError as err:
        summary["error"] = f"the pool does not load: {err}"
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(summary["error"], file=sys.stderr)
        return 2
    if sim_factory is None:
        from autofly_ue5.sim.airsim_backend import scene_config_factory
        from autofly_ue5.sim.process import instance_dir, route_client_log

        route_client_log(instance_dir(args.instance) / "client.log")
        sim = scene_config_factory(config)()
    else:
        sim = sim_factory()
    try:
        sim.launch(resolved.map_path, args.instance)
        summary.update(run_probe(sim, pool, registry, resolved.layout, out_dir=args.out, settle_steps=args.settle_steps))
    except Exception as err:
        traceback.print_exc()
        summary["error"] = f"{type(err).__name__}: {err}"
    finally:
        try:
            sim.close()
        except Exception as err:
            print(f"WARNING: close() raised {type(err).__name__}: {err}", file=sys.stderr)
        summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"pass": summary["pass"], "error": summary["error"], "failed": summary.get("failed"),
                      "missing_from_spawn_table": summary.get("spawn_table", {}).get("missing")}, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)  # the projectairsim client leaves a non-daemon thread that blocks normal interpreter exit
