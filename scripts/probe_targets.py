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
          (live, a target new to the session drew nothing there; runs/m4/targets_settle_diag/settle.json), and on every
          step's as `visible_per_step` and `first_visible_step` (the summary counts each): the render lag after a
          spawn, measured (beside 7 collectors on 2026-10-08, 59 of 60 targets showed only after step 1);
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
from collections import Counter  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autofly_ue5.dataset.targets import TargetPool, TargetPoolError  # noqa: E402
from autofly_ue5.expert.env import PREWARM_HOLD_STEPS  # noqa: E402
from autofly_ue5.expert.episode import _runtime_material_path, _spawn_asset_name  # noqa: E402
from autofly_ue5.scenes.model import AssetRegistry, Layout, load_registry  # noqa: E402
from autofly_ue5.sim.types import ObjectNotFoundError, Pose  # noqa: E402
# The view, the placement and the depth check, shared with the env's session warm-up (one implementation, 2026-10-08);
# their old private names stay importable from here.
from autofly_ue5.validate.target_probe import (  # noqa: E402,F401
    ALTITUDE_M,
    CHANGE_M,
    DEPTH_TOLERANCE_M,
    MAX_DISTANCE_M,
    MIN_DISTANCE_M,
    MIN_HIT_SHARE,
    MIN_TARGET_PX,
    REACH_M,
    VIEW_CLEARANCE_M,
    choose_view,
    placement,
)
from autofly_ue5.validate.target_probe import camera as _camera  # noqa: E402,F401
from autofly_ue5.validate.target_probe import depth_check as _depth_check  # noqa: E402
from autofly_ue5.validate.target_probe import expected_box as _expected_box  # noqa: E402
from autofly_ue5.validate.target_probe import hold as _hold  # noqa: E402
from autofly_ue5.validate.target_probe import project as _project  # noqa: E402,F401
from autofly_ue5.validate.target_probe import region as _region  # noqa: E402

SPAWN_NAME = "autofly_probe_target"
SIZE_TOLERANCE = 0.02  # of the axis, or SIZE_FLOOR_M: float32 bounds through the cook
SIZE_FLOOR_M = 0.02
CENTRE_FLOOR_M = 0.05
COLOUR_ORDER_MARGIN = 0.15  # check the order of two channels only where the colour separates them by this much
TILE_PX = 160
SHEET_COLUMNS = 8


def probe_entries(pool: TargetPool, registry: AssetRegistry):
    """What the runtime could spawn: drawn entries, and entries that carry an imported asset but are not drawn."""
    return [e for e in pool.entries if e.usable or (e.asset is not None and e.asset in registry.assets)]


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
        position, distance = placement(asset, entry.scale, view)  # 4-10 m ahead, standing on the ground
        centre, size = _expected_box(asset, entry.scale, position)
        row = {"index": k, "name": entry.name, "status": entry.status, "drawn": entry.usable, "asset": entry.asset,
               "short_name": short, "material_key": entry.material, "material": material, "scale": list(entry.scale),
               "distance_m": round(distance, 3), "position": [round(v, 4) for v in position], "listed": short in table,
               "spawned": False, "spawn_error": None, "collided": None, "bounds": _bounds_check(None, centre, size),
               "depth": {"ok": False}, "colour": None, "rgb_png": None, "depth_png": None, "settle_steps": settle_steps,
               "first_step_visible": None, "first_step_depth": None, "visible_per_step": None,
               "first_visible_step": None}
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
                # every step's frame judged (information): the render lag after a spawn, measured step by step
                row["visible_per_step"] = [row["first_step_visible"]]
                settled = obs1
                for _ in range(settle_steps):
                    settled = _hold(sim)
                    row["visible_per_step"].append(_depth_check(obs0.depth, settled.depth, region, region_box)[0]["ok"])
                row["first_visible_step"] = next((i + 1 for i, ok in enumerate(row["visible_per_step"]) if ok), None)
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
            # how many spawned targets showed first on step k (1 = the frame right after the spawn), or never
            "first_visible_step_counts": dict(sorted(Counter(
                str(r["first_visible_step"]) if r["first_visible_step"] is not None else "never"
                for r in rows if r["visible_per_step"] is not None).items())),
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
