"""Confirm a running simulator has a scene's map loaded by comparing every actor's world-axis bounding box with the
layout's prediction (validate/obstacles.py: cylinders, yawed boxes and imported meshes alike).

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.check_map --scene s01 --instance 0 --out runs/m1/check_map.json

M1's runbook runs this against a simulator launched separately; validate/live_m1.py has carried the same check inside
its own session (`map_matches_layout`) since 2026-10-07, so new scenes need no separate step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autofly_ue5.paths import CONFIGS_DIR
from autofly_ue5.scenes.model import load_registry
from autofly_ue5.scenes.resolve import resolve_scene
from autofly_ue5.sim.process import ports_for_instance, route_client_log
from autofly_ue5.validate.obstacles import MeshLibrary, bbox_error, mesh_keys, obstacles_from_layout_json

TOL_M = 0.05
GROUND_TAG = "AF_Ground"


def _bounds(world, name: str, alignment):
    box = world.get_3d_bounding_box(name, alignment)
    if not box:
        return None
    return ((box["center"]["x"], box["center"]["y"], box["center"]["z"]), (box["size"]["x"], box["size"]["y"], box["size"]["z"]))


def main(argv: list[str] | None = None) -> int:
    from projectairsim import ProjectAirSimClient, World
    from projectairsim.types import BoxAlignment

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="s01")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--scene-config", default=None, help="default: the scene's own")
    parser.add_argument("--layout", type=Path, default=None, help="default: the scene's stored layout")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    resolved = resolve_scene(args.scene)
    layout_path = args.layout or resolved.layout_path
    data = json.loads(layout_path.read_text())
    registry = load_registry()
    keys = mesh_keys(data, registry)
    obstacles = obstacles_from_layout_json(data, registry, MeshLibrary.from_files(keys) if keys else None)
    ports = ports_for_instance(args.instance)
    report: dict = {"scene": args.scene, "layout": str(layout_path), "obstacles": []}
    route_client_log(args.out.with_suffix(".client.log"))
    client = ProjectAirSimClient(port_topics=ports.topics, port_services=ports.services)
    try:
        client.connect()
        world = World(client, args.scene_config or resolved.default_scene_config, delay_after_load_sec=0, sim_config_path=str(CONFIGS_DIR))
        for o in obstacles:
            result = bbox_error(o, _bounds(world, o.tag, BoxAlignment.WORLD_AXIS))
            result["tag"], result["kind"] = o.tag, o.kind
            report["obstacles"].append(result)
            cards_tag = getattr(o, "cards_tag", None)
            if cards_tag:  # found, but boxless: a non-colliding actor has no bounds in the plugin's report
                result = bbox_error(o, _bounds(world, cards_tag, BoxAlignment.WORLD_AXIS), part="cards")
                result["tag"], result["kind"] = cards_tag, "cards"
                report["obstacles"].append(result)
        ground = _bounds(world, GROUND_TAG, BoxAlignment.WORLD_AXIS)
        report["ground_top_z_ned"] = ground[0][2] - ground[1][2] / 2.0 if ground else None
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
    finally:
        client.disconnect()
    found = [o for o in report["obstacles"] if o["found"]]
    boxed = [o for o in found if not o.get("box_unavailable")]
    report["max_center_error_m"] = max((o["center_error_m"] for o in boxed), default=None)
    report["max_size_error_m"] = max((o["size_error_m"] for o in boxed), default=None)
    report["pass"] = (
        "error" not in report and len(found) == len(report["obstacles"]) > 0 and len(boxed) > 0
        and all(o["center_error_m"] < TOL_M and o["size_error_m"] < TOL_M for o in boxed)
        and report.get("ground_top_z_ned") is not None and abs(report["ground_top_z_ned"]) < 0.02
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("pass", "max_center_error_m", "max_size_error_m", "ground_top_z_ned")} | {"error": report.get("error")}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
