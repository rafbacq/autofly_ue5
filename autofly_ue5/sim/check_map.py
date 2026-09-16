"""Confirm a running simulator has the generated map loaded by comparing obstacle bounding boxes with the layout.

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.check_map --instance 0 --layout runs/levels/s01.layout.json --out runs/m1/check_map.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from autofly_ue5.paths import CONFIGS_DIR
from autofly_ue5.sim.process import ports_for_instance, route_client_log

TOL_M = 0.05


def bbox_error(instance: dict, bbox: dict) -> dict:
    if not bbox:
        return {"found": False}
    center = (bbox["center"]["x"], bbox["center"]["y"], bbox["center"]["z"])
    center_error = math.dist(center, (instance["x"], instance["y"], instance["z_center"]))
    expected = (2 * instance["radius_m"], 2 * instance["radius_m"], instance["height_m"])
    size = (bbox["size"]["x"], bbox["size"]["y"], bbox["size"]["z"])
    return {"found": True, "center_error_m": center_error, "size_error_m": max(abs(a - b) for a, b in zip(size, expected))}


def main(argv: list[str] | None = None) -> int:
    from projectairsim import ProjectAirSimClient, World
    from projectairsim.types import BoxAlignment

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--scene-config", default="scene_autofly_s01.jsonc")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    instances = json.loads(args.layout.read_text())["layout"]["instances"]
    chosen = [instances[0], instances[len(instances) // 2], instances[-1]]
    ports = ports_for_instance(args.instance)
    report: dict = {"layout": str(args.layout), "obstacles": []}
    route_client_log(args.out.with_suffix(".client.log"))
    client = ProjectAirSimClient(port_topics=ports.topics, port_services=ports.services)
    try:
        client.connect()
        world = World(client, args.scene_config, delay_after_load_sec=0, sim_config_path=str(CONFIGS_DIR))
        for inst in chosen:
            result = bbox_error(inst, world.get_3d_bounding_box(inst["tag"], BoxAlignment.WORLD_AXIS))
            result["tag"] = inst["tag"]
            report["obstacles"].append(result)
        ground = world.get_3d_bounding_box("AF_Ground", BoxAlignment.WORLD_AXIS)
        report["ground_top_z_ned"] = ground["center"]["z"] - ground["size"]["z"] / 2.0 if ground else None
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
    finally:
        client.disconnect()
    report["pass"] = (
        "error" not in report
        and all(o["found"] and o["center_error_m"] < TOL_M and o["size_error_m"] < TOL_M for o in report["obstacles"])
        and report.get("ground_top_z_ned") is not None and abs(report["ground_top_z_ned"]) < 0.02
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
