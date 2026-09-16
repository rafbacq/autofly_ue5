"""Launch one owned simulator instance, wait for its ports and a client handshake, print a JSON line.

env -u PYTHONPATH .venv/bin/python scripts/launch_sim.py --mode editor --map /Game/BlocksMap --instance 0
"""

import argparse
import json
import sys

from autofly_ue5.gpu import check_gpu_for_launch, gpu_memory_mib
from autofly_ue5.paths import ZEN_DATA_DIR
from autofly_ue5.sim.process import (
    editor_game_command,
    handshake,
    instance_dir,
    launch_process,
    own_running_instances,
    packaged_command,
    ports_for_instance,
    route_client_log,
    sim_environment,
    stop,
    wait_ready,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["editor", "packaged"], required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    ports = ports_for_instance(args.instance)
    used, total = gpu_memory_mib()
    check_gpu_for_launch(used, total, own_running=len(own_running_instances()))
    directory = instance_dir(args.instance)
    directory.mkdir(parents=True, exist_ok=True)
    # ValidateDataPath (ZenServerInterface.cpp) rejects a -ZenDataPath it can't create; make it ourselves.
    ZEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    route_client_log(directory / "launch.client.log")
    log = directory / "sim.log"
    if args.mode == "editor":
        cmd = editor_game_command(args.map, ports, log, args.instance)
    else:
        cmd = packaged_command(args.map, ports, log, args.instance)
    sp = launch_process(cmd, args.instance, ports, sim_environment(editor_mode=args.mode == "editor"))
    try:
        ready_s = wait_ready(sp, args.timeout)
        handshake_s = handshake(ports, timeout_s=120.0)
    except Exception:
        print(f"launch failed; stopping pid {sp.pid}: {stop(args.instance)}", file=sys.stderr)
        raise
    print(json.dumps({
        "pid": sp.pid, "instance": args.instance, "ports": [ports.topics, ports.services],
        "ports_ready_s": round(ready_s, 1), "handshake_s": round(handshake_s, 1),
        "gpu_used_mib_before": used, "log": str(log),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
