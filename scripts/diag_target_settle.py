"""Diagnostic (not evidence): do the 5 invisible targets appear after more steps, or on a second spawn, in one session?"""
import json, os, sys, traceback
from pathlib import Path
ROOT = Path("/home/nvidiasims/research_uav/autofly_ue5"); sys.path.insert(0, str(ROOT))
import numpy as np, cv2
from scripts.probe_targets import (probe_entries, choose_view, _camera, _expected_box, _region, _depth_check, _hold, SPAWN_NAME)
from autofly_ue5.dataset.targets import TargetPool
from autofly_ue5.expert.episode import _spawn_asset_name, _runtime_material_path, standing_z_ned
from autofly_ue5.scenes.model import load_registry
from autofly_ue5.scenes.resolve import resolve_scene
from autofly_ue5.sim.types import Pose
from autofly_ue5.sim.airsim_backend import scene_config_factory
from autofly_ue5.sim.process import instance_dir, route_client_log
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); INSTANCE = int(sys.argv[2])
WANT = ["red sedan", "grandfather clock", "red fire extinguisher", "yellow generator", "man in brown jacket", "green hatchback"]
registry = load_registry(); pool = TargetPool.load(ROOT / "assets/targets.json", registry=registry)
entries = [e for e in probe_entries(pool, registry) if e.name in WANT]
resolved = resolve_scene("s01"); view = choose_view(resolved.layout)
route_client_log(instance_dir(INSTANCE) / "client.log")
sim = scene_config_factory("scene_autofly_s01_fast.jsonc")()
results = []
try:
    sim.launch(resolved.map_path, INSTANCE)
    for entry in entries:
        asset = registry.assets[entry.asset]; short = _spawn_asset_name(asset.ue_path)
        size = [b * s for b, s in zip(asset.base_size_m, entry.scale)]
        distance = float(np.clip(1.5 * max(size) + 3.0, 4.0, 10.0))
        cam, fwd, _r = _camera(view)
        position = [float(cam[0] + distance * fwd[0]), float(cam[1] + distance * fwd[1]), standing_z_ned(asset, entry.scale)]
        centre, bsize = _expected_box(asset, entry.scale, position)
        row = {"name": entry.name, "short": short, "captures": []}
        obs0 = sim.reset(view)
        for attempt in (1, 2):
            spawned = sim.spawn(SPAWN_NAME, short, Pose(position[0], position[1], position[2], 0.0), entry.scale,
                                _runtime_material_path(registry, entry.material))
            steps = 0
            for wanted in ((1, 5, 15, 40) if attempt == 1 else (1, 5)):
                while steps < wanted:
                    obs = _hold(sim); steps += 1
                region, box = _region(centre, bsize, view, obs.depth.shape[0])
                depth, _t = _depth_check(obs0.depth, obs.depth, region, box)
                row["captures"].append({"spawn": attempt, "steps": steps, "target_px": depth["target_px"], "changed_share": depth["changed_share"], "ok": depth["ok"]})
                cv2.imwrite(str(OUT / f"{short}_spawn{attempt}_step{steps}.png"), np.ascontiguousarray(obs.rgb[:, :, ::-1]))
            sim.destroy(spawned)
            _hold(sim)
        results.append(row); print(json.dumps(row), flush=True)
except Exception:
    traceback.print_exc()
finally:
    try: sim.close()
    except Exception: pass
    (OUT / "settle.json").write_text(json.dumps(results, indent=1))
    sys.stdout.flush(); os._exit(0)
