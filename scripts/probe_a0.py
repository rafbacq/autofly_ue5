"""Does a0's aligned start change how well the expert flies? A paired probe for M3's a0 decision (U1).

The first live collector smoke (2026-10-03) kept 3 of 5 episodes, against the M2 gate's 0.99 stochastic success for
the same checkpoint. a0 (`reset(options={"a0": "sector8"})`) was the one new input. This probe flies the same seeds
twice, back to back, with and without a0, so the only difference within a pair is the start yaw. The pairs are
interleaved to cancel any drift in load.

    env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 .venv/bin/python scripts/probe_a0.py \\
        --model runs/expert/s01_r2/best/best_model.zip --scene-config scene_autofly_s01_fast.jsonc --pairs 50 \\
        --instance <free slot> --out runs/probes/a0_paired.json

Seeds: PROBE_SEED_BASE + 500,000 + i (probe_crash_reset.py uses the first 30). Writes the JSON after every episode;
not evidence by default. Ends with os._exit, like every entry point that talks to projectairsim.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Any, Callable  # noqa: E402

from autofly_ue5.expert.seeds import PROBE_SEED_BASE  # noqa: E402
from autofly_ue5.expert.vec import teardown  # noqa: E402
from autofly_ue5.scenes.resolve import resolve_scene  # noqa: E402
from autofly_ue5.sim.airsim_backend import scene_config_factory, scene_config_record  # noqa: E402
from autofly_ue5.sim.process import SIM_RUN_DIR, slot_busy  # noqa: E402
from scripts.m2_gate import build_eval_env, checkpoint_obs_config, default_sac_loader  # noqa: E402

A0_PROBE_SEED_BASE = PROBE_SEED_BASE + 500_000
MODES = (("a0", {"a0": "sector8"}), ("random_yaw", None))


def fly(model, env, seed: int, options: dict | None, *, deterministic: bool, max_fault_retries: int = 10) -> dict:
    """One episode on `seed`; a backend fault replays the seed, as the gate and the collector do."""
    for retry in range(max_fault_retries + 1):
        obs, _info = env.reset(seed=seed, options=options)
        start_yaw = env.unwrapped.setup.start.yaw
        for step in range(1_000):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = env.step(action)
            if info.get("sim_fault"):
                break
            if terminated or truncated:
                return {"outcome": info["outcome"], "steps": step + 1, "start_yaw": start_yaw,
                        "collision_source": info.get("collision_source"), "oob_kind": info.get("oob_kind"),
                        "fault_retries": retry}
    return {"outcome": "fault_exhausted", "fault_retries": max_fault_retries}


def summarise(rows: list[dict]) -> dict:
    by_mode = {mode: [r for r in rows if r["mode"] == mode] for mode, _ in MODES}
    summary: dict[str, Any] = {mode: {"n": len(rs), "success": sum(r["outcome"] == "success" for r in rs),
                                      "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in {r["outcome"] for r in rs}}}
                               for mode, rs in by_mode.items()}
    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed"], {})[r["mode"]] = r["outcome"] == "success"
    complete = [p for p in pairs.values() if len(p) == len(MODES)]
    summary["pairs"] = {"complete": len(complete),
                        "both_succeed": sum(p["a0"] and p["random_yaw"] for p in complete),
                        "only_a0_succeeds": sum(p["a0"] and not p["random_yaw"] for p in complete),
                        "only_random_yaw_succeeds": sum(p["random_yaw"] and not p["a0"] for p in complete),
                        "both_fail": sum(not p["a0"] and not p["random_yaw"] for p in complete)}
    return summary


def run(*, model, env, pairs: int, out: Path, deterministic: bool, meta: dict,
        seed_base: int = A0_PROBE_SEED_BASE, log: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True)) -> dict:
    rows: list[dict] = []
    record = {**meta, "seed_base": seed_base, "deterministic": deterministic, "rows": rows, "summary": None}
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(pairs):
        seed = seed_base + i
        for mode, options in MODES:
            row = {"seed": seed, "mode": mode, **fly(model, env, seed, options, deterministic=deterministic)}
            rows.append(row)
            record["summary"] = summarise(rows)
            out.write_text(json.dumps(record, indent=1) + "\n")
            log(f"A0PAIR {json.dumps(row)}")
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="s01")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scene-config", required=True)
    p.add_argument("--pairs", type=int, default=50)
    p.add_argument("--instance", type=int, required=True)
    p.add_argument("--out", type=Path, default=_ROOT / "runs" / "probes" / "a0_paired.json")
    p.add_argument("--deterministic", action="store_true", help="default: stochastic, as collection flies")
    args = p.parse_args(argv)
    busy = slot_busy(args.instance, SIM_RUN_DIR)
    if busy:
        print(f"refusing to start: {busy}", file=sys.stderr)
        return 2
    resolved = resolve_scene(args.scene)
    meta = {"probe": "a0_paired", "scene": args.scene, "model": str(args.model),
            "scene_config": scene_config_record(args.scene_config), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    env = build_eval_env(args.scene, instance=args.instance,
                         sim_factory=scene_config_factory(args.scene_config, resolved.movable_objects),
                         seed_base=A0_PROBE_SEED_BASE, obs_config=checkpoint_obs_config(args.model))
    try:
        record = run(model=default_sac_loader(args.model), env=env, pairs=args.pairs, out=args.out,
                     deterministic=args.deterministic, meta=meta)
        print(f"A0SUMMARY {json.dumps(record['summary'])}", file=sys.stderr, flush=True)
        return 0
    finally:
        teardown(env, [args.instance], SIM_RUN_DIR)


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
