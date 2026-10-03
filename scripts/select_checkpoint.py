"""Choose a run's checkpoint on held-out validation episodes before gating it (2026-10-03).

Training-time evaluation scores 20 episodes, and SB3 keeps the best mean return. On s01d_r1 that picked best_model at
0.85, and the gate then measured 0.775. With a checkpoint every 10k steps, the better choice is to score many of them on
more episodes, on seeds of their own (SELECTION_SEED_BASE, never the gate's), and gate the winner.

The scoring reuses the gate itself, so faults, retries, records and slot ownership are its tried machinery: each slot
runs `scripts.m2_gate` over its share of the checkpoints, deterministic only, into runs/. This script plans the shares
and ranks the results.

    PY="env -u PYTHONPATH .venv/bin/python"
    $PY scripts/select_checkpoint.py plan --run-root runs/expert/s01d_r2 --every 20000 --min-steps 60000 \\
        --slots 0 1 2 3 --episodes 40 --scene s01d --scene-config scene_autofly_s01_fast.jsonc --launch
        # writes <run-root>/selection/stage1_plan.json; --launch starts one run_job per slot, 30 s apart (the
        # simulators' launches must not race the GPU guard), or omit it to print the commands
    $PY scripts/select_checkpoint.py rank --run-root runs/expert/s01d_r2      # after every part has finished
        # writes <run-root>/selection/stage1_ranking.json, best first (success rate, then mean return); exits 1 while
        # any part is missing, unfinished or not the plan's own (other names or seeds)

A second stage re-scores the top few on further validation episodes, so the winner is not just the luckiest of many
noisy estimates: `plan --stage 2 --names step_... step_... --seed-offset 1000 ...`, then `rank --stage 2`. A stage is
never planned twice, and no two stages share validation episodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

from autofly_ue5.expert.seeds import SELECTION_SEED_BASE  # noqa: E402

CHECKPOINT_RE = re.compile(r"^rl_model_(\d+)_steps\.zip$")
LAUNCH_STAGGER_S = 30  # each slot's first launch must not race the others' through the GPU guard (training's hazard #2)


def candidates(run_root: Path, *, every: int, min_steps: int, names: list[str] | None = None) -> dict[str, Path]:
    """name -> checkpoint: the periodic checkpoints at multiples of `every` from `min_steps` on, plus best_model and
    final when present. With `names`, exactly those (a second stage), refusing any that does not exist."""
    run_root = Path(run_root)
    found: dict[str, Path] = {}
    for path in sorted((run_root / "checkpoints").glob("rl_model_*_steps.zip")):
        m = CHECKPOINT_RE.match(path.name)
        if m:
            found[f"step_{int(m.group(1))}"] = path
    for name, path in (("best_model", run_root / "best" / "best_model.zip"), ("final", run_root / "final.zip")):
        if path.is_file():
            found[name] = path
    if names:
        missing = [n for n in names if n not in found]
        if missing:
            raise ValueError(f"no such checkpoint(s) in {run_root}: {missing}")
        return {n: found[n] for n in names}
    return {n: p for n, p in found.items()
            if not n.startswith("step_") or (int(n[5:]) >= min_steps and int(n[5:]) % every == 0)}


def split(names: list[str], n_slots: int) -> list[list[str]]:
    """Deal the candidates round-robin, so every slot gets early and late checkpoints alike."""
    return [names[i::n_slots] for i in range(n_slots)]


def plan_commands(run_root: Path, chosen: dict[str, Path], *, slots: list[int], episodes: int, scene: str,
                  scene_config: str, seed_offset: int = 0, stage: int = 1) -> list[dict]:
    out_dir = Path(run_root) / "selection"
    parts = []
    for slot, share in zip(slots, split(list(chosen), len(slots))):
        if not share:
            continue
        out = out_dir / f"stage{stage}_slot{slot}.json"
        argv = ["env", "-u", "PYTHONPATH", "DISPLAY=:1", "SDL_VIDEODRIVER=x11", ".venv/bin/python", "-m", "scripts.m2_gate",
                "--scene", scene, "--scene-config", scene_config, "--conditions", "deterministic",
                "--episodes", str(episodes), "--seed-base", str(SELECTION_SEED_BASE + seed_offset),
                "--instance", str(slot), "--out", str(out)]
        for name in share:
            argv += ["--model", f"{name}={chosen[name]}"]
        job = f"select_s{stage}_{slot}"
        parts.append({"slot": slot, "names": share, "out": str(out), "job": job,
                      "command": f"bash scripts/run_job.sh start {job} -- " + shlex.join(argv)})
    return parts


def plan_refusal(sel: Path, *, stage: int, names: list[str] | None, seed_offset: int, episodes: int) -> str | None:
    """Why this plan must not be made, or None. A stage is planned once (re-planning would point new jobs at its result
    files); a named second stage must say which stage it is; validation episodes stay inside SELECTION_SEED_BASE's
    range and are never shared between stages."""
    if seed_offset < 0 or seed_offset + episodes > 1_000_000:
        return "the validation episodes must stay inside SELECTION_SEED_BASE's 1e6 range"
    if names and stage == 1:
        return "a stage that re-scores named candidates is a later stage: pass --stage 2 (or higher)"
    if (sel / f"stage{stage}_plan.json").exists() or any(sel.glob(f"stage{stage}_slot*.json")):
        return f"stage {stage} is already planned in {sel}; plan another --stage"
    lo, hi = SELECTION_SEED_BASE + seed_offset, SELECTION_SEED_BASE + seed_offset + episodes
    for other in sorted(sel.glob("stage*_plan.json")):
        earlier = json.loads(other.read_text())
        a, b = earlier["seed_base"], earlier["seed_base"] + earlier["episodes"]
        if lo < b and a < hi:
            return f"its episodes [{lo}, {hi}) overlap stage {earlier['stage']}'s [{a}, {b}); pass another --seed-offset"
    return None


def part_problem(record: dict, part: dict, plan: dict) -> str | None:
    """Whether a part's gate record is the one the plan asked for: the same checkpoints on the plan's seeds."""
    if record.get("eval_seed_base") != plan["seed_base"]:
        return f"eval_seed_base {record.get('eval_seed_base')} is not the plan's {plan['seed_base']}"
    if sorted(record.get("checkpoints", {})) != sorted(part["names"]):
        return f"names {sorted(record.get('checkpoints', {}))} are not the part's {sorted(part['names'])}"
    return None


def _steps(name: str) -> int | None:
    return int(name[5:]) if name.startswith("step_") else None


def rank(records: list[dict]) -> list[dict]:
    """Every candidate scored in `records` (m2_gate output), best first: validation success rate, then mean return.
    A candidate whose combination did not finish is listed last with its status, never silently dropped."""
    rows = []
    for record in records:
        for name, ck in record["checkpoints"].items():
            det = ck.get("deterministic", {})
            done = det.get("status") == "ok"
            rows.append({"name": name, "steps": _steps(name), "path": ck.get("path"), "sha256": ck.get("sha256"),
                         "status": det.get("status"), "n_episodes": det.get("n_episodes", 0),
                         "success_rate": det.get("success_rate") if done else None,
                         "mean_return": det.get("mean_return") if done else None,
                         "collision_rate": det.get("collision_rate"), "out_of_bounds_rate": det.get("out_of_bounds_rate"),
                         "collision_sources": det.get("collision_sources")})
    return sorted(rows, key=lambda r: (r["success_rate"] is None, -(r["success_rate"] or 0.0), -(r["mean_return"] or 0.0)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--run-root", type=Path, required=True)
    pl.add_argument("--every", type=int, default=20_000)
    pl.add_argument("--min-steps", type=int, default=0)
    pl.add_argument("--names", nargs="+", default=None, help="a second stage: exactly these candidates")
    pl.add_argument("--slots", type=int, nargs="+", required=True)
    pl.add_argument("--episodes", type=int, default=40)
    pl.add_argument("--scene", required=True)
    pl.add_argument("--scene-config", required=True)
    pl.add_argument("--seed-offset", type=int, default=0, help="a later stage scores further validation episodes")
    pl.add_argument("--stage", type=int, default=1)
    pl.add_argument("--launch", action="store_true", help="start the parts with run_job.sh, LAUNCH_STAGGER_S apart")
    rk = sub.add_parser("rank")
    rk.add_argument("--run-root", type=Path, required=True)
    rk.add_argument("--stage", type=int, default=1)
    args = p.parse_args(argv)
    sel = args.run_root / "selection"
    if args.cmd == "plan":
        refusal = plan_refusal(sel, stage=args.stage, names=args.names, seed_offset=args.seed_offset,
                               episodes=args.episodes)
        chosen = {} if refusal else candidates(args.run_root, every=args.every, min_steps=args.min_steps, names=args.names)
        if not refusal and not chosen:
            refusal = f"no candidates in {args.run_root}"
        if refusal:
            print(f"refusing: {refusal}", file=sys.stderr)
            return 2
        parts = plan_commands(args.run_root, chosen, slots=args.slots, episodes=args.episodes, scene=args.scene,
                              scene_config=args.scene_config, seed_offset=args.seed_offset, stage=args.stage)
        sel.mkdir(parents=True, exist_ok=True)
        plan_path = sel / f"stage{args.stage}_plan.json"
        plan_path.write_text(json.dumps({"stage": args.stage, "seed_base": SELECTION_SEED_BASE + args.seed_offset,
                                         "episodes": args.episodes, "parts": parts}, indent=2) + "\n")
        for i, part in enumerate(parts):
            print(part["command"])
            if args.launch:
                if i:
                    time.sleep(LAUNCH_STAGGER_S)
                subprocess.run(part["command"], shell=True, check=True, cwd=_ROOT)
        return 0
    plan = json.loads((sel / f"stage{args.stage}_plan.json").read_text())
    records, missing, problems = [], [], []
    for part in plan["parts"]:
        path = Path(part["out"])
        if not path.is_file():
            missing.append(part["out"])
            continue
        record = json.loads(path.read_text())
        problem = part_problem(record, part, plan)
        if problem:
            problems.append(f"{part['out']}: {problem}")
        else:
            records.append(record)
    ranking = rank(records)
    unfinished = [row["name"] for row in ranking if row["status"] != "ok"]
    out = {"stage": args.stage, "seed_base": plan["seed_base"], "episodes": plan["episodes"], "missing_parts": missing,
           "problems": problems, "unfinished": unfinished, "ranking": ranking}
    (sel / f"stage{args.stage}_ranking.json").write_text(json.dumps(out, indent=2) + "\n")
    for row in ranking:
        print(f"{row['name']:>14}  success {row['success_rate']}  return {row['mean_return']}  "
              f"collisions {row['collision_rate']}  oob {row['out_of_bounds_rate']}  ({row['status']}, {row['n_episodes']} eps)")
    for line in [*(f"MISSING {m}" for m in missing), *(f"PROBLEM {p}" for p in problems), *(f"UNFINISHED {u}" for u in unfinished)]:
        print(line, file=sys.stderr)
    return 1 if (missing or problems or unfinished) else 0


if __name__ == "__main__":
    sys.exit(main())
