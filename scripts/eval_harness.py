"""The closed-loop evaluation harness from the command line (spec §12 M6; plan 6 item 6): write the frozen episode
files, or fly a policy on one condition and score it (autofly_ue5/evaluation).

    # once, then committed: assets/eval_episodes/{seen,test_seen,unseen_targets}.json (never written over)
    env -u PYTHONPATH .venv/bin/python scripts/eval_harness.py write-episodes

    # a policy on a condition, on a free simulator slot (a whole condition is a long job: scripts/run_job.sh)
    env -u PYTHONPATH .venv/bin/python scripts/eval_harness.py run --policy scripted --privileged-state \\
        --condition test_seen --instance 6 --out runs/eval/scripted_test_seen.json
    env -u PYTHONPATH .venv/bin/python scripts/eval_harness.py run \\
        --policy expert:runs/expert/s01d_r6/averaged/avg_250k_550k.zip --condition test_seen --instance 6 \\
        --out runs/eval/expert_test_seen.json

- `--policy`: `hover` (the floor), `scripted` (the privileged straight-line baseline: needs --privileged-state), or
  `expert:<checkpoint>` (stochastic, as it collected the dataset, unless --deterministic). A VLA adds its loader to
  `build_policy`.
- `--condition`: a condition file, or the name of one in assets/eval_episodes/. `--scenes` and `--episodes-per-scene`
  fly part of it (a smoke).
- `--privileged-state` hands the policy state[0] and state[1], the target's true distance and bearing (zeroed otherwise).
- Writes `--out` (JSON, after every episode) and its markdown summary beside it (.md).
- Refused before anything launches: a slot another live run holds, an --out that exists, a policy it cannot build, the
  privileged baseline without --privileged-state, and what run_condition refuses (a changed layout, a target split the
  env cannot draw, a pool a scene cannot draw from).

One simulator at a time on --instance, each scene's stopped through its own pid record before the next launches. Ends
with os._exit, like every entry point that talks to projectairsim (its client leaves a non-daemon thread behind).
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
import subprocess  # noqa: E402
from typing import Any  # noqa: E402

from autofly_ue5 import evidence  # noqa: E402
from autofly_ue5.evaluation.episodes import (  # noqa: E402
    CONDITIONS,
    DEFAULT_EPISODES_PER_SCENE,
    EPISODES_DIR,
    load_condition,
    write_condition,
)
from autofly_ue5.evaluation.harness import DEFAULT_MAX_STEPS, TARGETS_PATH, SimFactoryHook, run_condition  # noqa: E402
from autofly_ue5.evaluation.policy import ExpertPolicy, HoverPolicy, ScriptedPolicy  # noqa: E402
from autofly_ue5.paths import ROOT, RUNS_DIR  # noqa: E402
from autofly_ue5.scenes.resolve import resolve_scene  # noqa: E402
from autofly_ue5.sim.process import SIM_RUN_DIR, slot_busy  # noqa: E402

# run_condition's sim factory hook: None flies Project AirSim on each scene's config; the tests fly FakeSimulator.
SIM_FACTORY: SimFactoryHook | None = None
EXPERT_PREFIX = "expert:"


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_policy(spec: str, *, deterministic: bool = False, device: str = "auto") -> Any:
    """The policy `--policy` names: "hover", "scripted" or "expert:<checkpoint>"."""
    if spec.startswith(EXPERT_PREFIX):
        return ExpertPolicy.load(Path(spec[len(EXPERT_PREFIX):]), deterministic=deterministic, device=device)
    if deterministic:
        raise ValueError("--deterministic chooses how an expert samples its actions; the other policies have one way")
    if spec == "hover":
        return HoverPolicy()
    if spec == "scripted":
        return ScriptedPolicy()
    raise ValueError(f"unknown policy {spec!r}: hover, scripted, or expert:<checkpoint>")


def condition_path(value: str) -> Path:
    """A condition file's path, or the committed file of the condition named `value`."""
    return EPISODES_DIR / f"{value}.json" if value in CONDITIONS else Path(value)


def write_episodes(args: argparse.Namespace) -> int:
    conditions = args.condition or list(CONDITIONS)
    existing = [str(args.out_dir / f"{name}.json") for name in conditions if (args.out_dir / f"{name}.json").exists()]
    if existing:
        print(f"refusing: {', '.join(existing)} exist, and a frozen evaluation file is never written over (write a "
              f"missing one with --condition)", file=sys.stderr)
        return 2
    try:
        for name in conditions:
            path = write_condition(name, args.out_dir, episodes_per_scene=args.episodes_per_scene,
                                   resolve=lambda scene_id: resolve_scene(scene_id, levels_dir=args.levels_dir),
                                   gates_dir=args.gates_dir)
            print(f"wrote {path}", file=sys.stderr)
    except (FileNotFoundError, FileExistsError, ValueError) as err:
        print(f"refusing: {err}", file=sys.stderr)
        return 2
    return 0


def run(args: argparse.Namespace, argv: list[str]) -> int:
    try:
        out = args.out
        if out.exists():  # any record, a committed one under docs/gates/ (CLAUDE.md) or a run's own
            raise FileExistsError(f"{out} exists, and a record is never written over: pass a new --out")
        if out.suffix != ".json":
            raise ValueError(f"--out {out} must be a .json file (its markdown summary is written beside it)")
        busy = slot_busy(args.instance, args.sim_root)
        if busy:
            raise ValueError(f"{busy}; pass a free --instance")
        condition = load_condition(condition_path(args.condition))
        if args.policy == "scripted" and not args.privileged_state:
            raise ValueError("the scripted baseline steers by state[0] and state[1], the target's true distance and "
                             "bearing: run it with --privileged-state")
        policy = build_policy(args.policy, deterministic=args.deterministic, device=args.device)
        provenance = {"git_head": _git("rev-parse", "HEAD"), "tree_dirty": bool(_git("status", "--porcelain")),
                      "argv": list(argv)}
        record = run_condition(policy, condition, instance=args.instance, sim_root=args.sim_root,
                               privileged_state=args.privileged_state, max_steps=args.max_steps, out_path=out,
                               target_pool=args.targets, scenes=args.scenes,
                               max_episodes_per_scene=args.episodes_per_scene, sim_factory=SIM_FACTORY,
                               provenance=provenance)
    except (FileNotFoundError, FileExistsError, ValueError) as err:
        print(f"refusing to start: {err}", file=sys.stderr)
        return 2
    overall = record["overall"]
    print(json.dumps({"status": record["status"], "error": record["error"], "record": str(out),
                      "overall": {key: overall[key] for key in ("episodes", "success_rate", "collision_rate",
                                                                "out_of_bounds_rate", "timeout_rate", "per")}},
                     indent=2))
    return 0 if record["status"] == "ok" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write-episodes", help="write the frozen condition files (never over an existing one)")
    w.add_argument("--condition", nargs="+", choices=list(CONDITIONS), default=None,
                   help="the conditions to write (default: all)")
    w.add_argument("--episodes-per-scene", type=int, default=DEFAULT_EPISODES_PER_SCENE)
    w.add_argument("--out-dir", type=Path, default=EPISODES_DIR)
    w.add_argument("--levels-dir", type=Path, default=RUNS_DIR / "levels",
                   help="where the scenes' stored layouts are (scripts/build_scenes.py's output)")
    w.add_argument("--gates-dir", type=Path, default=evidence.COMMITTED_GATES_DIR,
                   help="where a gated scene's gate record is looked for (docs/gates/)")
    r = sub.add_parser("run", help="fly a policy on a condition and score it")
    r.add_argument("--policy", required=True, help="hover, scripted, or expert:<checkpoint>")
    r.add_argument("--condition", required=True,
                   help=f"a condition file, or one of {', '.join(CONDITIONS)} (assets/eval_episodes/<name>.json)")
    r.add_argument("--instance", type=int, required=True,
                   help="the simulator slot; refused while another live run holds it (an --instances N training run "
                        "holds 0..N)")
    r.add_argument("--out", type=Path, required=True, help="the record (.json; its summary goes beside it as .md)")
    r.add_argument("--privileged-state", action="store_true",
                   help="hand the policy state[0] and state[1], the target's true distance and bearing")
    r.add_argument("--deterministic", action="store_true", help="an expert's mean action (default: sampled)")
    r.add_argument("--scenes", nargs="+", default=None, help="fly only these of the condition's scenes")
    r.add_argument("--episodes-per-scene", type=int, default=None, help="fly only each scene's first N episodes")
    r.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="the step limit (spec §8: 300)")
    r.add_argument("--targets", type=Path, default=TARGETS_PATH, help="the target pool")
    r.add_argument("--device", default="auto", help="where an expert's network runs")
    r.add_argument("--sim-root", type=Path, default=SIM_RUN_DIR, help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(argv)
    if args.command == "write-episodes":
        return write_episodes(args)
    return run(args, argv)


if __name__ == "__main__":
    _code = main()
    # Not sys.exit(): the projectairsim client leaves a non-daemon thread alive that blocks interpreter shutdown
    # (scripts/m2_gate.py). The record is already written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
