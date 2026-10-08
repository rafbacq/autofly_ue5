"""Score a live run's checkpoints as they appear, without pausing training (2026-10-03).

SB3's EvalCallback stops all four training workers for each evaluation (about 10 minutes per 20 episodes on s01d),
and its simulator, once launched, renders unthrottled for the rest of the run, taking a share of the GPU's time slices
from the four that train. This job instead runs `scripts.m2_gate` on each new checkpoint at the interval
(deterministic, on the training-time evaluation seeds EVAL_CALLBACK_SEED_BASE, so the curve compares directly with
earlier runs'). One simulator is launched per evaluation and stopped after it, through the gate's own teardown. Train
with a huge --eval-freq (no in-training evaluation); selection then uses the periodic checkpoints
(`scripts/select_checkpoint.py`).

    $J start <run>_evalwatch -- env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 .venv/bin/python scripts/eval_watch.py \\
        --run-root runs/expert/<run> --job <training job> --every 50000 --instance 4 --scene s01d \\
        --scene-config scene_autofly_s01_fast.jsonc

Prints one line per evaluation ("EVALWATCH step <k>: success ... return ..."); results go to <run-root>/eval_watch/.
It stops after the training job's exit file appears and every due checkpoint is scored.

On a scene with mesh obstacles every evaluation scores the footprint shape the run trains under, read from the run's
sessions.json (expert/run_shape.py) once it exists: the trainer writes it seconds after it starts, so a watch started
beside it waits for it rather than refusing, until a checkpoint is due. `--footprint-shape` is needed only for a run
whose record names no shape, and is refused where it contradicts the run's.
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
import time  # noqa: E402
from typing import Callable  # noqa: E402

from autofly_ue5.expert.footprint_contact import SHAPES  # noqa: E402
from autofly_ue5.expert.run_shape import find_sessions, run_footprint_shape  # noqa: E402
from autofly_ue5.expert.seeds import EVAL_CALLBACK_SEED_BASE  # noqa: E402
from autofly_ue5.paths import RUNS_DIR  # noqa: E402
from autofly_ue5.scenes.resolve import resolve_scene  # noqa: E402
from scripts.select_checkpoint import CHECKPOINT_RE  # noqa: E402


def _gate_subprocess(argv: list[str]) -> int:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(DISPLAY=os.environ.get("DISPLAY", ":1"), SDL_VIDEODRIVER="x11")
    return subprocess.run([sys.executable, "-m", "scripts.m2_gate", *argv], cwd=_ROOT, env=env).returncode


def due_checkpoints(run_root: Path, every: int) -> list[tuple[int, Path]]:
    """(steps, path) of every checkpoint at a multiple of `every` without a result yet, in step order."""
    out_dir = Path(run_root) / "eval_watch"
    due = []
    for path in sorted((Path(run_root) / "checkpoints").glob("rl_model_*_steps.zip")):
        m = CHECKPOINT_RE.match(path.name)
        if m and int(m.group(1)) % every == 0 and not (out_dir / f"step_{int(m.group(1))}.json").is_file():
            due.append((int(m.group(1)), path))
    return sorted(due)


def evaluate_pending(run_root: Path, *, every: int, episodes: int, instance: int, scene: str, scene_config: str,
                     run_gate: Callable[[list[str]], int] = _gate_subprocess,
                     log: Callable[[str], None] = lambda line: print(line, flush=True),
                     footprint_shape: str | None = None) -> list[int]:
    """Score every checkpoint at a multiple of `every` that has no result yet; return the steps scored now. A failed
    evaluation leaves no result, so the next call tries it again. `footprint_shape`: the mesh scene's footprint shape
    the run trains under, passed to the gate (None: nothing passed; the gate then reads the checkpoint's run itself)."""
    out_dir = Path(run_root) / "eval_watch"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for steps, path in due_checkpoints(run_root, every):
        out = out_dir / f"step_{steps}.json"
        partial = out.with_suffix(".partial.json")
        argv = ["--scene", scene, "--scene-config", scene_config, "--model", f"step_{steps}={path}",
                "--conditions", "deterministic", "--episodes", str(episodes), "--seed-base", str(EVAL_CALLBACK_SEED_BASE),
                "--instance", str(instance), "--out", str(partial)]
        if footprint_shape is not None:
            argv += ["--footprint-shape", footprint_shape]
        code = run_gate(argv)
        record = json.loads(partial.read_text()) if partial.is_file() else None
        det = (record or {}).get("checkpoints", {}).get(f"step_{steps}", {}).get("deterministic", {})
        if code != 0 or det.get("status") != "ok":
            log(f"EVALWATCH step {steps}: FAILED (exit {code}, status {det.get('status')}); will retry")
            continue
        partial.rename(out)
        done.append(steps)
        log(f"EVALWATCH step {steps}: success {det['success_rate']:.2f} return {det['mean_return']:.2f} "
            f"collision {det['collision_rate']:.2f} oob {det['out_of_bounds_rate']:.2f} "
            f"sources {det.get('collision_sources')} ({det['n_episodes']} eps)")
    return done


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--job", required=True, help="the training job's run_job name: stop once its exit file appears")
    p.add_argument("--every", type=int, default=50_000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--instance", type=int, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--scene-config", required=True)
    p.add_argument("--poll-s", type=float, default=60.0)
    p.add_argument("--footprint-shape", choices=SHAPES, default=None,
                   help="a mesh scene's footprint shape, only for a run whose sessions.json names none; default: the "
                        "one the run trains under (expert/run_shape.py), and a flag that contradicts it is refused")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exit_file = RUNS_DIR / "jobs" / f"{args.job}.exit"
    footprint_shape, shape_known = None, False
    while True:
        finished = exit_file.is_file()  # read first: a checkpoint written before the exit must still be scored
        # The run's footprint shape, read once its record exists, or once a checkpoint is due (then a record still
        # missing is missing); refused, and the watch ended, where it cannot be known or contradicts the flag.
        if not shape_known and (find_sessions(args.run_root) is not None or due_checkpoints(args.run_root, args.every)):
            try:
                footprint_shape = run_footprint_shape(resolve_scene(args.scene).layout, args.run_root,
                                                      args.footprint_shape)
            except (FileNotFoundError, ValueError) as err:
                print(f"EVALWATCH refusing: {err}", file=sys.stderr, flush=True)
                return 2
            shape_known = True
        evaluate_pending(args.run_root, every=args.every, episodes=args.episodes, instance=args.instance,
                         scene=args.scene, scene_config=args.scene_config, run_gate=_gate_subprocess,
                         footprint_shape=footprint_shape)
        if finished:
            print("EVALWATCH training ended; every due checkpoint scored", flush=True)
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
