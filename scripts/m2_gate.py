"""M2 exit gate: evaluates the trained SAC expert(s) against spec Sec8/Sec9.5's acceptance bar (Task 9).

Gates BOTH checkpoints Task 8 produced -- `runs/expert/<scene>/best/best_model.zip` (EvalCallback's
best-REWARD save, spec Sec8) and `runs/expert/<scene>/final.zip` (the end-of-budget model) -- rather than
silently picking whichever scores higher: the Task 9 brief is explicit that which one (if either) flies
M3's data collection is the controller's decision, not this script's, and that reporting a flattering
number instead of a truthful one "would poison every dataset and every student-model comparison built on
top of it."

For each checkpoint, runs `--episodes` (>= 200 for the acceptance number itself, spec Sec8) DETERMINISTIC
episodes -- this is the number `gate_passes()` checks against 0.95, since a deterministic policy is what
will actually fly M3's collection -- and the same count of STOCHASTIC episodes (spec Sec9.5's own
collection protocol is stochastic; that number is reported for the record but is not gated). All four
(checkpoint, condition) combinations share the exact same `EVAL_SEED_BASE`-derived seed stream, so every
comparison is apples-to-apples. Priority order (brief's own, used when a run is stopped or fails partway):
(a) best_model deterministic, (b) final deterministic, (c) best_model stochastic, (d) final stochastic --
see `combo_order`.

Reuses, rather than reinvents: `autofly_ue5.expert.resilient.ResilientAutoFlyEnv` for the documented
recoverable backend hazards (`CameraPoseError`, `StepTimingError`, `StaleStateError`,
`CommandTimeoutError`, `pynng.exceptions.Timeout`); `autofly_ue5.expert.evaluate.evaluate_policy_episodes`
for the actual episode loop and its fault-vs-policy-outcome separation; `autofly_ue5.sim.process.sweep_orphaned_instances`
and `autofly_ue5.expert.vec.teardown` for GPU-safe startup/shutdown; `autofly_ue5.expert.train.sha256_of` for
checkpoint provenance.

Writes `docs/gates/m2_gate.json` incrementally -- after EVERY (checkpoint, condition) combination, not
just at the end -- so a run that is stopped partway, or that fails outright, still leaves an honest,
non-empty record: an un-attempted combination is marked `"status": "not_run"`, never silently dropped
(gate item 7). `main()` force-exits via `os._exit()`, exactly like `train.py`, because the `projectairsim`
client leaves a non-daemon thread alive that blocks normal interpreter shutdown forever.

    env -u PYTHONPATH .venv/bin/python -m scripts.m2_gate --episodes 200 --out docs/gates/m2_gate.json
    # (also runnable as `env -u PYTHONPATH .venv/bin/python scripts/m2_gate.py ...`)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap: a direct `python scripts/m2_gate.py` invocation puts `scripts/` (not the project root) at sys.path[0];
# putting the root first makes this checkout's `autofly_ue5` win even where the package is not installed editable.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable  # noqa: E402

from autofly_ue5.evidence import default_evidence_path, record_destination, refuse_existing_evidence  # noqa: E402
from autofly_ue5.expert.env import AutoFlyEnv  # noqa: E402
from autofly_ue5.expert.evaluate import (  # noqa: E402
    DEFAULT_MAX_FAULT_RETRIES_PER_EPISODE,
    DEFAULT_MAX_STEPS_PER_EPISODE,
    EvaluationInterrupted,
    evaluate_policy_episodes,
    fault_summary_delta,
)
from autofly_ue5.expert.faults import KNOWN_FAULT_NAMES  # noqa: E402
from autofly_ue5.expert.obs import ObsConfig, obs_config_from_space  # noqa: E402
from autofly_ue5.expert.resilient import ResilientAutoFlyEnv  # noqa: E402
from autofly_ue5.expert.reward import REWARD_VERSION  # noqa: E402
from autofly_ue5.expert.seeds import EVAL_SEED_BASE  # noqa: E402
from autofly_ue5.expert.train import sha256_of  # noqa: E402
from autofly_ue5.expert.vec import teardown  # noqa: E402
from autofly_ue5.paths import ROOT, RUNS_DIR  # noqa: E402
from autofly_ue5.scenes.resolve import resolve_scene  # noqa: E402
from autofly_ue5.sim.airsim_backend import (  # noqa: E402
    ProjectAirSimSimulator,
    scene_config_factory,
    scene_config_record,
)
from autofly_ue5.sim.process import (  # noqa: E402
    SIM_RUN_DIR,
    instance_dir,
    route_client_log,
    slot_busy,
    stop_instances,
    sweep_orphaned_instances,
)
from autofly_ue5.validate.engine_check import audit_engine_faults, boot_id, xid_count  # noqa: E402

# --------------------------------------------------------------------------------------------------------
# The gate itself: a pure function of the numbers, so it is trivially unit-testable without a live run.
# --------------------------------------------------------------------------------------------------------
GATE_MIN_SUCCESS_RATE = 0.95
GATE_MIN_EPISODES = 200
CONDITION_PRIORITY = ("deterministic", "stochastic")


def gate_passes(*, success_rate: float, n_episodes: int, faults_ok: bool) -> bool:
    """The M2 acceptance bar (spec Sec8/Sec9): >= 95% success over >= 200 episodes, with no engine-level
    fault (GPU Xid, reboot) during the run. Not rounded -- 0.9499 is a fail, not "basically 0.95"."""
    return success_rate >= GATE_MIN_SUCCESS_RATE and n_episodes >= GATE_MIN_EPISODES and bool(faults_ok)


def parse_model_args(specs: list[str] | None, scene: str, run_root: Path | None = None) -> dict[str, Path]:
    """`--model NAME=PATH` entries (repeatable) into an ordered {name: path} dict; the CLI default (no
    `--model` given) is BOTH of a training run's checkpoints under `run_root` (default `runs/expert/<scene>/`),
    best before final so `combo_order` runs the higher-priority checkpoint's deterministic condition first."""
    if not specs:
        run_root = run_root or RUNS_DIR / "expert" / scene
        return {"best_model": run_root / "best" / "best_model.zip", "final": run_root / "final.zip"}
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--model expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        out[name] = Path(path)
    return out


def combo_order(models: dict[str, Path], conditions: list[str]) -> list[tuple[str, str]]:
    """(checkpoint_name, condition) pairs in the Task 9 brief's own priority order: EVERY checkpoint's
    deterministic condition before ANY checkpoint's stochastic one -- (a) best det, (b) final det,
    (c) best stoch, (d) final stoch -- so a time-limited or interrupted run always has the
    highest-value combinations already on disk, never a silently truncated one."""
    ordered_conditions = [c for c in CONDITION_PRIORITY if c in conditions]
    return [(name, condition) for condition in ordered_conditions for name in models]


INSTANCES_PATH = ROOT / "docs" / "gates" / "m2_instances.json"


def default_instances_path(scene: str) -> Path | None:
    """The scene's own throughput measurement (scripts/measure_instances.py --scene), None for a scene without one."""
    try:
        return default_evidence_path(scene, "instances")
    except ValueError:
        return None


def checkpoint_obs_config(path: Path) -> ObsConfig:
    """The observation a checkpoint was trained on, read from its zip without building a model or an env: the gate
    must fly each checkpoint on its own observation (an s01d checkpoint sees a 3-frame float16 stack, M2's one float32
    frame)."""
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _params, _variables = load_from_zip_file(path, device="cpu", load_data=True)
    return obs_config_from_space(data["observation_space"])


def _load_throughput_projection(instances_path: Path = INSTANCES_PATH) -> dict | None:
    """Task 7's own instance-scaling measurement, carried into the gate record so M5's cost (10 more
    experts) is on the record alongside M2's pass/fail (gate item 5)."""
    if not instances_path.is_file():
        return None
    data = json.loads(instances_path.read_text())
    chosen_n = data.get("chosen_n")
    per_n = data.get("per_n", {}) or {}
    return {
        "source": str(instances_path),
        "chosen_n": chosen_n,
        "measured_env_steps_per_s_total": per_n.get(str(chosen_n), {}).get("env_steps_per_s_total") if chosen_n is not None else None,
        "projection": data.get("projection"),
    }


# --------------------------------------------------------------------------------------------------------
# Live env construction -- reuses AutoFlyEnv + ResilientAutoFlyEnv exactly as train.py builds them for one
# worker (no VecEnv needed: the gate steps a single environment sequentially, never in parallel).
# --------------------------------------------------------------------------------------------------------
def build_eval_env(scene: str, *, instance: int, sim_factory: Callable[[], Any], seed_base: int = EVAL_SEED_BASE,
                   sim_root: Path = SIM_RUN_DIR, obs_config: ObsConfig | None = None) -> ResilientAutoFlyEnv:
    # seed_base: every gate episode is an explicit reset(seed=...); the counter base only matters for a stray
    # seed=None reset, which then still draws from the gate's own range.
    resolved = resolve_scene(scene)
    scene_file, layout, map_path = resolved.scene, resolved.layout, resolved.map_path
    route_client_log(instance_dir(instance, sim_root) / "client.log")
    base = AutoFlyEnv(scene_file, layout, sim_factory, map_path=map_path, instance=instance, seed_base=seed_base,
                      obs_config=obs_config)
    return ResilientAutoFlyEnv(base, instance=instance, sim_root=sim_root)


def default_sac_loader(path: Path, *, device: str = "auto"):
    from stable_baselines3 import SAC

    return SAC.load(path, device=device)


# --------------------------------------------------------------------------------------------------------
# The run itself.
# --------------------------------------------------------------------------------------------------------
def run(
    *,
    scene: str,
    model_paths: dict[str, Path],
    conditions: list[str],
    n_episodes: int,
    seed_base: int,
    instance: int,
    out_path: Path,
    device: str = "auto",
    sim_factory: Callable[[], Any] = ProjectAirSimSimulator,
    load_model: Callable[[Path], Any] | None = None,
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE,
    max_fault_retries_per_episode: int = DEFAULT_MAX_FAULT_RETRIES_PER_EPISODE,
    sim_root: Path = SIM_RUN_DIR,
    scene_config: str | None = None,
    instances_path: Path | None = None,
    obs_config_reader: Callable[[Path], ObsConfig] | None = None,
) -> dict[str, Any]:
    load_model = load_model or (lambda p: default_sac_loader(p, device=device))
    obs_config_reader = obs_config_reader or checkpoint_obs_config
    instances_path = instances_path if instances_path is not None else default_instances_path(scene)
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    run_start_epoch = time.time()
    xid_before = xid_count(run_started)
    boot_before = boot_id()

    swept = sweep_orphaned_instances(sim_root)  # a crashed earlier run's simulators; never a live run's
    if swept:
        print(f"swept orphaned instances before starting: {swept}", file=sys.stderr)

    throughput_projection = _load_throughput_projection(instances_path) if instances_path is not None else None
    projection_missing = (None if throughput_projection is not None else
                          f"scene {scene} has no throughput measurement of its own: pass --out to "
                          f"scripts/measure_instances.py --scene {scene}" if instances_path is None else
                          f"{instances_path} does not exist: run scripts/measure_instances.py --scene {scene} before the "
                          f"gate (PLAN2 wants the throughput projection on the record)")
    combos = combo_order(model_paths, conditions)
    checkpoints: dict[str, dict[str, Any]] = {}
    for name, path in model_paths.items():
        checkpoints[name] = {
            "path": str(path),
            "sha256": sha256_of(path) if path.is_file() else None,
            **{cond: {"status": "not_run", "error": None} for cond in CONDITION_PRIORITY},
        }

    env: ResilientAutoFlyEnv | None = None
    status = "ok"
    error_message: str | None = None
    obs_config: ObsConfig | None = None

    def _write() -> dict[str, Any]:
        # Every log this slot wrote during the run (relaunches rotate sim.log) and a readable journal (C7).
        engine_faults = audit_engine_faults(since=run_started, since_epoch=run_start_epoch, xid_before=xid_before,
                                            boot_before=boot_before, log_dirs=[instance_dir(instance, sim_root)])
        faults_ok = engine_faults.pop("ok")
        cumulative = (
            env.get_fault_summary()
            if env is not None
            else {"instance": instance, "fault_counts": {n: 0 for n in KNOWN_FAULT_NAMES},
                  "recovered_counts": {n: 0 for n in KNOWN_FAULT_NAMES}, "relaunch_count": 0}
        )
        deterministic_gate_pass = {
            name: bool(
                checkpoints[name]["deterministic"].get("status") == "ok"
                and gate_passes(
                    success_rate=checkpoints[name]["deterministic"].get("success_rate", 0.0),
                    n_episodes=checkpoints[name]["deterministic"].get("n_episodes", 0),
                    faults_ok=faults_ok,
                )
            )
            for name in checkpoints
        }
        gate = {
            "description": f"Expert exit gate on scene {scene} (spec Sec8/Sec9.5; Task 9 of plan 2, M2d for s01d).",
            "scene": scene,
            "obs_config": obs_config.to_json() if obs_config is not None else None,
            "reward_version": REWARD_VERSION,
            "scene_config": scene_config_record(scene_config) if scene_config else None,
            "eval_seed_base": seed_base,
            "episodes_requested_per_condition": n_episodes,
            "conditions_requested": list(conditions),
            "priority_order": [f"{name}:{cond}" for name, cond in combos],
            "checkpoints": checkpoints,
            "deterministic_gate_pass_by_checkpoint": deterministic_gate_pass,
            "pass": any(deterministic_gate_pass.values()),
            "notes": [
                "Both checkpoints are gated and reported without selection; which checkpoint (if either) "
                "flies M3's data collection is the controller's decision, not this script's.",
                "A 20-episode eval cannot distinguish 0.85 from 1.0 against a 0.95 threshold; this run "
                "uses >= 200 episodes specifically so it can.",
                "Held out for both checkpoints: training-time evaluation (best_model.zip selection) draws its "
                "episodes from EVAL_CALLBACK_SEED_BASE (200,000,000+), disjoint from this gate's EVAL_SEED_BASE "
                "range, and training workers draw from ranges below both. (The 2026-09-17 run predates this: its "
                "EvalCallback walked seeds 100,000,000+0..~177, overlapping this gate's -- see "
                "docs/decisions/2026-09-25-code-review-findings.md.)",
            ],
            "cumulative_backend_faults": cumulative,
            "engine_faults": engine_faults,
            "faults_ok": faults_ok,
            "throughput_projection": throughput_projection,
            "throughput_projection_missing": projection_missing,
            "status": status,
            "error": error_message,
            "run_started": run_started,
            "run_finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Until an episode has been scored the record is not evidence: under docs/gates/ it would block the retry.
        evaluated = any(isinstance(c, dict) and c.get("n_episodes", 0) > 0 for ck in checkpoints.values() for c in ck.values())
        destination = record_destination(out_path, did_work=evaluated)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(gate, indent=2) + "\n")
        return gate

    try:
        # Each checkpoint's own observation, read before anything launches: checkpoints that disagree cannot share one
        # env, and comparing them on different observations would be no comparison.
        configs = {name: obs_config_reader(path) for name, path in model_paths.items() if Path(path).is_file()}
        if len(set(configs.values())) > 1:
            raise ValueError(f"the checkpoints' observations disagree ({ {k: v.to_json() for k, v in configs.items()} }); "
                             f"gate them in separate runs")
        obs_config = next(iter(configs.values()), None)
        env = build_eval_env(scene, instance=instance, sim_factory=sim_factory, seed_base=seed_base, sim_root=sim_root,
                             obs_config=obs_config)
        _write()  # an honest "everything not_run yet" record exists on disk even if launch itself fails
        for name, cond in combos:
            path = model_paths[name]
            if not path.is_file():
                checkpoints[name][cond] = {"status": "failed", "error": f"checkpoint not found: {path}"}
                status = "partial"
                _write()
                continue
            print(f"=== evaluating {name} ({cond}), {n_episodes} episodes ===", file=sys.stderr)
            model = load_model(path)
            before = env.get_fault_summary()
            try:
                report = evaluate_policy_episodes(
                    model, env, n_episodes, seed_base, deterministic=(cond == "deterministic"),
                    max_steps_per_episode=max_steps_per_episode,
                    max_fault_retries_per_episode=max_fault_retries_per_episode,
                )
                combo_status, combo_error = "ok", None
            except EvaluationInterrupted as err:
                report = err.partial_report
                combo_status, combo_error = "failed", str(err)
            after = env.get_fault_summary()
            checkpoints[name][cond] = {
                "status": combo_status,
                "error": combo_error,
                "episodes_requested": n_episodes,
                **report.to_dict(),
                "backend_faults": fault_summary_delta(before, after),
            }
            if combo_status != "ok":
                status = "partial"
            _write()
    except Exception as err:
        error_message = f"{type(err).__name__}: {err}"
        status = "failed"
        print(f"m2_gate failed: {error_message}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # Mirrors train.py's own finally block: every step wrapped, so a failure while cleaning up after a
        # failure still reaches the gate-record write below rather than crashing before anything is saved.
        if env is not None:
            try:
                teardown(env, [instance], sim_root)  # bounded close + stop this gate's own slot, nothing else
            except Exception as err:
                print(f"WARNING: teardown() raised {type(err).__name__}: {err}", file=sys.stderr)
                traceback.print_exc()
        else:
            try:
                stop_instances([instance], sim_root)
            except Exception as err:
                print(f"WARNING: stop_instances() raised {type(err).__name__}: {err}", file=sys.stderr)

    return _write()


# --------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="s01")
    p.add_argument(
        "--model", action="append", default=None, metavar="NAME=PATH",
        help="a checkpoint to gate, as NAME=PATH; repeatable. Default: both of Task 8's checkpoints "
             "(best_model=runs/expert/<scene>/best/best_model.zip, final=runs/expert/<scene>/final.zip).",
    )
    p.add_argument("--episodes", type=int, default=GATE_MIN_EPISODES)
    p.add_argument("--conditions", nargs="+", choices=CONDITION_PRIORITY, default=list(CONDITION_PRIORITY))
    p.add_argument("--out", type=Path, default=None,
                   help="the gate record; default docs/gates/<milestone>_gate.json (s01: m2, s01d: m2d). A committed "
                        "record is never written over")
    p.add_argument("--scene-config", default=None,
                   help="Project AirSim scene config in configs/ (default scene_autofly_<scene>.jsonc): gate on the clock the "
                        "expert was trained on")
    p.add_argument("--run-root", type=Path, default=None,
                   help="the training run whose best/best_model.zip and final.zip to gate (default runs/expert/<scene>)")
    p.add_argument("--instance", type=int, default=0,
                   help="the simulator slot; refused while another live run holds it (an --instances N training run "
                        "holds 0..N)")
    p.add_argument("--sim-root", type=Path, default=SIM_RUN_DIR, help=argparse.SUPPRESS)
    p.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE)
    p.add_argument("--max-fault-retries-per-episode", type=int, default=DEFAULT_MAX_FAULT_RETRIES_PER_EPISODE)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        out_path = args.out if args.out is not None else default_evidence_path(args.scene, "gate")
        refuse_existing_evidence(out_path)
        resolved = resolve_scene(args.scene)
        busy = slot_busy(args.instance, args.sim_root)
        if busy:
            raise ValueError(f"{busy}; pass a free --instance")
    except (FileNotFoundError, FileExistsError, ValueError) as err:
        print(f"refusing to start: {err}", file=sys.stderr)
        return 2
    model_paths = parse_model_args(args.model, args.scene, run_root=args.run_root)
    scene_config = args.scene_config or resolved.default_scene_config
    gate = run(
        scene=args.scene, model_paths=model_paths, conditions=list(args.conditions), n_episodes=args.episodes,
        seed_base=args.seed_base, instance=args.instance, out_path=out_path, device=args.device,
        max_steps_per_episode=args.max_steps_per_episode,
        max_fault_retries_per_episode=args.max_fault_retries_per_episode,
        sim_factory=scene_config_factory(scene_config, resolved.movable_objects, run_root=args.sim_root),
        scene_config=scene_config, sim_root=args.sim_root,
    )
    print(json.dumps(
        {"status": gate["status"], "pass": gate["pass"],
         "deterministic_gate_pass_by_checkpoint": gate["deterministic_gate_pass_by_checkpoint"]},
        indent=2,
    ))
    return 0 if gate["status"] == "ok" else 1


if __name__ == "__main__":
    _code = main()
    # Not sys.exit(): see train.py's own docstring/__main__ guard -- the projectairsim client leaves a
    # non-daemon thread alive that blocks CPython's interpreter-shutdown sequence forever. Everything
    # durable (the gate JSON) is already written by run() above; os._exit() guarantees this process
    # actually terminates instead of hanging "still running" forever.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
