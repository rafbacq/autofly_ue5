"""Closed-loop evaluation of a policy on a frozen condition file (spec §12 M6; plan 6 item 6, revision 1 items 4-5).

`run_condition` flies each episode of a condition file the way the dataset's episodes were flown, and scores it the way
the paper scores its models:

- The episode: AutoFlyEnv on the scene's own level (scenes/resolve.py), reset to the file's seed with the collector's a0
  aligned start (spec §9 step 4), its target and distractors drawn from the target pool (assets/targets.json by default),
  the instruction naming the drawn target. Lock-step: the simulator's clock advances only inside step(), so a policy's
  latency never changes the dynamics.
- The task's rules: physical contact on pillar and box scenes, the mover rule on s01d, and on a mesh scene the footprint
  rule measured to the model's outline (expert/footprint_contact.py), the task's rule whatever shape an expert trained
  under; the altitude band and the bounds; 5 m and 15 degrees for success; AutoFlyEnv's step limit for a timeout.
- What the policy sees: a PolicyObservation (evaluation/policy.py) built per step with the collector's own code
  (collect.collector.record_inputs), state[0] and state[1] zeroed unless privileged_state.
- Faults: a backend fault mid-episode (ResilientAutoFlyEnv) discards the attempt and flies the seed again from a reset
  policy, as the gate and the collector do; it is counted, never scored.
- Per episode: the outcome and what ended it, steps, the flown path's horizontal length L, L_opt's and PER's intervals
  (evaluation/metrics.py), the target's entry, the instruction, the actions' summary and the policy's time per action.
- Per scene and over all scenes: SR, CR, out-of-bounds and timeout rates with Wilson intervals, PER's interval over the
  successes. The JSON record is written durably after every episode, so a run that stops leaves what it flew, and a
  short markdown summary is written beside it.

Scenes fly one after another on one simulator slot, each on its own level (a launch per scene). A scene the file gates is
flown only once a gate record of it passes (episodes.m4_gate_status). Refused before anything launches: a scene whose file
or layout no longer matches the file's sha256, a policy that needs the privileged state without it, a target_split the
env cannot draw, a pool a scene cannot draw from. A policy that breaks stops the run (PolicyError, and the record says
so); a backend that cannot fly a scene fails that scene, and the run moves on to the next.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from autofly_ue5.collect.collector import record_inputs
from autofly_ue5.dataset.targets import VERSION as POOL_VERSION
from autofly_ue5.dataset.targets import TargetPool
from autofly_ue5.durable import write_text_durably
from autofly_ue5.evaluation.episodes import ConditionFile, EvalEpisode, load_condition, m4_gate_status
from autofly_ue5.evaluation.metrics import (
    L_OPT_INFLATE_M,
    L_OPT_RESOLUTION_M,
    horizontal_path_length_m,
    l_opt_bounds,
    per_bounds,
    summarise,
)
from autofly_ue5.evaluation.policy import (
    ACTION_FIELDS,
    PRIVILEGED_STATE_INDICES,
    PolicyError,
    PolicyObservation,
    checked_action,
    describe_policy,
    zero_privileged,
)
from autofly_ue5.evidence import COMMITTED_GATES_DIR, record_destination
from autofly_ue5.expert import env as env_module
from autofly_ue5.expert.env import AutoFlyEnv
from autofly_ue5.expert.footprint_contact import task_contact_rules
from autofly_ue5.expert.obs import ObsConfig
from autofly_ue5.expert.resilient import ResilientAutoFlyEnv
from autofly_ue5.expert.reward import RewardConfig, reward_config_for_scene
from autofly_ue5.expert.seeds import EVAL_HARNESS_SEED_BASE
from autofly_ue5.expert.vec import teardown
from autofly_ue5.paths import CONFIGS_DIR, ROOT
from autofly_ue5.scenes.resolve import ResolvedScene, resolve_scene
from autofly_ue5.sim.airsim_backend import scene_config_factory, scene_config_record
from autofly_ue5.sim.process import SIM_RUN_DIR, instance_dir, route_client_log, stop_instances, sweep_orphaned_instances
from autofly_ue5.validate.engine_check import audit_engine_faults, boot_id, xid_count

RECORD_VERSION = 1
TARGETS_PATH = ROOT / "assets" / "targets.json"
TASK_FOOTPRINT_SHAPE = "outline"  # the task's rule since 2026-10-08; the circle only scores runs that trained under it
DEFAULT_MAX_STEPS = RewardConfig().step_limit  # 300 (spec §8)
DEFAULT_MAX_FAULT_RETRIES = 20  # as the gate's and the collector's: 20 faults in a row on one seed is a broken backend
STEP_GUARD = 5  # the loop's own bound past the env's step limit, which always ends an episode first

SimFactoryHook = Callable[[ResolvedScene, str], Callable[[], Any]]


def env_accepts_target_split() -> bool:
    """Whether this checkout's AutoFlyEnv can draw a condition's target split (the unseen-target draw, plan 6 revision 1
    item 5, lands with the target work)."""
    return "target_split" in inspect.signature(env_module.AutoFlyEnv.__init__).parameters


def default_scene_config(resolved: ResolvedScene) -> str:
    """The scene's 1 ms config (scene_autofly_<level>_fast.jsonc), the clock every expert trained, was gated and collected
    on; its plain config where it has none. Both are steppable, so either is lock-step; the fast one runs faster."""
    fast = f"scene_autofly_{resolved.base_id}_fast.jsonc"
    return fast if (CONFIGS_DIR / fast).is_file() else resolved.default_scene_config


def build_env(resolved: ResolvedScene, *, instance: int, sim_factory: Callable[[], Any], sim_root: Path,
              obs_config: ObsConfig | None, target_pool: TargetPool, target_split: str | None,
              max_steps: int) -> ResilientAutoFlyEnv:
    """One scene's env as the harness flies it: the scene's level, the pool, the task's contact rules, the step limit, and
    `target_split` only when the condition asks for one (the scene's own split otherwise)."""
    split_kwargs = {"target_split": target_split} if target_split is not None else {}
    base = AutoFlyEnv(resolved.scene, resolved.layout, sim_factory, map_path=resolved.map_path, instance=instance,
                      seed_base=EVAL_HARNESS_SEED_BASE, max_episode_steps=max_steps, obs_config=obs_config,
                      target_pool=target_pool, footprint_shape=TASK_FOOTPRINT_SHAPE, **split_kwargs)
    return ResilientAutoFlyEnv(base, instance=instance, sim_root=sim_root)


def _never_launched() -> Any:
    raise AssertionError("the preflight env must never launch a simulator")


@dataclass(frozen=True)
class _ScenePlan:
    scene_id: str
    entry: dict  # the condition file's scene entry
    episodes: tuple[EvalEpisode, ...]
    resolved: ResolvedScene | None  # None while pending
    gate: str | None
    pending_reason: str | None


def _plan(condition: ConditionFile, *, scenes: Iterable[str] | None, max_episodes_per_scene: int | None,
          resolve: Callable[[str], ResolvedScene], gates_dir: Path) -> list[_ScenePlan]:
    chosen = condition.scene_ids if scenes is None else tuple(dict.fromkeys(scenes))
    unknown = [s for s in chosen if s not in condition.scenes]
    if unknown:
        raise ValueError(f"{condition.path} has no scene {', '.join(unknown)} (its scenes: {', '.join(condition.scene_ids)})")
    if max_episodes_per_scene is not None and max_episodes_per_scene < 1:
        raise ValueError(f"max_episodes_per_scene must be at least 1, got {max_episodes_per_scene}")
    plans, changed = [], []
    for scene_id in chosen:
        entry = condition.scenes[scene_id]
        episodes = condition.episodes_for(scene_id)[:max_episodes_per_scene]
        gate = None
        if entry.get("requires_gate"):
            status = m4_gate_status(scene_id, gates_dir)
            if not status["ready"]:
                plans.append(_ScenePlan(scene_id, entry, episodes, None, None, status["reason"]))
                continue
            gate = status["gate"]
        resolved = resolve(scene_id)
        if resolved.scene.sha256 != entry["scene_sha256"] or resolved.layout_sha256 != entry["layout_sha256"]:
            changed.append(f"{scene_id} (scene file {resolved.scene.sha256[:12]} against {entry['scene_sha256'][:12]}, "
                           f"layout {resolved.layout_sha256[:12]} against {entry['layout_sha256'][:12]})")
        plans.append(_ScenePlan(scene_id, entry, episodes, resolved, gate, None))
    if changed:
        raise ValueError(f"{condition.path} was written against other scene files or layouts: {'; '.join(changed)}. Its "
                         f"seeds are other episodes on these; write a new condition file rather than score them as these")
    return plans


def _call(policy: Any, name: str, method: str, argument: Any) -> Any:
    """policy.<method>(argument), anything it raises turned into a PolicyError that names the policy."""
    try:
        return getattr(policy, method)(argument)
    except PolicyError as err:
        raise PolicyError(f"policy {name}: {err}") from err
    except Exception as err:
        raise PolicyError(f"policy {name}: {method}() raised {type(err).__name__}: {err}") from err


def _action_summary(raw: list[np.ndarray], commanded: list[np.ndarray]) -> dict:
    """Per dimension (forward, yaw rate, vertical): the commanded actions' mean, std, min and max, and how often the
    policy asked for more than the action space allows (clipped as AutoFlyEnv clips)."""
    if not commanded:
        return {"fields": list(ACTION_FIELDS), "n": 0}
    flown, asked = np.asarray(commanded, dtype=np.float64), np.asarray(raw, dtype=np.float64)
    rounded = lambda values: [round(float(v), 4) for v in values]  # noqa: E731
    return {"fields": list(ACTION_FIELDS), "n": len(flown), "mean": rounded(flown.mean(axis=0)),
            "std": rounded(flown.std(axis=0)), "min": rounded(flown.min(axis=0)), "max": rounded(flown.max(axis=0)),
            "clipped_fraction": rounded((np.abs(asked - flown) > 1e-6).mean(axis=0))}


def _fly_once(policy: Any, env: ResilientAutoFlyEnv, episode: EvalEpisode, *, a0: str, privileged_state: bool,
              max_steps: int, layout, success_radius_m: float, name: str) -> dict:
    """One attempt at `episode`: its record, or {"fault": name} when a backend fault cut it short. `name` names the
    policy in its errors."""
    obs, info = env.reset(seed=episode.seed, options={"a0": a0})
    # the reset's settle hold, and whether it spawned an object new to the session (expert/env.py; with a pool)
    reset_facts = {key: info[key] for key in ("prewarm_steps", "first_use") if key in info}
    base = env.unwrapped
    setup = base.setup
    raw = base.last_observation
    _call(policy, name, "reset", setup.instruction)
    points = [(raw.pose.x, raw.pose.y)]
    asked, commanded, act_s = [], [], []
    low, high = env.action_space.low, env.action_space.high
    outcome = "exceeded_max_steps"
    for step in range(max_steps + STEP_GUARD):
        rgb, state = record_inputs(raw, setup)
        if not privileged_state:
            state = zero_privileged(state)
        handed = PolicyObservation(rgb=rgb, instruction=setup.instruction, state=state, privileged_state=privileged_state,
                                   step=step, extras={"expert_obs": obs})
        started = time.perf_counter()
        proposed = _call(policy, name, "act", handed)
        act_s.append(time.perf_counter() - started)
        try:
            action = checked_action(proposed)
        except PolicyError as err:
            raise PolicyError(f"policy {name}: {err}") from err
        asked.append(action)
        commanded.append(np.clip(action, low, high))
        obs, _reward, terminated, truncated, info = env.step(action)
        if info.get("sim_fault"):
            return {"fault": info["sim_fault"]}
        raw = base.last_observation
        points.append((raw.pose.x, raw.pose.y))
        if terminated or truncated:
            outcome = info["outcome"]
            break
    length = horizontal_path_length_m(points)
    l_opt = l_opt_bounds(layout, (setup.start.x, setup.start.y), setup.target_xy_z, success_radius_m=success_radius_m,
                         exclude_tags=[route.tag for route in setup.movers])
    target = setup.target
    return {
        "scene": episode.scene, "seed": episode.seed, "target_split": episode.target_split,
        "outcome": outcome, "steps": info.get("steps", len(commanded)) if outcome != "exceeded_max_steps" else len(commanded),
        "instruction": setup.instruction,
        "target": target.to_json() if target is not None else None,
        "distractors": [entry.name for entry in setup.distractor_entries],
        "start_pose": [setup.start.x, setup.start.y, setup.start.z, setup.start.yaw],
        "target_xyz": list(setup.target_xy_z),
        "final_pose": info.get("pose"), "final_distance_m": info.get("final_distance_m"),
        "final_bearing_deg": info.get("bearing_deg"),
        # PER is a successful trial's (the paper's Sec. 4.1): a short failure would read as perfectly efficient.
        "path_length_m": length, "l_opt": l_opt,
        "per": per_bounds(length, l_opt) if outcome == "success" else {"lower": None, "upper": None},
        "collision_source": info.get("collision_source"), "oob_kind": info.get("oob_kind"),
        "n_movers": info.get("n_movers", 0), "mover_in_view": info.get("mover_in_view"),
        "mover_contact": info.get("mover_contact"),
        # A mesh scene's ending, or the outline nearest a physical contact; a mover contact the env inferred from a fault.
        **{key: info[key] for key in ("footprint_contact", "footprint_nearest", "inferred_from") if key in info},
        "actions": _action_summary(asked, commanded),
        "policy_act_s": {"mean": sum(act_s) / len(act_s), "max": max(act_s)} if act_s else None,
        **reset_facts,
    }


def fly_episode(policy: Any, env: ResilientAutoFlyEnv, episode: EvalEpisode, *, a0: str, privileged_state: bool,
                max_steps: int, layout, success_radius_m: float,
                max_fault_retries: int = DEFAULT_MAX_FAULT_RETRIES, policy_name: str | None = None) -> dict:
    """`episode` flown until an attempt runs uninterrupted, with how many attempts faults cost (never scored).
    `policy_name` names the policy in its errors (default: describe_policy's, read here)."""
    name = policy_name if policy_name is not None else describe_policy(policy).get("name")
    faults: Counter[str] = Counter()
    for _attempt in range(max_fault_retries + 1):
        result = _fly_once(policy, env, episode, a0=a0, privileged_state=privileged_state, max_steps=max_steps,
                           layout=layout, success_radius_m=success_radius_m, name=name)
        if "fault" in result:
            faults[result["fault"]] += 1
            print(f"EVAL {episode.scene} seed {episode.seed}: a {result['fault']} cut the attempt short; flying it "
                  f"again ({sum(faults.values())})", file=sys.stderr, flush=True)
            continue
        return {**result, "retries": sum(faults.values()), "fault_counts": dict(faults)}
    raise RuntimeError(f"scene {episode.scene} seed {episode.seed}: no uninterrupted flight in {max_fault_retries} retries "
                       f"({dict(faults)}): a broken backend, not a policy outcome")


def _episode_line(result: dict) -> str:
    """One episode for the log (the gate and the pilot print nothing per episode, and silence never shows progress)."""
    line = f"{result['outcome']} after {result['steps']} steps, L {result['path_length_m']:.1f} m"
    per = result["per"]
    if per["lower"] is not None:
        line += f", PER {per['lower']:.3f}-" + (f"{per['upper']:.3f}" if per["upper"] is not None else "?")
    return line + (f", {result['retries']} attempt(s) replayed" if result.get("retries") else "")


def _pending_scene(plan: _ScenePlan) -> dict:
    return {"status": "pending", "reason": plan.pending_reason, "error": None,
            "requires_gate": plan.entry.get("requires_gate"), "gate": None,
            "scene_sha256": plan.entry["scene_sha256"], "layout_sha256": plan.entry["layout_sha256"], "map": None,
            "scene_config": None, "target_split": plan.entry["target_split"], "episodes_requested": len(plan.episodes),
            "contact_rules": None, "dynamic": None, "metrics": None, "backend_faults": None}


def _config_record(name: str) -> dict:
    try:
        return scene_config_record(name)
    except OSError:
        return {"file": name, "sha256": None}


def run_condition(policy: Any, condition_file: Path | ConditionFile, *, instance: int, sim_root: Path = SIM_RUN_DIR,
                  privileged_state: bool = False, max_steps: int = DEFAULT_MAX_STEPS, out_path: Path | None = None,
                  target_pool: TargetPool | Path = TARGETS_PATH, scenes: Iterable[str] | None = None,
                  max_episodes_per_scene: int | None = None, sim_factory: SimFactoryHook | None = None,
                  scene_config: Callable[[ResolvedScene], str] = default_scene_config,
                  resolve: Callable[[str], ResolvedScene] = resolve_scene, gates_dir: Path = COMMITTED_GATES_DIR,
                  max_fault_retries: int = DEFAULT_MAX_FAULT_RETRIES, provenance: dict | None = None) -> dict:
    """Fly `policy` on the condition file's episodes (all of them, or `scenes`, each cut to `max_episodes_per_scene`) on
    simulator slot `instance`, and return the record (written to `out_path` as it goes, with a markdown summary beside
    it). `sim_factory(resolved, scene_config)` replaces the Project AirSim factory (tests fly FakeSimulator); `resolve`
    and `gates_dir` replace where scenes and gate records are read. A refusal (ValueError, FileNotFoundError) comes before
    anything launches; after that the run's status is in the record: "ok", "partial" (a scene's backend failed) or
    "failed" (the policy broke)."""
    condition = condition_file if isinstance(condition_file, ConditionFile) else load_condition(condition_file)
    pool = target_pool if isinstance(target_pool, TargetPool) else TargetPool.load(target_pool)
    described = describe_policy(policy)  # once: a VLA's describe() may hash a checkpoint of many gigabytes
    name = described.get("name")
    if getattr(policy, "requires_privileged_state", False) and not privileged_state:
        raise ValueError(f"policy {name} steers by the privileged state (state[0], state[1]), which the harness zeroes: "
                         f"run it with privileged_state=True (--privileged-state)")
    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")
    if out_path is not None and Path(out_path).suffix != ".json":
        raise ValueError(f"the record {out_path} must be a .json file: its markdown summary is written beside it as .md")
    plans = _plan(condition, scenes=scenes, max_episodes_per_scene=max_episodes_per_scene, resolve=resolve,
                  gates_dir=gates_dir)
    ready = [p for p in plans if p.resolved is not None]
    splits = sorted({p.entry["target_split"] for p in ready} - {None})
    if splits and not env_accepts_target_split():
        raise ValueError(f"{condition.path} asks for target_split {', '.join(splits)}, and this checkout's AutoFlyEnv has "
                         f"no target_split to draw it (the unseen-target split lands with plan 6's target work)")
    obs_config = getattr(policy, "obs_config", None)
    for plan in ready:  # every scene's env built once, unlaunched: the pool, the fields, the split are refused here
        build_env(plan.resolved, instance=instance, sim_factory=_never_launched, sim_root=sim_root, obs_config=obs_config,
                  target_pool=pool, target_split=plan.entry["target_split"], max_steps=max_steps)

    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    run_start_epoch = time.time()
    xid_before, boot_before = xid_count(run_started), boot_id()
    episodes: list[dict] = []
    record: dict[str, Any] = {
        "harness": "autofly_ue5.evaluation", "record_version": RECORD_VERSION,
        "description": "Closed-loop evaluation (spec §12 M6, plan 6 item 6): SR, CR, out-of-bounds and timeout rates with "
                       "95 % Wilson intervals; PER = L_opt / max(L, L_opt) over successes, an interval between the "
                       "straight line to the success disc and the grid's shortest path.",
        "condition": {"name": condition.name, "file": None if condition.path is None else str(condition.path),
                      "sha256": condition.sha256, "episodes_per_scene": condition.episodes_per_scene,
                      "scenes": list(condition.scene_ids)},
        "policy": described,
        "privileged_state": bool(privileged_state), "privileged_state_indices": list(PRIVILEGED_STATE_INDICES),
        "a0": condition.a0, "max_steps": max_steps, "max_episodes_per_scene": max_episodes_per_scene,
        "contact_rule_shape": TASK_FOOTPRINT_SHAPE,
        "target_pool": {"path": pool.path, "sha256": pool.sha256, "version": POOL_VERSION},
        "l_opt": {"inflate_m": L_OPT_INFLATE_M, "resolution_m": L_OPT_RESOLUTION_M},
        "instance": instance, "provenance": provenance,
        "scenes": {}, "overall": summarise([]), "episodes": episodes,
        "status": "running", "error": None, "run_started": run_started, "run_finished": None,
    }
    for plan in plans:
        if plan.resolved is None:
            record["scenes"][plan.scene_id] = _pending_scene(plan)
            continue
        layout = plan.resolved.layout
        record["scenes"][plan.scene_id] = {
            "status": "not_run", "reason": None, "error": None, "requires_gate": plan.entry.get("requires_gate"),
            "gate": plan.gate, "scene_sha256": plan.entry["scene_sha256"], "layout_sha256": plan.entry["layout_sha256"],
            "map": plan.resolved.map_path, "scene_config": _config_record(scene_config(plan.resolved)),
            "target_split": plan.entry["target_split"], "episodes_requested": len(plan.episodes),
            "contact_rules": task_contact_rules(layout, shape=TASK_FOOTPRINT_SHAPE),
            "dynamic": plan.resolved.scene.dynamic is not None, "metrics": None, "backend_faults": None,
            # each simulator session's warm-up of the pool's pairs (expert/env.py), also in the next episode's record
            "session_warmups": []}

    def write(final: bool = False) -> None:
        if out_path is None:
            return
        destination = record_destination(out_path, did_work=bool(episodes))
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_text_durably(destination, json.dumps(record, indent=1) + "\n")
        if final:
            write_text_durably(destination.with_suffix(".md"), summary_markdown(record))

    swept = sweep_orphaned_instances(sim_root)  # a crashed earlier run's simulators; never a live run's
    if swept:
        print(f"swept orphaned instances before starting: {swept}", file=sys.stderr)
    route_client_log(instance_dir(instance, sim_root) / "client.log")
    status = "interrupted"  # until the loop finishes or a policy breaks: a run stopped from outside never reads "ok"
    backend_failed = False
    try:
        write()
        for plan in ready:
            scene = record["scenes"][plan.scene_id]
            scene["status"] = "running"
            env = None
            config = scene_config(plan.resolved)
            try:
                factory = (sim_factory(plan.resolved, config) if sim_factory is not None else
                           scene_config_factory(config, plan.resolved.movable_objects, run_root=sim_root))
                env = build_env(plan.resolved, instance=instance, sim_factory=factory, sim_root=sim_root,
                                obs_config=obs_config, target_pool=pool, target_split=plan.entry["target_split"],
                                max_steps=max_steps)
                success_radius_m = reward_config_for_scene(plan.resolved.scene).success_radius_m
                flown: list[dict] = []
                for episode in plan.episodes:
                    result = fly_episode(policy, env, episode, a0=condition.a0, privileged_state=privileged_state,
                                         max_steps=max_steps, layout=plan.resolved.layout,
                                         success_radius_m=success_radius_m, max_fault_retries=max_fault_retries,
                                         policy_name=name)
                    warmups = env.unwrapped.session_warmups[len(scene["session_warmups"]):]
                    if warmups:  # completed in this episode's reset, or in an attempt a fault cut short
                        scene["session_warmups"].extend(warmups)
                        result["session_warmup"] = warmups[-1]
                    flown.append(result)
                    episodes.append(result)
                    scene["metrics"], record["overall"] = summarise(flown), summarise(episodes)
                    print(f"EVAL {episode.scene} seed {episode.seed}: {_episode_line(result)} ({len(episodes)} flown)",
                          file=sys.stderr, flush=True)
                    write()
                scene["status"] = "ok"
            except PolicyError as err:
                scene["status"], scene["error"] = "failed", str(err)
                raise
            except Exception as err:  # the backend could not fly this scene: the next may still fly
                scene["status"], scene["error"] = "failed", f"{type(err).__name__}: {err}"
                backend_failed = True
                print(f"EVAL scene {plan.scene_id} failed: {scene['error']}", file=sys.stderr)
                traceback.print_exc()
            finally:
                try:
                    if env is not None:
                        scene["backend_faults"] = env.get_fault_summary()
                        teardown(env, [instance], sim_root)  # bounded close, then this slot only
                    else:
                        stop_instances([instance], sim_root)
                except Exception as err:
                    print(f"WARNING: teardown of scene {plan.scene_id} raised {type(err).__name__}: {err}", file=sys.stderr)
                write()
        status = "partial" if backend_failed else "ok"
    except PolicyError as err:
        status = "failed"
        record["error"] = str(err)
        print(f"EVAL stopped: {err}", file=sys.stderr)
    finally:
        record["status"] = status
        try:
            engine = audit_engine_faults(since=run_started, since_epoch=run_start_epoch, xid_before=xid_before,
                                         boot_before=boot_before, log_dirs=[instance_dir(instance, sim_root)])
            record["faults_ok"] = engine.pop("ok")
            record["engine_faults"] = engine
        except Exception as err:
            record["faults_ok"], record["engine_faults"] = False, {"error": f"{type(err).__name__}: {err}"}
        record["run_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write(final=True)
    return record


# --------------------------------------------------------------------------------------------------------
# The markdown summary.
# --------------------------------------------------------------------------------------------------------
def _rate(summary: dict, key: str, interval: bool = True) -> str:
    rate = summary[key]
    if rate["value"] is None:
        return ""
    if not interval or rate["wilson95"] is None:
        return f"{rate['value']:.3f}"
    lo, hi = rate["wilson95"]
    return f"{rate['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def _per(summary: dict) -> str:
    per = summary["per"]
    if per["lower"] is None:
        return ""
    return f"{per['lower']:.3f}-{per['upper']:.3f} (n={per['episodes']})"


def summary_markdown(record: dict) -> str:
    """A short human summary of a record: what was flown under which conditions, and the table of metrics."""
    condition, pool = record["condition"], record["target_pool"]
    faults = sum(e.get("retries", 0) for e in record["episodes"])
    lines = [
        f"# Evaluation: {record['policy'].get('name')} on {condition['name']}",
        "",
        f"- Policy: `{json.dumps(record['policy'], sort_keys=True)}`",
        f"- Condition file: {condition['file']} (sha256 {str(condition['sha256'])[:12]}), "
        f"{condition['episodes_per_scene']} episodes per scene"
        + (f", at most {record['max_episodes_per_scene']} flown" if record.get("max_episodes_per_scene") else ""),
        f"- a0: {record['a0']}; step limit {record['max_steps']}; mesh scenes' contact rule: the {record['contact_rule_shape']}",
        f"- State: state[0] and state[1] (the target's true distance and bearing) "
        f"{'given' if record['privileged_state'] else 'zeroed'}",
        f"- Target pool: {pool['path']} (sha256 {str(pool['sha256'])[:12]})",
        f"- Status: {record['status']}" + (f" ({record['error']})" if record.get("error") else "")
        + f"; {record['run_started']} to {record['run_finished']}; engine faults ok: {record.get('faults_ok')}; "
        f"attempts replayed after backend faults (never scored): {faults}",
        "",
        "| Scene | Status | Episodes | SR [95% CI] | CR [95% CI] | Out of bounds | Timeout | PER over successes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scene_id, scene in record["scenes"].items():
        metrics = scene.get("metrics")
        if metrics is None:
            note = scene.get("reason") or scene.get("error") or ""
            lines.append(f"| {scene_id} | {scene['status']} | 0 | {note} | | | | |")
            continue
        lines.append(f"| {scene_id} | {scene['status']} | {metrics['episodes']} | {_rate(metrics, 'success_rate')} | "
                     f"{_rate(metrics, 'collision_rate')} | {_rate(metrics, 'out_of_bounds_rate', False)} | "
                     f"{_rate(metrics, 'timeout_rate', False)} | {_per(metrics)} |")
    overall = record["overall"]
    if overall["episodes"]:
        lines.append(f"| all | {record['status']} | {overall['episodes']} | {_rate(overall, 'success_rate')} | "
                     f"{_rate(overall, 'collision_rate')} | {_rate(overall, 'out_of_bounds_rate', False)} | "
                     f"{_rate(overall, 'timeout_rate', False)} | {_per(overall)} |")
    lines += ["", f"PER is L_opt / max(L, L_opt) over successful episodes, L the flown horizontal path. L_opt lies between "
                  f"the straight line to the success disc (the lower PER) and the shortest path on the "
                  f"{record['l_opt']['resolution_m']} m grid inflated by {record['l_opt']['inflate_m']} m (the upper).", ""]
    return "\n".join(lines)
