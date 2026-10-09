"""The episode collector (spec §9; Plan 4 task B3): the expert flies, AutoFly's record is written.

For each seed from the collection range (`autofly_ue5.expert.seeds.COLLECTION_SEED_BASE`):

1. reset with a0, AutoFly's coarse guidance, applied before recording: the start yaw is set to the target's
   8-sector bearing (docs/decisions/2026-10-03-m3-a0-and-collection.md);
2. the expert flies, stochastically by default (spec §9 step 5; the M2 closeout). Each record holds the front RGB
   frame and state[9] the expert acted on, the action it was commanded (clipped as the env clips it), and the
   simulator time;
3. a successful episode goes to the store; any other goes to the rejects with its outcome as the reason (step 6).

A backend fault ends the attempt on its last real observation (`ResilientAutoFlyEnv`); its records are dropped and
the same seed is flown again, as the gate does (`autofly_ue5.expert.evaluate`). So a stored episode is always one
uninterrupted flight.

The target is named one of two ways (plan 5 C5). With a target pool in the env (`scripts/collect_dataset.py --targets`)
each episode's drawn entry names it: the sampler wrote the name into the instruction, and the provenance carries the
target's and every distractor's entry. Without one, M3's way: every episode is the orange cylinder under `target_name`.

With a pool, every episode, kept or rejected, also records whether its first recorded frame shows its target
(`first_frame_visibility`, validate/visibility.py: plan 6 revision 1's proof that the env's first-use hold suffices).
The summary counts each status, and every `missing` is said on stderr as it happens.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from autofly_ue5.dataset.raw import RawDatasetWriter
from autofly_ue5.dataset.state import autofly_state
from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES, EpisodeSetup, spawn_spec
from autofly_ue5.scenes.model import SceneFile
from autofly_ue5.sim.types import Observation
from autofly_ue5.validate.visibility import STATUSES as FIRST_FRAME_STATUSES
from autofly_ue5.validate.visibility import FirstFrameVisibility

DEFAULT_MAX_STEPS = 1_000  # far above AutoFlyEnv's 300-step limit: a guard, not a target
DEFAULT_MAX_FAULT_RETRIES = 20
POOL_NAME_STATUS = "pool"  # a name drawn from the target pool (plan 5 C5), not M3's placeholder


class EpisodeFault(Exception):
    """A backend fault cut the attempt short (never a policy outcome)."""


def dataset_instruction(scene: SceneFile, setup_instruction: str, target_name: str) -> str:
    """The episode's instruction with the target's real name. sample_setup drew the template (spec §9 step 3) but
    filled in Plan 2's placeholder "target" (D1); the same template gets the name the dataset uses."""
    for template in INSTRUCTION_TEMPLATES:
        if template.format(target="target", obstacle=scene.instruction_obstacle) == setup_instruction:
            return template.format(target=target_name, obstacle=scene.instruction_obstacle)
    raise ValueError(f"{setup_instruction!r} was not made from any instruction template")


def record_inputs(raw: Observation, setup: EpisodeSetup) -> tuple[np.ndarray, np.ndarray]:
    """The front RGB frame (a copy) and state[9] a record holds for the observation `raw` of the episode `setup`: what
    the dataset stores before each action, and what the evaluation harness hands a policy (autofly_ue5/evaluation), so
    a policy is scored on exactly the inputs it was trained on."""
    return np.array(raw.rgb, copy=True), autofly_state(raw, setup.start, setup.target_xy_z)


def _fly(model, env, seed: int, *, deterministic: bool, max_steps: int,
         visibility: FirstFrameVisibility | None = None) -> dict[str, Any]:
    obs, info = env.reset(seed=seed, options={"a0": "sector8"})
    # the reset's settle hold, and whether it spawned an object new to the session; with a pool (expert/env.py)
    prewarm_steps, first_use = info.get("prewarm_steps"), info.get("first_use")
    base = env.unwrapped
    setup = base.setup
    spawned = tuple(base.spawned)
    low, high = env.action_space.low, env.action_space.high
    raw = base.last_observation
    # the frame recorded first: does it show the target? (validate/visibility.py)
    first_frame = (visibility(raw, setup, prewarm_steps=prewarm_steps, first_use=first_use)
                   if visibility is not None else None)
    frames, states, actions, times, poses, movers = [], [], [], [], [], []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        command = np.clip(np.asarray(action, dtype=np.float32).ravel(), low, high)  # what AutoFlyEnv.step() flies
        rgb, state = record_inputs(raw, setup)
        frames.append(rgb)
        states.append(state)
        actions.append(command)
        times.append(raw.sim_time_ns)
        poses.append([raw.pose.x, raw.pose.y, raw.pose.z, raw.pose.yaw])
        movers.append(info.get("movers", []))
        obs, _reward, terminated, truncated, info = env.step(action)
        if info.get("sim_fault"):
            raise EpisodeFault(info["sim_fault"])
        raw = base.last_observation
        if terminated or truncated:
            return {"setup": setup, "spawned": spawned, "outcome": info["outcome"], "info": info, "frames": frames, "states": states,
                    "actions": actions, "times": times, "poses": poses, "movers": movers,
                    "final_pose": [raw.pose.x, raw.pose.y, raw.pose.z, raw.pose.yaw], "prewarm_steps": prewarm_steps,
                    "first_use": first_use, "first_frame": first_frame}
    return {"setup": setup, "spawned": spawned, "outcome": "exceeded_max_steps", "info": info, "frames": frames, "states": states,
            "actions": actions, "times": times, "poses": poses, "movers": movers, "final_pose": poses[-1],
            "prewarm_steps": prewarm_steps, "first_use": first_use, "first_frame": first_frame}


def _pool_object(entry, spawned: dict) -> dict:
    """A pool object's record: its entry (name, category, seen, asset, scale, material, status, and stand_in: a low-poly
    stand-in, plan 6 step 2b), the material the runtime applied (the spawner leaves the mesh's own where it cannot apply
    the entry's) and the height it stood at."""
    return {**entry.to_json(), "material_applied": spawned["material"] or "mesh_default", "spawn_z_ned": spawned["z_ned"]}


def _provenance(scene: SceneFile, layout_sha256: str | None, seed: int, flight: dict, *, deterministic: bool,
                target_name: str, target_name_status: str | None) -> dict:
    setup = flight["setup"]
    info = flight["info"]
    spec = spawn_spec(setup)
    names = list(flight["spawned"]) + [None] * (1 + len(setup.distractors) - len(flight["spawned"]))
    if setup.target is not None:
        # xyz stays the privileged target position the expert flew to (expert/episode.py TARGET_Z_NED); spawn_z_ned is
        # where the object stood.
        target = {**_pool_object(setup.target, spec["target"]), "name_status": target_name_status,
                  "spawned_as": names[0], "xyz": list(setup.target_xy_z)}
        distractors = [{**_pool_object(entry, spawned), "spawned_as": names[1 + i], "xyz": list(d)}
                       for i, (entry, spawned, d) in enumerate(zip(setup.distractor_entries, spec["distractors"],
                                                                   setup.distractors, strict=True))]
    else:
        target = {"name": target_name, "name_status": target_name_status, "asset": spec["asset"],
                  "material": spec["target_material"], "spawned_as": names[0], "xyz": list(setup.target_xy_z),
                  "scale": list(setup.target_scale)}
        distractors = [{"asset": spec["asset"], "material": spec["distractor_material"] or "mesh_default",
                        "spawned_as": names[1 + i], "xyz": list(d), "scale": list(setup.target_scale)}
                       for i, d in enumerate(setup.distractors)]
    return {
        "scene": {"id": scene.id, "file": scene.path, "sha256": scene.sha256},
        "layout_sha256": layout_sha256,
        "seed": seed,
        "target": target,
        "distractors": distractors,
        "start_pose": [setup.start.x, setup.start.y, setup.start.z, setup.start.yaw],
        "a0": {"mode": "sector8", "start_yaw_rad": setup.start.yaw},
        "movers": [r.to_json() for r in setup.movers],
        "mover_positions_per_record": flight["movers"],
        "poses_per_record": flight["poses"],
        "sim_time_ns": [int(t) for t in flight["times"]],
        "final_pose": flight["final_pose"],
        # With a pool: how many more hover steps the reset held before the first record (every pool episode: the render
        # lag after a spawn), and whether it spawned an object new to the simulator session (plan 6 step 2a). A pool-free
        # episode's provenance stays M3's, key for key.
        **({"prewarm_steps": flight["prewarm_steps"], "first_use": flight["first_use"]}
           if flight.get("prewarm_steps") is not None else {}),
        # With a pool: whether the first recorded frame shows the target (validate/visibility.py), kept or rejected.
        **({"first_frame_visibility": flight["first_frame"]} if flight.get("first_frame") is not None else {}),
        "termination": flight["outcome"],
        "collision_source": info.get("collision_source"),
        # A mesh scene's ending (expert/footprint_contact.py): which tree, how close, from where, how fast; or the circle
        # nearest a physical contact. Only where the env named one, so a pillar scene's provenance is as before.
        **{key: info[key] for key in ("footprint_contact", "footprint_nearest") if key in info},
        "oob_kind": info.get("oob_kind"),
        "final_distance_m": info.get("final_distance_m"),
        "deterministic": deterministic,
        "pilot": "sac_expert",
    }


def collect(model, env, *, scene: SceneFile, writer: RawDatasetWriter, seed_base: int, n_keep: int,
            target_name: str | None = None, deterministic: bool = False, max_attempts: int | None = None,
            layout_sha256: str | None = None, max_steps: int = DEFAULT_MAX_STEPS,
            max_fault_retries: int = DEFAULT_MAX_FAULT_RETRIES, progress: dict | None = None,
            target_name_status: str | None = None) -> dict:
    """Fly seeds seed_base, seed_base + 1, ... until `n_keep` episodes are kept or `max_attempts` seeds were flown.

    `target_name` names every episode's target when the env has no target pool (M3), and must be None when it has one:
    the pool's entries name each episode's own, and a second name beside them would be ignored or would contradict
    them. Refused either way before anything is flown.

    `progress` (returned, and updated after every episode) lets a caller that catches an exception still report what
    was flown: a crashed pilot's record must keep its counts (the M3 review, 2026-10-03)."""
    pool = env.unwrapped.target_pool
    if pool is None and target_name is None:
        raise ValueError("no target name: pass target_name, or give the env a target pool (plan 5 C5)")
    if pool is not None and target_name is not None:
        raise ValueError(f"the env's target pool names each episode's target; target_name={target_name!r} is the "
                         f"pool-free fallback and cannot be given beside it")
    max_attempts = max_attempts if max_attempts is not None else 10 * n_keep
    # With a pool, each first recorded frame is checked against the layout's geometry, built (a mesh scene's glTFs
    # loaded) once, before anything is flown.
    visibility = FirstFrameVisibility(env.unwrapped.layout) if pool is not None else None
    summary = progress if progress is not None else {}
    summary.update(status="running", kept=0, rejected=0, attempted=0, fault_retries=0, outcomes={},
                   seeds_flown=[seed_base, None])
    if visibility is not None:  # only with a pool: a pool-free collection's summary stays M3's
        summary["first_frame_visibility"] = {status: 0 for status in FIRST_FRAME_STATUSES}
    for seed in range(seed_base, seed_base + max_attempts):
        if summary["kept"] >= n_keep:
            break
        summary["seeds_flown"][1] = seed
        for retry in range(max_fault_retries + 1):
            try:
                flight = _fly(model, env, seed, deterministic=deterministic, max_steps=max_steps, visibility=visibility)
                break
            except EpisodeFault as fault:
                summary["fault_retries"] += 1
                print(f"COLLECT seed {seed}: {fault} cut the attempt short; flying it again ({retry + 1})", file=sys.stderr)
        else:
            raise RuntimeError(f"seed {seed}: no uninterrupted flight in {max_fault_retries} retries -- a broken backend")
        outcome = flight["outcome"]
        summary["attempted"] += 1
        summary["outcomes"][outcome] = summary["outcomes"].get(outcome, 0) + 1
        episode_id = f"{scene.id}_{seed}"
        first_frame = flight["first_frame"]
        if first_frame is not None:
            summary["first_frame_visibility"][first_frame["status"]] += 1
            if first_frame["status"] == "missing":  # the failure the hold guards against: said, never only counted
                print(f"FIRST-FRAME MISSING seed {seed}: {flight['setup'].target.name} in view and unobstructed at "
                      f"{first_frame['range_m']} m (pixel {first_frame['pixel']}), the frame's window reads "
                      f"{first_frame['depth_window_min_m']} m; prewarm_steps {first_frame['prewarm_steps']}",
                      file=sys.stderr, flush=True)
        setup = flight["setup"]
        if setup.target is not None:  # the sampler already wrote the drawn name into the instruction
            name, name_status, instruction = setup.target.name, POOL_NAME_STATUS, setup.instruction
        else:
            name, name_status = target_name, target_name_status
            instruction = dataset_instruction(scene, setup.instruction, target_name)
        provenance = _provenance(scene, layout_sha256, seed, flight, deterministic=deterministic, target_name=name,
                                 target_name_status=name_status)
        if outcome == "success":
            writer.write_episode(
                episode_id=episode_id, scene_id=scene.id, split=scene.split, seed=seed,
                instruction=instruction, target_name=name,
                frames=flight["frames"], states=np.stack(flight["states"]), actions=np.stack(flight["actions"]),
                sim_time_ns=np.asarray(flight["times"], dtype=np.int64), provenance=provenance)
            summary["kept"] += 1
        else:
            writer.write_reject(episode_id=episode_id, reason=outcome, provenance=provenance)
            summary["rejected"] += 1
        print(f"COLLECT seed {seed}: {outcome} after {len(flight['actions'])} records -- kept {summary['kept']}/{n_keep}, "
              f"rejected {summary['rejected']}", file=sys.stderr, flush=True)
    summary["status"] = "ok" if summary["kept"] >= n_keep else "incomplete"
    return summary
