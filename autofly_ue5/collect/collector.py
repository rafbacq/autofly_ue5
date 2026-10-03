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
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from autofly_ue5.dataset.raw import RawDatasetWriter
from autofly_ue5.dataset.state import autofly_state
from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES
from autofly_ue5.scenes.model import SceneFile

DEFAULT_MAX_STEPS = 1_000  # far above AutoFlyEnv's 300-step limit: a guard, not a target
DEFAULT_MAX_FAULT_RETRIES = 20


class EpisodeFault(Exception):
    """A backend fault cut the attempt short (never a policy outcome)."""


def dataset_instruction(scene: SceneFile, setup_instruction: str, target_name: str) -> str:
    """The episode's instruction with the target's real name. sample_setup drew the template (spec §9 step 3) but
    filled in Plan 2's placeholder "target" (D1); the same template gets the name the dataset uses."""
    for template in INSTRUCTION_TEMPLATES:
        if template.format(target="target", obstacle=scene.instruction_obstacle) == setup_instruction:
            return template.format(target=target_name, obstacle=scene.instruction_obstacle)
    raise ValueError(f"{setup_instruction!r} was not made from any instruction template")


def _fly(model, env, seed: int, *, deterministic: bool, max_steps: int) -> dict[str, Any]:
    obs, info = env.reset(seed=seed, options={"a0": "sector8"})
    base = env.unwrapped
    setup = base.setup
    low, high = env.action_space.low, env.action_space.high
    raw = base.last_observation
    frames, states, actions, times, poses, movers = [], [], [], [], [], []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        command = np.clip(np.asarray(action, dtype=np.float32).ravel(), low, high)  # what AutoFlyEnv.step() flies
        frames.append(np.array(raw.rgb, copy=True))
        states.append(autofly_state(raw, setup.start, setup.target_xy_z))
        actions.append(command)
        times.append(raw.sim_time_ns)
        poses.append([raw.pose.x, raw.pose.y, raw.pose.z, raw.pose.yaw])
        movers.append(info.get("movers", []))
        obs, _reward, terminated, truncated, info = env.step(action)
        if info.get("sim_fault"):
            raise EpisodeFault(info["sim_fault"])
        raw = base.last_observation
        if terminated or truncated:
            return {"setup": setup, "outcome": info["outcome"], "info": info, "frames": frames, "states": states,
                    "actions": actions, "times": times, "poses": poses, "movers": movers,
                    "final_pose": [raw.pose.x, raw.pose.y, raw.pose.z, raw.pose.yaw]}
    return {"setup": setup, "outcome": "exceeded_max_steps", "info": info, "frames": frames, "states": states,
            "actions": actions, "times": times, "poses": poses, "movers": movers, "final_pose": poses[-1]}


def _provenance(scene: SceneFile, layout_sha256: str | None, seed: int, flight: dict, *, deterministic: bool) -> dict:
    setup = flight["setup"]
    info = flight["info"]
    return {
        "scene": {"id": scene.id, "file": scene.path, "sha256": scene.sha256},
        "layout_sha256": layout_sha256,
        "seed": seed,
        "target": {"xyz": list(setup.target_xy_z), "scale": list(setup.target_scale)},
        "distractors": [list(d) for d in setup.distractors],
        "start_pose": [setup.start.x, setup.start.y, setup.start.z, setup.start.yaw],
        "a0": {"mode": "sector8", "start_yaw_rad": setup.start.yaw},
        "movers": [r.to_json() for r in setup.movers],
        "mover_positions_per_record": flight["movers"],
        "poses_per_record": flight["poses"],
        "sim_time_ns": [int(t) for t in flight["times"]],
        "final_pose": flight["final_pose"],
        "termination": flight["outcome"],
        "collision_source": info.get("collision_source"),
        "oob_kind": info.get("oob_kind"),
        "final_distance_m": info.get("final_distance_m"),
        "deterministic": deterministic,
        "pilot": "sac_expert",
    }


def collect(model, env, *, scene: SceneFile, writer: RawDatasetWriter, seed_base: int, n_keep: int, target_name: str,
            deterministic: bool = False, max_attempts: int | None = None, layout_sha256: str | None = None,
            max_steps: int = DEFAULT_MAX_STEPS, max_fault_retries: int = DEFAULT_MAX_FAULT_RETRIES) -> dict:
    """Fly seeds seed_base, seed_base + 1, ... until `n_keep` episodes are kept or `max_attempts` seeds were flown."""
    max_attempts = max_attempts if max_attempts is not None else 10 * n_keep
    kept = rejected = fault_retries = 0
    outcomes: dict[str, int] = {}
    last_flown = None
    for seed in range(seed_base, seed_base + max_attempts):
        if kept >= n_keep:
            break
        last_flown = seed
        for retry in range(max_fault_retries + 1):
            try:
                flight = _fly(model, env, seed, deterministic=deterministic, max_steps=max_steps)
                break
            except EpisodeFault as fault:
                fault_retries += 1
                print(f"COLLECT seed {seed}: {fault} cut the attempt short; flying it again ({retry + 1})", file=sys.stderr)
        else:
            raise RuntimeError(f"seed {seed}: no uninterrupted flight in {max_fault_retries} retries -- a broken backend")
        outcome = flight["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        episode_id = f"{scene.id}_{seed}"
        provenance = _provenance(scene, layout_sha256, seed, flight, deterministic=deterministic)
        if outcome == "success":
            writer.write_episode(
                episode_id=episode_id, scene_id=scene.id, split=scene.split, seed=seed,
                instruction=dataset_instruction(scene, flight["setup"].instruction, target_name), target_name=target_name,
                frames=flight["frames"], states=np.stack(flight["states"]), actions=np.stack(flight["actions"]),
                sim_time_ns=np.asarray(flight["times"], dtype=np.int64), provenance=provenance)
            kept += 1
        else:
            writer.write_reject(episode_id=episode_id, reason=outcome, provenance=provenance)
            rejected += 1
        print(f"COLLECT seed {seed}: {outcome} after {len(flight['actions'])} records -- kept {kept}/{n_keep}, "
              f"rejected {rejected}", file=sys.stderr, flush=True)
    return {"status": "ok" if kept >= n_keep else "incomplete", "kept": kept, "rejected": rejected,
            "attempted": rejected + kept, "fault_retries": fault_retries, "outcomes": outcomes,
            "seeds_flown": [seed_base, last_flown]}
