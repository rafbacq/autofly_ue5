"""Warm-start a new SAC from an earlier checkpoint whose observation lacked some inputs (2026-10-03).

s01d_r1's best_model scored 0.775 on the s01d gate. s01d_r2 adds the privileged mover input (`obs.encode_movers`),
which changes the observation space, so the checkpoint cannot simply be resumed. Instead every parameter is copied
across. The only shape that changes is the input width of each features extractor's head, which also takes the new
mover branch: its old columns are copied and the new ones start at zero. So at step 0 the new network acts and values
exactly like its source, and learns to use the movers from there (tests/test_warm_start.py). The entropy coefficient
comes across too.

SB3 takes uniform random actions until `learning_starts`; for a warm-started model that would fill the new replay
buffer with random flights. `PolicyWarmupSAC` acts with the policy instead (stochastically, as SAC does after
warm-up). Updates still wait for `learning_starts`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from stable_baselines3 import SAC

# The one parameter whose shape a new input changes: each features extractor's head (actor, critic, critic_target).
WIDENED_SUFFIX = "features_extractor.head.0.weight"
NEW_BRANCH_MARKER = ".mover_mlp."


class PolicyWarmupSAC(SAC):
    """SAC whose warm-up steps act with its (warm-started) policy, not uniform random actions."""

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)


def warm_start(model: SAC, source: Path) -> dict:
    """Copy `source`'s parameters into `model` in place, widening the extractors' heads with zero columns for a new
    input branch. Refuses (ValueError) any other difference: another network, depth stack or action space."""
    from stable_baselines3.common.save_util import load_from_zip_file

    _data, params, variables = load_from_zip_file(source, device=model.device)
    old = params["policy"]
    new = model.policy.state_dict()
    copied, widened, fresh = 0, [], []
    for name, tensor in new.items():
        before = old.get(name)
        if before is not None and before.shape == tensor.shape:
            new[name] = before.clone()
            copied += 1
        elif (before is not None and name.endswith(WIDENED_SUFFIX) and before.dim() == 2
              and before.shape[0] == tensor.shape[0] and before.shape[1] < tensor.shape[1]):
            # the new branch's features are concatenated last (DepthVectorExtractor.forward), so its columns are last
            widen = torch.zeros_like(tensor)
            widen[:, :before.shape[1]] = before
            new[name] = widen
            widened.append(name)
        elif before is None and NEW_BRANCH_MARKER in name:
            fresh.append(name)
        else:
            raise ValueError(f"cannot warm-start {name}: the source has "
                             f"{tuple(before.shape) if before is not None else 'no such parameter'}, the new model "
                             f"{tuple(tensor.shape)}")
    unused = sorted(set(old) - set(new))
    if unused:
        raise ValueError(f"the source has parameters the new model lacks: {unused}")
    model.policy.load_state_dict(new)
    if "log_ent_coef" not in variables or getattr(model, "log_ent_coef", None) is None:
        raise ValueError("warm start expects SAC's automatically tuned entropy coefficient in both models")
    with torch.no_grad():
        model.log_ent_coef.copy_(variables["log_ent_coef"].to(model.log_ent_coef.device))
    return {"copied": copied, "widened": widened, "fresh": fresh, "log_ent_coef": float(model.log_ent_coef)}
