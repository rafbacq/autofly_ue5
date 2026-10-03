"""Warm-start a new SAC from an earlier checkpoint whose observation lacked some inputs (2026-10-03).

s01d_r1's best_model scored 0.775 on the s01d gate. s01d_r2 adds the privileged mover input (`obs.encode_movers`),
which changes the observation space, so the checkpoint cannot simply be resumed. Instead every parameter is copied
across. The only shape that changes is the input width of each features extractor's head, which also takes the new
mover branch: its old columns are copied and the new ones start at zero. So at step 0 the new network acts and values
exactly like its source (tests/test_warm_start.py). The entropy coefficient comes across too.

The optimizers come across as well, by parameter name. A fresh Adam's first updates move every weight by the full
learning rate, which the 2026-10-03 review measured at 10-180x the steady-state step of s01d_r1's own optimizer (median
|m|/sqrt(v) 0.023 for the actor, 0.0056 for the critic). On a trained network that could undo the head start within
minutes. With the moments copied, the first updates to every shared weight are exactly the source's.
- The widened head's moments get zeros for the new columns' mean, and the old columns' average second moment, so the
  new columns start at a typical step size.
- The new branch's own parameters start with no state, as any new parameter does.

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
    """SAC whose warm-up steps act with its (warm-started) policy, not uniform random actions, and whose actor and
    entropy coefficient may sit out the first `actor_freeze_updates` gradient updates (a critic warm-up).

    Run 2 (2026-10-03) without one: within ~500 updates of a warm start, training success fell from ~0.8 to ~0.45 and
    the entropy coefficient doubled. The actor was chasing a critic that was re-learning a changed reward on a small,
    narrow buffer. With the actor frozen, r1's policy keeps flying (and filling the buffer) while the critic adapts.
    The freeze sets the two learning rates to zero, so Adam keeps tracking their gradients and the actor resumes with
    current moments."""

    actor_freeze_updates: int = 0

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)

    def _update_learning_rate(self, optimizers) -> None:
        super()._update_learning_rate(optimizers)
        if self._n_updates < self.actor_freeze_updates:
            for optimizer in (self.actor.optimizer, self.ent_coef_optimizer):
                if optimizer is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = 0.0


def plan(new_shapes: dict[str, tuple], old: dict[str, torch.Tensor]) -> dict[str, str]:
    """How each new policy parameter is filled from the source: "copy", "widen" or "fresh". Raises ValueError for
    anything else (another network, depth stack, slot count or action space), naming the parameter."""
    actions = {}
    for name, shape in new_shapes.items():
        before = old.get(name)
        if before is not None and tuple(before.shape) == tuple(shape):
            actions[name] = "copy"
        elif (before is not None and name.endswith(WIDENED_SUFFIX) and before.dim() == 2 and len(shape) == 2
              and before.shape[0] == shape[0] and before.shape[1] < shape[1]):
            actions[name] = "widen"  # the new branch's features are concatenated last (DepthVectorExtractor.forward)
        elif before is None and NEW_BRANCH_MARKER in name:
            actions[name] = "fresh"
        else:
            raise ValueError(f"cannot warm-start {name}: the source has "
                             f"{tuple(before.shape) if before is not None else 'no such parameter'}, the new model "
                             f"{tuple(shape)}")
    unused = sorted(set(old) - set(new_shapes))
    if unused:
        raise ValueError(f"the source has parameters the new model lacks: {unused}")
    return actions


def _widen(before: torch.Tensor, shape) -> torch.Tensor:
    out = torch.zeros(shape, dtype=before.dtype, device=before.device)
    out[:, :before.shape[1]] = before
    return out


def _transfer_optimizer(optimizer: torch.optim.Optimizer, saved: dict, old_names: list[str], new_names: list[str],
                        actions: dict[str, str], prefix: str) -> None:
    """Load `saved` (the source optimizer's state_dict, indexed in `old_names` order) into `optimizer` (indexed in
    `new_names` order), by parameter name."""
    old_index = {name: i for i, name in enumerate(old_names)}
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if len(params) != len(new_names):
        raise ValueError(f"{prefix} optimizer holds {len(params)} parameters, the network {len(new_names)}")
    state = {}
    for j, (name, param) in enumerate(zip(new_names, params)):
        action = actions[f"{prefix}.{name}"]
        if action == "fresh" or old_index.get(name) not in saved["state"]:
            continue
        before = saved["state"][old_index[name]]
        entry = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in before.items()}
        if action == "widen":
            entry["exp_avg"] = _widen(before["exp_avg"], param.shape)
            sq = _widen(before["exp_avg_sq"], param.shape)
            sq[:, before["exp_avg_sq"].shape[1]:] = before["exp_avg_sq"].mean()
            entry["exp_avg_sq"] = sq
        state[j] = entry
    current = optimizer.state_dict()
    optimizer.load_state_dict({"state": state, "param_groups": current["param_groups"]})


def warm_start(model: SAC, source: Path) -> dict:
    """Copy `source`'s parameters and optimizer state into `model` in place, widening the extractors' heads with zero
    columns for a new input branch. Refuses (ValueError) any other difference."""
    from stable_baselines3.common.save_util import load_from_zip_file

    _data, params, variables = load_from_zip_file(source, device=model.device)
    old = params["policy"]
    new = model.policy.state_dict()
    actions = plan({name: tuple(t.shape) for name, t in new.items()}, old)
    for name, action in actions.items():
        if action == "copy":
            new[name] = old[name].clone()
        elif action == "widen":
            new[name] = _widen(old[name], new[name].shape)
    model.policy.load_state_dict(new)
    if "log_ent_coef" not in variables or getattr(model, "log_ent_coef", None) is None:
        raise ValueError("warm start expects SAC's automatically tuned entropy coefficient in both models")
    with torch.no_grad():
        model.log_ent_coef.copy_(variables["log_ent_coef"].to(model.log_ent_coef.device))

    optimizers = {}
    for net in ("actor", "critic"):
        saved = params.get(f"{net}.optimizer")
        if saved is None:
            optimizers[net] = "fresh (the source saved none)"
            continue
        old_names = [k[len(net) + 1:] for k in old if k.startswith(f"{net}.")]
        new_names = [name for name, _ in getattr(model, net).named_parameters()]
        _transfer_optimizer(getattr(model, net).optimizer, saved, old_names, new_names, actions, net)
        optimizers[net] = "copied"
    saved = params.get("ent_coef_optimizer")
    if saved is not None and getattr(model, "ent_coef_optimizer", None) is not None:
        model.ent_coef_optimizer.load_state_dict({"state": saved["state"],
                                                  "param_groups": model.ent_coef_optimizer.state_dict()["param_groups"]})
        optimizers["ent_coef"] = "copied"
    else:
        optimizers["ent_coef"] = "fresh (the source saved none)"
    return {"copied": sum(a == "copy" for a in actions.values()),
            "widened": [n for n, a in actions.items() if a == "widen"],
            "fresh": [n for n, a in actions.items() if a == "fresh"],
            "log_ent_coef": float(model.log_ent_coef.detach()), "optimizers": optimizers}
