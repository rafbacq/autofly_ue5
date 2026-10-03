"""SAC feature extractor (spec §8): strided CNN on depth + MLP on the target vector.

Depth and vector are fused into one `features_dim`-wide embedding. Within SB3's SAC, the twin Q-heads of
one critic network share a single `DepthVectorExtractor` instance -- that pair genuinely is "shared", per
spec §8 -- but the actor network and the target critic network each get their own separate instance:
`POLICY_KWARGS` deliberately leaves `share_features_extractor` at SB3's default (`False`), which shares an
extractor only *within* a network, never *between* the actor and the critic. Sharing one extractor between
actor and critic would couple their gradients -- the actor's loss would backprop into the critic's
features -- a known instability source for off-policy algorithms like SAC, which is exactly why SB3
defaults it off. The cost of not sharing is one extra CNN forward pass per gradient step, which is
negligible next to the simulator's own throughput (on the order of 7 env steps/s dominates by orders of
magnitude). The CNN geometry below is the Nature-DQN stack, which is what fixes `DEPTH_SIZE` at 84 in
`obs.py`: those strides only land on an exact 7x7 output for an 84x84 input.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class DepthVectorExtractor(BaseFeaturesExtractor):
    """Strided CNN on `depth`, small MLP on `vector`, concatenated and projected to `features_dim`. With a `movers`
    key (a dynamic scene's privileged moving-pillar input, `obs.encode_movers`) a second small MLP joins the
    concatenation. Without one the module is exactly what every earlier checkpoint holds, so they all load unchanged."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, vector_dim: int = 64,
                 mover_dim: int = 64):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space["depth"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),     # 84 -> 20
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),    # 20 -> 9
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),    # 9 -> 7
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.mlp = nn.Sequential(nn.Linear(observation_space["vector"].shape[0], vector_dim), nn.ReLU())
        self.has_movers = "movers" in observation_space.spaces
        if self.has_movers:
            self.mover_mlp = nn.Sequential(nn.Linear(observation_space["movers"].shape[0], mover_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(n_flat + vector_dim + (mover_dim if self.has_movers else 0), features_dim),
                                  nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.cnn(observations["depth"]), self.mlp(observations["vector"])]
        if self.has_movers:
            parts.append(self.mover_mlp(observations["movers"]))
        return self.head(torch.cat(parts, dim=1))


POLICY_KWARGS = {
    "features_extractor_class": DepthVectorExtractor,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [256, 256],
}
