"""SAC feature extractor (spec §8): strided CNN on depth + MLP on the target vector, shared by actor and critics.

Depth and vector are fused into one `features_dim`-wide embedding so SAC's actor and twin critics all read
the same representation, rather than each learning their own -- the standard SB3 `BaseFeaturesExtractor`
pattern. The CNN geometry below is the Nature-DQN stack, which is what fixes `DEPTH_SIZE` at 84 in
`obs.py`: those strides only land on an exact 7x7 output for an 84x84 input.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class DepthVectorExtractor(BaseFeaturesExtractor):
    """Strided CNN on `depth`, small MLP on `vector`, concatenated and projected to `features_dim`."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, vector_dim: int = 64):
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
        self.head = nn.Sequential(nn.Linear(n_flat + vector_dim, features_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(torch.cat([self.cnn(observations["depth"]), self.mlp(observations["vector"])], dim=1))


POLICY_KWARGS = {
    "features_extractor_class": DepthVectorExtractor,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [256, 256],
}
