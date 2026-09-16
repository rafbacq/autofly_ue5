import numpy as np
import torch


def space():
    from gymnasium import spaces

    from autofly_ue5.expert.obs import DEPTH_SIZE, VECTOR_DIM

    return spaces.Dict({
        "depth": spaces.Box(0.0, 1.0, (1, DEPTH_SIZE, DEPTH_SIZE), np.float32),
        "vector": spaces.Box(-np.inf, np.inf, (VECTOR_DIM,), np.float32),
    })


def test_output_shape_and_dtype():
    from autofly_ue5.expert.features import DepthVectorExtractor

    ex = DepthVectorExtractor(space(), features_dim=256)
    out = ex({"depth": torch.zeros(4, 1, 84, 84), "vector": torch.zeros(4, 8)})
    assert out.shape == (4, 256) and out.dtype == torch.float32


def test_gradients_reach_both_branches():
    # A vector-only or depth-only extractor would silently train a blind or a target-less agent.
    from autofly_ue5.expert.features import DepthVectorExtractor

    ex = DepthVectorExtractor(space(), features_dim=256)
    obs = {"depth": torch.rand(2, 1, 84, 84, requires_grad=True),
           "vector": torch.rand(2, 8, requires_grad=True)}
    ex(obs).sum().backward()
    assert obs["depth"].grad is not None and obs["depth"].grad.abs().sum() > 0
    assert obs["vector"].grad is not None and obs["vector"].grad.abs().sum() > 0


def test_it_plugs_into_sac_and_takes_one_gradient_step():
    from stable_baselines3 import SAC

    from autofly_ue5.expert.features import POLICY_KWARGS
    from tests.test_expert_env import make_env

    model = SAC("MultiInputPolicy", make_env(), policy_kwargs=POLICY_KWARGS,
                buffer_size=500, learning_starts=10, batch_size=8, device="cpu", verbose=0)
    model.learn(total_timesteps=40)
    assert model.num_timesteps >= 40


def test_target_entropy_matches_the_spec():
    # spec 8: automatic entropy tuning with target entropy -3 (one per action dimension).
    from stable_baselines3 import SAC

    from autofly_ue5.expert.features import POLICY_KWARGS
    from tests.test_expert_env import make_env

    model = SAC("MultiInputPolicy", make_env(), policy_kwargs=POLICY_KWARGS,
                buffer_size=100, learning_starts=10, device="cpu")
    assert float(model.target_entropy) == -3.0
