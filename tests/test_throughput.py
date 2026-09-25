"""C3 (2026-09-24 review): throughput is measured the way training runs, and training uses what it measured."""

import pytest

from autofly_ue5.sim.fake import FakeSimulator
from tests.test_expert_episode import scene_and_layout


def test_gradient_steps_follow_the_number_of_collected_transitions():
    # gradient_steps=1 with n envs did ONE update per n transitions; -1 keeps one update per transition at any n.
    from stable_baselines3.common.vec_env import DummyVecEnv

    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.train import build_model

    scene, layout = scene_and_layout()
    make = lambda base: (lambda: AutoFlyEnv(scene, layout, FakeSimulator, map_path="/x", instance=0, seed_base=base))
    vec = DummyVecEnv([make(1_000_000), make(2_000_000)])
    model = build_model(vec, device="cpu", buffer_size=200, learning_starts=4, batch_size=4, seed=0, verbose=0)
    assert model.gradient_steps == -1
    model.learn(total_timesteps=20)
    # 10 vec steps of 2 transitions; training starts once num_timesteps > 4, i.e. on the vec steps ending at 6..20.
    assert model._n_updates == 16


def test_a_scene_config_factory_and_record():
    from autofly_ue5.sim.airsim_backend import scene_config_factory, scene_config_record

    sim = scene_config_factory("scene_autofly_s01_fast.jsonc")()
    assert sim._scene_config == "scene_autofly_s01_fast.jsonc"
    record = scene_config_record("scene_autofly_s01_fast.jsonc")
    assert record["file"] == "scene_autofly_s01_fast.jsonc" and len(record["sha256"]) == 64


@pytest.mark.parametrize("module", ["autofly_ue5.expert.train", "scripts.m2_gate"])
def test_training_and_the_gate_take_a_scene_config(module):
    import importlib

    parser = importlib.import_module(module).build_arg_parser()
    assert parser.parse_args(["--scene-config", "scene_autofly_s01_fast.jsonc"]).scene_config == "scene_autofly_s01_fast.jsonc"
    assert parser.parse_args([]).scene_config is None  # None = the scene's own scene_autofly_<scene>.jsonc


@pytest.mark.parametrize("n", [1, 2])
def test_measure_n_runs_resilient_workers_and_reports_their_faults(tmp_path, n):
    # The recorded n=2 measurement died on an unwrapped CameraPoseError hang, so its chosen_n=1 was a crash, not a
    # measurement. Workers now come from make_vec_env: resilient, staggered, bounded.
    from scripts.measure_instances import measure_n

    scene, layout = scene_and_layout()
    record = measure_n(n, scene, layout, warmup_s=0.2, timed_s=0.5, sim_factory=FakeSimulator,
                       vram_reader=lambda: (1000, 24000), sim_root=tmp_path / "sim", monitor_dir=tmp_path / "mon")
    assert record["env_steps_per_s_total"] > 0 and record["total_env_steps"] > 0
    assert record["backend_faults"]["relaunch_count"] == 0
    assert set(record["backend_faults"]["fault_counts"]) >= {"CameraPoseError", "Timeout"}
    assert len(record["per_instance_launch_s"]) == n
