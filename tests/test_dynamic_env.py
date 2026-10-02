"""AutoFlyEnv on a dynamic scene (s01d, spec §6.5), against FakeSimulator: movers follow their clocks, yield to the
drone, are scored by the swept-step contact rule, never leave a pillar displaced across resets or faults, and the
expert sees a three-frame float16 depth stack. Static s01 is pinned separately (tests/test_static_s01_golden.py)."""

from __future__ import annotations

import functools
import json
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.model import load_scene_file
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import CameraPoseError, KinematicsJumpError, Pose, SimRequestTimeoutError
from tests.test_expert_episode import scene_and_layout
from tests.test_scene_resolver import _dynamic_variant

ROTOR = 0.472
ZERO = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def dynamic_scene(tmp_path_factory=None):
    """s01d as a SceneFile (the committed scenes/s01d_moving_pillars.json when present) and s01's layout."""
    _static, layout = scene_and_layout()
    path = SCENES_DIR / "s01d_moving_pillars.json"
    if not path.is_file():
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "s01d_moving_pillars.json"
        path.write_text(json.dumps(_dynamic_variant()))
    return load_scene_file(path), layout


def scene_objects(layout) -> dict:
    """The built level's tagged pillars, as FakeSimulator scene objects that collide at the rotor tips."""
    return {i.tag: (Pose(i.x, i.y, i.z_center, 0.0), i.radius_m + ROTOR) for i in layout.instances}


def make_dynamic_env(sim_cls=FakeSimulator, **kw):
    from autofly_ue5.expert.env import AutoFlyEnv

    scene, layout = dynamic_scene()
    factory = functools.partial(sim_cls, scene_objects=scene_objects(layout))
    return AutoFlyEnv(scene, layout, factory, map_path="/Game/AutoFly/Maps/S01", instance=0, **kw)


class RecordingFake(FakeSimulator):
    """Records the order of the calls that matter for keeping the drone and the movers apart."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[tuple] = []

    def reset(self, pose):
        self.calls.append(("reset",))
        return super().reset(pose)

    def set_object_poses(self, poses):
        self.calls.append(("set_object_poses", dict(poses)))
        super().set_object_poses(poses)


def _home_positions(layout) -> dict:
    return {i.tag: (i.x, i.y, i.z_center) for i in layout.instances}


def _misplaced(env, layout, movers: set[str]) -> dict:
    """Pillars that are not at home although they are not this episode's movers."""
    home = _home_positions(layout)
    return {tag: (p.x, p.y, p.z) for tag, p in env._sim.scene_object_poses().items()
            if tag not in movers and (p.x, p.y, p.z) != home[tag]}


# --------------------------------------------------------------------------------------------------------
# Observation.
# --------------------------------------------------------------------------------------------------------
def test_a_dynamic_scene_stacks_three_float16_depth_frames_and_passes_the_checker():
    env = make_dynamic_env()
    depth = env.observation_space["depth"]
    assert depth.shape == (3, 84, 84) and depth.dtype == np.float16
    assert env.observation_space["vector"].dtype == np.float32
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=1_000_001)
    assert obs["depth"].dtype == np.float16 and env.observation_space.contains(obs)


def test_the_newest_frame_is_last_and_the_first_frame_fills_the_stack():
    class Countdown(FakeSimulator):
        def observe(self):
            import dataclasses

            obs = super().observe()
            return dataclasses.replace(obs, depth=np.full((32, 32), float(1 + self.steps_taken % 29), np.float32))

    env = make_dynamic_env(sim_cls=Countdown)
    obs, _ = env.reset(seed=1_000_002)
    first = obs["depth"][:, 0, 0].astype(np.float32)
    assert first[0] == first[1] == first[2]
    obs, *_ = env.step(ZERO)
    obs2, *_ = env.step(ZERO)
    stack = obs2["depth"][:, 0, 0].astype(np.float32)
    assert stack[0] == first[0] and stack[2] > stack[1] > stack[0], f"oldest to newest: {stack}"
    assert obs["depth"][2, 0, 0] == obs2["depth"][1, 0, 0]


def test_the_frame_count_can_be_overridden():
    from autofly_ue5.expert.obs import ObsConfig

    env = make_dynamic_env(obs_config=ObsConfig(depth_frames=1, depth_dtype="float32"))
    assert env.observation_space["depth"].shape == (1, 84, 84) and env.observation_space["depth"].dtype == np.float32


# --------------------------------------------------------------------------------------------------------
# Motion.
# --------------------------------------------------------------------------------------------------------
def test_movers_start_on_their_routes_and_follow_their_clocks():
    env = make_dynamic_env()
    _obs, info = env.reset(seed=1_000_003)
    routes = env._setup.movers
    assert len(routes) >= 2 and info["n_movers"] == len(routes)
    poses = env._sim.scene_object_poses()
    for route in routes:
        assert (poses[route.tag].x, poses[route.tag].y) == pytest.approx(route.position(0.0))
        assert poses[route.tag].z == route.z
    assert info["movers"] == [list(route.position(0.0)) for route in routes]
    before = {r.tag: route_xy for r, route_xy in zip(routes, info["movers"])}
    _obs, _r, _t, _tr, info = env.step(ZERO)
    for route, xy in zip(routes, info["movers"]):
        moved = math.dist(xy, before[route.tag])
        assert moved <= route.speed_m_s * 0.2 + 1e-9
        assert xy == pytest.approx(list(route.position(env._movers.taus[routes.index(route)])))
        poses = env._sim.scene_object_poses()
        assert (poses[route.tag].x, poses[route.tag].y) == pytest.approx(tuple(xy))


def test_a_hovering_drone_is_never_rammed():
    env = make_dynamic_env()
    for seed in range(1_000_010, 1_000_016):
        env.reset(seed=seed)
        for _ in range(300):
            _obs, _r, terminated, truncated, info = env.step(ZERO)
            if terminated or truncated:
                break
        assert info["outcome"] == "timeout", f"seed {seed}: a hovering drone ended in {info['outcome']}"


def _fly_at_nearest_mover(env, seed):
    """Reset, then turn toward the nearest mover and fly at it until the episode ends."""
    _obs, info = env.reset(seed=seed)
    for _ in range(300):
        pose = env._last_pose
        xy = min(info["movers"], key=lambda m: math.dist(m, (pose.x, pose.y)))
        bearing = math.atan2(xy[1] - pose.y, xy[0] - pose.x) - pose.yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        action = np.array([2.0 if abs(bearing) < 0.2 else 0.0, float(np.clip(2.0 * bearing, -1, 1)), 0.0], np.float32)
        _obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return reward, info
    raise AssertionError("never ended")


def test_flying_into_a_mover_is_a_mover_collision_seen_coming():
    env = make_dynamic_env()
    hits = 0
    for seed in range(1_000_020, 1_000_030):
        reward, info = _fly_at_nearest_mover(env, seed)
        if info["outcome"] == "collision" and info["collision_source"] == "mover":
            hits += 1
            assert info["mover_in_view"] is True, "it flew straight at it"
            assert reward < -5.0
    assert hits >= 5


# --------------------------------------------------------------------------------------------------------
# Resets never leave a pillar displaced, and keep the drone and the movers apart.
# --------------------------------------------------------------------------------------------------------
def test_displaced_pillars_are_parked_before_the_reset_and_returned_after_it():
    env = make_dynamic_env(sim_cls=RecordingFake)
    env.reset(seed=1_000_040)
    first = {r.tag for r in env._setup.movers}
    env.step(ZERO)
    env._sim.calls.clear()
    env.reset(seed=1_000_041)
    second = {r.tag for r in env._setup.movers}
    calls = env._sim.calls
    reset_at = calls.index(("reset",))
    park = calls[reset_at - 1]
    assert park[0] == "set_object_poses" and set(park[1]) == first
    assert all(p.z >= 40.0 for p in park[1].values()), "parked underground, out of the reset's path"
    batch = calls[reset_at + 1]
    assert batch[0] == "set_object_poses" and set(batch[1]) == first | second
    assert _misplaced(env, scene_and_layout()[1], second) == {}
    assert env._displaced == second


def test_a_static_scene_moves_nothing():
    from tests.test_expert_env import make_env

    env = make_env()
    env.reset(seed=3)
    assert env._displaced == set() and env._movers is None
    _obs, _r, _t, _tr, info = env.step(ZERO)
    assert info["n_movers"] == 0 and info["movers"] == [] and info["collision_source"] is None


def test_a_fault_mid_reset_leaves_no_untracked_displaced_pillar():
    class FailsTheFirstBatch(FakeSimulator):
        failed = False

        def set_object_poses(self, poses):
            if not FailsTheFirstBatch.failed and any(p.z < 40.0 for p in poses.values()) and len(poses) > 3:
                FailsTheFirstBatch.failed = True
                first_name = next(iter(poses))
                super().set_object_poses({first_name: poses[first_name]})  # one moved, then the request dies
                raise SimRequestTimeoutError("injected Fatal Timeout mid-batch")
            super().set_object_poses(poses)

    env = make_dynamic_env(sim_cls=FailsTheFirstBatch)
    with pytest.raises(SimRequestTimeoutError):
        env.reset(seed=1_000_050)
    env.reset(seed=1_000_051)  # same simulator (the fake's connection survives): must clear what the failure left
    assert _misplaced(env, scene_and_layout()[1], {r.tag for r in env._setup.movers}) == {}


def test_a_relaunched_simulator_starts_with_nothing_displaced():
    env = make_dynamic_env(sim_cls=RecordingFake)
    env.reset(seed=1_000_060)
    old = env._sim
    env._sim = None  # what ResilientAutoFlyEnv._relaunch does before the next reset
    env.reset(seed=1_000_061)
    assert env._sim is not old
    first_calls = env._sim.calls
    assert first_calls[0] == ("reset",), "a fresh level has every pillar home: nothing to park"


# --------------------------------------------------------------------------------------------------------
# Faults next to a mover.
# --------------------------------------------------------------------------------------------------------
class FaultsOnStep(FakeSimulator):
    error = CameraPoseError
    fail_at: int | None = None

    def step(self, dt=0.2):
        if FaultsOnStep.fail_at is not None and self.steps_taken + 1 == FaultsOnStep.fail_at:
            raise FaultsOnStep.error("injected")
        return super().step(dt)


@pytest.mark.parametrize("error", [CameraPoseError, KinematicsJumpError])
def test_a_fault_right_next_to_a_mover_is_scored_as_an_inferred_collision(error):
    env = make_dynamic_env(sim_cls=FaultsOnStep)
    env.reset(seed=1_000_070)
    route = env._setup.movers[0]
    x, y = route.position(env._movers.taus[0])
    # Put the drone 1.2 m from the mover's surface (inside the 1.64 m inference radius), then fail the next step.
    standoff = route.footprint.radius_m + 1.2
    from autofly_ue5.expert.obs import target_geometry

    env._sim._pose = env._last_pose = Pose(x - standoff, y, env._last_pose.z, 0.0)
    env._prev_dist = target_geometry(env._last_pose, env._setup.target_xy_z)[0]  # no progress paid for the teleport
    FaultsOnStep.error, FaultsOnStep.fail_at = error, env._sim.steps_taken + 1
    try:
        obs, reward, terminated, truncated, info = env.step(ZERO)
    finally:
        FaultsOnStep.fail_at = None
    assert terminated and not truncated and info["outcome"] == "collision"
    assert info["collision_source"] == "mover_inferred" and error.__name__ in info["inferred_from"]
    assert reward < -5.0


def test_a_fault_far_from_every_mover_is_still_a_fault():
    env = make_dynamic_env(sim_cls=FaultsOnStep)
    env.reset(seed=1_000_071)
    FaultsOnStep.error, FaultsOnStep.fail_at = CameraPoseError, env._sim.steps_taken + 1
    try:
        with pytest.raises(CameraPoseError):
            env.step(ZERO)  # at the start, >= 6 m from every mover's sweep
    finally:
        FaultsOnStep.fail_at = None


# --------------------------------------------------------------------------------------------------------
# SB3 with the float16 stack.
# --------------------------------------------------------------------------------------------------------
def test_sac_builds_steps_saves_and_reloads_on_the_float16_stack(tmp_path):
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    from autofly_ue5.expert.train import build_model

    vec = DummyVecEnv([make_dynamic_env])
    model = build_model(vec, device="cpu", buffer_size=200, learning_starts=10, batch_size=8, verbose=0)
    model.learn(total_timesteps=30)
    assert model.replay_buffer.observations["depth"].dtype == np.float16
    assert model.replay_buffer.observations["depth"].shape[1:] == (1, 3, 84, 84)
    model.save(tmp_path / "m.zip")
    loaded = SAC.load(tmp_path / "m.zip", env=vec, device="cpu")
    assert loaded.observation_space["depth"].dtype == np.float16
    action, _ = loaded.predict(vec.reset(), deterministic=True)
    assert action.shape == (1, 3)


def test_home_and_park_poses_keep_each_pillars_yaw():
    from autofly_ue5.expert.movers import home_poses, park_poses

    _scene, layout = dynamic_scene()
    inst = layout.instances[0]
    turned = {inst.tag: type(inst)(**{**inst.__dict__, "yaw": 0.4})}
    assert home_poses(turned, [inst.tag])[inst.tag].yaw == 0.4
    assert park_poses(turned, [inst.tag])[inst.tag].yaw == 0.4
