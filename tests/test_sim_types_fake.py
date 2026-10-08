import math

import numpy as np
import pytest

from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.types import ObjectNotFoundError, Pose, dt_to_ns


def test_dt_to_ns():
    assert dt_to_ns(0.2) == 200_000_000
    with pytest.raises(ValueError):
        dt_to_ns(0.203)
    with pytest.raises(ValueError):
        dt_to_ns(0.0)


def test_fake_implements_protocol():
    assert isinstance(FakeSimulator(), Simulator)


def test_step_requires_launch_and_command():
    sim = FakeSimulator()
    with pytest.raises(RuntimeError, match="not launched"):
        sim.step()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    with pytest.raises(RuntimeError, match="command_velocity"):
        sim.step()


def test_step_and_observe_require_reset_first():
    # Mirrors ProjectAirSimSimulator's reset-before-step precondition (spec §7.1) so an (M2/M3) caller
    # bug that only breaks against the real backend does not pass silently against the fake.
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    with pytest.raises(RuntimeError, match="before reset"):
        sim.observe()
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="before reset"):
        sim.step()
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.command_velocity(0.0, 0.0, 0.0)
    sim.step()  # no longer raises once reset() has run


def test_reset_advances_the_clock_like_the_backend():
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    obs = sim.reset(Pose(1.0, 2.0, -2.0, 0.5))
    assert sim.steps_taken == 4 and obs.sim_time_ns == 800_000_000
    assert obs.rgb_time_ns == obs.depth_time_ns == obs.kinematics_time_ns == 800_000_000
    assert obs.pose == Pose(1.0, 2.0, -2.0, 0.5) and obs.velocity_ned == (0.0, 0.0, 0.0) and obs.camera_pose_error_m == 0.0


def test_forward_yaw_and_vertical_integration():
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.command_velocity(2.0, 0.0, 0.0)
    assert sim.step(0.2) == 1_000_000_000
    obs = sim.observe()
    assert obs.pose.x == pytest.approx(0.4) and obs.sim_time_ns == 1_000_000_000 and sim.steps_taken == 5
    assert obs.rgb_time_ns == obs.depth_time_ns == obs.kinematics_time_ns == 1_000_000_000
    sim.reset(Pose(0.0, 0.0, -2.0, math.pi / 2))
    sim.command_velocity(1.0, 0.0, 0.5)
    sim.step()
    obs = sim.observe()
    assert obs.pose.y == pytest.approx(0.2) and obs.pose.x == pytest.approx(0.0, abs=1e-9)
    assert obs.pose.z == pytest.approx(-2.1)
    sim.command_velocity(0.0, 1.0, 0.0)
    sim.step()
    assert sim.observe().pose.yaw == pytest.approx(math.pi / 2 + 0.2)


def test_collision_sets_flag_until_reset():
    sim = FakeSimulator(obstacles=[(1.0, 0.0, 0.5)])
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    assert sim.observe().collided is False
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    obs = sim.observe()
    assert obs.collided is True and obs.step_collisions[0].object_name == "obstacle_0"
    assert sim.reset(Pose(0.0, 0.0, -2.0, 0.0)).collided is False


def test_rgb_depends_on_pose_only():
    sim = FakeSimulator(image_size=32)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    a = sim.reset(Pose(0.0, 0.0, -2.0, 0.0)).rgb
    b = sim.reset(Pose(5.0, 0.0, -2.0, 0.0)).rgb
    a2 = sim.reset(Pose(0.0, 0.0, -2.0, 0.0)).rgb
    assert a.shape == (32, 32, 3) and a.dtype == np.uint8
    assert not np.array_equal(a, b) and np.array_equal(a, a2)
    assert np.isinf(sim.observe().depth).all()


def test_spawn_unique_names_and_destroy():
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    first = sim.spawn("AF_Target", "Cylinder", Pose(1.0, 2.0, -1.0, 0.0), (1.0, 1.0, 1.0), "/Game/Geometry/Materials/M_Orange")
    second = sim.spawn("AF_Target", "Cylinder", Pose(3.0, 2.0, -1.0, 0.0), (1.0, 1.0, 1.0))
    assert first == "AF_Target" and second == "AF_Target1"
    sim.destroy(first)
    with pytest.raises(ObjectNotFoundError):
        sim.destroy(first)


def test_spawn_rejects_a_material_instance_as_the_real_server_does():
    # Project AirSim's setMaterial loads with an exact base-UMaterial class filter (WorldSimApi.cpp:1019-1033): a
    # material instance (UE's MI_ prefix: every registry color_instance, every imported glTF material) loads as null and
    # the spawn fails live. A target pool's palette-coloured primitive is exactly such a call (plan 5 C5).
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    for path in ("/Game/AutoFly/Materials/MI_Red", "/Game/AutoFly/Assets/tree_x/MI_leaves.MI_leaves"):
        with pytest.raises(ValueError, match="instance"):
            sim.spawn("AF_Target", "Cube", Pose(1.0, 2.0, -1.0, 0.0), (1.0, 1.0, 1.0), path)
    assert sim.spawn("AF_Target", "Cube", Pose(1.0, 2.0, -1.0, 0.0), (1.0, 1.0, 1.0),
                     "/Game/Geometry/Materials/M_Orange") == "AF_Target"


# --------------------------------------------------------------------------------------------------------
# Moving scene objects (spec §6.5, §7): set_object_poses and its typed errors.
# --------------------------------------------------------------------------------------------------------
PILLARS = {"obs_0000": (Pose(10.0, 0.0, -5.0, 0.0), 0.5), "obs_0001": (Pose(10.0, 6.0, -5.0, 0.0), 0.5)}


def test_backend_hazards_are_typed_in_the_interface_module():
    from autofly_ue5.sim import airsim_backend, types

    for name in ("ObjectPoseError", "SimRequestTimeoutError", "SimConnectionLostError", "CameraPoseError"):
        assert issubclass(getattr(types, name), RuntimeError), name
    # CameraPoseError moved here so the env can catch it without importing the backend; the backend re-exports it.
    assert airsim_backend.CameraPoseError is types.CameraPoseError


def test_fake_moves_named_scene_objects_before_reset():
    sim = FakeSimulator(scene_objects=PILLARS)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.set_object_poses({"obs_0000": Pose(11.0, 1.0, -5.0, 0.0)})  # the real server accepts this before any reset
    assert sim.scene_object_poses()["obs_0000"] == Pose(11.0, 1.0, -5.0, 0.0)
    assert sim.scene_object_poses()["obs_0001"] == Pose(10.0, 6.0, -5.0, 0.0)


def test_fake_rejects_unknown_names_like_the_server_and_applies_a_batch_in_order():
    from autofly_ue5.sim.types import ObjectPoseError

    sim = FakeSimulator(scene_objects=PILLARS)
    with pytest.raises(RuntimeError, match="not launched"):
        sim.set_object_poses({"obs_0000": Pose(0.0, 0.0, 0.0, 0.0)})
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    with pytest.raises(ObjectPoseError, match="obs_9999"):
        sim.set_object_poses({"obs_0000": Pose(12.0, 0.0, -5.0, 0.0), "obs_9999": Pose(0.0, 0.0, 0.0, 0.0),
                              "obs_0001": Pose(13.0, 0.0, -5.0, 0.0)})
    poses = sim.scene_object_poses()
    # One request per name, in order (the real backend has no batch setter): the first moved, the last did not.
    assert poses["obs_0000"].x == 12.0 and poses["obs_0001"].x == 10.0


def test_fake_collides_with_scene_objects_where_they_are_now():
    sim = FakeSimulator(scene_objects=PILLARS)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.set_object_poses({"obs_0000": Pose(0.9, 0.0, -5.0, 0.0)})  # moved into the flight line
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    obs = sim.observe()
    assert obs.collided and obs.step_collisions[0].object_name == "obs_0000"
    sim.set_object_poses({"obs_0000": Pose(10.0, 0.0, 45.0, 0.0)})  # parked underground: no longer in the way
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    assert not sim.observe().collided


def test_fake_depth_is_off_by_default_and_renders_its_pillars_when_asked():
    plain = FakeSimulator(obstacles=[(6.9, 0.0, 0.5)], image_size=32)
    plain.launch("/Game/AutoFly/Maps/S01", 0)
    assert not np.isfinite(plain.reset(Pose(0.0, 0.0, -2.0, 0.0)).depth).any()
    sim = FakeSimulator(obstacles=[(6.9, 0.0, 0.5)], image_size=256, render_depth=True, obstacle_height_m=4.0)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    depth = sim.reset(Pose(0.0, 0.0, -2.0, 0.0)).depth
    # the camera is 0.4 m ahead: the pillar's near surface is 6.0 m away at the image centre, and it fills a width of
    # 2 * atan(r / sqrt(D^2 - r^2)) of the 90 deg field: ~2 * 128 * 0.5 / 6.48 = 19.8 px in row 128
    assert float(np.median(depth[126:130, 126:130])) == pytest.approx(6.0, abs=0.01)
    row = depth[128]
    assert 18 <= int(np.isfinite(row).sum()) <= 22
    assert np.isinf(row[:100]).all() and np.isinf(row[156:]).all()
    # the pillar stands 4 m tall on the ground: looking 1.5 m up from 6 m away is still pillar, the top row (6 m up) is sky
    assert np.isfinite(depth[128 - int(1.5 / 6.0 * 128), 128]) and np.isinf(depth[0, 128])
    assert sim.object_bounds("obstacle_0") == ((6.9, 0.0, -2.0), (1.0, 1.0, 4.0)) and sim.object_bounds("obstacle_7") is None


# --------------------------------------------------------------------------------------------------------
# The runtime spawn table (plan 5 C6): Project AirSim spawns only a cooked asset it finds by short name, and lists them.
# --------------------------------------------------------------------------------------------------------
def test_fake_lists_its_spawn_table_by_full_match_and_refuses_an_asset_it_does_not_hold():
    # Server: ListAssets std::regex_match()es each short name (UnrealHelpers.cpp ListMatchingAssets); SpawnObject raises
    # "There were no objects with name X found in the Registry" for a name not in its maps (WorldSimApi.cpp spawnObject).
    sim = FakeSimulator(assets=["Cylinder", "Cone", "tgt_sedan", "tgt_sedan_old"])
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    assert sim.list_assets("tgt_sedan") == ["tgt_sedan"], "a full match, not a search"
    assert sim.list_assets("^tgt_.*$") == ["tgt_sedan", "tgt_sedan_old"]
    assert sim.list_assets() == ["Cone", "Cylinder", "tgt_sedan", "tgt_sedan_old"]
    assert sim.spawn("t", "tgt_sedan", Pose(5.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == "t"
    with pytest.raises(RuntimeError, match="no objects with name tgt_van found in the Registry"):
        sim.spawn("t2", "tgt_van", Pose(5.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert sim.object_bounds("t2") is None


def test_a_fake_without_a_spawn_table_spawns_any_short_name_and_lists_nothing():
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    assert sim.list_assets() == []
    assert sim.spawn("t", "anything", Pose(5.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == "t"
