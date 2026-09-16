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
    first = sim.spawn("AF_Target", "cylinder", Pose(1.0, 2.0, -1.0, 0.0), (1.0, 1.0, 1.0), "white")
    second = sim.spawn("AF_Target", "cylinder", Pose(3.0, 2.0, -1.0, 0.0), (1.0, 1.0, 1.0))
    assert first == "AF_Target" and second == "AF_Target1"
    sim.destroy(first)
    with pytest.raises(ObjectNotFoundError):
        sim.destroy(first)
