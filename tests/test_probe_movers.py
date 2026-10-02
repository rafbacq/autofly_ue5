"""scripts/probe_movers.py: the M2d go/no-go must pass a simulator that moves, renders and collides with moved pillars,
and fail one that does not -- checked offline against fakes, never a live simulator."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import load_registry, load_scene_file
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import Pose
from autofly_ue5.validate.geometry import CAMERA_OFFSET_M, ray_circle_distance

LAYOUT = generate_layout(load_scene_file(SCENES_DIR / "s01_white_pillars.json"), load_registry())
ROTOR = 0.472


class _RenderingFake(FakeSimulator):
    """FakeSimulator whose depth image is the distance along the camera's centre ray to the nearest above-ground scene
    object, and which reads object poses back like the backend's get_object_poses."""

    def __init__(self, **kwargs):
        self.radius = {i.tag: i.radius_m for i in LAYOUT.instances}
        super().__init__(image_size=32, scene_objects={
            i.tag: (Pose(i.x, i.y, i.z_center, 0.0), i.radius_m + ROTOR) for i in LAYOUT.instances}, **kwargs)

    def get_object_poses(self, names):
        poses = self.scene_object_poses()
        return {n: poses.get(n) for n in names}

    def observe(self):
        obs = super().observe()
        p = obs.pose
        cx, cy = p.x + CAMERA_OFFSET_M * math.cos(p.yaw), p.y + CAMERA_OFFSET_M * math.sin(p.yaw)
        hits = [ray_circle_distance(cx, cy, p.yaw, q.x, q.y, self.radius[n])
                for n, q in self.scene_object_poses().items() if q.z < 0.0]
        nearest = min((h for h in hits if h is not None), default=math.inf)
        return dataclasses.replace(obs, depth=np.full((32, 32), nearest, dtype=np.float32))


class _SlowToSpeedUp(_RenderingFake):
    """Forward speed follows the command with a first-order lag, as the real drone does: M0 flew 2.33 m in its first
    10 steps at a 2 m/s command (docs/gates/m0_smoke_inst0.json), which a 0.95 s time constant reproduces. The plain
    fake reaches full speed at once, which hid that a hovering drone cannot reach a pillar 0.33 m away in one step."""

    TAU_S = 0.95

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._speed = 0.0

    def reset(self, pose):
        self._speed = 0.0
        return super().reset(pose)

    def step(self, dt=0.2):
        v_cmd, yaw_rate, v_z = self._pending
        self._speed += (v_cmd - self._speed) * (1.0 - math.exp(-dt / self.TAU_S))
        self._pending = (self._speed, yaw_rate, v_z)
        return super().step(dt)


class _IgnoresMoves(_RenderingFake):
    """The failure the probe exists to catch: the server accepts the request but the baked actor stays put."""

    def set_object_poses(self, poses):
        self._require_launched()


class _CollisionLagsAMove(_RenderingFake):
    """Collision geometry that only follows a moved pillar one step later (rendering is fine)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._previous = None

    def set_object_poses(self, poses):
        self._previous = dict(self._scene_objects)
        super().set_object_poses(poses)

    def step(self, dt=0.2):
        if self._previous is None:
            return super().step(dt)
        current, self._scene_objects = self._scene_objects, self._previous
        try:
            return super().step(dt)
        finally:
            self._scene_objects, self._previous = current, None


def _launched(cls):
    sim = cls()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    return sim


def test_a_simulator_that_moves_renders_and_collides_with_moved_pillars_passes():
    from scripts.probe_movers import run_probe

    batches = []
    report = run_probe(_launched(_RenderingFake), LAYOUT, async_batch=batches.append)
    assert report["pass"], report["checks"]
    assert set(report["checks"]) == {"a_move", "b_restore", "c_depth_same_step", "d_collision", "e_vacated_home",
                                     "f_park_and_restore"}
    assert report["checks"]["c_depth_same_step"]["depth_after_move_m"] == np.float32(4.0)
    assert report["checks"]["d_collision"]["d2"]["collided_on_first_step"] is True
    latency = report["latency"]
    assert latency["batch_size"] == 12 and latency["sequential_batch_ms"]["n"] == 20
    assert "step_ms_without_moves" in latency and "step_ms_with_10_moves" in latency
    assert len(batches) == 20 and all(len(b) == 12 for b in batches)


def test_an_actor_that_does_not_move_fails_the_probe():
    from scripts.probe_movers import run_probe

    report = run_probe(_launched(_IgnoresMoves), LAYOUT)
    checks = report["checks"]
    assert report["pass"] is False
    assert not checks["a_move"]["pass"] and checks["a_move"]["moved_error_m"] > 1.0
    assert not checks["c_depth_same_step"]["pass"]
    assert not checks["e_vacated_home"]["pass"], "the 'vacated' home spot still holds the pillar"


def test_collision_geometry_that_lags_a_move_fails_the_first_step_check():
    from scripts.probe_movers import run_probe

    report = run_probe(_launched(_CollisionLagsAMove), LAYOUT)
    d = report["checks"]["d_collision"]
    assert d["pass"] is False and d["d2"]["collided_on_first_step"] is False
    assert report["checks"]["c_depth_same_step"]["pass"], "only the collision lags in this fake"


def test_a_check_that_raises_is_recorded_and_the_others_still_run():
    from scripts.probe_movers import run_probe

    class _ReadBackFails(_RenderingFake):
        def get_object_poses(self, names):
            raise RuntimeError("ERROR code: -32603, message: injected")

    report = run_probe(_launched(_ReadBackFails), LAYOUT)
    assert report["pass"] is False
    assert "injected" in report["checks"]["a_move"]["error"]
    assert report["checks"]["c_depth_same_step"]["pass"] and report["checks"]["d_collision"]["pass"]


def test_the_probe_passes_a_drone_that_is_slow_to_speed_up_like_the_real_one():
    from scripts.probe_movers import FRONT_TIP_M, run_probe

    sim = _launched(_SlowToSpeedUp)
    report = run_probe(sim, LAYOUT)
    assert report["pass"], report["checks"]
    d2 = report["checks"]["d_collision"]["d2"]
    assert 0.25 < d2["last_runway_step_m"] < 0.4, "up to speed, but not at the commanded 2 m/s"
    assert d2["rotor_gap_at_move_m"] < d2["last_runway_step_m"] and d2["collided_on_first_step"]
    assert report["checks"]["e_vacated_home"]["passed_home_spot"]
    assert FRONT_TIP_M == pytest.approx(0.3673)


def test_a_hovering_drone_cannot_reach_a_pillar_a_third_of_a_metre_away_in_one_step():
    # Why d2 needs its runway (2026-10-02 review): from hover, one step covers centimetres.
    sim = _launched(_SlowToSpeedUp)
    sim.reset(Pose(-30.0, 0.0, -2.0, math.pi / 2))
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    assert sim.observe().pose.y < 0.1
