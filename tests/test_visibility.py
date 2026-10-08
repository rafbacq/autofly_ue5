"""autofly_ue5/validate/visibility.py: does an episode's first recorded frame show its target (plan 6 revision 1, the
"Targets" done check)? Against depth frames rendered here from the same geometry (the ground plane, the layout's pillars,
the spawned objects' boxes) with the live checks' pinhole: a target rendered, one not rendered, one behind a pillar, one
behind a distractor, one behind the camera, one too small to tell."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from autofly_ue5.scenes.model import Bounds, Instance, Layout
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import Observation, Pose

N = 256
POSE = Pose(0.0, 0.0, -2.0, 0.0)  # 2 m up, looking north; the camera 0.4 m ahead


def render_depth(pose: Pose, *, boxes=(), cylinders=(), n: int = N) -> np.ndarray:
    """Planar depth of the ground plane (z = 0), upright cylinders (x, y, r, height) and axis-aligned boxes
    ((cx, cy, cz), (sx, sy, sz)), NED, from the front camera (90 deg, 0.4 m ahead, level): validate/obstacles.py's model."""
    f = (n / 2) / math.tan(math.radians(45.0))
    k = (np.arange(n) + 0.5 - n / 2) / f
    b, a = np.meshgrid(k, k, indexing="ij")  # rows (down), cols (right)
    fwd, right = np.array([math.cos(pose.yaw), math.sin(pose.yaw), 0.0]), np.array([-math.sin(pose.yaw), math.cos(pose.yaw), 0.0])
    d = fwd[None, None, :] + a[..., None] * right[None, None, :] + b[..., None] * np.array([0.0, 0.0, 1.0])
    o = np.array([pose.x, pose.y, pose.z]) + 0.4 * fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(d[..., 2] > 0, -o[2] / d[..., 2], np.inf)
        for (cx, cy, cz), (sx, sy, sz) in boxes:
            lo, hi = np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2]), np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2])
            t1, t2 = (lo - o) / d, (hi - o) / d
            near, far = np.nanmax(np.minimum(t1, t2), axis=-1), np.nanmin(np.maximum(t1, t2), axis=-1)
            depth = np.where((near <= far) & (near > 0) & (near < depth), near, depth)
        for x, y, r, height in cylinders:
            fx, fy = o[0] - x, o[1] - y
            qa = d[..., 0] ** 2 + d[..., 1] ** 2
            qb = fx * d[..., 0] + fy * d[..., 1]
            disc = qb ** 2 - qa * (fx ** 2 + fy ** 2 - r ** 2)
            t = (-qb - np.sqrt(np.maximum(disc, 0))) / qa
            z = o[2] + t * d[..., 2]
            hit = (disc >= 0) & (t > 0) & (z <= 0) & (z >= -height) & (t < depth)
            depth = np.where(hit, t, depth)
    return depth.astype(np.float32)


class RenderingFake(FakeSimulator):
    """Renders the ground and every spawned object as its registry box (render_depth) on a reset's frames, up to the
    first-use hold's last step; later frames keep the fake's blank depth, since only the first recorded frame is read
    for visibility (and rendering every step of every episode would make the collector's tests slow)."""

    def __init__(self, *args, **kwargs):
        from autofly_ue5.scenes.model import load_registry

        super().__init__(*args, **kwargs)
        self._by_short: dict = {}
        for asset in load_registry().assets.values():
            self._by_short.setdefault(asset.ue_path.rsplit("/", 1)[-1], asset)
        self._reset_at = 0

    def reset(self, pose):
        self._reset_at = self.steps_taken
        return super().reset(pose)

    def observe(self):
        from autofly_ue5.expert.env import PREWARM_HOLD_STEPS
        from autofly_ue5.validate.visibility import spawned_box

        obs = super().observe()
        if self.steps_taken - self._reset_at > self._reset_steps + 1 + PREWARM_HOLD_STEPS:
            return obs
        boxes = []
        for name, (asset, pose, scale, _material) in self._objects.items():
            box = spawned_box(name, self._by_short[asset], scale, pose.x, pose.y)
            boxes.append(((box.x, box.y, box.z_center), tuple(2 * h for h in box.half)))
        return dataclasses.replace(obs, depth=render_depth(obs.pose, boxes=boxes))


def _observation(depth: np.ndarray, pose: Pose = POSE) -> Observation:
    return Observation(rgb=np.zeros((N, N, 3), np.uint8), depth=depth, pose=pose, velocity_ned=(0.0, 0.0, 0.0),
                       yaw_rate=0.0, sim_time_ns=0, collided=False, step_collisions=(), rgb_time_ns=0, depth_time_ns=0,
                       kinematics_time_ns=0, camera_pose_error_m=0.0)


def _pillar(tag: str, x: float, y: float, r: float = 0.5, h: float = 10.0) -> Instance:
    return Instance(tag=tag, asset="cylinder", x=x, y=y, z_center=-h / 2, yaw=0.0, scale=(2 * r, 2 * r, h),
                    material="white", radius_m=r, height_m=h)


def _layout(*instances: Instance) -> Layout:
    return Layout(scene_id="s01", seed=0, bounds=Bounds(-35, 35, -35, 35), instances=tuple(instances))


def _entry(name="red cube", asset="cube", scale=(1.2, 1.2, 1.2)):
    from autofly_ue5.dataset.targets import TargetEntry

    return TargetEntry(name=name, category="geometry", seen=True, asset=asset, scale=scale, material="runtime_red",
                       status="primitive")


def _setup(target_xy, entry=None, distractors=()):
    """An episode setup whose target (an entry) stands at target_xy, with (entry, (x, y)) distractors."""
    from autofly_ue5.expert.episode import TARGET_SCALE, TARGET_Z_NED, EpisodeSetup

    entry = entry or _entry()
    return EpisodeSetup(scene_id="s01", seed=0, start=POSE, target_xy_z=(*target_xy, TARGET_Z_NED),
                        target_scale=TARGET_SCALE if entry is None else entry.scale,
                        distractors=tuple((x, y, TARGET_Z_NED) for _e, (x, y) in distractors),
                        instruction="go through and avoid the white pillars or other obstacles to reach the red cube",
                        start_edge="x_min", target_edge="x_max", obstacle_material="white", target=entry,
                        distractor_entries=tuple(e for e, _xy in distractors))


def _box(entry, xy):
    """The spawned object's box as the renderer draws it: the registry's size at the entry's scale, standing."""
    from autofly_ue5.scenes.model import load_registry
    from autofly_ue5.validate.visibility import spawned_box

    box = spawned_box("x", load_registry().assets[entry.asset], entry.scale, *xy)
    return (box.x, box.y, box.z_center), tuple(2 * h for h in box.half)


def _check(layout, setup, depth, pose=POSE):
    from autofly_ue5.validate.visibility import FirstFrameVisibility

    return FirstFrameVisibility(layout)(_observation(depth, pose), setup, prewarm_steps=5)


def test_a_target_the_frame_shows_is_visible():
    setup = _setup((20.0, 0.0))
    record = _check(_layout(_pillar("obs_0000", 5.0, 8.0)), setup, render_depth(POSE, boxes=[_box(setup.target, (20.0, 0.0))],
                                                                                    cylinders=[(5.0, 8.0, 0.5, 10.0)]))
    assert record["status"] == "visible" and record["prewarm_steps"] == 5
    assert record["pixel"] == [int(128 + 128 * 1.4 / 19.6), 128] and record["range_m"] == pytest.approx(math.hypot(19.6, 1.4), abs=1e-3)
    assert record["depth_window_min_m"] == pytest.approx(19.0, abs=0.05), "the cube's near face, 0.6 m before its centre"
    assert record["matching_px"] > 0 and record["occluder"] is None


def test_a_target_in_view_and_unobstructed_that_the_frame_lacks_is_missing():
    setup = _setup((20.0, 0.0))
    record = _check(_layout(_pillar("obs_0000", 5.0, 8.0)), setup, render_depth(POSE, cylinders=[(5.0, 8.0, 0.5, 10.0)]))
    assert record["status"] == "missing" and record["informative_px"] > 0 and record["matching_px"] == 0
    assert record["depth_window_min_m"] > 22.0, "the ground behind it"


def test_a_target_behind_a_pillar_is_occluded_by_that_pillar():
    setup = _setup((20.0, 0.0))
    layout = _layout(_pillar("obs_0007", 10.0, 0.0))
    depth = render_depth(POSE, boxes=[_box(setup.target, (20.0, 0.0))], cylinders=[(10.0, 0.0, 0.5, 10.0)])
    record = _check(layout, setup, depth)
    assert record["status"] == "occluded" and record["occluder"] == {"by": "obs_0007", "part": "solid"}


def test_a_target_behind_a_distractor_is_occluded_by_that_distractor():
    setup = _setup((20.0, 0.0), distractors=[(_entry("big box", "cube", (2.0, 2.0, 2.0)), (12.0, 0.0))])
    depth = render_depth(POSE, boxes=[_box(setup.target, (20.0, 0.0)), _box(setup.distractor_entries[0], (12.0, 0.0))])
    record = _check(_layout(), setup, depth)
    assert record["status"] == "occluded" and record["occluder"] == {"by": "distractor_0", "part": "box"}


def test_a_target_behind_the_camera_or_beside_its_view_is_out_of_view():
    for xy in ((-20.0, 0.0), (3.0, 20.0)):
        setup = _setup(xy)
        record = _check(_layout(), setup, render_depth(POSE, boxes=[_box(setup.target, xy)]))
        assert record["status"] == "out_of_view" and record["pixel"] is None, xy


def test_a_target_too_small_to_cover_a_pixel_of_the_window_is_undetermined():
    # a 6 cm cube 50 m off whose centre falls on a pixel corner: no pixel ray of the window meets it
    # (projected at row 133.04, col 128.0: the nearest pixel centres are 0.46 px off, 18 cm at 50 m, past its 3 cm)
    tiny = _entry("tiny cube", "cube", (0.06, 0.06, 0.06))
    x = 0.4 + 50.0
    setup = _setup((x, 0.0), entry=tiny)
    record = _check(_layout(), setup, render_depth(POSE, boxes=[_box(tiny, (x, 0.0))]))
    assert record["pixel"] == [133, 128]
    assert record["status"] == "undetermined" and record["informative_px"] == 0


def test_the_record_says_which_depth_it_read():
    setup = _setup((20.0, 0.0))
    record = _check(_layout(), setup, render_depth(POSE, boxes=[_box(setup.target, (20.0, 0.0))]))
    assert record["depth_frame"] == [N, N] and record["tolerance_m"] == pytest.approx(1.0)
    assert set(record) >= {"status", "range_m", "pixel", "depth_window_min_m", "prewarm_steps"}


def test_a_spawned_box_stands_as_apply_setup_stands_it():
    from autofly_ue5.expert.episode import standing_z_ned
    from autofly_ue5.scenes.model import load_registry
    from autofly_ue5.validate.visibility import spawned_box

    registry = load_registry()
    cube = spawned_box("t", registry.assets["cube"], (1.2, 1.2, 1.2), 3.0, 4.0)
    assert (cube.x, cube.y, cube.z_center, cube.yaw) == (3.0, 4.0, pytest.approx(-0.6), 0.0)
    assert cube.half == pytest.approx((0.6, 0.6, 0.6))
    boulder = registry.assets["boulder_01"]  # an imported model stands on its own origin, its box about its bounds
    box = spawned_box("t", boulder, (1.5, 1.5, 1.5), 0.0, 0.0)
    o = boulder.bounds_origin_cm or (0.0, 0.0, 0.0)
    assert box.z_center == pytest.approx(standing_z_ned(boulder, (1.5, 1.5, 1.5)) - o[2] * 1.5 / 100)
    assert box.half == pytest.approx(tuple(s * 1.5 / 2 for s in boulder.base_size_m))
    assert dataclasses.is_dataclass(box)
