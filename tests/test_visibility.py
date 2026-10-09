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


# --------------------------------------------------------------------------------------------------------
# 'missing' only when the target's pixels read what lies behind it; nearer readings are an unmodelled occluder
# (2026-10-08: the thin slice's first four 'missing' records, re-derived below from their stored numbers).
# --------------------------------------------------------------------------------------------------------
def test_an_unmodelled_surface_in_front_of_the_target_is_occluded_unmodeled_with_its_depth():
    # A box the layout does not know (a rock edge, a branch) stands between the camera and the target: every informative
    # pixel reads it, nearer than the target's near face by more than the tolerance.
    setup = _setup((20.0, 0.0))
    wall = ((12.0, 0.0, -1.0), (0.4, 3.0, 2.0))  # 11.4 m planar from the camera, not in the layout
    record = _check(_layout(), setup, render_depth(POSE, boxes=[_box(setup.target, (20.0, 0.0)), wall]))
    assert record["status"] == "occluded_unmodeled" and record["matching_px"] == 0
    assert record["nearer_px"] == record["informative_px"] > 0 and record["farther_px"] == 0
    assert record["unmodeled_depth_m"] == pytest.approx(11.4, abs=0.01)


def test_a_target_whose_pixels_read_the_background_behind_it_stays_missing():
    setup = _setup((20.0, 0.0))
    record = _check(_layout(), setup, render_depth(POSE))
    assert record["status"] == "missing" and record["nearer_px"] == 0
    assert record["farther_px"] == record["informative_px"] > 0 and record["unmodeled_depth_m"] is None


def test_the_record_keeps_its_window_so_every_verdict_can_be_re_derived():
    setup = _setup((20.0, 0.0))
    record = _check(_layout(), setup, render_depth(POSE, boxes=[_box(setup.target, (20.0, 0.0))]))
    window = record["window"]
    rows, cols = window["rows"], window["cols"]
    assert rows == [record["pixel"][0] - 2, record["pixel"][0] + 2] and cols == [record["pixel"][1] - 2, record["pixel"][1] + 2]
    for key in ("observed_m", "background_m", "target_m"):
        assert len(window[key]) == 5 and all(len(row) == 5 for row in window[key]), key
    assert window["observed_m"][2][2] == pytest.approx(19.0, abs=0.01), "the target's near face at the centre"
    centre_row = record["pixel"][0]
    assert window["background_m"][2][2] == pytest.approx(2.0 / ((centre_row + 0.5 - 128) / 128), abs=0.01), \
        "the ground the model predicts behind the target"
    assert window["target_m"][2][2] == pytest.approx([19.0, 20.2], abs=0.01)
    from autofly_ue5.validate.visibility import _grid

    assert _grid([math.inf, math.nan, 1.234], 3) == [[None, None, 1.23]], "no hit (inf) and NaN are null in JSON"


# The four 'missing' first frames of the thin slice's first ~80 episodes (2026-10-08 evening), their 5 x 5 windows
# reconstructed from each record's stored pose, target and distractors on main's built layout (read only): each pixel's
# predicted depth without the target, and the target box's interval on the informative ones (*). The stored window
# minimum equals the prediction to 1-10 cm in all four, so every nearer reading there is predicted geometry; what no
# stored number says is what each informative pixel itself read.
def _window(predicted: list[float], informative: dict[int, tuple[float, float]]) -> tuple[list, list, list]:
    intervals = [informative.get(k) for k in range(len(predicted))]
    return list(predicted), list(predicted), intervals


INF = math.inf
FOUR_CASES = {
    # s05 seed 451000001, bronze whale statue at 63.6 m (prewarm 5): stored window min 56.5 m = the predicted snow 56.6 m
    # at the window's bottom row; four informative pixels with ground behind (69-89 m)
    "s05_451000001": dict(tolerance=3.176, stored_min=56.5, predicted=[207.5] * 5 + [124.5] * 5 + [88.9] * 5 + [69.2] * 5
                          + [56.6] * 5, informative={11: (62.84, 64.2), 12: (62.84, 64.2), 16: (62.84, 64.2),
                                                     17: (62.84, 64.2)}, neighbour=None),
    # s04 seed 441000009, old television at 63.5 m: one informative pixel, a rock at 29.0 m beside it and below it
    "s04_441000009": dict(tolerance=3.114, stored_min=28.547, predicted=[167.3] * 5 + [29.4, 100.4, 100.4, 100.4, 100.4]
                          + [29.0, 29.0, 71.7, 71.7, 71.7] + [28.7, 28.9, 29.4, 55.8, 55.8]
                          + [28.6, 28.8, 45.6, 45.6, 45.6], informative={12: (62.04, 62.51)}, neighbour=29.0),
    # s06 seed 461000000, green armchair at 74.3 m: one informative pixel among trees at 10.4 m
    "s06_461000000": dict(tolerance=3.564, stored_min=8.797, predicted=[216.0, 216.0, 9.0, 216.0, 8.8]
                          + [10.4, 10.5, 129.6, 129.6, 129.6] + [10.5, 10.4, 92.6, 92.6, 92.6]
                          + [48.1, 10.4, 10.4, 72.0, 72.0] + [10.4, 10.3, 10.3, 58.9, 10.7],
                          informative={12: (70.95, 71.69)}, neighbour=10.4),
    # s02 seed 421000010, orange cone at 74.3 m: one informative pixel, trees at 51.5 m and 52.6 m on either side
    "s02_421000010": dict(tolerance=3.621, stored_min=50.188, predicted=[50.2, 51.9, 50.3, 50.7, 50.7]
                          + [51.1, 51.5, 183.8, 52.6, 51.3] + [110.3, 110.3, 52.6, 52.3, 110.3]
                          + [78.8, 78.8, 52.3, 52.2, 78.8] + [61.3, 61.3, 61.3, 52.6, 61.3],
                          informative={7: (71.83, 72.8)}, neighbour=51.5),
}


@pytest.mark.parametrize("case", sorted(FOUR_CASES))
def test_the_four_stored_missing_records_re_derived_by_the_new_rule(case):
    from autofly_ue5.validate.visibility import classify_window

    c = FOUR_CASES[case]
    observed, background, intervals = _window(c["predicted"], c["informative"])
    assert min(observed) == pytest.approx(c["stored_min"], abs=0.11), "the stored minimum is predicted geometry"
    # the informative pixels reading what lies behind the target: the target is genuinely not drawn
    verdict = classify_window(observed, background, intervals, c["tolerance"])
    assert verdict["status"] == "missing" and verdict["informative_px"] == len(c["informative"])
    if c["neighbour"] is not None:
        # the occluder beside the single informative pixel, one pixel over (the RGB showed a rock edge, trees): nearer
        covered = [c["neighbour"] if interval is not None else value for value, interval in zip(observed, intervals)]
        verdict = classify_window(covered, background, intervals, c["tolerance"])
        assert verdict["status"] == "occluded_unmodeled" and verdict["unmodeled_depth_m"] == pytest.approx(c["neighbour"])


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
