"""autofly_ue5/validate/target_probe.py: the clear view, the placement ahead of it and the depth check that tell whether a
spawned target is drawn, one implementation for scripts/probe_targets.py (the live probe of every pool target) and the
env's session warm-up (expert/env.py). The probe's own tests (tests/test_probe_targets.py) hold its behaviour."""

from __future__ import annotations

import pytest


def test_the_probe_and_the_env_share_one_view_placement_and_depth_check():
    import scripts.probe_targets as probe
    from autofly_ue5.validate import target_probe

    assert probe.choose_view is target_probe.choose_view
    assert probe._region is target_probe.region and probe._depth_check is target_probe.depth_check
    assert probe._expected_box is target_probe.expected_box and probe._camera is target_probe.camera
    assert probe.REACH_M == target_probe.REACH_M and probe.VIEW_CLEARANCE_M == target_probe.VIEW_CLEARANCE_M


@pytest.mark.parametrize("scale, bounds, expected", [
    ((1.0, 1.0, 1.0), (4.0, 10.0), 4.5),  # a 1 m cube: 1.5 x 1 + 3 m
    ((1.0, 1.0, 1.0), (6.0, 10.0), 6.0),  # the env's warm-up stands it 6 m off at the least
    ((5.0, 5.0, 5.0), (6.0, 10.0), 10.0),  # a large one at most 10 m off
])
def test_a_target_stands_straight_ahead_of_the_view_at_its_size_s_distance(scale, bounds, expected):
    import math

    from autofly_ue5.scenes.model import load_registry
    from autofly_ue5.sim.types import Pose
    from autofly_ue5.validate.geometry import CAMERA_OFFSET_M
    from autofly_ue5.validate.target_probe import placement

    view = Pose(1.0, 2.0, -2.0, math.pi / 2)
    position, distance = placement(load_registry().assets["cube"], scale, view, min_distance_m=bounds[0],
                                   max_distance_m=bounds[1])
    assert distance == pytest.approx(expected)
    assert position[0] == pytest.approx(1.0) and position[1] == pytest.approx(2.0 + CAMERA_OFFSET_M + expected)
    assert position[2] == pytest.approx(-scale[2] / 2), "standing on the ground, as apply_setup stands it"
