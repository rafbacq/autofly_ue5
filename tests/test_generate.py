import itertools
import json
import math
import random

import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import UnsupportedPlacementError, generate_layout, jittered_grid
from autofly_ue5.scenes.model import Bounds, load_registry, load_scene_file

S01 = SCENES_DIR / "s01_white_pillars.json"
BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)


def test_jittered_grid_count_margin_and_spacing():
    points = jittered_grid(random.Random(7), BOUNDS, 80, margin_m=8.0, jitter_m=1.0)
    assert len(points) == 80
    for x, y in points:
        assert -28.0 <= x <= 28.0 and -28.0 <= y <= 28.0
    closest = min(math.dist(a, b) for a, b in itertools.combinations(points, 2))
    assert closest >= 4.0 - 1e-9  # 6 m cells minus twice the 1 m jitter


def test_jittered_grid_is_deterministic():
    a = jittered_grid(random.Random(3), BOUNDS, 20, margin_m=8.0, jitter_m=1.0)
    b = jittered_grid(random.Random(3), BOUNDS, 20, margin_m=8.0, jitter_m=1.0)
    assert a == b


def test_jitter_larger_than_half_cell_is_rejected():
    with pytest.raises(ValueError, match="jitter"):
        jittered_grid(random.Random(1), BOUNDS, 80, margin_m=8.0, jitter_m=3.5)


def test_generate_s01_layout():
    scene, registry = load_scene_file(S01), load_registry()
    layout = generate_layout(scene, registry)
    assert layout.scene_id == "s01" and layout.seed == 1001 and len(layout.instances) == 80
    assert [i.tag for i in layout.instances[:2]] == ["obs_0000", "obs_0001"]
    assert len({i.tag for i in layout.instances}) == 80
    for inst in layout.instances:
        assert 0.4 <= inst.radius_m <= 0.6 and 8.0 <= inst.height_m <= 12.0
        assert inst.z_center == pytest.approx(-inst.height_m / 2, abs=1e-3)
        assert inst.scale[0] == inst.scale[1] and inst.material == "white" and inst.asset == "cylinder"
    assert generate_layout(scene, registry) == layout
    assert generate_layout(scene, registry, seed=2002) != layout
    assert json.loads(json.dumps(layout.to_json()))["instances"][0]["tag"] == "obs_0000"


def test_unimplemented_placement_is_explicit(tmp_path):
    data = json.loads(S01.read_text())
    data["obstacle_groups"][0]["placement"] = {"type": "poisson", "margin_m": 8.0, "min_distance_m": 3.0}
    path = tmp_path / "poisson.json"
    path.write_text(json.dumps(data))
    with pytest.raises(UnsupportedPlacementError, match="poisson"):
        generate_layout(load_scene_file(path), load_registry())
