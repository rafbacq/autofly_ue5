from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import load_registry, load_scene_file
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.validate.geometry import choose_depth_probes, pillars_from_layout_json
from autofly_ue5.validate.live_m1 import check_crash, check_one_step, check_rgb_changes, check_tracking

SCENE = load_scene_file(SCENES_DIR / "s01_white_pillars.json")
LAYOUT_JSON = {"layout": generate_layout(SCENE, load_registry()).to_json()}


def _probes():
    return choose_depth_probes(pillars_from_layout_json(LAYOUT_JSON), SCENE.bounds)


def test_depth_probes_exist_for_the_generated_s01_layout():
    probes = _probes()
    assert [p.distance_m for p in probes] == [3.0, 6.0, 12.0]
    assert all(abs(p.expected_depth_m - p.distance_m) < 1e-6 for p in probes)


def test_gate_checks_pass_on_the_fake_simulator():
    pillars = pillars_from_layout_json(LAYOUT_JSON)
    sim = FakeSimulator(obstacles=[(p.x, p.y, p.radius) for p in pillars], image_size=32)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    assert check_rgb_changes(sim)["pass"] is True
    crash = check_crash(sim, _probes()[1])
    assert crash["pass"] is True and crash["reset_after_crash"]["depth"]["mask_agreement"] == 1.0
    one_step = check_one_step(sim)
    assert one_step["pass"] is True and all(r["frame_times_match"] for r in one_step["rows"])
    assert check_tracking(sim)["pass"] is True
