import pytest
from projectairsim.utils import load_scene_config_as_dict

from autofly_ue5.paths import CONFIGS_DIR


@pytest.mark.parametrize("scene", ["scene_autofly_m0.jsonc", "scene_autofly_m0_fast.jsonc", "scene_autofly_s01.jsonc",
                                   "scene_autofly_s01_fast.jsonc"])
def test_scene_validates_against_project_airsim_schema(scene):
    config, _paths = load_scene_config_as_dict(scene, str(CONFIGS_DIR))
    assert config["clock"]["type"] == "steppable"
    assert config["clock"]["step-ns"] == 5_000_000
    assert config["clock"]["pause-on-start"] is True
    robot = config["actors"][0]["robot-config"]
    camera = next(s for s in robot["sensors"] if s["id"] == "FrontCamera")
    assert camera["capture-interval"] == 0.001
    settings = {c["image-type"]: c for c in camera["capture-settings"]}
    assert set(settings) == {0, 1}
    for c in settings.values():
        assert (c["width"], c["height"], c["fov-degrees"]) == (256, 256, 90)
        assert c["capture-enabled"] is True and c["compress"] is False
    assert camera["origin"]["xyz"] == "0.40 0.0 0.0"


def test_s01_start_is_in_the_south_start_band_at_2m():
    config, _paths = load_scene_config_as_dict("scene_autofly_s01.jsonc", str(CONFIGS_DIR))
    x, y, z = (float(v) for v in config["actors"][0]["origin"]["xyz"].split())
    assert 2.0 <= x - (-35.0) <= 6.0 and z == -2.0


def test_the_fast_s01_config_differs_from_s01_only_in_its_id_and_clock_rate():
    # C3: M0 measured 18.0 steps/s at a 1 ms real-time update rate against 7.4 at 3 ms, lock-step intact; the fast
    # s01 config must be the same scene in every other respect so it can replace s01's once M1 passes on it.
    base, _ = load_scene_config_as_dict("scene_autofly_s01.jsonc", str(CONFIGS_DIR))
    fast, _ = load_scene_config_as_dict("scene_autofly_s01_fast.jsonc", str(CONFIGS_DIR))
    assert base["clock"]["real-time-update-rate"] == 3_000_000 and fast["clock"]["real-time-update-rate"] == 1_000_000
    assert fast["id"] == "SceneAutoFlyS01Fast"
    for config in (base, fast):
        config.pop("id")
        config["clock"].pop("real-time-update-rate")
    assert fast == base

