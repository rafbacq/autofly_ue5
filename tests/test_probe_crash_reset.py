"""scripts/probe_crash_reset.py: measures, live, whether a reset straight after a crash lands where it was asked (C9)."""

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import load_registry, load_scene_file
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.validate.geometry import choose_depth_probes, pillars_from_layout_json

SCENE = load_scene_file(SCENES_DIR / "s01_white_pillars.json")
LAYOUT = generate_layout(SCENE, load_registry())
PILLARS = pillars_from_layout_json({"layout": LAYOUT.to_json()})
PROBE = choose_depth_probes(PILLARS, SCENE.bounds)[1]


class _StuckAfterEveryOtherCrash(FakeSimulator):
    """The suspected live failure: the first reset after a crash leaves the drone at the crash site."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._crashes = 0
        self._flew = False

    def command_velocity(self, v_forward, yaw_rate, v_z):
        self._flew = self._flew or v_forward > 0
        super().command_velocity(v_forward, yaw_rate, v_z)

    def reset(self, pose):
        crashed_in_flight, self._flew = self._collided and self._flew, False
        if crashed_in_flight:
            self._crashes += 1
            if self._crashes % 2 == 1:
                return super().reset(self._pose)  # "accepted", but the drone stays where it crashed
        return super().reset(pose)


def _sim(cls):
    sim = cls(obstacles=[(p.x, p.y, p.radius) for p in PILLARS], image_size=32)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    return sim


def test_the_probe_reports_clean_resets_on_a_well_behaved_simulator():
    from scripts.probe_crash_reset import run_probe

    report = run_probe(_sim(FakeSimulator), PROBE, SCENE, LAYOUT, trials=4)
    assert report["summary"]["trials"] == 4 and report["summary"]["crashed"] == 4
    assert report["summary"]["bad_first_reset"] == 0 and report["summary"]["render_step_collided"] == 0


def test_the_probe_detects_a_reset_that_stays_at_the_crash_site_and_whether_a_second_one_fixes_it():
    from scripts.probe_crash_reset import run_probe

    report = run_probe(_sim(_StuckAfterEveryOtherCrash), PROBE, SCENE, LAYOUT, trials=4)
    summary = report["summary"]
    assert summary["crashed"] == 4
    assert summary["bad_first_reset"] == 2
    assert summary["fixed_by_second_reset"] == 2
    bad = [t for t in report["trials"] if t["first_reset"]["bad"]]
    assert all(t["first_reset"]["position_error_m"] > 1.0 for t in bad)
