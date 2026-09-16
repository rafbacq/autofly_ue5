import math

import numpy as np
import pytest

from autofly_ue5.paths import ROOT
from autofly_ue5.scenes.model import load_scene_file
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import Pose


def scene_and_layout():
    import json

    from autofly_ue5.scenes.model import Bounds, Instance, Layout

    scene = load_scene_file(ROOT / "scenes" / "s01_white_pillars.json")
    raw = json.loads((ROOT / "runs" / "levels" / "s01.layout.json").read_text())["layout"]
    b = Bounds(**raw["bounds"])
    inst = tuple(Instance(**i) for i in raw["instances"])
    return scene, Layout(scene_id=raw["scene_id"], seed=raw["seed"], bounds=b, instances=inst)


def test_instruction_templates_are_verbatim_including_the_misspelling():
    from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES

    joined = " | ".join(INSTRUCTION_TEMPLATES)
    assert "avioding" in joined, "the released episodes misspell 'avoiding'; do not correct it"
    assert "go through and avoid the {obstacle} or other obstacles to reach the {target}" in INSTRUCTION_TEMPLATES


def test_start_and_target_are_on_opposite_edges():
    from autofly_ue5.expert.episode import OPPOSITE, sample_setup

    scene, layout = scene_and_layout()
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        assert OPPOSITE[s.start_edge] == s.target_edge, "a same-edge episode would be a metre long"


def test_start_is_in_the_start_band_and_altitude_band():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    lo, hi = scene.start_band
    a_lo, a_hi = scene.altitude_band
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        b = layout.bounds
        d = min(s.start.x - b.x_min, b.x_max - s.start.x, s.start.y - b.y_min, b.y_max - s.start.y)
        assert lo - 1e-6 <= d <= hi + 1e-6
        assert a_lo - 1e-6 <= -s.start.z <= a_hi + 1e-6, "z is NED; altitude is -z"


def test_start_altitude_is_inset_from_the_hard_altitude_band():
    """Task 3's altitude band is a hard boundary with no margin: at v_z in [-1, 1] m/s over a 0.2 s
    control step, one step covers 0.2 m, so a start drawn from within a few centimetres of either edge
    could terminate the episode on its first downward step through no fault of the policy. The start
    must be drawn from the band's *interior*, inset by START_ALTITUDE_MARGIN_M at each end -- the band
    itself (where the drone may fly) stays the full spec range; only where it may *begin* is narrowed.
    This pins the inset explicitly so a later edit that widens the start back to the full band fails
    loudly, even though such an edit would still satisfy the looser altitude-band assertion above.
    """
    from autofly_ue5.expert.episode import START_ALTITUDE_MARGIN_M, sample_setup

    scene, layout = scene_and_layout()
    a_lo, a_hi = scene.altitude_band
    inset_lo, inset_hi = a_lo + START_ALTITUDE_MARGIN_M, a_hi - START_ALTITUDE_MARGIN_M
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        altitude = -s.start.z
        assert inset_lo - 1e-6 <= altitude <= inset_hi + 1e-6, (
            f"seed {seed}: altitude {altitude:.3f} m is outside the inset start band "
            f"[{inset_lo}, {inset_hi}]; start altitude must not use the full hard band"
        )


def test_the_crossing_is_long():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        d = math.hypot(s.target_xy_z[0] - s.start.x, s.target_xy_z[1] - s.start.y)
        assert d > 50.0, f"seed {seed} produced a {d:.1f} m episode; the scene is 70 m across"


def test_nothing_spawns_inside_an_obstacle():
    from autofly_ue5.expert.episode import sample_setup, spawn_clearance_m

    scene, layout = scene_and_layout()
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        points = [(s.start.x, s.start.y), (s.target_xy_z[0], s.target_xy_z[1])]
        points += [(d[0], d[1]) for d in s.distractors]
        for px, py in points:
            for inst in layout.instances:
                gap = math.hypot(px - inst.x, py - inst.y) - inst.radius_m
                assert gap >= spawn_clearance_m(), f"seed {seed}: point {px:.1f},{py:.1f} sits in {inst.tag}"


def test_distractor_count_and_spacing():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    for seed in range(25):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        assert 3 <= len(s.distractors) <= 5
        pts = [s.target_xy_z] + list(s.distractors)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) >= 4.0 - 1e-6


def test_distractors_keep_clear_of_the_start_pose():
    """A distractor spawning on top of the drone's start pose makes the episode unwinnable from step 0
    (an immediate collision) regardless of the policy, which silently eats into the M2 gate's success-rate
    budget. Measured before the fix, over 2000 seeds: a distractor landed within 8 m of the start in
    ~22 % of episodes, within 5 m in 12.5 %, within 1 m in about 1 seed in 200, closest 0.07 m -- i.e.
    the outright unwinnable cases are rare while the keep-out this test pins is violated often. Checked
    across many seeds either way: a single seed proves nothing here.
    """
    from autofly_ue5.expert.episode import START_KEEPOUT_M, sample_setup

    scene, layout = scene_and_layout()
    worst = math.inf
    for seed in range(500):
        s = sample_setup(scene, layout, np.random.default_rng(seed))
        for dx, dy, _ in s.distractors:
            d = math.hypot(dx - s.start.x, dy - s.start.y)
            worst = min(worst, d)
            assert d >= START_KEEPOUT_M - 1e-6, f"seed {seed}: distractor {dx:.2f},{dy:.2f} is {d:.2f} m from the start"
    assert worst < math.inf  # sanity: the loop actually exercised at least one distractor


def test_sampling_is_deterministic_for_a_seed():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    a = sample_setup(scene, layout, np.random.default_rng(7))
    b = sample_setup(scene, layout, np.random.default_rng(7))
    assert a == b
    assert a != sample_setup(scene, layout, np.random.default_rng(8))


def test_apply_then_clear_round_trips_against_the_fake():
    from autofly_ue5.expert.episode import apply_setup, clear_setup, sample_setup

    scene, layout = scene_and_layout()
    setup = sample_setup(scene, layout, np.random.default_rng(3))
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(setup.start)
    names = apply_setup(sim, setup)
    assert len(names) == 1 + len(setup.distractors)
    clear_setup(sim, names)
    clear_setup(sim, names)  # must tolerate an already-destroyed name, so a failed episode can always clean up


def test_apply_setup_tears_down_a_partial_spawn_on_failure():
    """The real backend's spawn() can raise AFTER the actor already exists server-side (WorldSimApi
    creates the object, then raises if set_object_material fails on it) -- apply_setup only reports its
    name list on a full, uncaught success, so a failure partway through the loop must not leave 1-5
    actors that nothing will ever destroy (the caller's own bookkeeping never learns their names, since
    apply_setup never returns). Plan 1 named exactly this kind of actor accumulation as a trap, and a
    training run calls apply_setup tens of thousands of times.

    A minimal simulator double stands in for FakeSimulator here (rather than FakeSimulator itself)
    because FakeSimulator's own spawn() never raises -- there is no way to make the closed fake fail
    mid-spawn without a double that can.
    """
    from autofly_ue5.expert.episode import apply_setup, sample_setup

    class FailingThirdSpawn:
        def __init__(self) -> None:
            self.spawned: list[str] = []
            self.destroyed: list[str] = []
            self._calls = 0

        def spawn(self, name, asset, pose, scale, material=None):
            self._calls += 1
            if self._calls == 3:
                raise RuntimeError("set_object_material failed after the actor was already created")
            self.spawned.append(name)
            return name

        def destroy(self, name):
            self.destroyed.append(name)

    scene, layout = scene_and_layout()
    # N_DISTRACTORS_DEFAULT is (3, 5), so every setup spawns at least 1 + 3 = 4 objects -- the 3rd spawn
    # call always exists to fail, regardless of seed.
    setup = sample_setup(scene, layout, np.random.default_rng(0))
    sim = FailingThirdSpawn()

    with pytest.raises(RuntimeError):
        apply_setup(sim, setup)

    assert sim.spawned, "the failure must happen after at least one object was actually spawned"
    assert sorted(sim.destroyed) == sorted(sim.spawned), (
        "every object spawned before the failure must be torn down before the error propagates"
    )


def test_the_target_material_is_one_the_packaged_binary_actually_accepts():
    """apply_setup paints the target with "orange" if the registry has it, else the first material that is
    not the obstacle's. With only {grid, white} registered and s01's obstacles painted "white", that
    fallback could only ever resolve to "grid" -- the engine WorldGridMaterial -- which the packaged binary
    rejects in set_object_material. That is not seed-dependent: it failed 100% of live episodes and blocked
    Task 7's measurement and Task 8's training entirely, while every offline test stayed green because
    FakeSimulator accepts any material string.

    docs/gates/m1_gate.json is the evidence for the fix: on this exact SHA-256-verified binary, spawning a
    cube with /Game/Geometry/Materials/M_Orange produced centre RGB [97, 55, 36] and orange_dominant=True.
    Adding "orange" to the registry does not touch the level spec -- layout_to_level_spec emits only the
    materials the layout actually uses -- so the M1 provenance chain and its recorded hash are unaffected.
    """
    import json

    from autofly_ue5.expert.episode import sample_setup
    from autofly_ue5.paths import ROOT
    from autofly_ue5.scenes.model import load_registry

    registry = load_registry()
    assert "orange" in registry.materials, (
        "the target material is gone; live spawns will fall back to the ground material and every "
        "episode will fail against the real simulator while every offline test still passes"
    )
    proven = json.loads((ROOT / "docs" / "gates" / "m1_gate.json").read_text())
    assert registry.materials["orange"].ue_path == proven["checks"]["spawn_destroy_packaged"]["material"], (
        "the target material must be the one M1 proved the packaged binary accepts"
    )
    # And the selection logic must actually reach it rather than the ground material.
    scene, layout = scene_and_layout()
    setup = sample_setup(scene, layout, np.random.default_rng(3))
    chosen = "orange" if "orange" in registry.materials else next(
        (n for n in registry.materials if n != setup.obstacle_material), None)
    assert chosen == "orange" and chosen != scene.ground


def test_spawn_asset_name_is_the_last_path_segment():
    from autofly_ue5.expert.episode import _spawn_asset_name

    # Confirmed live (Task 7) via world.list_assets() against the packaged s01 binary: Project AirSim's
    # runtime spawn table keys every static mesh -- engine content included -- by this short name, not by
    # its full package path. "/Engine/BasicShapes/Cylinder" -> "Cylinder" and M1's own
    # "/Game/Geometry/Meshes/1M_Cube" -> "1M_Cube" are both present verbatim in that live probe.
    assert _spawn_asset_name("/Engine/BasicShapes/Cylinder") == "Cylinder"
    assert _spawn_asset_name("/Game/Geometry/Meshes/1M_Cube") == "1M_Cube"


def test_apply_setup_forwards_a_resolved_asset_and_material_not_bare_registry_keys():
    """The bug this guards against, found live in Task 7: apply_setup used to pass FakeSimulator a bare
    registry key ("cylinder", "orange"/"white"/"grid") for both `asset` and `material`. FakeSimulator's
    old spawn() accepted that silently -- a bare string is a perfectly good dict key -- so 207 green tests
    never caught it, but the real backend's spawn_object()/set_object_material() rejected it on 100% of
    live calls, because Simulator.spawn()'s contract (protocol.py) requires a slash-free short spawn name
    for `asset` and a full UE package path for `material`. FakeSimulator.spawn() now enforces exactly that
    contract (autofly_ue5/sim/fake.py), so this test would fail loudly, not silently, if the bare-key bug
    ever returned.
    """
    from autofly_ue5.expert.episode import _spawn_asset_name, apply_setup, sample_setup
    from autofly_ue5.scenes.model import load_registry

    registry = load_registry()
    scene, layout = scene_and_layout()
    setup = sample_setup(scene, layout, np.random.default_rng(3))
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(setup.start)

    names = apply_setup(sim, setup)

    expected_asset = _spawn_asset_name(registry.assets["cylinder"].ue_path)
    target_material_name = "orange" if "orange" in registry.materials else next(
        n for n in registry.materials if n != setup.obstacle_material)
    expected_target_material = registry.materials[target_material_name].ue_path
    expected_obstacle_material = registry.materials[setup.obstacle_material].ue_path

    target_asset, _pose, _scale, target_material = sim._objects[names[0]]
    assert target_asset == expected_asset and "/" not in target_asset
    assert target_material == expected_target_material and target_material.startswith("/")

    for name in names[1:]:
        asset, _pose, _scale, material = sim._objects[name]
        assert asset == expected_asset and "/" not in asset
        assert material == expected_obstacle_material and material.startswith("/")
