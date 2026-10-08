import dataclasses
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

    target_asset, _pose, _scale, target_material = sim._objects[names[0]]
    assert target_asset == expected_asset and "/" not in target_asset
    assert target_material == expected_target_material and target_material.startswith("/")

    # Distractors spawn UNPAINTED. The obstacle material is a color_instance, which the runtime spawn path
    # categorically cannot apply (see test_a_color_instance_material_is_never_forwarded_to_a_runtime_spawn);
    # an earlier revision of this test asserted they were painted with it, which is why that assertion has
    # been replaced rather than relaxed -- it pinned a behaviour the real backend rejects 100% of the time.
    for name in names[1:]:
        asset, _pose, _scale, material = sim._objects[name]
        assert asset == expected_asset and "/" not in asset
        assert material is None


def test_a_color_instance_material_is_never_forwarded_to_a_runtime_spawn():
    """Project AirSim's setMaterial uses StaticLoadObject with an EXACT base-UMaterial class filter
    (WorldSimApi.cpp:1019-1033), so a UMaterialInstanceConstant load returns null and set_object_material
    fails however correct the path is. Our registry's scene palette is built from color_instances
    ("white" -> MI_White), which build_level.py applies happily at EDITOR build time -- a different code
    path with different class requirements. M0 measured this exact failure on M_Blue and wrote it down in
    sim/smoke_m0.py:44-45; the knowledge never reached episode.py, and rediscovering it cost three blocked
    live runs. This test is where that knowledge now lives for the runtime path.
    """
    from autofly_ue5.expert.episode import _runtime_material_path, apply_setup, sample_setup
    from autofly_ue5.scenes.model import load_registry

    registry = load_registry()
    assert registry.materials["white"].kind == "color_instance"
    assert _runtime_material_path(registry, "white") is None, "a color_instance can never be applied at runtime"
    assert _runtime_material_path(registry, "orange") == "/Game/Geometry/Materials/M_Orange"
    assert _runtime_material_path(registry, None) is None

    # And nothing a color_instance resolves from may reach spawn(): record what apply_setup forwards.
    scene, layout = scene_and_layout()
    setup = sample_setup(scene, layout, np.random.default_rng(3))
    forwarded: list[str | None] = []

    class RecordingSim(FakeSimulator):
        def spawn(self, name, asset, pose, scale, material=None):
            forwarded.append(material)
            return super().spawn(name, asset, pose, scale, material)

    sim = RecordingSim()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(setup.start)
    apply_setup(sim, setup)

    instance_paths = {m.ue_path for m in registry.materials.values() if m.kind == "color_instance"}
    assert forwarded[0] == "/Game/Geometry/Materials/M_Orange", "the target keeps the M1-proven base material"
    assert all(m is None for m in forwarded[1:]), "distractors must spawn unpainted, not with a color_instance"
    assert not (set(forwarded) & instance_paths)


def test_a0_aligns_the_start_to_the_target_s_8_sector_bearing_and_nothing_else():
    # M3's a0 (docs/decisions/2026-10-03-m3-a0-and-collection.md): the start yaw is the bearing to the target rounded
    # to the nearest 45 degrees; everything the seed drew stays as it was.
    import dataclasses

    from autofly_ue5.expert.episode import a0_aligned, sample_setup

    scene, layout = scene_and_layout()
    for seed in range(40):
        setup = sample_setup(scene, layout, np.random.default_rng(seed))
        aligned = a0_aligned(setup)
        bearing = math.atan2(setup.target_xy_z[1] - setup.start.y, setup.target_xy_z[0] - setup.start.x)
        off = math.atan2(math.sin(bearing - aligned.start.yaw), math.cos(bearing - aligned.start.yaw))
        assert abs(off) <= math.radians(22.5) + 1e-9
        assert round(aligned.start.yaw / (math.pi / 4)) * (math.pi / 4) == pytest.approx(aligned.start.yaw)
        assert dataclasses.replace(aligned, start=setup.start) == setup


# --------------------------------------------------------------------------------------------------------
# The target pool (plan 5 C5, "Design (2026-10-08)"): drawn after every existing draw, spawned from its entries.
# --------------------------------------------------------------------------------------------------------
def _template_of(instruction: str, target: str, obstacle: str) -> str:
    from autofly_ue5.expert.episode import INSTRUCTION_TEMPLATES

    (template,) = [t for t in INSTRUCTION_TEMPLATES if t.format(target=target, obstacle=obstacle) == instruction]
    return template


def test_without_a_pool_the_setup_is_drawn_exactly_as_before():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    for seed in range(20):
        plain = sample_setup(scene, layout, np.random.default_rng(seed))
        explicit = sample_setup(scene, layout, np.random.default_rng(seed), pool=None)
        assert explicit == plain and explicit.target is None and explicit.distractor_entries == ()
        _template_of(plain.instruction, "target", scene.instruction_obstacle)  # Plan 2's placeholder name, as before


def test_a_pool_changes_no_earlier_draw_and_the_instruction_names_the_drawn_target(tmp_path):
    import dataclasses

    from autofly_ue5.expert.episode import sample_setup
    from tests.test_dataset_targets import primitive_pool

    scene, layout = scene_and_layout()
    pool = primitive_pool(tmp_path)
    for seed in range(30):
        plain = sample_setup(scene, layout, np.random.default_rng(seed))
        drawn = sample_setup(scene, layout, np.random.default_rng(seed), pool=pool)
        assert dataclasses.replace(drawn, target=None, distractor_entries=(), instruction=plain.instruction,
                                   target_scale=plain.target_scale) == plain, f"seed {seed}: an earlier draw moved"
        assert drawn.target in pool.for_split("train") and drawn.target_scale == drawn.target.scale
        assert len(drawn.distractor_entries) == len(drawn.distractors)
        names = [drawn.target.name] + [e.name for e in drawn.distractor_entries]
        assert len(set(names)) == len(names), "distractors are drawn without replacement, never the target"
        template = _template_of(plain.instruction, "target", scene.instruction_obstacle)
        assert drawn.instruction == template.format(target=drawn.target.name, obstacle=scene.instruction_obstacle)


def test_the_pool_s_draws_are_pinned_the_target_then_one_entry_per_distractor_position(tmp_path):
    # Step 10: the target, uniform over the split's usable entries in file order. Step 11: for each distractor position
    # already drawn, in order, one of the remaining entries, uniform, removed from the rest. Both by rng.integers, after
    # every draw a pool-free setup makes from the same stream.
    from autofly_ue5.expert.episode import sample_setup
    from tests.test_dataset_targets import primitive_pool

    scene, layout = scene_and_layout()
    pool = primitive_pool(tmp_path)
    for seed in (0, 1, 7, 400_000_000, 100_000_061):
        rng = np.random.default_rng(seed)
        plain = sample_setup(scene, layout, rng)
        rest = list(pool.for_split(scene.split))
        target = rest.pop(int(rng.integers(len(rest))))
        expected = tuple(rest.pop(int(rng.integers(len(rest)))) for _ in plain.distractors)
        drawn = sample_setup(scene, layout, np.random.default_rng(seed), pool=pool)
        assert drawn.target == target and drawn.distractor_entries == expected, f"seed {seed}"


def test_the_pool_s_draws_follow_the_scene_s_exclusions_in_the_same_pinned_form(tmp_path):
    # Plan 6 step 2c: the scene's own entry of the pool's exclusion table removes entries before step 10, keeping file
    # order, so the draws keep their pinned form over the scene's list (here s01 loses its cylinders).
    from autofly_ue5.expert.episode import sample_setup
    from tests.test_dataset_targets import wide_pool

    scene, layout = scene_and_layout()
    pool = wide_pool(tmp_path)
    drawn_names = set()
    for seed in range(60):
        rng = np.random.default_rng(seed)
        plain = sample_setup(scene, layout, rng)
        rest = list(pool.for_split(scene.split, scene_id=scene.id))
        target = rest.pop(int(rng.integers(len(rest))))
        expected = tuple(rest.pop(int(rng.integers(len(rest)))) for _ in plain.distractors)
        drawn = sample_setup(scene, layout, np.random.default_rng(seed), pool=pool)
        assert drawn.target == target and drawn.distractor_entries == expected, f"seed {seed}"
        objects = (drawn.target, *drawn.distractor_entries)
        assert all(e.asset != "cylinder" for e in objects), "an excluded shape is never drawn, target or distractor"
        drawn_names |= {e.name for e in objects}
    assert {"orange cylinder", "grid pole"}.isdisjoint(drawn_names) and "orange cone" in drawn_names


def test_a_train_scene_draws_every_seen_entry_and_never_an_unseen_one(tmp_path):
    import dataclasses

    from autofly_ue5.expert.episode import sample_setup
    from tests.test_dataset_targets import SEEN_USABLE, primitive_pool

    scene, layout = scene_and_layout()
    pool = primitive_pool(tmp_path)
    targets = set()
    for seed in range(120):
        setup = sample_setup(scene, layout, np.random.default_rng(seed), pool=pool)
        targets.add(setup.target.name)
        assert all(e.seen for e in (setup.target, *setup.distractor_entries))
    assert targets == set(SEEN_USABLE)
    unseen_scene = dataclasses.replace(scene, split="test_unseen")
    drawn = set()
    for seed in range(120):
        setup = sample_setup(unseen_scene, layout, np.random.default_rng(seed), pool=pool)
        drawn |= {e.name for e in (setup.target, *setup.distractor_entries)}
    assert "orange slab" in drawn, "a test_unseen scene draws from the seen and the unseen entries"


def test_a_pool_too_small_for_the_distractor_positions_is_refused(tmp_path):
    from autofly_ue5.expert.episode import sample_setup
    from tests.test_dataset_targets import primitive_pool

    scene, layout = scene_and_layout()
    pool = primitive_pool(tmp_path)  # six seen usable entries: a target and five distractors at most
    with pytest.raises(ValueError, match="distractor"):
        sample_setup(scene, layout, np.random.default_rng(0), pool=pool, n_distractors_range=(6, 6))
    assert len(sample_setup(scene, layout, np.random.default_rng(0), pool=pool,
                            n_distractors_range=(5, 5)).distractor_entries) == 5


def _pool_setup(tmp_path, target: str, distractors: tuple[str, ...], seed: int = 3):
    import dataclasses

    from autofly_ue5.expert.episode import sample_setup
    from tests.test_dataset_targets import primitive_pool

    scene, layout = scene_and_layout()
    pool = primitive_pool(tmp_path)
    by_name = {e.name: e for e in pool.entries}
    setup = sample_setup(scene, layout, np.random.default_rng(seed), pool=pool)
    return dataclasses.replace(setup, target=by_name[target], target_scale=by_name[target].scale,
                               distractor_entries=tuple(by_name[n] for n in distractors[:len(setup.distractors)]))


class _RecordingSim(FakeSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spawn_calls = []

    def spawn(self, name, asset, pose, scale, material=None):
        self.spawn_calls.append((name, asset, pose, tuple(scale), material))
        return super().spawn(name, asset, pose, scale, material)


def _spawned(setup):
    from autofly_ue5.expert.episode import apply_setup

    sim = _RecordingSim()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(setup.start)
    names = apply_setup(sim, setup)
    return names, sim


def test_apply_setup_stands_each_pool_entry_on_the_ground_by_its_short_name_scale_and_material(tmp_path):
    # Centre pivot: the centre at half the scaled height (z = -h s_z / 2, NED); base pivot: z = 0. The entry's material
    # only when it is a base UMaterial; none keeps the mesh's own.
    setup = _pool_setup(tmp_path, "grey boulder", ("orange cylinder", "grid pole", "white box", "orange cube", "grid cube"))
    names, sim = _spawned(setup)
    expected = {
        "grey boulder": ("boulder_01", 0.0, None),
        "orange cylinder": ("Cylinder", -1.0, "/Game/Geometry/Materials/M_Orange"),
        "grid pole": ("Cylinder", -1.5, "/Engine/EngineMaterials/WorldGridMaterial"),
        "white box": ("Cube", -0.5, None),
        "orange cube": ("Cube", -0.6, "/Game/Geometry/Materials/M_Orange"),
        "grid cube": ("Cube", -0.75, "/Engine/EngineMaterials/WorldGridMaterial"),
    }
    entries = (setup.target, *setup.distractor_entries)
    positions = (setup.target_xy_z, *setup.distractors)
    assert len(names) == len(sim.spawn_calls) == len(entries) == 1 + len(setup.distractors)
    for i, (call, entry, xyz) in enumerate(zip(sim.spawn_calls, entries, positions)):
        name, asset, pose, scale, material = call
        exp_asset, exp_z, exp_material = expected[entry.name]
        assert name == ("target" if i == 0 else f"distractor_{i - 1}")
        assert asset == exp_asset and "/" not in asset
        assert (pose.x, pose.y, pose.yaw) == (xyz[0], xyz[1], 0.0) and pose.z == pytest.approx(exp_z)
        assert scale == entry.scale and material == exp_material


def test_the_privileged_target_position_the_expert_flies_to_does_not_follow_the_entry(tmp_path):
    # The expert's vector observation holds the target's height difference (expert/obs.py), so the setup keeps the
    # cylinder's centre height whatever stands there; only the spawn pose stands the entry on the ground.
    from autofly_ue5.expert.episode import TARGET_Z_NED

    setup = _pool_setup(tmp_path, "grey boulder", ("orange cylinder", "grid pole", "white box", "orange cube", "grid cube"))
    assert setup.target_xy_z[2] == TARGET_Z_NED and all(d[2] == TARGET_Z_NED for d in setup.distractors)


def test_the_orange_cylinder_from_a_pool_spawns_exactly_as_the_pool_free_target(tmp_path):
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = scene_and_layout()
    plain = sample_setup(scene, layout, np.random.default_rng(3))
    pooled = _pool_setup(tmp_path, "orange cylinder", ("grid pole", "white box", "orange cube", "grid cube", "grey boulder"))
    _names, plain_sim = _spawned(plain)
    _names, pooled_sim = _spawned(pooled)
    assert pooled_sim.spawn_calls[0] == plain_sim.spawn_calls[0]


def test_an_entry_whose_material_the_runtime_cannot_apply_spawns_with_the_mesh_s_own(tmp_path):
    # The pool's loader refuses such an entry; the spawner keeps the runtime's rule on its own all the same.
    import dataclasses

    setup = _pool_setup(tmp_path, "orange cube", ("grid pole", "white box", "orange cylinder", "grid cube", "grey boulder"))
    setup = dataclasses.replace(setup, target=dataclasses.replace(setup.target, material="red"))
    _names, sim = _spawned(setup)
    assert sim.spawn_calls[0][4] is None


def test_spawn_spec_describes_each_pool_entry_by_registry_key(tmp_path):
    from autofly_ue5.expert.episode import spawn_spec

    setup = _pool_setup(tmp_path, "grey boulder", ("orange cylinder", "white box", "grid pole", "orange cube", "grid cube"))
    spec = spawn_spec(setup)
    assert spec["target"] == {"asset": "boulder_01", "material": None, "scale": [1.5, 1.5, 1.5], "z_ned": 0.0}
    assert spec["distractors"][:2] == [
        {"asset": "cylinder", "material": "orange", "scale": [1.0, 1.0, 2.0], "z_ned": -1.0},
        {"asset": "crate", "material": None, "scale": [2.0, 1.0, 1.0], "z_ned": -0.5},
    ]
    assert len(spec["distractors"]) == len(setup.distractors)


# --------------------------------------------------------------------------------------------------------
# A scene's own spawn clearance (2026-10-08): the sampler keeps starts, targets and distractors farther from every
# obstacle so a training margin can be wider (docs/decisions/2026-10-08-footprint-contact-rule.md, known limit 1).
# --------------------------------------------------------------------------------------------------------
def _edge_pillar_layout():
    """s01's bounds with pillars every 5 m along a line 4 m inside each edge: in the start band and next to the target
    band, so a good share of the points the sampler draws lands 1.4-1.7 m from a pillar (s01's own field stays 8 m in,
    where no start, target or distractor ever comes within 1.7 m of it)."""
    from autofly_ue5.scenes.model import Instance

    _scene, s01 = scene_and_layout()
    b, inset, r = s01.bounds, 4.0, 0.4
    along = [b.x_min + inset + 5.0 * k for k in range(13)]
    points = ([(x, b.y_min + inset) for x in along] + [(x, b.y_max - inset) for x in along]
              + [(b.x_min + inset, y) for y in along[1:-1]] + [(b.x_max - inset, y) for y in along[1:-1]])
    return dataclasses.replace(s01, instances=tuple(
        Instance(f"obs_{i:04d}", "cylinder", x, y, -5.0, 0.0, (2 * r, 2 * r, 10.0), "white", r, 10.0)
        for i, (x, y) in enumerate(points)))


def _points(setup):
    return [(setup.start.x, setup.start.y), setup.target_xy_z[:2]] + [d[:2] for d in setup.distractors]


def _nearest_gap(layout, xy):
    return min(math.hypot(xy[0] - inst.x, xy[1] - inst.y) - inst.radius_m for inst in layout.instances)


def test_the_scene_spawn_clearance_is_the_scenes_own_or_the_samplers():
    from autofly_ue5.expert.episode import scene_spawn_clearance_m, spawn_clearance_m

    scene, _layout = scene_and_layout()
    assert scene_spawn_clearance_m(scene) == spawn_clearance_m() == 1.4
    assert scene_spawn_clearance_m(dataclasses.replace(scene, spawn_clearance_m=1.7)) == 1.7
    with pytest.raises(ValueError, match="1.4"):  # a SceneFile built in code, past the schema
        scene_spawn_clearance_m(dataclasses.replace(scene, spawn_clearance_m=1.2))


def test_a_scene_at_the_samplers_own_clearance_draws_exactly_as_one_without_it():
    from autofly_ue5.expert.episode import sample_setup

    scene, _s01 = scene_and_layout()
    layout = _edge_pillar_layout()
    for seed in range(60):
        plain_rng, same_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        assert (sample_setup(dataclasses.replace(scene, spawn_clearance_m=1.4), layout, same_rng)
                == sample_setup(scene, layout, plain_rng))
        assert same_rng.bit_generator.state == plain_rng.bit_generator.state


def test_a_wider_spawn_clearance_only_rejects_closer_points_and_adds_no_draw():
    # The field adds no draw and moves none: it only rejects a candidate the default clearance would have accepted. So
    # a seed whose every placed point already clears 1.7 m draws the same episode with the same rng consumption, and a
    # seed with a point 1.4-1.7 m from a pillar draws another one, every point of it clear of 1.7 m.
    from autofly_ue5.expert.episode import sample_setup, spawn_clearance_m

    scene, _s01 = scene_and_layout()
    wide_scene = dataclasses.replace(scene, spawn_clearance_m=1.7)
    layout = _edge_pillar_layout()
    same = moved = 0
    for seed in range(60):
        plain_rng, wide_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        plain, wide = sample_setup(scene, layout, plain_rng), sample_setup(wide_scene, layout, wide_rng)
        assert all(_nearest_gap(layout, xy) >= 1.7 for xy in _points(wide)), f"seed {seed}"
        plain_gaps = [_nearest_gap(layout, xy) for xy in _points(plain)]
        assert min(plain_gaps) >= spawn_clearance_m()
        if min(plain_gaps) >= 1.7:
            assert wide == plain and wide_rng.bit_generator.state == plain_rng.bit_generator.state, f"seed {seed}"
            same += 1
        else:
            assert wide != plain, f"seed {seed}"
            moved += 1
    assert same >= 5 and moved >= 5, (same, moved)  # both branches exercised: the layout puts points in the ring


def test_the_mover_guard_keeps_the_layouts_acceptance_inflation(monkeypatch):
    # The path-length guard measures paths on the occupancy grid the layout was accepted on (spec §6.5, §6.2); a
    # scene's wider start clearance must not narrow that grid, or a layout proven connected only at 1.4 m could read
    # as unreachable and silently lose every mover.
    import autofly_ue5.expert.episode as episode
    from tests.test_dynamic_env import dynamic_scene

    scene, layout = dynamic_scene()
    seen = []
    real = episode.sample_movers

    def spy(*args, **kwargs):
        seen.append(kwargs["inflate_m"])
        return real(*args, **kwargs)

    monkeypatch.setattr(episode, "sample_movers", spy)
    episode.sample_setup(dataclasses.replace(scene, spawn_clearance_m=1.7), layout, np.random.default_rng(3))
    assert seen == [episode.spawn_clearance_m()] == [1.4]
