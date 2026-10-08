"""The task's contact rule for mesh obstacles (2026-10-07, plan 5 / M4 C8): in a scene whose obstacles are imported
models (trees, rocks), an episode ends as a collision when the drone's swept step passes within CIRCLE_CONTACT_M of
an obstacle's footprint circle, the band radius the layout already carries. A body that reaches wood can wedge the
simulator's swept actor beyond any teleport (docs/gates/m4_s02_wood_crash_reset_probe.json: 2 of 4 crashes), and a
wedged actor cannot fly, so the body must never reach it. The physical collision event stays as a backstop. The rule
is the task's (the gate scores it, the collector rejects by it); a training run may add a margin on top, as for
movers, because every expert flies to the boundary it trains on (MEMORY.md). Static s01 is pinned separately
(tests/test_static_s01_golden.py): a pillar scene has no footprint obstacle, so nothing there may change.

Two shapes since 2026-10-08 (afternoon). The rule first measured to the footprint circle; it now measures to the model's
outline (scenes/footprint_field.py), at OUTLINE_CONTACT_M = 1.05 m for the drone's measured collision reach, and the
circle (0.9 m) is kept so s02_r1's models can be evaluated under the rule they trained on. The tests written before the
outline pin the circle exactly as it was: their envs and rules are built with `footprint_shape="circle"`
(make_tree_env's and _rule's default here), and test_the_circle_shape_is_the_static_boundary_it_always_was holds it to
the original computation. The outline, the env's and the trainer's default, has its own tests in the last section, on a
synthetic leaning tree and on the real fields (assets/footprints/; they skip, saying so, on a checkout without them).
The trainer tests run its default, the outline; the gate's and the collector's read the shape from the model's run
(tests/test_run_shape.py), and their checkpoints outside any run name it."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autofly_ue5.scenes.model import DRONE_COLLISION_REACH_M, DRONE_HALF_SPAN_M, Instance, load_registry, load_scene_file
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import StartCollisionError

WOOD_RADIUS_M = 0.3  # the fake's physical trunk, well inside every tree's footprint circle: the backstop, never the rule


# --------------------------------------------------------------------------------------------------------
# A small tree scene for the fake: real registry assets, a seeded layout, trunks that collide.
# --------------------------------------------------------------------------------------------------------
def _tree_groups() -> list[dict]:
    group = lambda asset, count, dist: {  # noqa: E731
        "asset": asset, "count": count, "scale_range": {"xy": [0.9, 1.1], "z": [0.9, 1.1]}, "palette": ["mesh"],
        "placement": {"type": "poisson", "margin_m": 8.0, "min_distance_m": dist}}
    return [group("island_tree_02", 3, 12.0), group("searsia_lucida_b", 4, 8.0), group("quiver_tree_01", 4, 7.0)]


def tree_scene_dict(seed: int = 7002) -> dict:
    return {
        "id": "s99", "split": "train", "seed": seed,
        "bounds": {"x_min": -35.0, "x_max": 35.0, "y_min": -35.0, "y_max": 35.0}, "ground": "grass",
        "obstacle_groups": _tree_groups(),
        "start_band": [2.0, 6.0], "target_band": [0.0, 3.0], "altitude_band": [1.0, 3.0], "instruction_obstacle": "trees",
    }


def mixed_scene_dict() -> dict:
    """Pillars and trees in one scene: a grid of ten cylinders first (a grid cannot move its points), then the trees
    scattered around them."""
    pillars = {"asset": "cylinder", "count": 10, "scale_range": {"xy": [0.8, 1.2], "z": [8.0, 12.0]}, "palette": ["white"],
               "placement": {"type": "jittered_grid", "margin_m": 8.0, "jitter_m": 1.0}}
    return {**tree_scene_dict(), "obstacle_groups": [pillars, *_tree_groups()], "instruction_obstacle": "pillars and trees"}


_SCENES: dict[str, tuple] = {}  # each test scene and its layout, generated once per session
_SCENE_ROOT: list[Path] = []  # where this module's fixture puts the scene files: pytest's session directory


@pytest.fixture(scope="session", autouse=True)
def _scene_root(tmp_path_factory):
    """The scene JSONs go under pytest's own temporary directory (pruned with every other tmp_path), never a directory
    left behind in /tmp on every run."""
    _SCENE_ROOT.append(tmp_path_factory.mktemp("footprint_scenes"))
    yield
    _SCENE_ROOT.clear()


def _generated(key: str, data: dict):
    if key not in _SCENES:
        from autofly_ue5.scenes.generate import generate_layout

        with contextlib.ExitStack() as stack:
            # Another test module importing these helpers (test_footprint_rule_figure) runs without this module's
            # fixture: its file goes in a scratch directory removed once loaded, as nothing reads it again (scene.path
            # is only recorded).
            root = _SCENE_ROOT[0] if _SCENE_ROOT else Path(stack.enter_context(tempfile.TemporaryDirectory()))
            path = root / key / f"{data['id']}_test_trees.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            scene = load_scene_file(path)
        _SCENES[key] = (scene, generate_layout(scene, load_registry()))
    return _SCENES[key]


def tree_scene_and_layout(seed: int = 7002):
    scene, layout = _generated(f"trees_{seed}", tree_scene_dict(seed))
    assert len(layout.instances) == 11 and all(inst.material == "mesh" for inst in layout.instances)
    return scene, layout


def mixed_scene_and_layout():
    scene, layout = _generated("mixed", mixed_scene_dict())
    assert [inst.asset for inst in layout.instances[:10]] == ["cylinder"] * 10 and len(layout.instances) == 21
    return scene, layout


def wood(layout) -> list[tuple[float, float, float]]:
    """Each tree's trunk as a fake obstacle circle: physical contact, inside the footprint circle."""
    return [(inst.x, inst.y, WOOD_RADIUS_M * inst.scale[0]) for inst in layout.instances]


def outline_fields_for(layout):
    """The real outline fields of the layout's assets (assets/footprints/, built by scripts/footprint_fields.py and
    committed), or a skip saying so: a checkout without them cannot run the outline's tests."""
    from autofly_ue5.scenes.footprint_field import FootprintFields

    try:
        return FootprintFields.load({inst.asset for inst in layout.instances})
    except ValueError as err:
        pytest.skip(f"the outline fields are not on this checkout: {err}")


def outline_rules(fields, obstacles: int) -> dict:
    """task_contact_rules' record of the outline rule (1.05 m) over `obstacles` mesh obstacles measured with `fields`."""
    return {"footprint": {"contact_m": 1.05, "obstacles": obstacles, "shape": "outline", "fields": fields.digest(),
                          "fields_manifest_sha256": fields.manifest_sha256}}


def make_tree_env(sim_cls=FakeSimulator, obstacles=None, **kw):
    """The tree scene's env on the fake, under the circle unless `footprint_shape` says otherwise (see the docstring)."""
    from autofly_ue5.expert.env import AutoFlyEnv

    kw.setdefault("footprint_shape", "circle")
    scene, layout = tree_scene_and_layout()
    return AutoFlyEnv(scene, layout, functools.partial(sim_cls, obstacles=wood(layout) if obstacles is None else obstacles),
                      map_path="/Game/AutoFly/Maps/S99", instance=0, **kw)


def _trees(env) -> list[Instance]:
    return [inst for inst in env._layout.instances if inst.material == "mesh"]


def _nearest_tree(env, pose) -> Instance:
    return min(_trees(env), key=lambda inst: math.hypot(inst.x - pose.x, inst.y - pose.y) - inst.radius_m)


def _toward(pose, target: Instance, speed: float = 2.0) -> np.ndarray:
    """Turn on the spot until the target's pivot is ahead, then fly at it at `speed`."""
    bearing = math.atan2(target.y - pose.y, target.x - pose.x) - pose.yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    return np.array([speed if abs(bearing) < 0.2 else 0.0, float(np.clip(2.0 * bearing, -1, 1)), 0.0], np.float32)


def _fly_at_a_tree(env, seed, speed: float = 2.0):
    """Turn to the nearest tree's pivot and fly straight at it; returns each step's (reward, info) and observation."""
    obs, _info = env.reset(seed=seed)
    target = _nearest_tree(env, env._last_pose)
    steps, observations = [], [obs]
    for _ in range(300):
        obs, reward, terminated, truncated, info = env.step(_toward(env._last_pose, target, speed))
        steps.append((reward, info))
        observations.append(obs)
        if terminated or truncated:
            return steps, observations
    raise AssertionError("never ended")


class AtTheNearestTree:
    """A scripted pilot for harnesses that build their own env (the gate, the collector): it reads the drone's pose
    from the newest fake simulator `sims` holds and flies as _fly_at_a_tree does, at whichever circle is nearest.
    Turning happens on the spot, so the nearest tree cannot change while it turns."""

    def __init__(self, layout, sims: list):
        self._trees = [inst for inst in layout.instances if inst.material == "mesh"]
        self._sims = sims

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        pose = self._sims[-1]._pose
        target = min(self._trees, key=lambda inst: math.hypot(inst.x - pose.x, inst.y - pose.y) - inst.radius_m)
        return _toward(pose, target), None


def recording_tree_factory(layout, sims: list):
    """A FakeSimulator factory with the trees' wood that keeps each simulator it makes, for AtTheNearestTree."""

    def factory():
        sims.append(FakeSimulator(obstacles=wood(layout)))
        return sims[-1]

    return factory


SEEDS = range(1_000_040, 1_000_050)


# --------------------------------------------------------------------------------------------------------
# The rule itself (pure).
# --------------------------------------------------------------------------------------------------------
def _rule(**kw):
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    trees = (Instance("t0", "island_tree_02", 0.0, 0.0, 0.0, 0.0, (1.0, 1.0, 1.0), "mesh", 3.0, 3.4),
             Instance("t1", "quiver_tree_01", 20.0, 0.0, 0.0, 0.0, (1.0, 1.0, 1.0), "mesh", 0.7, 2.7))
    return FootprintContactRule(trees, **{"shape": "circle", **kw})  # made-up radii: the circle's geometry


def test_the_contact_distance_keeps_the_rotor_span_outside_the_circle_on_the_ending_step_too():
    # The circle contains every branch the band meets; the drone's rotor tips reach DRONE_HALF_SPAN_M from its centre;
    # the rule is checked after the step, so the ending step can carry the centre a full step (0.4 m) past the boundary;
    # a full-rate turn at full speed bows that step's chord by 2 (1 - cos 0.1) = 0.01 m. The rule must cover all three.
    # (0.5 m, the first version, let 40 of 40 scripted flights end with a rotor tip inside the circle.) The circle keeps
    # this, s02_r1's rule, exactly: its slack (it reaches past the outline everywhere but at the model's widest point)
    # covers the farther reach of the drone's collision below.
    from autofly_ue5.expert.footprint_contact import CIRCLE_CONTACT_M, contact_distance_m
    from autofly_ue5.expert.movers import FORWARD_SPEED_MAX_M_S
    from autofly_ue5.sim.types import CONTROL_DT_S

    assert CIRCLE_CONTACT_M - FORWARD_SPEED_MAX_M_S * CONTROL_DT_S >= DRONE_HALF_SPAN_M + 0.01
    assert CIRCLE_CONTACT_M == 0.9 and contact_distance_m("circle") == 0.9


def test_the_outlines_contact_distance_keeps_the_drones_measured_collision_reach_off_it_on_the_ending_step_too():
    # Physical collisions register farther out than the rotor tips: the drone's centre was 0.51 m from a pillar's
    # surface at the mover probe's contact (d1, docs/decisions/2026-10-02-dynamic-obstacles.md) and 0.52-0.63 m at run
    # 5's gate static collisions (docs/decisions/2026-10-06-s01d-r6-plan.md). The circle hid that in its slack; on a
    # trunk-side approach the outline has none, so its distance keeps a full-speed ending step (0.4 m) and a full-rate
    # turn's chord bow (0.01 m) outside the measured reach: 1.05 - 0.41 = 0.64 >= 0.63. Neither the ending step nor the
    # reset's straight climb from where it ended can then reach the model.
    from autofly_ue5.expert.footprint_contact import CHORD_BOW_M, OUTLINE_CONTACT_M, contact_distance_m
    from autofly_ue5.expert.movers import FORWARD_SPEED_MAX_M_S
    from autofly_ue5.scenes.model import DRONE_COLLISION_REACH_M
    from autofly_ue5.sim.types import CONTROL_DT_S

    assert DRONE_COLLISION_REACH_M == 0.63 and DRONE_HALF_SPAN_M == 0.48, "the mover invariant keeps the rotor span"
    assert OUTLINE_CONTACT_M == 1.05 and contact_distance_m("outline") == 1.05
    assert OUTLINE_CONTACT_M - FORWARD_SPEED_MAX_M_S * CONTROL_DT_S - CHORD_BOW_M >= DRONE_COLLISION_REACH_M
    with pytest.raises(ValueError, match="shape"):
        contact_distance_m("square")


def test_the_outlines_contact_distance_lies_inside_every_start_the_sampler_draws():
    # Starts keep 1.4 m from every circle (and so from every outline inside it), and a reset may land 0.3 m off: the
    # task's own rule must never meet a start, or the gate's episodes would begin inside it.
    from autofly_ue5.expert.episode import spawn_clearance_m
    from autofly_ue5.expert.footprint_contact import OUTLINE_CONTACT_M
    from autofly_ue5.sim.airsim_backend import RESET_POSITION_TOLERANCE_M

    assert OUTLINE_CONTACT_M <= spawn_clearance_m() - RESET_POSITION_TOLERANCE_M


def test_each_shape_ends_and_records_at_its_own_distance(tmp_path):
    from autofly_ue5.expert.footprint_contact import FootprintContactRule, task_contact_rules

    _registry, fields, inst = lean_tree(tmp_path)
    outline = FootprintContactRule([inst], fields=fields)
    wide = FootprintContactRule([inst], fields=fields, training_margin_m=0.2)
    circle = FootprintContactRule([inst], shape="circle", training_margin_m=0.2)
    assert (outline.contact_m, outline.termination_m) == (1.05, 1.05)
    assert wide.contact_m == 1.05 and wide.termination_m == pytest.approx(1.25)
    assert circle.contact_m == 0.9 and circle.termination_m == pytest.approx(1.1)
    assert wide.to_json()["contact_m"] == 1.05 and circle.to_json()["contact_m"] == 0.9
    assert wide.source(1.05) == "footprint" and wide.source(1.06) == "footprint_margin"
    a, b = (-0.05 - 1.0, -0.2), (-0.05 - 1.0, 0.2)  # 1.0 m off the trunk's face, on the side away from the crown
    assert outline.contact(a, b) is not None and outline.contact(a, b)[1] == pytest.approx(1.0, abs=0.015)
    _scene, layout = tree_scene_and_layout()
    assert task_contact_rules(layout, shape="outline")["footprint"]["contact_m"] == 1.05
    assert task_contact_rules(layout, shape="circle")["footprint"]["contact_m"] == 0.9


@pytest.mark.parametrize("gap, fires", [(1.2, False), (0.9001, False), (0.9, True), (0.2, True), (-0.5, True)])
def test_a_point_fires_within_the_contact_distance_of_the_circle(gap, fires):
    point = (-3.0 - gap, 0.0)  # t0's circle has radius 3.0: `gap` from its edge
    hit = _rule().contact(point, point)
    assert (hit is not None) == fires
    if fires:
        assert hit[0] == "t0" and hit[1] == pytest.approx(gap)


def test_a_step_that_sweeps_past_the_circle_fires_with_both_ends_outside():
    hit = _rule().contact((-6.0, 3.4), (6.0, 3.4))  # 0.4 m from the edge at the closest point, 3 m away at both ends
    assert hit is not None and hit[0] == "t0" and hit[1] == pytest.approx(0.4)
    assert _rule().contact((-6.0, 4.0), (6.0, 4.0)) is None  # 1.0 m at the closest point


def test_the_training_margin_widens_where_it_fires_and_each_ending_names_its_rule():
    plain, wide = _rule(), _rule(training_margin_m=0.2)
    assert plain.termination_m == pytest.approx(0.9) and wide.termination_m == pytest.approx(1.1)
    for gap, fires_plain, fires_wide in ((1.2, False, False), (1.0, False, True), (0.4, True, True)):
        point = (-3.0 - gap, 0.0)
        assert (plain.contact(point, point) is not None) == fires_plain, gap
        assert (wide.contact(point, point) is not None) == fires_wide, gap
    assert wide.source(0.4) == "footprint" and wide.source(0.9) == "footprint"
    assert wide.source(1.0) == "footprint_margin"
    assert plain.contact_m == wide.contact_m, "the task's rule is the same whatever the margin"


def test_nearest_gap_measures_from_the_circle():
    assert _rule().nearest_gap((-5.0, 0.0)) == pytest.approx(2.0)
    assert _rule().nearest_gap((20.0, 2.0)) == pytest.approx(1.3)


def test_an_episodes_movers_are_left_out():
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    _scene, layout = tree_scene_and_layout()
    moving = {layout.instances[0].tag}
    rule = FootprintContactRule.for_episode(layout.instances, moving)
    assert set(rule.tags) == {inst.tag for inst in layout.instances} - moving
    assert set(FootprintContactRule.for_episode(layout.instances, ()).tags) == {inst.tag for inst in layout.instances}


@pytest.mark.parametrize("kw", [{"training_margin_m": -0.1}, {"training_margin_m": math.nan}, {"training_margin_m": math.inf},
                                {"contact_m": 0.0}, {"contact_m": -1.0}])
def test_a_distance_that_could_not_mean_anything_is_refused(kw):
    with pytest.raises(ValueError):
        _rule(**kw)


def test_the_rule_serialises_what_a_record_needs():
    rule = _rule(training_margin_m=0.2)
    assert rule.to_json() == {"contact_m": 0.9, "training_margin_m": 0.2, "termination_m": 1.1, "obstacles": 2,
                              "shape": "circle"}


# --------------------------------------------------------------------------------------------------------
# Which obstacles the rule covers: measured models, never engine primitives.
# --------------------------------------------------------------------------------------------------------
def test_only_measured_models_are_footprint_obstacles():
    from autofly_ue5.expert.footprint_contact import footprint_obstacles, task_contact_rules
    from tests.test_expert_episode import scene_and_layout

    registry = load_registry()
    _scene, trees = tree_scene_and_layout()
    assert footprint_obstacles(trees.instances, registry) == trees.instances
    _scene, pillars = scene_and_layout()
    assert footprint_obstacles(pillars.instances, registry) == ()
    boxes = (Instance("b0", "crate", 0.0, 0.0, -0.5, 0.3, (1.0, 1.0, 1.0), "wood", 0.7071, 1.0),)
    assert footprint_obstacles(boxes, registry) == (), "a box is a convex solid: physical contact, which never wedges"
    assert task_contact_rules(pillars) is None
    assert task_contact_rules(trees, shape="circle") == {"footprint": {"contact_m": 0.9, "obstacles": 11, "shape": "circle"}}


def test_a_measured_model_of_another_role_in_a_layout_is_refused():
    # Targets are measured too (a radius profile sizes them, plan 5 C5) but are spawned per episode, never placed in a
    # layout, and scripts/footprint_fields.py builds fields for the obstacle role only. One definition of a mesh
    # obstacle, then: a measured model in the obstacle role. A target in a layout is a mistake upstream, refused saying
    # so, rather than read as a footprint obstacle (or refused under the outline for a field it was never meant to have).
    from autofly_ue5.expert.footprint_contact import footprint_obstacles, is_footprint_obstacle

    registry = load_registry()
    target = dataclasses.replace(registry.assets["quiver_tree_01"], name="tgt_quiver", role="target")
    with_target = dataclasses.replace(registry, assets={**registry.assets, "tgt_quiver": target})
    placed = (Instance("obs_0007", "tgt_quiver", 0.0, 0.0, 0.0, 0.0, (1.0, 1.0, 1.0), "mesh", 0.7, 2.7),)
    with pytest.raises(ValueError, match="obs_0007 uses asset 'tgt_quiver', a measured model in the 'target' role"):
        footprint_obstacles(placed, with_target)
    assert not is_footprint_obstacle(target) and is_footprint_obstacle(registry.assets["quiver_tree_01"])
    primitive = dataclasses.replace(registry.assets["cylinder"], role="target")  # the orange cylinder: no profile
    assert not is_footprint_obstacle(primitive)


def test_an_asset_the_registry_does_not_know_is_refused():
    # A ValueError, which the trainer's pre-claim checks report as a refusal (a KeyError escaped them as a traceback).
    from autofly_ue5.expert.footprint_contact import footprint_obstacles

    unknown = (Instance("x", "no_such_asset", 0.0, 0.0, 0.0, 0.0, (1.0, 1.0, 1.0), "mesh", 1.0, 1.0),)
    with pytest.raises(ValueError, match="no_such_asset"):
        footprint_obstacles(unknown, load_registry())


# --------------------------------------------------------------------------------------------------------
# The environment.
# --------------------------------------------------------------------------------------------------------
def test_a_pillar_scene_has_no_footprint_rule():
    # Static s01 is pinned by test_static_s01_golden; this says why nothing there can change: no rule exists for it.
    from tests.test_expert_env import make_env

    env = make_env()
    assert env.contact_rules is None
    env.reset(seed=3)
    assert env._footprint is None
    _obs, _r, _t, _tr, info = env.step(np.array([1.0, 0.0, 0.0], np.float32))
    assert not any(k.startswith("footprint") for k in info)


def test_a_tree_scene_names_its_rule():
    env = make_tree_env()
    assert env.contact_rules == {"footprint": {"contact_m": 0.9, "training_margin_m": 0.0, "termination_m": 0.9, "obstacles": 11,
                                               "shape": "circle"}}
    assert make_tree_env(footprint_contact_margin_m=0.2).contact_rules["footprint"]["termination_m"] == pytest.approx(1.1)


def test_flying_at_a_tree_ends_at_the_footprint_circle_and_never_reaches_wood():
    env = make_tree_env()
    endings = 0
    for seed in SEEDS:
        steps, _obs = _fly_at_a_tree(env, seed)
        reward, end = steps[-1]
        assert end["collision_source"] != "sim", f"seed {seed}: the body reached the fake's trunk"
        if end["collision_source"] != "footprint":
            continue
        endings += 1
        contact = end["footprint_contact"]
        assert end["outcome"] == "collision" and end["is_success"] is False
        assert contact["gap_m"] <= 0.9 + 1e-9 and contact["gap_m"] > 0.9 - 0.41, "a full-speed step covers 0.4 m"
        assert contact["gap_m"] >= DRONE_HALF_SPAN_M, "no rotor tip inside the circle, on the ending step either"
        assert contact["tag"] == _nearest_tree(env, env._last_pose).tag and contact["asset"] == env._instances[contact["tag"]].asset
        assert contact["rule"] == "footprint" and -180.0 <= contact["bearing_deg"] <= 180.0
        assert reward <= -10.0 + 0.5, "scored as the collision it stands for (r_collision 10)"
    assert endings >= 7, f"flying straight at a tree must end at the circle ({endings} of {len(SEEDS)})"


def test_the_physical_backstop_still_scores_a_body_that_reaches_wood(monkeypatch):
    # Were the rule ever to miss (a branch outside its circle), the simulator's own event still ends the episode.
    import autofly_ue5.expert.env as env_module
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    class Blind(FootprintContactRule):
        def contact(self, a_xy, b_xy):
            return None

    monkeypatch.setattr(env_module, "FootprintContactRule", Blind)
    env = make_tree_env()
    sim_ends = 0
    for seed in SEEDS:
        steps, _ = _fly_at_a_tree(env, seed)
        end = steps[-1][1]
        if end["collision_source"] == "sim":
            sim_ends += 1
            assert "footprint_contact" not in end
            nearest = end["footprint_nearest"]
            assert nearest["tag"] == _nearest_tree(env, env._last_pose).tag, "the trunk the fake's wood is"
            assert nearest["gap_m"] < 0, "a rule miss reads as a centre inside the circle"
    assert sim_ends >= 7


def test_the_training_margin_ends_sooner_and_sees_the_same_on_the_way():
    # At 0.5 m/s a step covers 0.1 m, less than the 0.2 m margin band, so a flight straight in always ends inside the
    # band: at full speed (0.4 m) about half the ending steps jump it and cross both lines at once.
    plain, wide = make_tree_env(), make_tree_env(footprint_contact_margin_m=0.2)
    margin_hits = 0
    for seed in SEEDS:
        (a, obs_a), (b, obs_b) = _fly_at_a_tree(plain, seed, speed=0.5), _fly_at_a_tree(wide, seed, speed=0.5)
        end_a, end_b = a[-1][1], b[-1][1]
        assert len(b) <= len(a), "the wider rule can only end an episode sooner"
        for (r_a, i_a), (r_b, i_b) in zip(a[:len(b) - 1], b[:len(b) - 1]):
            assert (r_a, i_a) == (r_b, i_b), "every step before the margin's is untouched"
        for o_a, o_b in zip(obs_a[:len(b)], obs_b[:len(b)]):
            assert all(np.array_equal(o_a[k], o_b[k]) for k in o_a), "what the expert saw on the way: the same flight"
        # Under a margin, "footprint" names a step that also crossed the task's line, so the trainer's
        # outcomes/collision_footprint_margin counts only the endings the margin alone caused.
        if end_b["collision_source"] != "footprint_margin":
            assert a == b, "a flight the margin never ended is the plain flight, step for step"
            continue
        margin_hits += 1
        contact = end_b["footprint_contact"]
        assert 0.9 < contact["gap_m"] <= 1.1 + 1e-9 and contact["rule"] == "footprint_margin"
        assert end_b["outcome"] == "collision" and b[-1][0] <= -10.0 + 0.5
        assert end_a["collision_source"] == "footprint", "without the margin the same flight ends at the task's rule"
    assert margin_hits >= 8, f"slow flights straight in end in the margin band ({margin_hits} of {len(SEEDS)})"


def _swollen_wood(layout, beyond_circle_m: float = 1.0) -> list[tuple[float, float, float]]:
    """Physical wood reaching `beyond_circle_m` outside each footprint circle: 0.1 m outside the rule's 0.9 m line, so
    the simulator's own event ends the flight first, on a step that may or may not also cross the rule."""
    return [(inst.x, inst.y, inst.radius_m + beyond_circle_m) for inst in layout.instances]


def test_a_physical_ending_names_the_nearest_circle_so_wood_a_spawn_or_a_rule_miss_can_be_told_apart():
    _scene, layout = tree_scene_and_layout()
    env = make_tree_env(obstacles=_swollen_wood(layout))
    sim_ends = both_crossed = 0
    for seed in SEEDS:
        steps, _ = _fly_at_a_tree(env, seed)
        end = steps[-1][1]
        if end["collision_source"] != "sim":
            continue
        sim_ends += 1
        tree = _nearest_tree(env, env._last_pose)
        gap = math.hypot(tree.x - env._last_pose.x, tree.y - env._last_pose.y) - tree.radius_m
        assert end["footprint_nearest"] == {"tag": tree.tag, "asset": tree.asset, "gap_m": pytest.approx(gap, abs=1e-3),
                                            "shape": "circle", "circle_gap_m": end["footprint_nearest"]["gap_m"]}
        assert end["footprint_nearest"]["gap_m"] <= 1.0 + 1e-9, "inside the fake's wood"
        assert "footprint_contact" not in end, "a physical contact is scored as the contact it is"
        if gap <= 0.9:  # this step crossed the rule's line too: still the simulator's ending, scored as one
            both_crossed += 1
            assert end["outcome"] == "collision" and steps[-1][0] <= -10.0 + 0.5
    assert sim_ends >= 7 and both_crossed >= 1, (sim_ends, both_crossed)
    for seed in SEEDS:  # every other step of a tree scene, and every step that ended otherwise, carries no such key
        steps, _ = _fly_at_a_tree(make_tree_env(), seed)
        assert all("footprint_nearest" not in info for _reward, info in steps)


def test_a_step_after_a_reset_that_did_not_complete_is_refused(monkeypatch):
    # reset() stores the episode before its start checks; a step after one of them refused the start would fly without
    # the rule, so a body could reach wood unscored.
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    env = make_tree_env()
    env.reset(seed=1_000_041)
    with monkeypatch.context() as m:
        m.setattr(FootprintContactRule, "nearest_gap", lambda self, xy: 0.45)
        with pytest.raises(StartCollisionError):
            env.reset(seed=1_000_042)
    with pytest.raises(RuntimeError, match="reset\\(\\) did not complete"):
        env.step(np.array([1.0, 0.0, 0.0], np.float32))
    env.reset(seed=1_000_042)
    env.step(np.array([1.0, 0.0, 0.0], np.float32))  # a completed reset flies again


def test_a_scene_with_moving_pillars_and_mesh_obstacles_is_refused():
    # Moving a mesh obstacle would leave its <tag>_cards actor (the leaves, no collision) behind, and how the mover rule
    # and the footprint rule share an obstacle is not designed: refused until it is.
    from autofly_ue5.expert.env import AutoFlyEnv
    from tests.test_dynamic_env import dynamic_scene

    scene, layout = tree_scene_and_layout()
    moving = dataclasses.replace(scene, dynamic=dynamic_scene()[0].dynamic)
    with pytest.raises(ValueError, match="_cards"):
        AutoFlyEnv(moving, layout, functools.partial(FakeSimulator, obstacles=wood(layout)),
                   map_path="/Game/AutoFly/Maps/S99", instance=0)


def test_a_static_boundary_is_refused_on_a_scene_of_mesh_obstacles_only():
    # The boundary covers pillars and boxes; on a scene of trees alone it would end nothing, yet the run's identity
    # would name it.
    with pytest.raises(ValueError, match="no pillar or box"):
        make_tree_env(static_contact_m=1.1)
    make_tree_env(static_contact_m=0.0)


def _make_mixed_env(**kw):
    from autofly_ue5.expert.env import AutoFlyEnv

    kw.setdefault("footprint_shape", "circle")
    scene, layout = mixed_scene_and_layout()
    physical = [(i.x, i.y, WOOD_RADIUS_M * i.scale[0] if i.material == "mesh" else i.radius_m) for i in layout.instances]
    return AutoFlyEnv(scene, layout, functools.partial(FakeSimulator, obstacles=physical),
                      map_path="/Game/AutoFly/Maps/S99", instance=0, **kw)


def test_a_mixed_scene_splits_pillars_and_trees_between_the_two_rules():
    env = _make_mixed_env(static_contact_m=1.1)
    env.reset(seed=1_000_041)
    pillars = {inst.tag for inst in env._layout.instances if inst.asset == "cylinder"}
    trees = {inst.tag for inst in env._layout.instances if inst.material == "mesh"}
    assert len(pillars) == 10 and len(trees) == 11
    assert set(env._static.tags) == pillars and set(env._footprint.tags) == trees


def test_on_a_step_that_crosses_a_tree_circle_and_a_pillar_boundary_the_footprint_rule_wins(monkeypatch):
    from autofly_ue5.expert.footprint_contact import FootprintContactRule
    from autofly_ue5.expert.static_contact import StaticContactBoundary

    hit = {"now": False}
    real = FootprintContactRule.contact

    def footprint_spy(self, a_xy, b_xy):
        result = real(self, a_xy, b_xy)
        hit["now"] = result is not None
        return result

    monkeypatch.setattr(FootprintContactRule, "contact", footprint_spy)
    monkeypatch.setattr(StaticContactBoundary, "contact",
                        lambda self, a, b: (self.tags[0], 0.5) if hit["now"] else None)  # fires only with a tree
    env = _make_mixed_env(static_contact_m=1.1)
    endings = 0
    for seed in SEEDS:
        steps, _ = _fly_at_a_tree(env, seed)
        end = steps[-1][1]
        assert end["collision_source"] != "static_margin", f"seed {seed}: the boundary fired alone"
        if end["collision_source"] == "footprint":
            endings += 1
            assert "static_contact" not in end and end["footprint_contact"]["tag"] in env._footprint.tags
    assert endings >= 5, endings


def test_a_start_inside_the_rule_is_retried_by_the_wrapper_and_the_episode_starts_normally(tmp_path, monkeypatch):
    from autofly_ue5.expert.footprint_contact import FootprintContactRule
    from autofly_ue5.expert.resilient import ResilientAutoFlyEnv

    real = FootprintContactRule.nearest_gap
    calls = []

    def once_inside(self, xy):
        calls.append(xy)
        return 0.45 if len(calls) == 1 else real(self, xy)

    monkeypatch.setattr(FootprintContactRule, "nearest_gap", once_inside)
    env = ResilientAutoFlyEnv(make_tree_env(), instance=0, sim_root=tmp_path / "sim")
    obs, info = env.reset(seed=1_000_041)
    summary = env.get_fault_summary()
    assert summary["fault_counts"]["StartCollisionError"] == 1 and summary["recovered_counts"]["StartCollisionError"] == 1
    assert summary["relaunch_count"] == 0 and len(calls) == 2
    assert info["outcome"] == "running" and env.unwrapped._footprint is not None
    _obs, _r, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0], np.float32))
    assert info["outcome"] == "running" and not (terminated or truncated)


def test_each_step_checks_the_path_it_flew_not_only_where_it_ended(monkeypatch):
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    real = FootprintContactRule.contact
    for seed in SEEDS:
        calls = []

        def spy(self, a_xy, b_xy):
            calls.append((tuple(a_xy), tuple(b_xy)))
            return real(self, a_xy, b_xy)

        monkeypatch.setattr(FootprintContactRule, "contact", spy)
        steps, _ = _fly_at_a_tree(make_tree_env(), seed)
        if steps[-1][1]["collision_source"] != "footprint":
            continue
        ends = [tuple(info["pose"][:2]) for _reward, info in steps]
        assert len(calls) == len(steps)
        for (_a, b), end in zip(calls, ends):
            assert b == pytest.approx(end)
        for (a, _b), previous_end in zip(calls[1:], ends[:-1]):
            assert a == pytest.approx(previous_end)
        assert sum(math.dist(a, b) > 0.3 for a, b in calls) >= 3, "full-speed steps are checked as segments"
        return
    raise AssertionError("no flight ended at the circle")


def test_a_physical_contact_in_a_mesh_scene_is_logged_with_the_nearest_circle(capsys):
    # A "sim" ending in a tree scene is what the rule exists to prevent; training keeps counts only, so the log line
    # is the only way a 12 h run can say what the body hit (a spawned cylinder, wood outside its circle, or a miss).
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    _scene, layout = tree_scene_and_layout()
    monkeypatch_rule = FootprintContactRule.contact
    FootprintContactRule.contact = lambda self, a, b: None  # the rule never fires: the fake's wood ends the flight
    try:
        env = make_tree_env()
        for seed in SEEDS:
            steps, _ = _fly_at_a_tree(env, seed)
            end = steps[-1][1]
            if end["collision_source"] == "sim":
                near = end["footprint_nearest"]
                lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("SIM-CONTACT")]
                assert lines[-1] == (f"SIM-CONTACT instance 0: physical contact with the nearest circle {near['tag']} "
                                     f"({near['asset']}) {near['gap_m']:.2f} m away, forward 2.00 m/s")
                return
        raise AssertionError("no flight reached the fake's wood")
    finally:
        FootprintContactRule.contact = monkeypatch_rule


def test_each_ending_is_logged_with_where_it_happened(capsys):
    env = make_tree_env()
    for seed in SEEDS:
        steps, _ = _fly_at_a_tree(env, seed)
        if steps[-1][1]["collision_source"] == "footprint":
            c = steps[-1][1]["footprint_contact"]
            lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("FOOTPRINT")]
            assert lines == [f"FOOTPRINT instance 0: {c['tag']} ({c['asset']}) at {c['gap_m']:.2f} m from its circle "
                             f"({c['rule']}), bearing {c['bearing_deg']:+.0f} deg, forward {c['drone_forward_m_s']:.2f} m/s"]
            return
    raise AssertionError("no flight ended at the circle")


def test_the_margin_is_refused_on_a_scene_without_mesh_obstacles():
    from tests.test_expert_env import make_env

    with pytest.raises(ValueError, match="no mesh obstacle"):
        make_env(footprint_contact_margin_m=0.2)


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf])
def test_the_env_refuses_a_margin_that_could_not_mean_anything(value):
    with pytest.raises(ValueError, match="footprint"):
        make_tree_env(footprint_contact_margin_m=value)


def test_a_start_inside_the_rule_is_retried_not_charged(monkeypatch):
    # Starts keep 1.4 m from every footprint circle (episode.spawn_clearance_m), so the rule never meets one; a reset
    # that landed inside it would lose its first step to an unavoidable -10, so it is retried like a start in contact.
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    monkeypatch.setattr(FootprintContactRule, "nearest_gap", lambda self, xy: 0.45)
    with pytest.raises(StartCollisionError, match="footprint"):
        make_tree_env().reset(seed=1_000_070)


def edge_tree_scene_and_layout(**fields):
    """Trees down to 4 m from the edges, inside the start band, so a share of starts lands near a circle (the main tree
    scene keeps every circle 4.8 m or more from any start drawn, so no start ever comes near one). `fields`: top-level
    scene fields to add, e.g. spawn_clearance_m."""
    group = lambda asset, count: {  # noqa: E731
        "asset": asset, "count": count, "scale_range": {"xy": [0.9, 1.1], "z": [0.9, 1.1]}, "palette": ["mesh"],
        "placement": {"type": "poisson", "margin_m": 4.0, "min_distance_m": 6.0}}
    groups = [group("searsia_lucida_b", 40), group("quiver_tree_01", 20)]
    data = {**tree_scene_dict(), "obstacle_groups": groups, **fields}
    return _generated("edge_trees_" + "_".join(f"{k}{v}" for k, v in sorted(fields.items())), data)


class LandsOffTowardTheNearestCircle(FakeSimulator):
    """A reset that lands the backend's whole tolerance (RESET_POSITION_TOLERANCE_M, which it accepts) off its pose,
    straight toward the nearest footprint circle: the worst start the trainer's margin ceiling allows for."""

    def __init__(self, circles, **kw):
        super().__init__(**kw)
        self._circles = circles

    def reset(self, pose):
        from autofly_ue5.sim.airsim_backend import RESET_POSITION_TOLERANCE_M
        from autofly_ue5.sim.types import Pose

        c = min(self._circles, key=lambda i: math.hypot(i.x - pose.x, i.y - pose.y) - i.radius_m)
        d = math.hypot(c.x - pose.x, c.y - pose.y)
        k = RESET_POSITION_TOLERANCE_M / d
        return super().reset(Pose(pose.x + k * (c.x - pose.x), pose.y + k * (c.y - pose.y), pose.z, pose.yaw))


def test_starts_at_a_scenes_wider_clearance_never_meet_the_margin_it_allows_even_landing_off_toward_a_tree():
    # spawn_clearance_m 1.7 lets a run train at 0.9 + 0.5 = 1.4 m = 1.7 - 0.3 (train.py's ceiling). Twenty seeds whose
    # start the default 1.4 m sampler puts 1.4-1.7 m from a circle, each reset landing 0.3 m toward that circle: every
    # start is drawn clear of 1.7 m instead, lands outside the 1.4 m boundary, and none is lost as a
    # StartCollisionError.
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = edge_tree_scene_and_layout(spawn_clearance_m=1.7)
    assert scene.spawn_clearance_m == 1.7 and layout == edge_tree_scene_and_layout()[1]

    def gap(xy):
        return min(math.hypot(xy[0] - i.x, xy[1] - i.y) - i.radius_m for i in layout.instances)

    default = dataclasses.replace(scene, spawn_clearance_m=None)
    starts = ((seed, sample_setup(default, layout, np.random.default_rng(seed)).start)
              for seed in range(1_000_000, 1_002_000))
    seeds = [seed for seed, start in starts if 1.4 <= gap((start.x, start.y)) < 1.7][:20]
    assert len(seeds) == 20
    env = AutoFlyEnv(scene, layout, functools.partial(LandsOffTowardTheNearestCircle, layout.instances,
                                                      obstacles=wood(layout)),
                     map_path="/Game/AutoFly/Maps/S99", instance=0, footprint_contact_margin_m=0.5,
                     footprint_shape="circle")
    for seed in seeds:
        env.reset(seed=seed)  # a StartCollisionError here fails the test
        drawn, landed = gap((env._setup.start.x, env._setup.start.y)), gap((env._last_pose.x, env._last_pose.y))
        assert drawn >= 1.7 and 1.4 < landed <= drawn - 0.3 + 1e-6, (seed, drawn, landed)


def test_the_outcome_callback_logs_footprint_collisions_on_their_own_curves():
    from types import SimpleNamespace

    from stable_baselines3.common.logger import Logger

    from autofly_ue5.expert.train import OutcomeHistogramCallback

    logger = Logger(folder=None, output_formats=[])
    cb = OutcomeHistogramCallback()
    cb.model = SimpleNamespace(logger=logger)
    for info in ({"outcome": "collision", "collision_source": "footprint"},
                 {"outcome": "collision", "collision_source": "footprint_margin"}, {"outcome": "success"}):
        cb.update_locals({"infos": [info], "dones": [True]})
        cb._on_step()
    values = logger.name_to_value
    assert values["outcomes/collision_footprint"] == pytest.approx(1 / 3)
    assert values["outcomes/collision_footprint_margin"] == pytest.approx(1 / 3)
    assert cb.collision_sources == {"footprint": 1, "footprint_margin": 1}


def test_the_vec_env_hands_it_to_every_worker(tmp_path):
    from autofly_ue5.expert.vec import make_vec_env

    scene, layout = tree_scene_and_layout()
    fields = outline_fields_for(layout)
    for kw, shape in (({}, "outline"), ({"footprint_shape": "circle"}, "circle")):
        vec = make_vec_env(scene, layout, 1, map_path="/Game/AutoFly/Maps/S99", monitor_dir=tmp_path / "m",
                           sim_factory=functools.partial(FakeSimulator, obstacles=wood(layout)), sim_root=tmp_path / "sim",
                           footprint_contact_margin_m=0.2, footprint_fields=fields, **kw)
        try:
            env = vec.envs[0].unwrapped
            assert env._footprint_contact_margin_m == 0.2 and env._footprint_shape == shape
            assert env._footprint_fields is (fields if shape == "outline" else None)
        finally:
            vec.close()


# --------------------------------------------------------------------------------------------------------
# The trainer, the gate and the watcher.
# --------------------------------------------------------------------------------------------------------
def _tree_resolved(tmp_path, scene=None, layout=None):
    from autofly_ue5.scenes.resolve import ResolvedScene

    tree_scene, tree_layout = tree_scene_and_layout()
    scene, layout = scene or tree_scene, layout or tree_layout
    return ResolvedScene(scene=scene, base_scene=scene, layout=layout, layout_path=tmp_path / "s99.layout.json",
                         layout_sha256="0" * 64, map_path="/Game/AutoFly/Maps/S99",
                         default_scene_config="scene_autofly_s02_fast.jsonc")


def _train(tmp_path, monkeypatch, *flags, trees=True, scene=None, layout=None, registries=None, fields=None,
           shapes=None, scenes=None):
    """train.main() on the tree scene (or s01), or on `scene` / `layout` in its place; `registries`, `fields`, `shapes`
    and `scenes` collect the registry, the outline fields, the footprint shape and the scene each make_vec_env call was
    handed (training's first, evaluation's second)."""
    from autofly_ue5.expert import train
    from tests.test_dynamic_plumbing import _no_launch

    _no_launch(monkeypatch)
    if trees:
        _scene, tree_layout = tree_scene_and_layout()
        monkeypatch.setattr(train, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path, scene, layout))
        obstacles = wood(layout if layout is not None else tree_layout)
        monkeypatch.setattr(train, "scene_config_factory",
                            lambda config, movable=(), **kw: functools.partial(FakeSimulator, obstacles=obstacles))
    else:
        monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: FakeSimulator)
    seen = []
    real = train.make_vec_env

    def spy(*args, **kwargs):
        seen.append(kwargs.get("footprint_contact_margin_m"))
        if registries is not None:
            registries.append(kwargs.get("registry"))
        if fields is not None:
            fields.append(kwargs.get("footprint_fields"))
        if shapes is not None:
            shapes.append(kwargs.get("footprint_shape"))
        if scenes is not None:
            scenes.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "make_vec_env", spy)
    out = tmp_path / "train.json"
    code = train.main(["--scene", "s99" if trees else "s01", *flags, "--run-root", str(tmp_path / "run"), "--out",
                       str(out), "--device", "cpu", "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "10",
                       "--learning-starts", "100", "--buffer-size", "100", "--eval-freq", "1000"])
    return code, out, seen


def test_a_training_run_takes_the_margin_from_the_command_line_and_records_the_task_rule(tmp_path, monkeypatch):
    # 1.05 + 0.05 = 1.1 m = 1.4 - 0.3: at the sampler's own 1.4 m, the widest margin the outline leaves.
    fields = outline_fields_for(tree_scene_and_layout()[1])
    code, out, seen = _train(tmp_path, monkeypatch, "--footprint-contact-margin", "0.05")
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["identity"]["footprint_contact"] == {"contact_m": 1.05, "training_margin_m": 0.05, "shape": "outline",
                                                       "fields": fields.digest()}
    assert record["config"]["footprint_contact_margin_m"] == 0.05 and record["config"]["footprint_shape"] == "outline"
    assert record["contact_rules"] == outline_rules(fields, 11)
    assert seen == [0.05, 0.0], "training ends episodes at the margin; evaluation keeps the task's rule"


def test_a_tree_scene_without_the_margin_still_names_the_task_rule_in_its_identity(tmp_path, monkeypatch):
    fields = outline_fields_for(tree_scene_and_layout()[1])
    code, out, _seen = _train(tmp_path, monkeypatch)
    record = json.loads(out.read_text())
    assert code == 0 and record["identity"]["footprint_contact"] == {"contact_m": 1.05, "training_margin_m": 0.0,
                                                                     "shape": "outline", "fields": fields.digest()}
    assert "spawn_clearance_m" not in record["identity"] and record["config"]["spawn_clearance_m"] == 1.4


def test_a_pillar_scenes_identity_and_record_are_as_before(tmp_path, monkeypatch):
    code, out, seen = _train(tmp_path, monkeypatch, trees=False)
    record = json.loads(out.read_text())
    assert code == 0 and "footprint_contact" not in record["identity"] and record["contact_rules"] is None
    assert record["config"]["footprint_contact_margin_m"] == 0.0 and seen == [0.0, 0.0]
    assert "spawn_clearance_m" not in record["identity"], "every earlier run's identity must still match"
    assert "training_spawn_clearance_m" not in record["identity"] and record["config"]["training_spawn_clearance_m"] is None
    assert "footprint_shape" not in record["config"], "a pillar scene has no footprint rule, so no shape to record"


def test_a_training_run_keeps_its_own_starts_farther_out_and_its_evaluation_keeps_the_scenes(tmp_path, monkeypatch):
    # --spawn-clearance-m: the training workers' sampler keeps starts, targets and distractors M from every circle; the
    # evaluation env keeps the scene's own, as the gate and the collector do. (A scene-file clearance would change the
    # gate's episodes for the same seeds, and so every comparison of two runs on the same 200 gate episodes.) The trees
    # come down to 4 m from the edges here, so the scene's own 1.4 m lets starts come closer than 1.9 m to a circle.
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = edge_tree_scene_and_layout()
    outline_fields_for(layout)
    scenes = []
    code, out, seen = _train(tmp_path, monkeypatch, "--spawn-clearance-m", "1.9", "--footprint-contact-margin", "0.5",
                             scene=scene, layout=layout, scenes=scenes)
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    training, evaluation = scenes
    assert training == dataclasses.replace(scene, spawn_clearance_m=1.9) and evaluation is scene
    assert record["identity"]["training_spawn_clearance_m"] == 1.9 and "spawn_clearance_m" not in record["identity"]
    assert record["identity"]["footprint_contact"]["training_margin_m"] == 0.5 and seen == [0.5, 0.0]
    assert record["config"]["training_spawn_clearance_m"] == 1.9 and record["config"]["spawn_clearance_m"] == 1.4

    def gap(start):
        return min(math.hypot(start.x - i.x, start.y - i.y) - i.radius_m for i in layout.instances)

    seeds = range(1_000_000, 1_000_400)
    trained = [gap(sample_setup(training, layout, np.random.default_rng(s)).start) for s in seeds]
    evaluated = [gap(sample_setup(evaluation, layout, np.random.default_rng(s)).start) for s in seeds]
    assert min(trained) >= 1.9, "no training start within M of a circle"
    assert min(evaluated) >= 1.4 and sum(g < 1.9 for g in evaluated) >= 10, "the evaluation's starts are the scene's"


@pytest.mark.parametrize("flags, ok, why", [
    (("--spawn-clearance-m", "1.9", "--footprint-contact-margin", "0.5"), True, None),  # 1.05 + 0.5 <= 1.9 - 0.3
    (("--spawn-clearance-m", "1.8", "--footprint-contact-margin", "0.5"), False,
     "puts the boundary at 1.55 m from the outline, beyond the 1.5 m a start can be from a footprint circle (training "
     "starts keep 1.8 m"),
    (("--spawn-clearance-m", "1.3"), False, "below the 1.4 m the scene's own episodes keep"),
    (("--spawn-clearance-m", "nan"), False, "--spawn-clearance-m must be a number of metres"),
    (("--spawn-clearance-m", "inf"), False, "--spawn-clearance-m must be a number of metres"),
])
def test_the_footprint_margins_ceiling_follows_the_training_clearance(tmp_path, monkeypatch, capsys, flags, ok, why):
    code, out, seen = _train(tmp_path, monkeypatch, *flags)
    if ok:
        assert code == 0 and json.loads(out.read_text())["status"] == "ok"
        return
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


@pytest.mark.parametrize("flags, ok, why", [
    (("--spawn-clearance-m", "1.7", "--static-contact-m", "1.4"), True, None),
    (("--spawn-clearance-m", "1.7", "--static-contact-m", "1.5"), False,
     "beyond the 1.4 m a start can be from a pillar (training starts keep 1.7 m"),
])
def test_the_static_boundarys_ceiling_follows_the_training_clearance(tmp_path, monkeypatch, capsys, flags, ok, why):
    scene, layout = mixed_scene_and_layout()
    code, out, seen = _train(tmp_path, monkeypatch, *flags, scene=scene, layout=layout)
    if ok:
        record = json.loads(out.read_text())
        assert code == 0 and record["status"] == "ok", record["error"]
        assert record["identity"]["static_contact_m"] == 1.4 and record["identity"]["training_spawn_clearance_m"] == 1.7
        return
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


def test_a_training_clearance_may_not_undercut_a_scenes_own(tmp_path, monkeypatch, capsys):
    scene, layout = wide_tree_scene_and_layout()  # the scene keeps 1.7 m
    code, _out, seen = _train(tmp_path, monkeypatch, "--spawn-clearance-m", "1.6", scene=scene, layout=layout)
    assert code == 2 and "below the 1.7 m the scene's own episodes keep" in capsys.readouterr().err and seen == []
    assert not (tmp_path / "run").exists()
    scenes = []
    code, _out, _seen = _train(tmp_path, monkeypatch, "--spawn-clearance-m", "1.7", scene=scene, layout=layout,
                               scenes=scenes)
    assert code == 0 and scenes[0].spawn_clearance_m == 1.7 and scenes[1] is scene


def test_the_trainers_help_says_why_the_training_clearance_is_not_the_scenes():
    from autofly_ue5.expert.train import build_arg_parser

    action = next(a for a in build_arg_parser()._actions if "--spawn-clearance-m" in a.option_strings)
    assert "training only" in action.help and "gate's episodes" in action.help and action.default is None


def wide_tree_scene_and_layout():
    """The tree scene with `spawn_clearance_m: 1.7` (2026-10-08): the same layout, starts kept 0.3 m farther out."""
    scene, layout = _generated("trees_7002_clear1.7", {**tree_scene_dict(), "spawn_clearance_m": 1.7})
    assert scene.spawn_clearance_m == 1.7 and layout == tree_scene_and_layout()[1]
    return scene, layout


def wide_mixed_scene_and_layout():
    scene, layout = _generated("mixed_clear1.7", {**mixed_scene_dict(), "spawn_clearance_m": 1.7})
    assert scene.spawn_clearance_m == 1.7 and layout == mixed_scene_and_layout()[1]
    return scene, layout


def test_a_scene_that_keeps_its_starts_farther_out_allows_a_wider_margin_and_names_it_in_its_identity(
        tmp_path, monkeypatch):
    # 1.05 + 0.35 = 1.4 m = 1.7 - 0.3: the widest boundary a start drawn 1.7 m out and landing 0.3 m off can never meet.
    scene, layout = wide_tree_scene_and_layout()
    fields = outline_fields_for(layout)
    code, out, seen = _train(tmp_path, monkeypatch, "--footprint-contact-margin", "0.35", scene=scene, layout=layout)
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["identity"]["spawn_clearance_m"] == 1.7 and record["config"]["spawn_clearance_m"] == 1.7
    assert record["identity"]["footprint_contact"] == {"contact_m": 1.05, "training_margin_m": 0.35, "shape": "outline",
                                                       "fields": fields.digest()}
    assert seen == [0.35, 0.0]


def test_a_static_boundary_may_reach_as_far_as_the_scenes_starts_allow(tmp_path, monkeypatch):
    scene, layout = wide_mixed_scene_and_layout()
    code, out, _seen = _train(tmp_path, monkeypatch, "--static-contact-m", "1.4", scene=scene, layout=layout)
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["identity"]["static_contact_m"] == 1.4 and record["identity"]["spawn_clearance_m"] == 1.7


@pytest.mark.parametrize("flags, wide, why", [
    (("--footprint-contact-margin", "0.6"), True,
     "beyond the 1.4 m a start can be from a footprint circle (starts keep 1.7 m"),
    (("--static-contact-m", "1.5"), True, "beyond the 1.4 m a start can be from a pillar (starts keep 1.7 m"),
    (("--static-contact-m", "1.4"), False, "beyond the 1.1 m a start can be from a pillar (starts keep 1.4 m"),
])
def test_the_ceilings_follow_the_scenes_own_start_clearance(tmp_path, monkeypatch, capsys, flags, wide, why):
    scene, layout = (wide_mixed_scene_and_layout() if "--static-contact-m" in flags else wide_tree_scene_and_layout()) \
        if wide else mixed_scene_and_layout()
    code, _out, seen = _train(tmp_path, monkeypatch, *flags, scene=scene, layout=layout)
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


@pytest.mark.parametrize("flags, trees, why", [
    (("--footprint-contact-margin", "0.2"), False, "no mesh obstacle"),
    (("--footprint-contact-margin", "-0.2"), True, "non-negative"),
    (("--footprint-contact-margin", "nan"), True, "non-negative"),
    (("--footprint-contact-margin", "0.3"), True, "1.1 m"),  # 1.05 + 0.3 is beyond what a start can be from a circle
    (("--footprint-contact-margin", "0.06"), True, "1.1 m"),  # so is 1.05 + 0.06: the outline leaves 0.05 m at 1.4 m
    (("--static-contact-m", "1.1"), True, "no pillar or box"),  # every obstacle is a tree: the boundary ends nothing
])
def test_a_margin_that_cannot_apply_is_refused_before_the_run_root_is_claimed(tmp_path, monkeypatch, capsys, flags, trees, why):
    code, _out, seen = _train(tmp_path, monkeypatch, *flags, trees=trees)
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


def test_an_excessive_margin_is_reported_as_one_not_as_a_field_that_cannot_reach_it(tmp_path, monkeypatch, capsys):
    # The ceiling is checked before the rule is built: a 3 m margin would also outrun the fields' 3 m reach, and that
    # refusal ("rebuild it with a wider margin") would send the reader to the wrong fix.
    code, _out, seen = _train(tmp_path, monkeypatch, "--footprint-contact-margin", "3.0")
    err = capsys.readouterr().err
    assert code == 2 and "puts the boundary at 4.05 m from the outline, beyond the 1.1 m a start can be" in err
    assert "reach" not in err and not (tmp_path / "run").exists() and seen == []


def test_the_trainers_help_states_the_task_rule_from_its_constant():
    from autofly_ue5.expert.footprint_contact import CIRCLE_CONTACT_M, OUTLINE_CONTACT_M
    from autofly_ue5.expert.train import build_arg_parser

    action = next(a for a in build_arg_parser()._actions if "--footprint-contact-margin" in a.option_strings)
    assert (f"{OUTLINE_CONTACT_M:g} m from the obstacle's outline, or {CIRCLE_CONTACT_M:g} m from its circle under "
            f"--footprint-shape circle" in action.help)
    assert "0.5 m" not in action.help


def test_a_layout_with_an_asset_the_registry_does_not_know_is_refused_before_the_run_root_is_claimed(
        tmp_path, monkeypatch, capsys):
    _scene, layout = tree_scene_and_layout()
    bogus = dataclasses.replace(layout, instances=(dataclasses.replace(layout.instances[0], asset="no_such_asset"),
                                                   *layout.instances[1:]))
    code, _out, seen = _train(tmp_path, monkeypatch, layout=bogus)
    err = capsys.readouterr().err
    assert code == 2 and "refusing to start" in err and "no_such_asset" in err
    assert not (tmp_path / "run").exists() and seen == []


def test_a_scene_with_moving_pillars_and_mesh_obstacles_is_refused_before_the_run_root_is_claimed(
        tmp_path, monkeypatch, capsys):
    from tests.test_dynamic_env import dynamic_scene

    scene, _layout = tree_scene_and_layout()
    code, _out, seen = _train(tmp_path, monkeypatch, scene=dataclasses.replace(scene, dynamic=dynamic_scene()[0].dynamic))
    err = capsys.readouterr().err
    assert code == 2 and "refusing to start" in err and "_cards" in err
    assert not (tmp_path / "run").exists() and seen == []


def test_the_identity_the_record_and_every_worker_read_one_registry_snapshot(tmp_path, monkeypatch):
    # M4 keeps re-importing assets: a registry read again hours later, in the record's unguarded assembly, could raise
    # and lose the run's record (as s01d_r1 lost its own, MEMORY.md 2026-10-02), or disagree with the one the run
    # trained on. So the trainer reads it once, before the run root is claimed, and hands that copy to everything.
    import autofly_ue5.expert.env as env_module
    import autofly_ue5.expert.footprint_contact as footprint_module
    from autofly_ue5.expert import train

    loaded = []

    def load_once():
        loaded.append(load_registry())
        return loaded[-1]

    def second_read():
        raise AssertionError("the registry was read a second time")

    fields = outline_fields_for(tree_scene_and_layout()[1])
    monkeypatch.setattr(train, "load_registry", load_once)
    monkeypatch.setattr(env_module, "load_registry", second_read)
    monkeypatch.setattr(footprint_module, "load_registry", second_read)
    # the outline fields likewise: read once before the run root is claimed, then handed to every env
    real_fields = train.FootprintFields.load
    fields_loaded = []

    def fields_once(assets, *args, **kwargs):
        fields_loaded.append(real_fields(assets, *args, **kwargs))
        return fields_loaded[-1]

    monkeypatch.setattr(train.FootprintFields, "load", fields_once)
    registries, handed = [], []
    code, out, _seen = _train(tmp_path, monkeypatch, "--footprint-contact-margin", "0.05", registries=registries,
                              fields=handed)
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert len(loaded) == 1 and registries == [loaded[0], loaded[0]], "training and evaluation workers"
    assert len(fields_loaded) == 1 and handed == [fields_loaded[0], fields_loaded[0]]
    assert record["contact_rules"] == outline_rules(fields, 11)
    assert record["identity"]["footprint_contact"] == {"contact_m": 1.05, "training_margin_m": 0.05, "shape": "outline",
                                                       "fields": fields.digest()}


def test_run_identity_reads_the_registry_it_is_given():
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.train import run_identity

    resolved = _tree_resolved(Path("unused"))
    registry = load_registry()
    primitives = dataclasses.replace(registry, assets={name: dataclasses.replace(entry, radius_profile_m=None)
                                                       for name, entry in registry.assets.items()})
    assert "footprint_contact" in run_identity(resolved, ObsConfig(), registry=registry, footprint_shape="circle")
    assert "footprint_contact" not in run_identity(resolved, ObsConfig(), registry=primitives)


def test_the_gate_records_the_task_contact_rule_it_scored(tmp_path, monkeypatch):
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.seeds import EVAL_SEED_BASE
    from scripts import m2_gate
    from tests.test_m2_gate import _StraightAtTargetModel

    _scene, layout = tree_scene_and_layout()
    monkeypatch.setattr(m2_gate, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a real model; the fake loader ignores it")
    gate = m2_gate.run(scene="s99", model_paths={"model": checkpoint}, conditions=["deterministic"], n_episodes=2,
                       seed_base=EVAL_SEED_BASE, instance=5, out_path=tmp_path / "gate.json",
                       sim_factory=functools.partial(FakeSimulator, obstacles=wood(layout)),
                       load_model=lambda path: _StraightAtTargetModel(), sim_root=tmp_path / "sim",
                       instances_path=tmp_path / "no_measurement.json", obs_config_reader=lambda path: ObsConfig(),
                       footprint_shape="outline")  # a checkpoint outside any run: the shape must be given
    assert gate["status"] == "ok"
    assert gate["contact_rules"] == outline_rules(outline_fields_for(layout), 11)
    assert json.loads((tmp_path / "gate.json").read_text())["contact_rules"] == gate["contact_rules"]


def test_the_gate_records_which_tree_ended_each_footprint_episode_and_from_where(tmp_path, monkeypatch):
    # The 2026-10-03 s01d gate counted mover collisions without saying which mover, how close, from where or how fast:
    # each footprint ending carries all of it into the per-episode record.
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.seeds import EVAL_SEED_BASE
    from scripts import m2_gate

    _scene, layout = tree_scene_and_layout()
    monkeypatch.setattr(m2_gate, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a real model; the fake loader ignores it")
    outline_fields_for(layout)
    sims = []
    gate = m2_gate.run(scene="s99", model_paths={"model": checkpoint}, conditions=["deterministic"], n_episodes=4,
                       seed_base=EVAL_SEED_BASE, instance=5, out_path=tmp_path / "gate.json",
                       sim_factory=recording_tree_factory(layout, sims), footprint_shape="outline",
                       load_model=lambda path: AtTheNearestTree(layout, sims), sim_root=tmp_path / "sim",
                       instances_path=tmp_path / "no_measurement.json", obs_config_reader=lambda path: ObsConfig())
    report = gate["checkpoints"]["model"]["deterministic"]
    endings = [e for e in report["per_episode"] if e.get("collision_source") == "footprint"]
    assert endings and report["collision_sources"]["footprint"] == len(endings)
    tags = {inst.tag: inst.asset for inst in layout.instances}
    for episode in endings:
        contact = episode["footprint_contact"]
        assert episode["outcome"] == "collision" and contact["rule"] == "footprint"
        assert set(contact) == {"tag", "asset", "gap_m", "rule", "bearing_deg", "drone_forward_m_s", "shape", "circle_gap_m"}
        assert tags[contact["tag"]] == contact["asset"] and DRONE_COLLISION_REACH_M <= contact["gap_m"] <= 1.05 + 1e-9
        assert contact["shape"] == "outline" and contact["circle_gap_m"] <= contact["gap_m"] + 0.02
    assert all("footprint_contact" not in e for e in report["per_episode"] if e.get("collision_source") != "footprint")
    stored = json.loads((tmp_path / "gate.json").read_text())["checkpoints"]["model"]["deterministic"]
    assert [e.get("footprint_contact") for e in stored["per_episode"]] == [e.get("footprint_contact") for e in report["per_episode"]]


def test_the_watcher_counts_footprint_endings_apart_from_the_training_margins(tmp_path):
    # Under a margin run most endings are the margin's: one counter would hide how often the task's own line is crossed.
    from scripts.watch_training import render_text, summarize
    from tests.test_watch_training import _run

    root, log = _run(tmp_path)
    line = ("FOOTPRINT instance 2: obs_0003 (island_tree_02) at {gap} m from its circle ({rule}), bearing +12 deg, "
            "forward 2.00 m/s\n")
    with open(log, "a") as handle:
        handle.write(line.format(gap="0.48", rule="footprint"))
        handle.write(line.format(gap="1.02", rule="footprint_margin") * 3)
    s = summarize(root, log, window=8, host=False)
    assert s["log"]["counts"]["footprint"] == 1 and s["log"]["counts"]["footprint_margin"] == 3
    assert "footprint endings 1 (+3 at the training margin)" in render_text(s)


# --------------------------------------------------------------------------------------------------------
# The outline (2026-10-08, afternoon): the distance is to the model's real outline, not its circle.
# --------------------------------------------------------------------------------------------------------
LEAN = "synthetic_lean"  # tests/test_footprint_field.py's leaning tree: a 0.1 m trunk on the pivot, a crown 1 m off it
LEAN_RADIUS_M = math.hypot(1.5, 0.5)  # its widest reach from the pivot: the crown's far corners


def lean_tree(tmp_path, *, x=0.0, y=0.0, yaw=0.0, scale=1.0):
    """The synthetic leaning tree as a registry entry, an outline field (both in tmp_path, never assets/) and one
    instance: (registry, fields, instance). Its crown leans toward the instance's +x (local glTF x)."""
    from autofly_ue5.scenes.footprint_field import FootprintFields
    from tests.test_footprint_field import leaning_tree_parts, synthetic_field

    field = synthetic_field(tmp_path, LEAN, leaning_tree_parts())
    base = load_registry()
    entry = dataclasses.replace(base.assets["quiver_tree_01"], name=LEAN, radius_profile_m=(0.0707,) * 8 + (LEAN_RADIUS_M,) * 6)
    registry = dataclasses.replace(base, assets={**base.assets, LEAN: entry})
    inst = Instance("lean_0", LEAN, x, y, 0.0, yaw, (scale, scale, scale), "mesh", round(LEAN_RADIUS_M * scale, 4), 3.5 * scale)
    return registry, FootprintFields({LEAN: field}), inst


def _lean_rule(tmp_path, shape, **kw):
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    _registry, fields, inst = lean_tree(tmp_path, **kw)
    return FootprintContactRule([inst], shape=shape, fields=fields if shape == "outline" else None), inst


def _scripted(rule, start, heading, steps: int = 60, step_m: float = 0.4):
    """Full-speed steps from `start` along `heading` (rad), each checked by the rule as the env checks it; returns
    (index of the ending step or None, its (tag, gap) or None, the steps flown as (a, b))."""
    a = np.asarray(start, dtype=float)
    u = np.array([math.cos(heading), math.sin(heading)])
    flown = []
    for k in range(steps):
        b = a + step_m * u
        flown.append((tuple(a), tuple(b)))
        hit = rule.contact(tuple(a), tuple(b))
        if hit is not None:
            return k, hit, flown
        a = b
    return None, None, flown


def test_a_flight_past_the_side_away_from_the_lean_is_ended_by_the_circle_but_not_by_the_outline(tmp_path):
    # 1.2 m from the pivot on the side away from the crown: inside the 1.58 m circle (the rule's line is 2.48 m out),
    # 1.15 m from the trunk, which is all the model has on that side at any height.
    for shape, ends in (("circle", True), ("outline", False)):
        rule, _inst = _lean_rule(tmp_path / shape, shape)
        k, hit, flown = _scripted(rule, (-1.2, -6.0), math.pi / 2, steps=30)
        assert (k is not None) == ends, (shape, k, hit)
        if ends:
            assert hit[1] <= 0.9 and math.hypot(*flown[k][1]) > 1.2, "the circle ends it on the way in"
        else:
            assert flown[-1][1][1] > 5.0, "the outline lets it fly past"


@pytest.mark.parametrize("heading_deg", [180.0, 150.0, 210.0, 135.0])
def test_a_flight_into_the_crown_ends_within_the_contact_distance_of_the_outline_never_inside_the_rotor_reach(
        tmp_path, heading_deg):
    from autofly_ue5.scenes.footprint_field import outline_distance

    rule, inst = _lean_rule(tmp_path, "outline")
    heading = math.radians(heading_deg)
    start = (1.0 - 6.0 * math.cos(heading), -6.0 * math.sin(heading))  # 6 m out from the crown's centre (1, 0)
    k, hit, flown = _scripted(rule, start, heading)
    assert k is not None and hit[0] == "lean_0", "flying straight at the crown ends there"
    a, b = flown[k]
    field = rule.fields[LEAN]
    assert hit[1] == pytest.approx(outline_distance(field, inst, a, b), abs=1e-12)
    assert DRONE_COLLISION_REACH_M < hit[1] <= 1.05, "within the rule's distance, the collision reach clear of it"
    assert all(outline_distance(field, inst, *step) > 1.05 for step in flown[:k]), "no earlier step came that close"
    # the crown is leaf cards 2-3.5 m up; the drone flies at 1-3 m under or beside it: the column counts it
    assert min(math.dist(b, (x, z)) for x in (0.5, 1.5) for z in (-0.5, 0.5)) > 0.4


def test_the_outline_checks_the_swept_segment_not_only_its_ends(tmp_path):
    # A step passing the trunk 1.03 m from its face, both ends ~1.17 m from it: the pass is what the rule must see.
    rule, _inst = _lean_rule(tmp_path, "outline")
    a, b = (-1.08, -0.6), (-1.08, 0.6)
    assert rule.contact(a, a) is None and rule.contact(b, b) is None
    hit = rule.contact(a, b)
    assert hit is not None and hit[1] == pytest.approx(1.03, abs=0.015) and hit[1] <= 1.03 + 3e-3


def test_a_non_finite_position_never_raises_from_the_rule_and_reads_as_no_contact(tmp_path):
    # Non-finite kinematics are the backend's to refuse; the rule reads them as no contact (+inf) under the outline and
    # never raises (math.ceil of a NaN length did: a bare ValueError, which ends a run).
    rule, _inst = _lean_rule(tmp_path, "outline")
    nan, inf = math.nan, math.inf
    for bad in ((nan, 0.0), (inf, 1.0), (0.0, -inf)):
        assert rule.contact(bad, (8.0, 8.0)) is None and rule.contact((8.0, 8.0), bad) is None
        assert rule.contact(bad, bad) is None
        tag, gap = rule.nearest(bad, (8.0, 8.0))
        assert tag == "lean_0" and gap == inf
        assert rule.nearest_gap(bad) == inf


def test_a_damaged_field_file_is_refused_before_the_run_root_is_claimed(tmp_path, monkeypatch, capsys):
    import shutil

    from autofly_ue5.expert import train
    from autofly_ue5.paths import ASSET_FOOTPRINTS
    from autofly_ue5.scenes.footprint_field import FootprintFields

    outline_fields_for(tree_scene_and_layout()[1])  # skips on a checkout without the fields
    root = tmp_path / "footprints"
    shutil.copytree(ASSET_FOOTPRINTS, root)
    victim = root / "island_tree_02.npz"
    victim.write_bytes(victim.read_bytes()[:1000])
    real = FootprintFields.load.__func__
    monkeypatch.setattr(train.FootprintFields, "load", classmethod(lambda cls, assets: real(cls, assets, root)))
    code, _out, seen = _train(tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 2 and "refusing to start" in err and "island_tree_02.npz: unreadable footprint field" in err
    assert not (tmp_path / "run").exists() and seen == []


def test_the_circle_shape_is_the_static_boundary_it_always_was():
    # s02_r1 trained under the circle; its evaluations must score exactly what its training scored: the circle shape is
    # the static boundary's geometry over the layout's radii, step for step, gap for gap.
    from autofly_ue5.expert.footprint_contact import FootprintContactRule
    from autofly_ue5.expert.static_contact import StaticContactBoundary

    _scene, layout = tree_scene_and_layout()
    for margin in (0.0, 0.2):
        rule = FootprintContactRule(layout.instances, shape="circle", training_margin_m=margin)
        legacy = StaticContactBoundary([(i.tag, i.x, i.y, i.radius_m) for i in layout.instances], 0.9 + margin)
        rng = np.random.default_rng(8)
        for _ in range(2000):
            a = rng.uniform(-35, 35, size=2)
            b = a + rng.uniform(-0.4, 0.4, size=2)
            assert rule.contact(a, b) == legacy.contact(a, b)
            gaps = legacy.gaps(a, b)
            assert rule.nearest(a, b) == (legacy.tags[int(np.argmin(gaps))], float(gaps.min()))
            assert rule.nearest_gap(a) == legacy.nearest_gap(a)


def test_an_obstacle_far_from_its_circle_is_never_measured_against_its_outline(tmp_path, monkeypatch):
    # The per-step cost: the circle holds the outline, so a step farther than termination + 0.1 m from a circle cannot
    # be within termination of that outline, and its field is not read (termination 1.05 m: the screen reads to 1.15).
    import autofly_ue5.expert.footprint_contact as footprint_module

    rule, _inst = _lean_rule(tmp_path, "outline")
    reach = rule.fields[LEAN].pivot_reach_m  # the screen's radius: the field's own reach (next test)
    calls = []
    real = footprint_module.outline_distance
    monkeypatch.setattr(footprint_module, "outline_distance", lambda *a, **k: calls.append(a) or real(*a, **k))
    assert rule.contact((reach + 1.2, 0.0), (reach + 1.2, 0.4)) is None and calls == []
    assert rule.contact((reach + 1.1, 0.0), (reach + 1.1, 0.4)) is None and len(calls) == 1


def test_the_screen_is_the_fields_own_reach_not_the_layouts_radius(tmp_path):
    # Every layout generated since 2026-10-08 gives a mesh the circle of its whole reach, which holds the outline; an
    # older one gave the flight band's (smaller). The screen must not depend on which: a crown contact is found even
    # when the instance's radius_m is far too small, and the circle shape still reads the layout's radius as it was.
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    _registry, fields, inst = lean_tree(tmp_path)
    small = dataclasses.replace(inst, radius_m=0.1)
    a, b = (1.5 + 0.85, -0.2), (1.5 + 0.85, 0.2)  # 0.85 m off the crown's face, 2.25 m from the pivot
    hit = FootprintContactRule([small], fields=fields).contact(a, b)
    assert hit is not None and hit[1] == pytest.approx(0.85, abs=0.015)
    assert FootprintContactRule([small], shape="circle").contact(a, b) is None
    assert fields[LEAN].pivot_reach_m == pytest.approx(LEAN_RADIUS_M, abs=0.04)
    assert fields[LEAN].pivot_reach_m >= LEAN_RADIUS_M, "a bound: never short of the geometry"


def test_the_outline_is_every_entry_points_default_and_an_unknown_shape_is_refused(tmp_path):
    from autofly_ue5.expert.footprint_contact import DEFAULT_SHAPE, FootprintContactRule, task_contact_rules

    assert DEFAULT_SHAPE == "outline"
    _scene, layout = tree_scene_and_layout()
    fields = outline_fields_for(layout)
    assert task_contact_rules(layout) == outline_rules(fields, 11)
    assert FootprintContactRule(layout.instances).shape == "outline"
    with pytest.raises(ValueError, match="shape"):
        FootprintContactRule(layout.instances, shape="square")
    with pytest.raises(ValueError, match="shape"):
        make_tree_env(footprint_shape="square")
    with pytest.raises(ValueError, match="shape"):
        task_contact_rules(layout, shape="square")


def test_a_mesh_asset_without_a_field_is_refused_under_the_outline_never_treated_as_a_circle(tmp_path):
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.footprint_contact import FootprintContactRule
    from autofly_ue5.scenes.footprint_field import FootprintFields

    registry, fields, inst = lean_tree(tmp_path)
    scene, layout = tree_scene_and_layout()
    alone = dataclasses.replace(layout, instances=(inst,))
    make = functools.partial(AutoFlyEnv, scene, alone, FakeSimulator, map_path="/Game/AutoFly/Maps/S99", instance=0,
                             registry=registry)
    with pytest.raises(ValueError, match=f"no footprint field for {LEAN}"):
        make()  # the default fields (assets/footprints/) have none for it
    with pytest.raises(ValueError, match=f"no footprint field for {LEAN}"):
        make(footprint_fields=FootprintFields({}))
    with pytest.raises(ValueError, match=f"no footprint field for {LEAN}"):
        FootprintContactRule([inst], fields=FootprintFields({}))
    make(footprint_fields=fields)  # with its field it is measured
    make(footprint_shape="circle")  # the circle needs none


@pytest.mark.parametrize("z_center, ok", [(0.0, True), (-0.0, True), (5e-7, True), (2e-6, False), (0.5, False),
                                          (-0.2, False)])
def test_under_the_outline_a_mesh_instance_must_stand_on_the_ground(tmp_path, z_center, ok):
    # The outline is the model cut at the ground (footprint_field.clip_above_ground, glTF y = 0 at its base pivot): an
    # instance whose pivot is not on the ground would be measured from a cut at the wrong height. Every stored layout's
    # meshes stand at z_center 0; the circle has no cut and takes any.
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    _registry, fields, inst = lean_tree(tmp_path)
    moved = dataclasses.replace(inst, z_center=z_center)
    if ok:
        FootprintContactRule([moved], fields=fields)
    else:
        with pytest.raises(ValueError, match="stand on the ground"):
            FootprintContactRule([moved], fields=fields)
    FootprintContactRule([moved], shape="circle")


def test_a_field_that_cannot_reach_the_rules_distance_or_an_unequal_scale_is_refused(tmp_path):
    from autofly_ue5.expert.footprint_contact import FootprintContactRule
    from autofly_ue5.scenes.footprint_field import FootprintFields, build_field, outline_triangles
    from autofly_ue5.validate.obstacles import MeshLibrary
    from tests.test_footprint_field import leaning_tree_parts, synthetic_library

    _registry, fields, inst = lean_tree(tmp_path / "a")
    library = MeshLibrary.from_files({LEAN}, **synthetic_library(tmp_path / "b", {LEAN: leaning_tree_parts()}))
    short = FootprintFields({LEAN: build_field(outline_triangles(library[LEAN]), margin_m=0.8)})  # reach 0.79 m
    with pytest.raises(ValueError, match="reach"):
        FootprintContactRule([inst], fields=short)
    with pytest.raises(ValueError, match="scale"):
        FootprintContactRule([dataclasses.replace(inst, scale=(1.0, 1.1, 1.0))], fields=fields)
    FootprintContactRule([dataclasses.replace(inst, scale=(1.0, 1.1, 1.0))], shape="circle")  # radii only


class StartsAt(FakeSimulator):
    """A fake whose reset puts the drone where the test says, whatever the episode sampled: scripted flights."""

    start = None

    def reset(self, pose):
        return super().reset(self.start if self.start is not None else pose)


def _lean_env(tmp_path, shape, start, **kw):
    """The synthetic leaning tree alone in the test scene, on a fake whose wood is its trunk, the drone at `start`."""
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.sim.types import Pose

    registry, fields, inst = lean_tree(tmp_path, x=5.0, y=-3.0, yaw=0.6)
    scene, layout = tree_scene_and_layout()
    alone = dataclasses.replace(layout, instances=(inst,))
    StartsAt.start = Pose(*start)
    env = AutoFlyEnv(scene, alone, functools.partial(StartsAt, obstacles=[(inst.x, inst.y, 0.05)]),
                     map_path="/Game/AutoFly/Maps/S99", instance=0, registry=registry, footprint_shape=shape,
                     footprint_fields=fields if shape == "outline" else None, **kw)
    return env, inst


def _env_flight(env, steps: int = 40):
    """Reset on a seed whose target is far from the scripted path, then fly straight ahead at 2 m/s."""
    for seed in range(1_000_100, 1_000_200):
        env.reset(seed=seed)
        tx, ty, _ = env._setup.target_xy_z
        if math.hypot(tx - env._last_pose.x, ty - env._last_pose.y) > 25.0:
            break
    infos = []
    for _ in range(steps):
        _obs, _r, terminated, truncated, info = env.step(np.array([2.0, 0.0, 0.0], np.float32))
        infos.append(info)
        if terminated or truncated:
            break
    return infos


def _world(inst, local_xy):
    from autofly_ue5.scenes.footprint_field import to_world

    return tuple(to_world(inst, np.array([local_xy]))[0])


def test_in_the_env_the_outline_lets_a_flight_pass_the_lean_and_ends_one_into_the_crown(tmp_path, capsys):
    from autofly_ue5.scenes.footprint_field import outline_distance

    _registry, _fields, inst = lean_tree(tmp_path / "probe", x=5.0, y=-3.0, yaw=0.6)
    # past the side away from the crown, 1.2 m from the pivot, heading along the instance's local +z
    x0, y0 = _world(inst, (-1.2, -6.0))
    past = (x0, y0, -2.0, inst.yaw + math.pi / 2)
    for shape in ("circle", "outline"):
        env, _ = _lean_env(tmp_path / shape, shape, past)
        infos = _env_flight(env, steps=30)
        ended = [i for i in infos if i.get("collision_source") == "footprint"]
        assert bool(ended) == (shape == "circle"), (shape, [i["outcome"] for i in infos][-3:])
    # straight into the crown, from 6 m out along the instance's local -x
    x1, y1 = _world(inst, (7.0, 0.0))
    env, inst = _lean_env(tmp_path / "into", "outline", (x1, y1, -2.0, inst.yaw + math.pi))
    capsys.readouterr()
    infos = _env_flight(env)
    end = infos[-1]
    assert end["collision_source"] == "footprint" and end["outcome"] == "collision"
    contact = end["footprint_contact"]
    assert contact["shape"] == "outline" and DRONE_COLLISION_REACH_M < contact["gap_m"] <= 1.05
    before, after = infos[-2]["pose"][:2] if len(infos) > 1 else None, end["pose"][:2]
    assert contact["gap_m"] == pytest.approx(round(outline_distance(env._footprint_fields[LEAN], inst, before, after), 4))
    # on the crown's own side the circle (through the crown's far corners, 1.58 m out) and the outline (its face, 1.5 m)
    # nearly agree; it is on the far side that they part (the flight past above)
    assert 0.0 <= contact["gap_m"] - contact["circle_gap_m"] < 0.1
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("FOOTPRINT")]
    assert lines == [f"FOOTPRINT instance 0: lean_0 ({LEAN}) at {contact['gap_m']:.2f} m from its outline (footprint), "
                     f"bearing {contact['bearing_deg']:+.0f} deg, forward 2.00 m/s"]


def test_the_outline_tree_scene_names_its_rule_and_its_fields(tmp_path):
    _scene, layout = tree_scene_and_layout()
    fields = outline_fields_for(layout)
    env = make_tree_env(footprint_shape="outline", footprint_contact_margin_m=0.2)
    assert env.contact_rules == {"footprint": {"contact_m": 1.05, "training_margin_m": 0.2, "termination_m": 1.25,
                                               "obstacles": 11, "shape": "outline", "fields": fields.digest(),
                                               "fields_manifest_sha256": fields.manifest_sha256}}


def _outline_gap(env, inst, a_xy, b_xy) -> float:
    from autofly_ue5.scenes.footprint_field import outline_distance

    return outline_distance(env._footprint_fields[inst.asset], inst, a_xy, b_xy)


def test_flying_at_a_tree_ends_at_its_outline_and_never_reaches_wood():
    env = make_tree_env(footprint_shape="outline")
    outline_fields_for(env._layout)
    endings = 0
    for seed in SEEDS:
        steps, _obs = _fly_at_a_tree(env, seed)
        reward, end = steps[-1]
        assert end["collision_source"] != "sim", f"seed {seed}: the body reached the fake's trunk"
        if end["collision_source"] != "footprint":
            continue
        endings += 1
        contact = end["footprint_contact"]
        a, b = (steps[-2][1]["pose"][:2] if len(steps) > 1 else None), end["pose"][:2]
        nearest = min(_trees(env), key=lambda inst: _outline_gap(env, inst, a, b))
        assert contact["tag"] == nearest.tag and contact["shape"] == "outline"
        assert contact["gap_m"] == pytest.approx(round(_outline_gap(env, nearest, a, b), 4))
        assert DRONE_COLLISION_REACH_M < contact["gap_m"] <= 1.05 + 1e-9 and contact["rule"] == "footprint"
        circle = math.hypot(nearest.x - b[0], nearest.y - b[1]) - nearest.radius_m
        assert contact["circle_gap_m"] <= round(circle, 4) + 1e-4, "the swept step's circle gap, at most the end's"
        assert reward <= -10.0 + 0.5
    assert endings >= 7, f"flying straight at a tree must end at its outline ({endings} of {len(SEEDS)})"


def test_under_the_outline_a_rule_miss_names_the_nearest_outline(monkeypatch, capsys):
    import autofly_ue5.expert.env as env_module
    from autofly_ue5.expert.footprint_contact import FootprintContactRule

    class Blind(FootprintContactRule):
        def contact(self, a_xy, b_xy):
            return None

    monkeypatch.setattr(env_module, "FootprintContactRule", Blind)
    env = make_tree_env(footprint_shape="outline")
    outline_fields_for(env._layout)
    for seed in SEEDS:
        steps, _ = _fly_at_a_tree(env, seed)
        end = steps[-1][1]
        if end["collision_source"] != "sim":
            continue
        near = end["footprint_nearest"]
        assert near["shape"] == "outline" and set(near) == {"tag", "asset", "gap_m", "shape", "circle_gap_m"}
        assert near["gap_m"] < 1.05, "a rule miss reads as a body inside the rule's line"
        assert near["circle_gap_m"] < 0, "the fake's trunk is deep inside the circle"
        lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("SIM-CONTACT")]
        assert lines[-1] == (f"SIM-CONTACT instance 0: physical contact with the nearest outline {near['tag']} "
                             f"({near['asset']}) {near['gap_m']:.2f} m away, forward 2.00 m/s")
        return
    raise AssertionError("no flight reached the fake's wood")


@pytest.mark.parametrize("shape, contact_m", [("circle", 0.9), ("outline", 1.05)])
def test_under_either_shape_the_training_margin_ends_sooner_in_its_own_band(shape, contact_m):
    plain, wide = make_tree_env(footprint_shape=shape), make_tree_env(footprint_shape=shape, footprint_contact_margin_m=0.2)
    if shape == "outline":
        outline_fields_for(plain._layout)
    margin_hits = 0
    for seed in SEEDS:
        (a, _), (b, _) = _fly_at_a_tree(plain, seed, speed=0.5), _fly_at_a_tree(wide, seed, speed=0.5)
        assert len(b) <= len(a)
        end_b = b[-1][1]
        if end_b["collision_source"] == "footprint_margin":
            margin_hits += 1
            assert contact_m < end_b["footprint_contact"]["gap_m"] <= contact_m + 0.2 + 1e-9
            assert end_b["footprint_contact"]["shape"] == shape
            assert a[-1][1]["collision_source"] == "footprint"
    assert margin_hits >= 8, margin_hits


def test_a_resume_under_another_shape_or_other_fields_is_refused_and_a_circle_run_still_resumes():
    # s02_r1's identity predates the shape (its entry names none): it trained under the circle, so a resume under the
    # circle matches it and one under the outline does not. Pillar scenes name no footprint rule at all.
    from autofly_ue5.expert.train import _identity_mismatch

    legacy = {"scene": "s02", "footprint_contact": {"contact_m": 0.9, "training_margin_m": 0.2}}
    circle = {"scene": "s02", "footprint_contact": {"contact_m": 0.9, "training_margin_m": 0.2, "shape": "circle"}}
    outline = {"scene": "s02", "footprint_contact": {"contact_m": 1.05, "training_margin_m": 0.2, "shape": "outline",
                                                     "fields": {"quiver_tree_01": "a" * 64}}}
    other = {**outline, "footprint_contact": {**outline["footprint_contact"], "fields": {"quiver_tree_01": "b" * 64}}}
    assert _identity_mismatch(legacy, circle) == [] and _identity_mismatch(circle, circle) == []
    assert _identity_mismatch(legacy, outline) == ["footprint_contact"]
    assert _identity_mismatch(circle, outline) == ["footprint_contact"]
    assert _identity_mismatch(outline, other) == ["footprint_contact"]
    assert _identity_mismatch({"scene": "s01"}, {"scene": "s01"}) == []


def test_a_training_run_under_the_circle_names_it_and_hands_it_to_every_env(tmp_path, monkeypatch):
    shapes, handed = [], []
    code, out, seen = _train(tmp_path, monkeypatch, "--footprint-shape", "circle", "--footprint-contact-margin", "0.2",
                             shapes=shapes, fields=handed)
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["identity"]["footprint_contact"] == {"contact_m": 0.9, "training_margin_m": 0.2, "shape": "circle"}
    assert record["config"]["footprint_shape"] == "circle"
    assert record["contact_rules"] == {"footprint": {"contact_m": 0.9, "obstacles": 11, "shape": "circle"}}
    assert shapes == ["circle", "circle"] and handed == [None, None] and seen == [0.2, 0.0]


def test_a_tree_scene_whose_fields_are_missing_is_refused_before_the_run_root_is_claimed(tmp_path, monkeypatch, capsys):
    from autofly_ue5.expert import train

    registry, _fields, inst = lean_tree(tmp_path / "lean")
    _scene, layout = tree_scene_and_layout()
    monkeypatch.setattr(train, "load_registry", lambda: registry)
    code, _out, seen = _train(tmp_path, monkeypatch, layout=dataclasses.replace(layout, instances=(*layout.instances, inst)))
    err = capsys.readouterr().err
    assert code == 2 and "refusing to start" in err and f"no footprint field for {LEAN}" in err
    assert not (tmp_path / "run").exists() and seen == []
    code, _out, _seen = _train(tmp_path, monkeypatch, "--footprint-shape", "circle",
                               layout=dataclasses.replace(layout, instances=(*layout.instances, inst)))
    assert code == 0, "the circle needs no field"


def test_the_gate_scores_and_records_the_shape_it_is_given(tmp_path, monkeypatch):
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.seeds import EVAL_SEED_BASE
    from scripts import m2_gate

    _scene, layout = tree_scene_and_layout()
    monkeypatch.setattr(m2_gate, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a real model; the fake loader ignores it")
    sims = []
    gate = m2_gate.run(scene="s99", model_paths={"model": checkpoint}, conditions=["deterministic"], n_episodes=3,
                       seed_base=EVAL_SEED_BASE, instance=5, out_path=tmp_path / "gate.json",
                       sim_factory=recording_tree_factory(layout, sims), footprint_shape="circle",
                       load_model=lambda path: AtTheNearestTree(layout, sims), sim_root=tmp_path / "sim",
                       instances_path=tmp_path / "no_measurement.json", obs_config_reader=lambda path: ObsConfig())
    assert gate["contact_rules"] == {"footprint": {"contact_m": 0.9, "obstacles": 11, "shape": "circle"}}
    endings = [e["footprint_contact"] for e in gate["checkpoints"]["model"]["deterministic"]["per_episode"]
               if "footprint_contact" in e]
    assert endings and all(c["shape"] == "circle" and c["gap_m"] == c["circle_gap_m"] for c in endings)
    assert m2_gate.build_arg_parser().parse_args([]).footprint_shape is None, "no default: the run's identity decides"
    assert m2_gate.build_arg_parser().parse_args(["--footprint-shape", "circle"]).footprint_shape == "circle"


def test_the_gate_refuses_a_mesh_asset_without_a_field_before_launching_anything(tmp_path, monkeypatch, capsys):
    import autofly_ue5.expert.footprint_contact as footprint_module
    from scripts import m2_gate

    registry, _fields, inst = lean_tree(tmp_path / "lean")
    scene, layout = tree_scene_and_layout()
    with_lean = dataclasses.replace(layout, instances=(*layout.instances, inst))
    monkeypatch.setattr(footprint_module, "load_registry", lambda: registry)
    monkeypatch.setattr(m2_gate, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path, scene, with_lean))
    out = tmp_path / "gate.json"
    assert m2_gate.main(["--scene", "s99", "--out", str(out), "--instance", "6", "--sim-root", str(tmp_path / "sim"),
                         "--model", f"m={tmp_path / 'model.zip'}", "--footprint-shape", "outline"]) == 2
    err = capsys.readouterr().err
    assert "refusing to start" in err and f"no footprint field for {LEAN}" in err and not out.exists()


def _gate_main(tmp_path, monkeypatch, *argv, scene=None, layout=None):
    """m2_gate.main() on the tree scene (or `scene` / `layout`) with run() replaced: (exit code, the keyword arguments
    run() was called with, or None if it was not)."""
    from scripts import m2_gate

    monkeypatch.setattr(m2_gate, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path, scene, layout))
    called = []

    def fake_run(**kwargs):
        called.append(kwargs)
        return {"status": "ok", "pass": False, "deterministic_gate_pass_by_checkpoint": {}}

    monkeypatch.setattr(m2_gate, "run", fake_run)
    code = m2_gate.main(["--scene", "s99", "--out", str(tmp_path / "gate.json"), "--instance", "6", "--sim-root",
                         str(tmp_path / "sim"), *argv])
    return code, (called[0] if called else None)


def test_the_gate_scores_a_model_under_the_shape_its_run_trained_under_without_being_told(tmp_path, monkeypatch, capsys):
    # s02_r1's pre-registered commands name no shape: the gate reads its run's identity (the circle) instead of
    # defaulting to the outline.
    from tests.test_run_shape import LEGACY, OUTLINE, run_root

    s02_r1 = run_root(tmp_path, "s02_r1", LEGACY)
    code, kwargs = _gate_main(tmp_path, monkeypatch, "--model", f"avg_200k_end={s02_r1 / 'averaged' / 'avg_1k_end.zip'}",
                              "--model", f"final={s02_r1 / 'final.zip'}")
    assert code == 0 and kwargs["footprint_shape"] == "circle" and kwargs["against_trained_shape"] is False
    assert "footprint shape: circle (from the run's identity" in capsys.readouterr().err
    r2 = run_root(tmp_path, "s02_r2", OUTLINE)
    code, kwargs = _gate_main(tmp_path, monkeypatch, "--model", f"final={r2 / 'final.zip'}")
    assert code == 0 and kwargs["footprint_shape"] == "outline"
    code, kwargs = _gate_main(tmp_path, monkeypatch, "--run-root", str(s02_r1))  # the default models: best and final
    assert code == 0 and kwargs["footprint_shape"] == "circle"


@pytest.mark.parametrize("case, why", [
    ("contradiction", "--footprint-shape outline contradicts"),
    ("disagreement", "score them in separate runs"),
    ("unknown", "pass --footprint-shape outline or --footprint-shape circle"),
    ("against_alone", "--against-trained-shape needs --footprint-shape"),
])
def test_the_gate_refuses_a_shape_it_cannot_know_or_that_contradicts_the_run(tmp_path, monkeypatch, capsys, case, why):
    from tests.test_run_shape import LEGACY, OUTLINE, run_root

    r1, r2 = run_root(tmp_path, "s02_r1", LEGACY), run_root(tmp_path, "s02_r2", OUTLINE)
    loose = tmp_path / "model.zip"
    loose.write_bytes(b"x")
    argv = {"contradiction": ["--model", f"final={r1 / 'final.zip'}", "--footprint-shape", "outline"],
            "disagreement": ["--model", f"r1={r1 / 'final.zip'}", "--model", f"r2={r2 / 'final.zip'}"],
            "unknown": ["--model", f"m={loose}"],
            "against_alone": ["--model", f"final={r1 / 'final.zip'}", "--against-trained-shape"]}[case]
    code, kwargs = _gate_main(tmp_path, monkeypatch, *argv)
    err = capsys.readouterr().err
    assert code == 2 and kwargs is None and "refusing to start" in err and why in err
    assert not (tmp_path / "gate.json").exists()


def test_the_gate_takes_the_flag_where_no_run_names_a_shape_and_ignores_it_on_a_pillar_scene(tmp_path, monkeypatch):
    from tests.test_expert_episode import scene_and_layout

    loose = tmp_path / "model.zip"
    loose.write_bytes(b"x")
    code, kwargs = _gate_main(tmp_path, monkeypatch, "--model", f"m={loose}", "--footprint-shape", "circle")
    assert code == 0 and kwargs["footprint_shape"] == "circle"
    pillars, pillar_layout = scene_and_layout()
    code, kwargs = _gate_main(tmp_path, monkeypatch, "--model", f"m={loose}", scene=pillars, layout=pillar_layout)
    assert code == 0, "a pillar scene has no footprint rule: nothing to ask for"


def test_the_gate_scores_a_run_under_the_other_shape_only_on_purpose_and_records_both(tmp_path, monkeypatch):
    # The evaluation plan's addendum: s02_r1's pick (circle-trained) is also flown under the outline, as a separately
    # named record. The record names the shape scored and the one the run trained under.
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.seeds import EVAL_SEED_BASE
    from scripts import m2_gate
    from tests.test_run_shape import LEGACY, run_root

    s02_r1 = run_root(tmp_path, "s02_r1", LEGACY)
    code, kwargs = _gate_main(tmp_path, monkeypatch, "--model", f"final={s02_r1 / 'final.zip'}", "--footprint-shape",
                              "outline", "--against-trained-shape")
    assert code == 0 and kwargs["footprint_shape"] == "outline" and kwargs["against_trained_shape"] is True
    monkeypatch.undo()
    _scene, layout = tree_scene_and_layout()
    monkeypatch.setattr(m2_gate, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    sims = []
    gate = m2_gate.run(scene="s99", model_paths={"final": s02_r1 / "final.zip"}, conditions=["deterministic"],
                       n_episodes=1, seed_base=EVAL_SEED_BASE, instance=5, out_path=tmp_path / "outline.json",
                       sim_factory=recording_tree_factory(layout, sims), footprint_shape="outline",
                       against_trained_shape=True, load_model=lambda path: AtTheNearestTree(layout, sims),
                       sim_root=tmp_path / "sim", instances_path=tmp_path / "none.json",
                       obs_config_reader=lambda path: ObsConfig())
    assert gate["status"] == "ok", gate["error"]
    expected = outline_rules(outline_fields_for(layout), 11)
    expected["footprint"]["trained_shape"] = "circle"
    assert gate["contact_rules"] == expected


def test_the_gates_library_call_refuses_a_mesh_scene_whose_shape_it_cannot_know(tmp_path, monkeypatch):
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.seeds import EVAL_SEED_BASE
    from scripts import m2_gate

    _scene, layout = tree_scene_and_layout()
    monkeypatch.setattr(m2_gate, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path))
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"x")
    with pytest.raises(ValueError, match="--footprint-shape"):
        m2_gate.run(scene="s99", model_paths={"model": checkpoint}, conditions=["deterministic"], n_episodes=1,
                    seed_base=EVAL_SEED_BASE, instance=5, out_path=tmp_path / "gate.json",
                    sim_factory=functools.partial(FakeSimulator, obstacles=wood(layout)),
                    load_model=lambda path: AtTheNearestTree(layout, []), sim_root=tmp_path / "sim",
                    instances_path=tmp_path / "none.json", obs_config_reader=lambda path: ObsConfig())
    assert not (tmp_path / "gate.json").exists()
