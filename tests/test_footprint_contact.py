"""The task's contact rule for mesh obstacles (2026-10-07, plan 5 / M4 C8): in a scene whose obstacles are imported
models (trees, rocks), an episode ends as a collision when the drone's swept step passes within FOOTPRINT_CONTACT_M of
an obstacle's footprint circle, the band radius the layout already carries. A body that reaches wood can wedge the
simulator's swept actor beyond any teleport (docs/gates/m4_s02_wood_crash_reset_probe.json: 2 of 4 crashes), and a
wedged actor cannot fly, so the body must never reach it. The physical collision event stays as a backstop. The rule
is the task's (the gate scores it, the collector rejects by it); a training run may add a margin on top, as for
movers, because every expert flies to the boundary it trains on (MEMORY.md). Static s01 is pinned separately
(tests/test_static_s01_golden.py): a pillar scene has no footprint obstacle, so nothing there may change."""

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

from autofly_ue5.scenes.model import DRONE_HALF_SPAN_M, Instance, load_registry, load_scene_file
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


def make_tree_env(sim_cls=FakeSimulator, obstacles=None, **kw):
    from autofly_ue5.expert.env import AutoFlyEnv

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
    return FootprintContactRule(trees, **kw)


def test_the_contact_distance_keeps_the_rotor_span_outside_the_circle_on_the_ending_step_too():
    # The circle contains every branch the band meets; the drone's rotor tips reach DRONE_HALF_SPAN_M from its centre;
    # the rule is checked after the step, so the ending step can carry the centre a full step (0.4 m) past the boundary;
    # a full-rate turn at full speed bows that step's chord by 2 (1 - cos 0.1) = 0.01 m. The rule must cover all three.
    # (0.5 m, the first version, let 40 of 40 scripted flights end with a rotor tip inside the circle.)
    from autofly_ue5.expert.footprint_contact import FOOTPRINT_CONTACT_M
    from autofly_ue5.expert.movers import FORWARD_SPEED_MAX_M_S
    from autofly_ue5.sim.types import CONTROL_DT_S

    assert FOOTPRINT_CONTACT_M - FORWARD_SPEED_MAX_M_S * CONTROL_DT_S >= DRONE_HALF_SPAN_M + 0.01
    assert FOOTPRINT_CONTACT_M == 0.9


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
    assert rule.to_json() == {"contact_m": 0.9, "training_margin_m": 0.2, "termination_m": 1.1, "obstacles": 2}


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
    assert task_contact_rules(trees) == {"footprint": {"contact_m": 0.9, "obstacles": 11}}


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
    assert env.contact_rules == {"footprint": {"contact_m": 0.9, "training_margin_m": 0.0, "termination_m": 0.9, "obstacles": 11}}
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
        assert end["footprint_nearest"] == {"tag": tree.tag, "asset": tree.asset, "gap_m": pytest.approx(gap, abs=1e-3)}
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
    vec = make_vec_env(scene, layout, 1, map_path="/Game/AutoFly/Maps/S99", monitor_dir=tmp_path / "m",
                       sim_factory=functools.partial(FakeSimulator, obstacles=wood(layout)), sim_root=tmp_path / "sim",
                       footprint_contact_margin_m=0.2)
    try:
        assert vec.envs[0].unwrapped._footprint_contact_margin_m == 0.2
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


def _train(tmp_path, monkeypatch, *flags, trees=True, scene=None, layout=None, registries=None):
    """train.main() on the tree scene (or s01), or on `scene` / `layout` in its place; `registries` collects the
    registry each make_vec_env call was handed."""
    from autofly_ue5.expert import train
    from tests.test_dynamic_plumbing import _no_launch

    _no_launch(monkeypatch)
    if trees:
        _scene, tree_layout = tree_scene_and_layout()
        monkeypatch.setattr(train, "resolve_scene", lambda scene_id: _tree_resolved(tmp_path, scene, layout))
        monkeypatch.setattr(train, "scene_config_factory",
                            lambda config, movable=(), **kw: functools.partial(FakeSimulator, obstacles=wood(tree_layout)))
    else:
        monkeypatch.setattr(train, "scene_config_factory", lambda config, movable=(), **kw: FakeSimulator)
    seen = []
    real = train.make_vec_env

    def spy(*args, **kwargs):
        seen.append(kwargs.get("footprint_contact_margin_m"))
        if registries is not None:
            registries.append(kwargs.get("registry"))
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "make_vec_env", spy)
    out = tmp_path / "train.json"
    code = train.main(["--scene", "s99" if trees else "s01", *flags, "--run-root", str(tmp_path / "run"), "--out",
                       str(out), "--device", "cpu", "--sim-root", str(tmp_path / "sim"), "--total-timesteps", "10",
                       "--learning-starts", "100", "--buffer-size", "100", "--eval-freq", "1000"])
    return code, out, seen


def test_a_training_run_takes_the_margin_from_the_command_line_and_records_the_task_rule(tmp_path, monkeypatch):
    code, out, seen = _train(tmp_path, monkeypatch, "--footprint-contact-margin", "0.2")
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert record["identity"]["footprint_contact"] == {"contact_m": 0.9, "training_margin_m": 0.2}
    assert record["config"]["footprint_contact_margin_m"] == 0.2
    assert record["contact_rules"] == {"footprint": {"contact_m": 0.9, "obstacles": 11}}
    assert seen == [0.2, 0.0], "training ends episodes at the margin; evaluation keeps the task's rule"


def test_a_tree_scene_without_the_margin_still_names_the_task_rule_in_its_identity(tmp_path, monkeypatch):
    code, out, _seen = _train(tmp_path, monkeypatch)
    record = json.loads(out.read_text())
    assert code == 0 and record["identity"]["footprint_contact"] == {"contact_m": 0.9, "training_margin_m": 0.0}


def test_a_pillar_scenes_identity_and_record_are_as_before(tmp_path, monkeypatch):
    code, out, seen = _train(tmp_path, monkeypatch, trees=False)
    record = json.loads(out.read_text())
    assert code == 0 and "footprint_contact" not in record["identity"] and record["contact_rules"] is None
    assert record["config"]["footprint_contact_margin_m"] == 0.0 and seen == [0.0, 0.0]


@pytest.mark.parametrize("flags, trees, why", [
    (("--footprint-contact-margin", "0.2"), False, "no mesh obstacle"),
    (("--footprint-contact-margin", "-0.2"), True, "non-negative"),
    (("--footprint-contact-margin", "nan"), True, "non-negative"),
    (("--footprint-contact-margin", "0.3"), True, "1.1 m"),  # 0.9 + 0.3 is beyond what a start can be from a circle
    (("--static-contact-m", "1.1"), True, "no pillar or box"),  # every obstacle is a tree: the boundary ends nothing
])
def test_a_margin_that_cannot_apply_is_refused_before_the_run_root_is_claimed(tmp_path, monkeypatch, capsys, flags, trees, why):
    code, _out, seen = _train(tmp_path, monkeypatch, *flags, trees=trees)
    assert code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and seen == []


def test_the_trainers_help_states_the_task_rule_from_its_constant():
    from autofly_ue5.expert.footprint_contact import FOOTPRINT_CONTACT_M
    from autofly_ue5.expert.train import build_arg_parser

    action = next(a for a in build_arg_parser()._actions if "--footprint-contact-margin" in a.option_strings)
    assert f"{FOOTPRINT_CONTACT_M:g} m from the circle" in action.help and "0.5 m" not in action.help


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

    monkeypatch.setattr(train, "load_registry", load_once)
    monkeypatch.setattr(env_module, "load_registry", second_read)
    monkeypatch.setattr(footprint_module, "load_registry", second_read)
    registries = []
    code, out, _seen = _train(tmp_path, monkeypatch, "--footprint-contact-margin", "0.2", registries=registries)
    record = json.loads(out.read_text())
    assert code == 0 and record["status"] == "ok", record["error"]
    assert len(loaded) == 1 and registries == [loaded[0], loaded[0]], "training and evaluation workers"
    assert record["contact_rules"] == {"footprint": {"contact_m": 0.9, "obstacles": 11}}
    assert record["identity"]["footprint_contact"] == {"contact_m": 0.9, "training_margin_m": 0.2}


def test_run_identity_reads_the_registry_it_is_given():
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.train import run_identity

    resolved = _tree_resolved(Path("unused"))
    registry = load_registry()
    primitives = dataclasses.replace(registry, assets={name: dataclasses.replace(entry, radius_profile_m=None)
                                                       for name, entry in registry.assets.items()})
    assert "footprint_contact" in run_identity(resolved, ObsConfig(), registry=registry)
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
                       instances_path=tmp_path / "no_measurement.json", obs_config_reader=lambda path: ObsConfig())
    assert gate["status"] == "ok"
    assert gate["contact_rules"] == {"footprint": {"contact_m": 0.9, "obstacles": 11}}
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
    sims = []
    gate = m2_gate.run(scene="s99", model_paths={"model": checkpoint}, conditions=["deterministic"], n_episodes=4,
                       seed_base=EVAL_SEED_BASE, instance=5, out_path=tmp_path / "gate.json",
                       sim_factory=recording_tree_factory(layout, sims),
                       load_model=lambda path: AtTheNearestTree(layout, sims), sim_root=tmp_path / "sim",
                       instances_path=tmp_path / "no_measurement.json", obs_config_reader=lambda path: ObsConfig())
    report = gate["checkpoints"]["model"]["deterministic"]
    endings = [e for e in report["per_episode"] if e.get("collision_source") == "footprint"]
    assert endings and report["collision_sources"]["footprint"] == len(endings)
    tags = {inst.tag: inst.asset for inst in layout.instances}
    for episode in endings:
        contact = episode["footprint_contact"]
        assert episode["outcome"] == "collision" and contact["rule"] == "footprint"
        assert set(contact) == {"tag", "asset", "gap_m", "rule", "bearing_deg", "drone_forward_m_s"}
        assert tags[contact["tag"]] == contact["asset"] and DRONE_HALF_SPAN_M <= contact["gap_m"] <= 0.9 + 1e-9
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
