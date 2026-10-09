"""The session warm-up (expert/env.py, 2026-10-08 evening): at the first pool reset of every simulator session, before
the episode's own reset, every (asset, material) pair the env could draw is spawned once ahead of a clear view and held
until the probe's depth check sees it (scripts/probe_targets.py's procedure, validate/target_probe.py), so an object's
first use (slow under load: ~20 steps for s05 seed 451000001) stays out of the episodes. Against a fake that draws each
asset only N steps after its spawn (N per asset; never, for one)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from autofly_ue5.sim.fake import FakeSimulator
from tests.test_expert_episode import scene_and_layout
from tests.test_visibility import render_depth

# Short names in the spawn table: the primitive pool's cylinder, crate (Cube) and boulder.
DELAYS = {"Cylinder": 1, "Cube": 3, "boulder_01": None}  # None: never drawn


class LaggingFake(FakeSimulator):
    """Draws the ground and every spawned object as its registry box, each only once it has existed for its asset's
    delay in steps (DELAYS; None: never). Counts resets and spawns."""

    delays = DELAYS

    def __init__(self, *args, **kwargs):
        from autofly_ue5.scenes.model import load_registry

        super().__init__(*args, **kwargs)
        self._by_short = {}
        for asset in load_registry().assets.values():
            self._by_short.setdefault(asset.ue_path.rsplit("/", 1)[-1], asset)
        self.spawned_at: dict[str, int] = {}
        self.resets = 0
        self.spawns: list[str] = []

    def reset(self, pose):
        self.resets += 1
        return super().reset(pose)

    def spawn(self, name, asset, pose, scale, material=None):
        unique = super().spawn(name, asset, pose, scale, material)
        self.spawned_at[unique] = self.steps_taken
        self.spawns.append(asset)
        return unique

    def observe(self):
        from autofly_ue5.validate.visibility import spawned_box

        obs = super().observe()
        boxes = []
        for name, (asset, pose, scale, _material) in self._objects.items():
            delay = self.delays.get(asset, 0)
            if delay is None or self.steps_taken - self.spawned_at[name] < delay:
                continue
            box = spawned_box(name, self._by_short[asset], scale, pose.x, pose.y)
            boxes.append(((box.x, box.y, box.z_center), tuple(2 * h for h in box.half)))
        return dataclasses.replace(obs, depth=render_depth(obs.pose, boxes=boxes))


def _env(tmp_path, *, sim=LaggingFake, pool=None, **kw):
    from autofly_ue5.expert.env import AutoFlyEnv
    from tests.test_dataset_targets import primitive_pool

    scene, layout = scene_and_layout()
    pool = pool if pool is not None else primitive_pool(tmp_path)
    return AutoFlyEnv(scene, layout, sim, map_path="/Game/AutoFly/Maps/S01", instance=0, target_pool=pool, **kw)


def _pairs(pool, split="train", scene_id="s01"):
    targets, rest = pool.draw_lists(split, scene_id=scene_id)
    return {(e.asset, e.material) for e in (*targets, *rest)}


def test_the_first_pool_reset_of_a_session_warms_every_pair_it_could_draw(tmp_path, capsys):
    from autofly_ue5.expert.env import WARMUP_MAX_STEPS

    env = _env(tmp_path)
    _obs, info = env.reset(seed=3)
    warmup = info["session_warmup"]
    per_pair = {(p["asset"], p["material"]): p for p in warmup["per_pair"]}
    assert set(per_pair) == _pairs(env.target_pool) and warmup["pairs"] == len(per_pair) == 6
    # steps to visible follow each asset's delay (a hover step 1 after the spawn at the earliest)
    for (asset, _material), p in per_pair.items():
        delay = {"cylinder": 1, "crate": 3, "boulder_01": None}[asset]
        assert p["steps"] == delay and (p["error"] is None), p
    assert [f["asset"] for f in warmup["failed"]] == ["boulder_01"] and warmup["warmed"] == 5
    assert warmup["steps"] == 2 * 1 + 3 * 3 + WARMUP_MAX_STEPS and warmup["wall_s"] >= 0
    # a pair never shown stays cold: the episode is a first use exactly when it drew it
    drew_boulder = any(e.asset == "boulder_01" for e in (env.setup.target, *env.setup.distractor_entries))
    assert info["first_use"] is drew_boulder and _pairs(env.target_pool) - {("boulder_01", None)} <= env._warm
    assert env.session_warmups == [warmup]
    err = capsys.readouterr().err
    assert err.count("WARMUP instance 0:") == 1 and "WARMUP-NEVER instance 0: grey boulder" in err
    # the episode itself starts as before: its own reset at its start, after the warm-up's resets at the view
    assert env._sim.resets == 6 + 1 and env.setup is not None and info["prewarm_steps"] == 5


def test_a_session_is_warmed_once_and_a_relaunched_one_again(tmp_path):
    from autofly_ue5.expert.resilient import ResilientAutoFlyEnv

    env = _env(tmp_path)
    wrapper = ResilientAutoFlyEnv(env, instance=0, sim_root=tmp_path / "sim")
    assert "session_warmup" in env.reset(seed=3)[1]
    sim = env._sim
    resets, spawns = sim.resets, len(sim.spawns)
    _obs, info = env.reset(seed=4)
    assert "session_warmup" not in info and sim.resets == resets + 1, "the same session: no warm-up again"
    assert len(sim.spawns) == spawns + 1 + len(env.setup.distractors), "only the episode's own spawns"
    wrapper._relaunch()
    _obs, info = env.reset(seed=4)
    assert env._sim is not sim and "session_warmup" in info and len(env.session_warmups) == 2


def test_an_unseen_target_evaluation_warms_the_unseen_entries_too(tmp_path):
    from tests.test_dataset_targets import unseen_pool

    pool = unseen_pool(tmp_path)
    env = _env(tmp_path, pool=pool, target_split="unseen")
    warmup = env.reset(seed=3)[1]["session_warmup"]
    pairs = {(p["asset"], p["material"]) for p in warmup["per_pair"]}
    assert pairs == _pairs(pool, "unseen") and {(e.asset, e.material) for e in pool.unseen_targets(scene_id="s01")} <= pairs


def test_a_depth_camera_that_shows_nothing_skips_the_warm_up_once_and_says_why(tmp_path, capsys):
    class Blank(LaggingFake):
        def observe(self):
            return FakeSimulator.observe(self)  # depth all inf: no ground, nothing to judge against

    env = _env(tmp_path, sim=Blank)
    warmup = env.reset(seed=3)[1]["session_warmup"]
    assert warmup["error"] and "depth" in warmup["error"] and warmup["per_pair"] == []
    assert env._sim.resets == 1 + 1, "one reset at the view, then the episode's own"
    assert len(env._sim.spawns) == 1 + len(env.setup.distractors), "nothing spawned but the episode's objects"
    assert "WARMUP instance 0" in capsys.readouterr().err
    assert "session_warmup" not in env.reset(seed=4)[1], "not retried at every reset"


def test_a_warm_up_cut_short_by_a_fault_resumes_where_it_stopped(tmp_path):
    from autofly_ue5.sim.types import CameraPoseError

    calls = {"n": 0}

    class FaultsOnce(LaggingFake):
        def step(self, dt=0.2):
            calls["n"] += 1
            if calls["n"] == 4:  # inside the warm-up, after its first pair showed
                raise CameraPoseError("camera 1.2 m from kinematics")
            return super().step(dt)

    env = _env(tmp_path, sim=FaultsOnce)
    with pytest.raises(CameraPoseError):
        env.reset(seed=3)
    assert len(env._warm) >= 1 and env.session_warmups == []
    warmed_before = set(env._warm)
    warmup = env.reset(seed=3)[1]["session_warmup"]
    assert warmup["warmed"] == 5 and {(p["asset"], p["material"]) for p in warmup["per_pair"]} == _pairs(env.target_pool)
    assert all(p["attempt"] == 1 for p in warmup["per_pair"] if (p["asset"], p["material"]) in warmed_before)


def test_a_spawn_that_fails_for_the_backend_is_a_fault_and_one_the_runtime_refuses_is_a_failed_pair(tmp_path, capsys):
    from autofly_ue5.sim.types import SimConnectionLostError

    class SpawnFails(LaggingFake):
        error: Exception = RuntimeError("There were no objects with name Cube found in the Registry")

        def spawn(self, name, asset, pose, scale, material=None):
            if asset == "Cube" and name.startswith("autofly_warmup"):
                raise type(self).error
            return super().spawn(name, asset, pose, scale, material)

    env = _env(tmp_path, sim=SpawnFails)
    warmup = env.reset(seed=3)[1]["session_warmup"]
    refused = {f["name"]: f["reason"] for f in warmup["failed"]}
    assert set(refused) == {"orange cube", "grid cube", "white box", "grey boulder"} and "RuntimeError" in refused["grid cube"]
    assert "WARMUP-NEVER instance 0: orange cube" in capsys.readouterr().err

    class ConnectionLost(SpawnFails):
        error = SimConnectionLostError("the simulator stopped answering")

    env = _env(tmp_path / "b", sim=ConnectionLost)
    with pytest.raises(SimConnectionLostError):
        env.reset(seed=3)
    assert env.session_warmups == [] and all(p["asset"] != "crate" for p in env._warmup_record["per_pair"]), \
        "never recorded as a pair the runtime cannot spawn: the next reset (a relaunch) tries it again"


def test_the_collector_records_each_session_warm_up_and_gives_it_to_the_first_episode_after_it(tmp_path, capsys):
    from autofly_ue5.collect.collector import collect
    from autofly_ue5.dataset.raw import RawDatasetWriter
    from autofly_ue5.expert.resilient import ResilientAutoFlyEnv
    from tests.test_collect import PROVENANCE, SEED_BASE
    from tests.test_m2_gate import _StraightAtTargetModel

    scene, layout = scene_and_layout()
    base = _env(tmp_path, max_episode_steps=3)
    env = ResilientAutoFlyEnv(base, instance=0, sim_root=tmp_path / "sim")
    writer = RawDatasetWriter(tmp_path / "data", "warm", scene=scene, provenance=PROVENANCE)
    summary = collect(_StraightAtTargetModel(), env, scene=scene, writer=writer, seed_base=SEED_BASE, n_keep=1,
                      max_attempts=3, layout_sha256="test")
    assert len(summary["session_warmups"]) == 1 and summary["session_warmups"][0] == base.session_warmups[0]
    rejects = sorted((tmp_path / "data" / "rejects" / "warm").glob("*.json"))  # 3 steps: every flight times out
    import json

    provenance = [json.loads(p.read_text())["provenance"] for p in rejects]
    first = min(provenance, key=lambda prov: prov["seed"])
    assert first["session_warmup"] == base.session_warmups[0]
    assert all("session_warmup" not in prov for prov in provenance if prov is not first)
    assert "WARMUP instance 0:" in capsys.readouterr().err


def test_the_harness_records_each_session_warm_up_with_its_scene_and_the_first_episode_after_it(tmp_path):
    # The evaluation harness flies its scenes with the real pool (assets/targets.json): the same warm-up, recorded.
    from tests.test_eval_harness import Constant, _condition, _run

    record = _run(tmp_path, Constant(), _condition(tmp_path, n=2), max_steps=2,
                  sim_factory=lambda resolved, config: LaggingFake)
    scene = record["scenes"]["s01"]
    assert len(scene["session_warmups"]) == 1 and scene["session_warmups"][0]["pairs"] == 43, "s01's 43 seen pairs"
    first, second = record["episodes"]
    assert first["session_warmup"] == scene["session_warmups"][0] and "session_warmup" not in second
    assert first["prewarm_steps"] == second["prewarm_steps"] == 5 and isinstance(second["first_use"], bool)


def test_without_a_pool_there_is_no_warm_up():
    from autofly_ue5.expert.env import AutoFlyEnv

    scene, layout = scene_and_layout()
    env = AutoFlyEnv(scene, layout, LaggingFake, map_path="/Game/AutoFly/Maps/S01", instance=0)
    _obs, info = env.reset(seed=3)
    assert "session_warmup" not in info and env.session_warmups == [] and env._sim.resets == 1
    assert np.isfinite(env.last_observation.depth).any()
