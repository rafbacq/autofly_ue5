"""The closed-loop evaluation harness (spec §12 M6; plan 6 item 6, revision 1 items 4-5) against FakeSimulator: a policy
flies a frozen condition file's episodes as the dataset's were flown (a0, the target pool, lock-step, the task's rules),
sees exactly what a record holds, never the privileged target fields unless asked, has its faulted attempts replayed and
never scored, and is scored per scene and overall (SR, CR, out-of-bounds, timeout, PER) in a record written as it goes.

AutoFlyEnv's `target_split` (the unseen-target draw) lands with plan 6's target work: the test that needs it skips until
then, and the harness's plumbing and refusal are tested against stand-ins either way."""

from __future__ import annotations

import functools
import inspect
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.types import CameraPoseError
from tests.test_dynamic_env import scene_objects
from tests.test_eval_episodes import fake_resolver


def fake_sims(resolved, scene_config):
    """The harness's sim_factory hook on the fake: a dynamic scene's pillars are scene objects the movers can move."""
    if resolved.scene.dynamic is not None:
        return functools.partial(FakeSimulator, scene_objects=scene_objects(resolved.layout))
    return FakeSimulator


def _condition(tmp_path, name="seen", scenes=("s01",), n=3, resolve=None, gates=None) -> Path:
    from autofly_ue5.evaluation.episodes import write_condition
    from autofly_ue5.scenes.resolve import resolve_scene

    return write_condition(name, tmp_path / "eval_episodes", scenes=scenes, episodes_per_scene=n,
                           resolve=resolve or resolve_scene, gates_dir=gates or tmp_path / "no_gates")


def _run(tmp_path, policy, condition, *, out="record.json", **kw):
    from autofly_ue5.evaluation.harness import run_condition

    kw.setdefault("sim_factory", fake_sims)
    return run_condition(policy, condition, instance=5, sim_root=tmp_path / "sim", out_path=tmp_path / "out" / out, **kw)


class Recording:
    """Wraps a policy and keeps everything it was handed."""

    name = "recording"

    def __init__(self, inner):
        self.inner = inner
        self.observations = []
        self.resets = []

    @property
    def requires_privileged_state(self):
        return getattr(self.inner, "requires_privileged_state", False)

    @property
    def obs_config(self):
        return getattr(self.inner, "obs_config", None)

    def reset(self, instruction):
        self.resets.append(instruction)
        self.inner.reset(instruction)

    def act(self, obs):
        self.observations.append(obs)
        return self.inner.act(obs)


class Constant:
    """The same action whatever it sees."""

    name = "constant"

    def __init__(self, action=(1.0, 0.1, 0.0)):
        self.action = np.array(action, np.float32)

    def reset(self, instruction):
        pass

    def act(self, obs):
        return self.action


class LaunchCounter:
    """A sim_factory hook that counts launches (and refuses them)."""

    def __init__(self):
        self.launches = 0

    def __call__(self, resolved, scene_config):
        counter = self

        class Counted(FakeSimulator):
            def launch(self, map_path, instance):
                counter.launches += 1
                raise AssertionError("launched")

        return Counted


# --------------------------------------------------------------------------------------------------------
# A whole condition, scored.
# --------------------------------------------------------------------------------------------------------
def test_a_policy_flies_every_episode_of_a_condition_and_the_record_scores_it(tmp_path):
    from autofly_ue5.evaluation.episodes import load_condition
    from autofly_ue5.evaluation.harness import TARGETS_PATH
    from autofly_ue5.evaluation.metrics import wilson_interval
    from autofly_ue5.evaluation.policy import ScriptedPolicy

    path = _condition(tmp_path, n=3)
    condition = load_condition(path)
    policy = Recording(ScriptedPolicy())
    record = _run(tmp_path, policy, path, privileged_state=True)
    assert record["status"] == "ok" and record["error"] is None
    assert record["condition"] == {"name": "seen", "file": str(path), "sha256": condition.sha256,
                                   "episodes_per_scene": 3, "scenes": ["s01"]}
    assert record["policy"]["name"] == "recording" and record["privileged_state"] is True and record["a0"] == "sector8"
    assert record["target_pool"]["path"] == str(TARGETS_PATH) and len(record["target_pool"]["sha256"]) == 64
    assert record["contact_rule_shape"] == "outline" and record["max_steps"] == 300
    episodes = record["episodes"]
    assert [(e["scene"], e["seed"]) for e in episodes] == [(e.scene, e.seed) for e in condition.episodes]
    assert all(e["outcome"] == "success" for e in episodes), "nothing stands in a straight line on the fake"
    for e in episodes:
        assert e["target"]["name"] in e["instruction"] and e["target"]["seen"] is True and e["target"]["category"]
        assert e["l_opt"]["straight_m"] > 40.0 and e["l_opt"]["grid_m"] >= e["l_opt"]["straight_m"]
        assert e["path_length_m"] >= e["l_opt"]["straight_m"] - 0.5
        assert 0.95 <= e["per"]["lower"] <= e["per"]["upper"] <= 1.0
        assert e["actions"]["n"] == e["steps"] and e["actions"]["max"][0] == pytest.approx(2.0)
        assert e["retries"] == 0 and e["policy_act_s"]["max"] >= 0.0
    scene = record["scenes"]["s01"]
    assert scene["status"] == "ok" and scene["episodes_requested"] == 3 and scene["contact_rules"] is None
    assert scene["scene_config"]["file"] == "scene_autofly_s01_fast.jsonc"
    for summary in (scene["metrics"], record["overall"]):
        assert summary["episodes"] == 3 and summary["success_rate"] == {"value": 1.0, "wilson95": list(wilson_interval(3, 3))}
        assert summary["per"]["episodes"] == 3 and summary["per"]["lower"] <= summary["per"]["upper"]
    first = policy.observations[0]
    assert first.step == 0 and abs(first.state[1]) <= math.pi / 8 + 1e-6, "the a0-aligned start faces the target's sector"
    assert json.loads((tmp_path / "out" / "record.json").read_text()) == json.loads(json.dumps(record))
    summary_md = (tmp_path / "out" / "record.md").read_text()
    assert "| s01 | ok | 3 |" in summary_md and "1.000 [" in summary_md


def test_the_privileged_fields_are_zeroed_unless_asked_and_everything_else_is_the_same(tmp_path):
    path = _condition(tmp_path, n=2)
    seen = {}
    for privileged in (False, True):
        policy = Recording(Constant())
        _run(tmp_path, policy, path, privileged_state=privileged, max_steps=8, out=f"{privileged}.json")
        seen[privileged] = policy.observations
    assert len(seen[False]) == len(seen[True]) == 16
    for hidden, given in zip(seen[False], seen[True]):
        assert hidden.privileged_state is False and given.privileged_state is True
        assert hidden.state.dtype == np.float32 and hidden.state.shape == (9,)
        assert hidden.state[0] == 0.0 and hidden.state[1] == 0.0 and given.state[0] > 10.0
        assert np.array_equal(hidden.state[2:], given.state[2:]) and np.array_equal(hidden.rgb, given.rgb)
        assert hidden.rgb.shape == (256, 256, 3) and hidden.rgb.dtype == np.uint8
        assert "expert_obs" in hidden.extras and hidden.instruction == given.instruction


def test_a_policy_sees_exactly_what_the_collector_records(tmp_path):
    # The same seed flown by the collector and through the harness, with the same pilot: every frame, state and the
    # instruction a policy is handed are those the dataset stores (collect.collector.record_inputs).
    from autofly_ue5.collect.collector import collect
    from autofly_ue5.dataset.raw import RawDatasetWriter
    from autofly_ue5.dataset.targets import TargetPool
    from autofly_ue5.evaluation.episodes import load_condition
    from autofly_ue5.evaluation.harness import TARGETS_PATH
    from autofly_ue5.evaluation.policy import ExpertPolicy
    from autofly_ue5.expert.env import AutoFlyEnv
    from autofly_ue5.expert.obs import ObsConfig
    from autofly_ue5.expert.resilient import ResilientAutoFlyEnv
    from autofly_ue5.scenes.resolve import resolve_scene
    from tests.test_m2_gate import _StraightAtTargetModel

    path = _condition(tmp_path, n=1)
    seed = load_condition(path).episodes[0].seed
    pool = TargetPool.load(TARGETS_PATH)
    resolved = resolve_scene("s01")
    base = AutoFlyEnv(resolved.scene, resolved.layout, FakeSimulator, map_path=resolved.map_path, instance=0,
                      target_pool=pool)
    env = ResilientAutoFlyEnv(base, instance=0, sim_root=tmp_path / "collector_sim")
    writer = RawDatasetWriter(tmp_path / "data", "same", scene=resolved.scene, provenance={})
    assert collect(_StraightAtTargetModel(), env, scene=resolved.scene, writer=writer, seed_base=seed, n_keep=1,
                   max_attempts=1)["kept"] == 1
    root = tmp_path / "data" / "same"
    entry = json.loads((root / "manifest.json").read_text())["episodes"][0]
    states = np.load(root / entry["path"] / "steps.npz")["state"]
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1] for p in sorted((root / entry["path"] / "frames").glob("*.png"))]

    policy = Recording(ExpertPolicy(_StraightAtTargetModel(), deterministic=True, obs_config=ObsConfig()))
    record = _run(tmp_path, policy, path, privileged_state=True, target_pool=pool)
    assert len(policy.observations) == entry["steps"] == len(frames)
    for obs, frame, state in zip(policy.observations, frames, states):
        assert np.array_equal(obs.rgb, frame) and np.array_equal(obs.state, state)
        assert obs.instruction == entry["instruction"]
    assert record["episodes"][0]["target"]["name"] == entry["target_name"] and record["episodes"][0]["outcome"] == "success"


def test_a_backend_fault_replays_the_seed_from_a_fresh_policy_and_is_never_scored(tmp_path):
    from tests.test_expert_train import _FlakyFakeSimulator

    path = _condition(tmp_path, n=1)
    policy = Recording(Constant((2.0, 0.0, 0.0)))
    flaky = functools.partial(_FlakyFakeSimulator, fail_on_step_calls=(3,), error=CameraPoseError)
    record = _run(tmp_path, policy, path, max_steps=10, sim_factory=lambda resolved, config: flaky)
    (episode,) = record["episodes"]
    assert episode["retries"] == 1 and episode["fault_counts"] == {"CameraPoseError": 1}
    assert episode["outcome"] == "timeout" and episode["steps"] == 10, "the replayed attempt, flown whole"
    assert len(policy.resets) == 2 and policy.resets[0] == policy.resets[1], "the replay starts the policy afresh"
    assert record["overall"]["episodes"] == 1 and record["overall"]["outcomes"]["timeout"] == 1
    assert record["scenes"]["s01"]["backend_faults"]["fault_counts"]["CameraPoseError"] == 1


def test_a_policy_is_described_once_per_run(tmp_path):
    # A VLA's describe() may hash a checkpoint of many gigabytes: once for the record, never per episode.
    class Described(Constant):
        calls = 0

        def describe(self):
            Described.calls += 1
            return {"name": "described", "checkpoint_sha256": "0" * 64}

    record = _run(tmp_path, Described(), _condition(tmp_path, n=3), max_steps=2)
    assert Described.calls == 1 and record["policy"] == {"name": "described", "checkpoint_sha256": "0" * 64}


def test_episode_limits_and_a_scene_subset_select_part_of_a_condition(tmp_path):
    path = _condition(tmp_path, scenes=("s01", "s01d"), n=3)
    record = _run(tmp_path, Constant(), path, scenes=["s01d"], max_episodes_per_scene=2, max_steps=4)
    assert list(record["scenes"]) == ["s01d"] and len(record["episodes"]) == 2
    assert record["scenes"]["s01d"]["episodes_requested"] == 2 and record["max_episodes_per_scene"] == 2
    with pytest.raises(ValueError, match="s07"):
        _run(tmp_path, Constant(), path, scenes=["s07"], out="other.json")


def test_a_dynamic_scene_flies_its_movers_and_leaves_them_off_the_grid(tmp_path):
    from autofly_ue5.evaluation.policy import HoverPolicy

    path = _condition(tmp_path, scenes=("s01d",), n=2)
    record = _run(tmp_path, HoverPolicy(), path, max_steps=3)
    assert record["status"] == "ok" and record["scenes"]["s01d"]["dynamic"] is True
    for episode in record["episodes"]:
        assert episode["outcome"] == "timeout" and episode["n_movers"] >= 1
        assert episode["l_opt"]["excluded_obstacles"] == episode["n_movers"], "the movers' homes are left off the grid"
        assert episode["per"] == {"lower": None, "upper": None}, "PER is a successful trial's (a short failure is no feat)"


def test_a_mesh_scene_is_scored_under_the_outline_rule(tmp_path):
    # The task's rule (expert/footprint_contact.py), never the circle an expert may have trained under.
    import dataclasses

    from autofly_ue5.expert.footprint_contact import OUTLINE_CONTACT_M
    from tests.test_footprint_contact import (
        AtTheNearestTree,
        _tree_resolved,
        outline_fields_for,
        recording_tree_factory,
        tree_scene_and_layout,
    )

    scene, layout = tree_scene_and_layout()
    outline_fields_for(layout)  # skips on a checkout without assets/footprints/
    trees = _tree_resolved(tmp_path)
    as_s02 = dataclasses.replace(trees, scene=dataclasses.replace(scene, id="s02"))
    path = _condition(tmp_path, scenes=("s02",), n=3, resolve=lambda scene_id: as_s02)
    sims: list = []

    class AtATree:
        name = "at_a_tree"

        def __init__(self):
            self.pilot = AtTheNearestTree(layout, sims)

        def reset(self, instruction):
            pass

        def act(self, obs):
            return self.pilot.predict(None)[0]

    record = _run(tmp_path, AtATree(), path, resolve=lambda scene_id: as_s02,
                  sim_factory=lambda resolved, config: recording_tree_factory(layout, sims))
    rules = record["scenes"]["s02"]["contact_rules"]["footprint"]
    assert rules["shape"] == "outline" and rules["contact_m"] == OUTLINE_CONTACT_M
    endings = [e for e in record["episodes"] if e["collision_source"] == "footprint"]
    assert endings and all(e["footprint_contact"]["shape"] == "outline" for e in endings)
    assert record["overall"]["collision_sources"].get("footprint", 0) == len(endings)


# --------------------------------------------------------------------------------------------------------
# Refused before anything launches.
# --------------------------------------------------------------------------------------------------------
def test_the_scripted_baseline_is_refused_without_the_privileged_state_before_anything_launches(tmp_path):
    from autofly_ue5.evaluation.policy import ScriptedPolicy

    launches = LaunchCounter()
    with pytest.raises(ValueError, match="privileged"):
        _run(tmp_path, ScriptedPolicy(), _condition(tmp_path, n=1), sim_factory=launches)
    assert launches.launches == 0 and not (tmp_path / "out").exists()


def test_a_record_that_is_not_json_is_refused_as_its_summary_would_take_its_place(tmp_path):
    launches = LaunchCounter()
    with pytest.raises(ValueError, match=".json"):
        _run(tmp_path, Constant(), _condition(tmp_path, n=1), out="record.md", sim_factory=launches)
    assert launches.launches == 0


def test_a_scene_whose_layout_changed_since_the_file_was_written_is_refused(tmp_path):
    import dataclasses

    resolve = fake_resolver()
    path = _condition(tmp_path, n=1, resolve=resolve)
    rebuilt = lambda scene_id: dataclasses.replace(resolve(scene_id), layout_sha256="rebuilt")  # noqa: E731
    launches = LaunchCounter()
    with pytest.raises(ValueError, match="layout"):
        _run(tmp_path, Constant(), path, resolve=rebuilt, sim_factory=launches)
    assert launches.launches == 0 and not (tmp_path / "out").exists()


def test_a_pending_scene_is_listed_and_flown_once_a_gate_record_of_it_passes(tmp_path):
    from autofly_ue5.evaluation.policy import HoverPolicy
    from tests.test_eval_episodes import _gates

    gates = _gates(tmp_path, m4_s05r_gate=True)
    path = _condition(tmp_path, name="test_seen", scenes=("s05r", "s06r"), n=2, resolve=fake_resolver(), gates=gates)
    record = _run(tmp_path, HoverPolicy(), path, max_steps=2, resolve=fake_resolver(), gates_dir=gates)
    assert record["status"] == "ok" and record["scenes"]["s05r"]["status"] == "ok"
    pending = record["scenes"]["s06r"]
    assert pending["status"] == "pending" and "m4_s06r_gate.json" in pending["reason"] and pending["metrics"] is None
    assert {e["scene"] for e in record["episodes"]} == {"s05r"} and record["overall"]["episodes"] == 2
    assert "| s06r | pending |" in (tmp_path / "out" / "record.md").read_text()
    _gates(tmp_path, m4_s06r_gate=True)
    later = _run(tmp_path, HoverPolicy(), path, max_steps=2, resolve=fake_resolver(), gates_dir=gates, out="later.json")
    assert later["scenes"]["s06r"]["status"] == "ok" and later["scenes"]["s06r"]["gate"] == "m4_s06r_gate.json"
    assert later["overall"]["episodes"] == 4


# --------------------------------------------------------------------------------------------------------
# A policy that breaks.
# --------------------------------------------------------------------------------------------------------
class Breaks(Constant):
    name = "breaks"

    def __init__(self, after: int, bad=None):
        super().__init__((2.0, 0.0, 0.0))
        self.left = after
        self.bad = bad

    def act(self, obs):
        self.left -= 1
        if self.left < 0:
            if self.bad is not None:
                return self.bad
            raise RuntimeError("the model crashed")
        return self.action


@pytest.mark.parametrize("bad, says", [(None, "the model crashed"), (np.array([np.nan, 0.0, 0.0]), "finite"),
                                       (np.zeros(2), "3 numbers")])
def test_a_policy_that_breaks_stops_the_run_and_its_record_keeps_what_was_flown(tmp_path, bad, says):
    record = _run(tmp_path, Breaks(after=7, bad=bad), _condition(tmp_path, n=3), max_steps=5)
    assert record["status"] == "failed" and says in record["error"] and "breaks" in record["error"]
    assert len(record["episodes"]) == 1 and record["scenes"]["s01"]["status"] == "failed"
    on_disk = json.loads((tmp_path / "out" / "record.json").read_text())
    assert on_disk["status"] == "failed" and len(on_disk["episodes"]) == 1
    assert (tmp_path / "out" / "record.md").is_file()


def test_a_run_stopped_from_outside_says_so_in_its_record(tmp_path):
    class Interrupted(Constant):
        name = "interrupted"

        def __init__(self):
            super().__init__()
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            if self.calls == 7:
                raise KeyboardInterrupt
            return self.action

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, Interrupted(), _condition(tmp_path, n=3), max_steps=5)
    on_disk = json.loads((tmp_path / "out" / "record.json").read_text())
    assert on_disk["status"] == "interrupted" and len(on_disk["episodes"]) == 1, "never 'ok' for a run that did not finish"


def test_a_backend_that_cannot_fly_a_scene_fails_that_scene_and_the_run_goes_on(tmp_path):
    from autofly_ue5.evaluation.policy import HoverPolicy

    class Broken(FakeSimulator):
        def launch(self, map_path, instance):
            raise RuntimeError("launch refused: a foreign GPU job holds 4.4 GB")

    path = _condition(tmp_path, scenes=("s01", "s01d"), n=1)
    record = _run(tmp_path, HoverPolicy(), path, max_steps=2,
                  sim_factory=lambda resolved, config: Broken if resolved.scene.id == "s01" else fake_sims(resolved, config))
    assert record["status"] == "partial" and "foreign GPU job" in record["scenes"]["s01"]["error"]
    assert record["scenes"]["s01"]["status"] == "failed" and record["scenes"]["s01d"]["status"] == "ok"
    assert [e["scene"] for e in record["episodes"]] == ["s01d"]


# --------------------------------------------------------------------------------------------------------
# The unseen-target split (AutoFlyEnv's target_split, landing with plan 6's target work).
# --------------------------------------------------------------------------------------------------------
def test_an_unseen_condition_is_refused_before_launch_when_the_env_cannot_draw_unseen_targets(tmp_path, monkeypatch):
    import autofly_ue5.evaluation.harness as harness

    monkeypatch.setattr(harness, "env_accepts_target_split", lambda: False)
    launches = LaunchCounter()
    with pytest.raises(ValueError, match="target_split"):
        _run(tmp_path, Constant(), _condition(tmp_path, name="unseen_targets", n=1), sim_factory=launches)
    assert launches.launches == 0


def test_the_harness_passes_target_split_only_when_the_condition_asks_for_it(tmp_path, monkeypatch):
    import autofly_ue5.evaluation.harness as harness
    from autofly_ue5.expert.env import AutoFlyEnv

    handed = []

    def recording_env(*args, **kwargs):
        handed.append(dict(kwargs))
        kwargs.pop("target_split", None)  # this checkout's AutoFlyEnv may not take it yet
        return AutoFlyEnv(*args, **kwargs)

    monkeypatch.setattr(harness, "env_accepts_target_split", lambda: True)
    monkeypatch.setattr(harness, "AutoFlyEnv", recording_env)
    _run(tmp_path, Constant(), _condition(tmp_path, n=1), max_steps=2)
    assert handed and all("target_split" not in kwargs for kwargs in handed), "a seen condition leaves the scene's split"
    handed.clear()
    _run(tmp_path, Constant(), _condition(tmp_path, name="unseen_targets", n=1), max_steps=2, out="unseen.json")
    assert handed and all(kwargs["target_split"] == "unseen" for kwargs in handed)


def _env_takes_target_split() -> bool:
    from autofly_ue5.expert.env import AutoFlyEnv

    return "target_split" in inspect.signature(AutoFlyEnv.__init__).parameters


@pytest.mark.skipif(not _env_takes_target_split(),
                    reason="AutoFlyEnv does not accept target_split yet (the unseen-target split lands with plan 6's "
                           "target work); this runs once it does")
def test_an_unseen_condition_draws_every_target_from_the_unseen_entries(tmp_path):
    record = _run(tmp_path, Constant(), _condition(tmp_path, name="unseen_targets", n=4), max_steps=2)
    assert record["status"] == "ok" and len(record["episodes"]) == 4
    for episode in record["episodes"]:
        assert episode["target_split"] == "unseen" and episode["target"]["seen"] is False
        assert all(name != episode["target"]["name"] for name in episode["distractors"])
