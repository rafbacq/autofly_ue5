"""The evaluation harness's policy interface (plan 6 item 6): what a policy sees (the dataset's own RGB, instruction and
state[9], the privileged target fields zeroed unless asked), what it returns (AutoFly's action[3]), and the reference
adapters every VLA is compared with: the SAC expert, a privileged straight-line baseline and a hover floor."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest


def _obs(state=None, *, privileged=True, extras=None, instruction="advance to red barrel while avioding trees and other "
                                                                    "obstacles"):
    from autofly_ue5.evaluation.policy import PolicyObservation

    return PolicyObservation(rgb=np.zeros((256, 256, 3), np.uint8), instruction=instruction,
                             state=np.asarray(state if state is not None else np.zeros(9), np.float32),
                             privileged_state=privileged, step=0, extras=extras or {})


def _state(distance=40.0, bearing=0.0, altitude=2.0) -> np.ndarray:
    from autofly_ue5.dataset.state import STATE2_OFFSET_M

    return np.array([distance, bearing, altitude - STATE2_OFFSET_M, 0.0, 0.0, 0.0, 0.0, 0.0, altitude], np.float32)


def test_the_reference_adapters_are_policies():
    from autofly_ue5.evaluation.policy import ExpertPolicy, HoverPolicy, Policy, ScriptedPolicy
    from autofly_ue5.expert.obs import ObsConfig

    for policy in (HoverPolicy(), ScriptedPolicy(), ExpertPolicy(object(), deterministic=True, obs_config=ObsConfig())):
        assert isinstance(policy, Policy)


def test_the_privileged_fields_are_the_target_s_distance_and_bearing_and_zeroing_keeps_the_rest():
    # Plan 6 revision 1 item 4: state[0] and state[1] come from the target's true position; a policy that reads them could
    # fly from them alone, so the harness hands them over only when told to.
    from autofly_ue5.dataset.state import STATE_FIELDS
    from autofly_ue5.evaluation.policy import PRIVILEGED_STATE_INDICES, zero_privileged

    assert [STATE_FIELDS[i]["name"] for i in PRIVILEGED_STATE_INDICES] == ["distance_to_target_m", "bearing_to_target_rad"]
    state = np.arange(1, 10, dtype=np.float32)
    zeroed = zero_privileged(state)
    assert zeroed.tolist() == [0.0, 0.0, 3, 4, 5, 6, 7, 8, 9] and zeroed.dtype == np.float32
    assert state.tolist() == list(range(1, 10)), "the caller's array is left alone"


def test_hover_commands_nothing():
    from autofly_ue5.evaluation.policy import HoverPolicy

    policy = HoverPolicy()
    policy.reset("anything")
    action = policy.act(_obs(privileged=False))
    assert action.dtype == np.float32 and action.tolist() == [0.0, 0.0, 0.0]


def test_the_scripted_baseline_refuses_to_fly_without_the_privileged_state_and_says_so():
    from autofly_ue5.evaluation.policy import PolicyError, PrivilegedStateRequired, ScriptedPolicy

    assert ScriptedPolicy.requires_privileged_state is True
    with pytest.raises(PrivilegedStateRequired, match="privileged") as err:
        ScriptedPolicy().act(_obs(_state(), privileged=False))
    assert isinstance(err.value, PolicyError)


@pytest.mark.parametrize("bearing, forward, turning", [(0.0, 2.0, 0.0), (0.1, 2.0, 0.2), (-0.1, 2.0, -0.2),
                                                       (0.5, 0.0, 1.0), (-2.0, 0.0, -1.0)])
def test_the_scripted_baseline_turns_to_the_target_and_flies_at_two_metres_a_second_once_facing_it(bearing, forward,
                                                                                                      turning):
    from autofly_ue5.evaluation.policy import ScriptedPolicy

    action = ScriptedPolicy().act(_obs(_state(bearing=bearing)))
    assert action.dtype == np.float32 and action.shape == (3,)
    assert action[0] == pytest.approx(forward) and action[1] == pytest.approx(turning) and action[2] == pytest.approx(0.0)


@pytest.mark.parametrize("altitude, climb", [(2.0, 0.0), (1.5, 0.5), (2.5, -0.5), (0.5, 1.0), (3.9, -1.0)])
def test_the_scripted_baseline_holds_two_metres_of_altitude(altitude, climb):
    from autofly_ue5.evaluation.policy import ScriptedPolicy

    assert ScriptedPolicy().act(_obs(_state(altitude=altitude)))[2] == pytest.approx(climb)


class _RecordingModel:
    """An SB3 model's predict(), recording what it was called with."""

    def __init__(self):
        self.calls = []

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.calls.append((observation, deterministic))
        return np.array([1.5, -0.25, 0.5], np.float32), None


@pytest.mark.parametrize("deterministic", [True, False])
def test_the_expert_flies_on_its_own_observation_from_extras(deterministic):
    from autofly_ue5.evaluation.policy import ExpertPolicy
    from autofly_ue5.expert.obs import ObsConfig

    model = _RecordingModel()
    policy = ExpertPolicy(model, deterministic=deterministic, obs_config=ObsConfig())
    own = {"depth": np.zeros((1, 84, 84), np.float32), "vector": np.zeros(8, np.float32)}
    policy.reset("go through and avoid the trees or other obstacles to reach the red barrel")
    action = policy.act(_obs(_state(), privileged=False, extras={"expert_obs": own}))
    assert action.dtype == np.float32 and action.tolist() == [1.5, -0.25, 0.5]
    assert model.calls == [(own, deterministic)], "SB3's predict on the env's observation, with the asked-for mode"


def test_the_expert_refuses_an_observation_without_its_own():
    from autofly_ue5.evaluation.policy import ExpertPolicy, PolicyError
    from autofly_ue5.expert.obs import ObsConfig

    with pytest.raises(PolicyError, match="expert_obs"):
        ExpertPolicy(_RecordingModel(), deterministic=True, obs_config=ObsConfig()).act(_obs(_state()))


def test_the_expert_loads_its_checkpoint_with_the_observation_it_was_trained_on(tmp_path):
    import hashlib

    from autofly_ue5.evaluation.policy import ExpertPolicy
    from autofly_ue5.expert.obs import ObsConfig

    checkpoint = tmp_path / "avg.zip"
    checkpoint.write_bytes(b"a checkpoint")
    stack = ObsConfig(3, "float16", mover_slots=4)
    loaded = []
    policy = ExpertPolicy.load(checkpoint, deterministic=False, loader=lambda path: loaded.append(path) or "model",
                               obs_config_reader=lambda path: stack)
    assert loaded == [checkpoint] and policy.model == "model" and policy.obs_config == stack
    described = policy.describe()
    assert described["name"] == "expert" and described["checkpoint"] == str(checkpoint)
    assert described["sha256"] == hashlib.sha256(b"a checkpoint").hexdigest() and described["deterministic"] is False
    assert described["obs_config"] == stack.to_json() and described["privileged_observation"] is True


def test_an_expert_checkpoint_that_does_not_exist_is_refused(tmp_path):
    from autofly_ue5.evaluation.policy import ExpertPolicy

    with pytest.raises(FileNotFoundError):
        ExpertPolicy.load(tmp_path / "missing.zip", deterministic=True)


def test_a_policy_s_action_must_be_three_finite_numbers():
    from autofly_ue5.evaluation.policy import PolicyError, checked_action

    assert checked_action([2.0, 0.5, -0.5]).tolist() == [2.0, 0.5, -0.5]
    assert checked_action(np.array([[1.0, 0.0, 0.0]])).shape == (3,), "a batch of one is one action"
    for bad in ([1.0, 2.0], [1.0, math.nan, 0.0], [math.inf, 0.0, 0.0], "fly", None):
        with pytest.raises(PolicyError):
            checked_action(bad)


def test_describe_names_any_policy():
    from autofly_ue5.evaluation.policy import HoverPolicy, ScriptedPolicy, describe_policy

    class Bare:
        def reset(self, instruction):
            pass

        def act(self, obs):
            return np.zeros(3, np.float32)

    assert describe_policy(Bare()) == {"name": "Bare"}
    assert describe_policy(HoverPolicy())["name"] == "hover"
    scripted = describe_policy(ScriptedPolicy())
    assert scripted["name"] == "scripted" and scripted["requires_privileged_state"] is True


# --------------------------------------------------------------------------------------------------------
# De-normalising a VLA's action with the dataset's statistics (OpenVLA's BOUNDS_Q99).
# --------------------------------------------------------------------------------------------------------
STATS = {"q01": [0.4, -0.8, -0.6], "q99": [2.0, 0.8, 0.6], "mean": [1.8, 0.0, 0.0], "std": [0.3, 0.2, 0.1],
         "min": [0.0, -1.0, -1.0], "max": [2.0, 1.0, 1.0]}


def test_unnormalising_inverts_the_q01_q99_scaling_a_vla_was_trained_on():
    from autofly_ue5.evaluation.policy import unnormalize_action

    q01, q99 = np.array(STATS["q01"]), np.array(STATS["q99"])
    assert np.allclose(unnormalize_action([-1.0, -1.0, -1.0], STATS), q01)
    assert np.allclose(unnormalize_action([1.0, 1.0, 1.0], STATS), q99)
    action = np.array([1.2, 0.3, -0.2])
    normalized = np.clip(2 * (action - q01) / (q99 - q01) - 1, -1, 1)  # how OpenVLA normalises a training action
    assert np.allclose(unnormalize_action(normalized, STATS), action)
    masked = unnormalize_action([0.5, 0.5, 0.5], {**STATS, "mask": [True, True, False]})
    assert masked[2] == pytest.approx(0.5), "a masked dimension passes through as OpenVLA's gripper does"
    assert unnormalize_action([0.0, 0.0, 0.0], STATS).dtype == np.float32


def test_the_action_statistics_load_from_openvla_s_dataset_statistics_file(tmp_path):
    from autofly_ue5.evaluation.policy import load_action_statistics

    keyed = tmp_path / "dataset_statistics.json"
    keyed.write_text(json.dumps({"autofly_ue5_proto": {"action": STATS, "proprio": STATS, "num_transitions": 10,
                                                       "num_trajectories": 1}}))
    assert load_action_statistics(keyed, "autofly_ue5_proto")["q99"] == STATS["q99"]
    assert load_action_statistics(keyed)["q01"] == STATS["q01"], "one dataset: its name may be left out"
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"action": STATS, "num_transitions": 10}))
    assert load_action_statistics(flat)["q01"] == STATS["q01"]
    two = tmp_path / "two.json"
    two.write_text(json.dumps({"a": {"action": STATS}, "b": {"action": STATS}}))
    with pytest.raises(ValueError, match="dataset"):
        load_action_statistics(two)
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"action": {"q01": [0.0, 0.0, 0.0]}}))
    with pytest.raises(ValueError, match="q99"):
        load_action_statistics(broken)
