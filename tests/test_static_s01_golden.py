"""Golden record of static s01, taken from the code that flew M2 (5d0abf5) before moving obstacles were added.

M2's gate, `scripts/audit_m2_gate.py` and `scripts/render_episodes.py` replay exact episodes from their seeds, and
every M2 checkpoint was trained on the observation space below. Adding moving pillars (scene s01d) touches the
episode sampler, the env loop and the observation, so this file pins what static s01 must keep doing, byte for
byte: the episode each seed draws, the simulator calls the env makes (and never a `set_object_poses`), the rewards,
outcomes and observations of a scripted flight, and the info keys (which may only gain additions).

The fixture is regenerated only on purpose, from a commit known to behave like M2:

    env -u PYTHONPATH .venv/bin/python -m tests.test_static_s01_golden --write
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys

import numpy as np
from gymnasium import spaces

from autofly_ue5.paths import FIXTURES_DIR
from autofly_ue5.sim.fake import FakeSimulator

GOLDEN = FIXTURES_DIR / "golden" / "s01_static.json"

# Training workers (sessions 0 and 1), SB3's explicit first-reset seeds, the M2 gate's episodes (including the ones
# rendered on 2026-09-26) and training-time evaluation. Literal values, not imports from seeds.py: the fixture must
# not follow a change of those constants.
SETUP_SEEDS = (
    0, 1, 2, 3,
    1_000_000, 1_000_001, 1_000_002, 1_100_000, 1_100_001,
    2_000_000, 2_000_001, 3_000_000, 3_000_001, 4_000_000, 4_000_001,
    100_000_000, 100_000_001, 100_000_002, 100_000_027, 100_000_061, 100_000_097, 100_000_124, 100_000_175,
    100_000_179, 100_000_199,
    200_000_000, 200_000_001, 200_000_019,
)
# (seed, whether the fake knows the pillars, yaw noise): pillar-free flights reach the target, the others crash.
FLIGHTS = ((5, False, 0.0), (100_000_061, False, 0.3), (2_000_003, True, 0.6), (1_000_004, True, 0.6))
FLIGHT_STEPS = 320  # past the 300-step limit: every flight ends
SPAWNABLE_MOVER_CALLS = ("set_object_poses",)


def _scene_and_layout():
    from tests.test_expert_episode import scene_and_layout

    return scene_and_layout()


def _setup_record(setup) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(setup)))


def _obs_digest(obs: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(obs):
        a = np.ascontiguousarray(obs[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class TracingSimulator:
    """FakeSimulator that records every call the env makes to it, with its arguments."""

    trace: list = []

    obstacles: list = []

    def __init__(self) -> None:
        self._inner = FakeSimulator(obstacles=TracingSimulator.obstacles)

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def _call(*args, **kwargs):
            TracingSimulator.trace.append([name, json.loads(json.dumps([_plain(a) for a in args])),
                                           json.loads(json.dumps({k: _plain(v) for k, v in kwargs.items()}))])
            return attr(*args, **kwargs)

        return _call

    @property
    def steps_taken(self) -> int:
        return self._inner.steps_taken


def _plain(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _scripted_action(env, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Points at the target and flies, with seeded noise so turns, climbs and pillar hits are exercised."""
    from autofly_ue5.expert.obs import target_geometry

    _dist, bearing, _dz = target_geometry(env.unwrapped._sim.observe().pose, env.unwrapped._setup.target_xy_z)
    yaw_rate = float(np.clip(bearing * 2.0 + rng.normal(0.0, noise), -1.0, 1.0))
    v = 2.0 if abs(bearing) < 0.3 else 0.3
    return np.array([v, yaw_rate, float(rng.uniform(-0.3, 0.3))], dtype=np.float32)


def _flight_record(seed: int, pillars: bool, noise: float) -> dict:
    from autofly_ue5.expert.env import AutoFlyEnv

    scene, layout = _scene_and_layout()
    TracingSimulator.trace = []
    # The real pillars, inflated by the rotor-tip half-span, so the fake registers pillar hits.
    TracingSimulator.obstacles = [(inst.x, inst.y, inst.radius_m + 0.47) for inst in layout.instances] if pillars else []
    env = AutoFlyEnv(scene, layout, TracingSimulator, map_path="/Game/AutoFly/Maps/S01", instance=0)
    rng = np.random.default_rng(seed + 7)
    obs, info = env.reset(seed=seed)
    steps = [{"obs": _obs_digest(obs), "info": json.loads(json.dumps(info))}]
    for _ in range(FLIGHT_STEPS):
        obs, reward, terminated, truncated, info = env.step(_scripted_action(env, rng, noise))
        steps.append({"obs": _obs_digest(obs), "reward": reward, "terminated": terminated, "truncated": truncated,
                      "info": json.loads(json.dumps(info))})
        if terminated or truncated:
            obs, info = env.reset()  # a counter-driven reset after the end, as training does
            steps.append({"obs": _obs_digest(obs), "info": json.loads(json.dumps(info))})
            break
    return {"seed": seed, "pillars": pillars, "noise": noise, "steps": steps, "trace": TracingSimulator.trace,
            "steps_taken": env._sim.steps_taken}


def _encoded_fixture_depth_digest() -> str:
    from autofly_ue5.expert.obs import encode_depth
    from autofly_ue5.sim.decode import decode_depth

    msg = json.loads((FIXTURES_DIR / "pas" / "depth_msg.json").read_text())
    msg["data"] = (FIXTURES_DIR / "pas" / "depth_msg.bin").read_bytes()
    encoded = encode_depth(decode_depth(msg))
    return hashlib.sha256(str(encoded.dtype).encode() + str(encoded.shape).encode() + encoded.tobytes()).hexdigest()


def build_golden() -> dict:
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = _scene_and_layout()
    return {
        "source": "static s01 as flown by M2 (commit 5d0abf5), before moving obstacles",
        "setups": {str(seed): _setup_record(sample_setup(scene, layout, np.random.default_rng(seed)))
                   for seed in SETUP_SEEDS},
        "encoded_depth_fixture_sha256": _encoded_fixture_depth_digest(),
        "flights": [_flight_record(*flight) for flight in FLIGHTS],
    }


def _golden() -> dict:
    return json.loads(GOLDEN.read_text())


# --------------------------------------------------------------------------------------------------------
# The tests.
# --------------------------------------------------------------------------------------------------------
def test_every_seed_still_draws_the_same_episode():
    from autofly_ue5.expert.episode import sample_setup

    scene, layout = _scene_and_layout()
    for seed, expected in _golden()["setups"].items():
        got = _setup_record(sample_setup(scene, layout, np.random.default_rng(int(seed))))
        assert {k: got[k] for k in expected} == expected, f"seed {seed} draws a different s01 episode than M2 did"
        added = {k: v for k, v in got.items() if k not in expected}
        assert all(v in (None, [], ()) for v in added.values()), (
            f"seed {seed}: fields added to EpisodeSetup must be empty for a static scene, got {added}")


def test_the_observation_space_is_still_the_one_every_m2_checkpoint_was_trained_on():
    from tests.test_expert_env import make_env

    space = make_env().observation_space
    expected = spaces.Dict({
        "depth": spaces.Box(0.0, 1.0, (1, 84, 84), dtype=np.float32),
        "vector": spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
    })
    assert space == expected
    assert space["depth"].dtype == np.float32 and space["vector"].dtype == np.float32


def test_a_real_depth_frame_still_encodes_bit_identically():
    assert _encoded_fixture_depth_digest() == _golden()["encoded_depth_fixture_sha256"]


def test_scripted_flights_make_the_same_calls_and_score_the_same():
    for expected in _golden()["flights"]:
        got = _flight_record(expected["seed"], expected["pillars"], expected["noise"])
        names = {call[0] for call in got["trace"]}
        assert not names & set(SPAWNABLE_MOVER_CALLS), f"static s01 must never move scene objects: {names}"
        assert got["trace"] == expected["trace"], f"seed {expected['seed']}: the simulator call sequence changed"
        assert got["steps_taken"] == expected["steps_taken"]
        assert len(got["steps"]) == len(expected["steps"])
        for i, (g, e) in enumerate(zip(got["steps"], expected["steps"])):
            where = f"seed {expected['seed']} step {i}"
            assert g["obs"] == e["obs"], f"{where}: the observation changed"
            for key in ("reward", "terminated", "truncated"):
                assert g.get(key) == e.get(key), f"{where}: {key} changed"
            assert {k: g["info"].get(k) for k in e["info"]} == e["info"], f"{where}: an info value changed"
            assert set(e["info"]) <= set(g["info"]), f"{where}: info keys may only gain additions"


def test_the_scripted_flights_reach_real_outcomes():
    # The fixture is only worth something if the flights it pins end in different ways.
    outcomes = {step["info"]["outcome"] for f in _golden()["flights"] for step in f["steps"]}
    assert {"running", "success", "collision"} <= outcomes, outcomes


if __name__ == "__main__":
    if "--write" not in sys.argv:
        print(f"refusing to touch {GOLDEN} without --write (see the module docstring)", file=sys.stderr)
        sys.exit(2)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(build_golden(), indent=1) + "\n")
    print(f"wrote {GOLDEN}")
