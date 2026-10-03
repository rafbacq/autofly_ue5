"""scripts/probe_a0.py flies each seed twice, with and without a0, and pairs the outcomes (FakeSimulator)."""

from __future__ import annotations

import json
import math

from tests.test_collect import _Crashes, _env
from tests.test_m2_gate import _StraightAtTargetModel


def test_each_seed_is_flown_with_and_without_a0_and_paired(tmp_path):
    from scripts.probe_a0 import A0_PROBE_SEED_BASE, run

    _scene, env = _env(tmp_path)
    record = run(model=_StraightAtTargetModel(), env=env, pairs=3, out=tmp_path / "a0.json", deterministic=True,
                 meta={"probe": "a0_paired"}, log=lambda line: None)
    rows = record["rows"]
    assert [(r["seed"], r["mode"]) for r in rows] == [(A0_PROBE_SEED_BASE + i, m) for i in range(3) for m in ("a0", "random_yaw")]
    for a0_row, random_row in zip(rows[::2], rows[1::2]):
        assert abs(math.remainder(a0_row["start_yaw"], math.pi / 4)) < 1e-9, "a0 starts on a sector centre"
        assert a0_row["start_yaw"] != random_row["start_yaw"] or abs(math.remainder(random_row["start_yaw"], math.pi / 4)) < 1e-9
    assert record["summary"]["pairs"] == {"complete": 3, "both_succeed": 3, "only_a0_succeeds": 0,
                                          "only_random_yaw_succeeds": 0, "both_fail": 0}
    assert json.loads((tmp_path / "a0.json").read_text())["summary"]["a0"]["success"] == 3


def test_failures_are_counted_per_mode(tmp_path):
    from scripts.probe_a0 import run

    _scene, env = _env(tmp_path)
    record = run(model=_Crashes(), env=env, pairs=2, out=tmp_path / "a0.json", deterministic=True, meta={},
                 log=lambda line: None)
    assert record["summary"]["a0"]["outcomes"] == {"out_of_bounds": 2}
    assert record["summary"]["pairs"]["both_fail"] == 2
