import threading
import time

import pytest

from autofly_ue5.sim.events import CollisionLog, collisions_after, event_from_step, event_from_topic
from autofly_ue5.sim.sync import FrameCollector, FrameTimeoutError, FrameTimestampError
from autofly_ue5.sim.types import CollisionEvent


def _msg(t_ns: int) -> dict:
    return {"time_stamp": t_ns, "encoding": "BGR", "height": 1, "width": 1, "data": b"\x00\x00\x00"}


def test_collector_returns_frames_at_target():
    fc = FrameCollector()
    fc.arm(200)
    fc.callback("rgb")(None, _msg(195))  # older frame is ignored
    fc.callback("rgb")(None, _msg(200))
    fc.callback("depth")(None, _msg(200))
    frames = fc.wait(timeout_s=0.5)
    assert frames["rgb"]["time_stamp"] == 200 and frames["depth"]["time_stamp"] == 200
    assert fc.received == 3


def test_collector_waits_for_late_frames_from_another_thread():
    fc = FrameCollector()
    fc.arm(400)

    def deliver():
        time.sleep(0.05)
        fc.callback("rgb")(None, _msg(400))
        fc.callback("depth")(None, _msg(400))

    threading.Thread(target=deliver).start()
    assert set(fc.wait(timeout_s=2.0)) == {"rgb", "depth"}


def test_collector_times_out_naming_missing_stream():
    fc = FrameCollector()
    fc.arm(600)
    fc.callback("rgb")(None, _msg(600))
    with pytest.raises(FrameTimeoutError, match="depth"):
        fc.wait(timeout_s=0.05)


def test_collector_rejects_frames_past_target():
    fc = FrameCollector()
    fc.arm(800)
    fc.callback("rgb")(None, _msg(805))
    fc.callback("depth")(None, _msg(800))
    with pytest.raises(FrameTimestampError):
        fc.wait(timeout_s=0.5)


def test_arm_clears_previous_frames():
    fc = FrameCollector()
    fc.arm(200)
    fc.callback("rgb")(None, _msg(200))
    fc.callback("depth")(None, _msg(200))
    fc.arm(400)
    with pytest.raises(FrameTimeoutError):
        fc.wait(timeout_s=0.05)


def test_unknown_stream_is_rejected():
    with pytest.raises(KeyError):
        FrameCollector().callback("segmentation")


def test_collector_keeps_first_exact_target_frame_despite_late_duplicate():
    fc = FrameCollector()
    fc.arm(1000)
    fc.callback("rgb")(None, _msg(1000))
    fc.callback("depth")(None, _msg(1000))
    fc.callback("rgb")(None, _msg(1000 + 5_000_000))  # late duplicate/reorder for an already-filled key
    frames = fc.wait(timeout_s=0.5)
    assert frames["rgb"]["time_stamp"] == 1000 and frames["depth"]["time_stamp"] == 1000
    assert fc.received == 3


def test_collector_still_rejects_when_only_a_late_frame_ever_arrives():
    fc = FrameCollector()
    fc.arm(1000)
    fc.callback("rgb")(None, _msg(1000 + 5_000_000))  # no exact-target frame for rgb ever arrives
    fc.callback("depth")(None, _msg(1000))
    with pytest.raises(FrameTimestampError):
        fc.wait(timeout_s=0.5)


STEP_EVENT = {"type": "collision", "sim_time_ns": 400, "object_name": "StaticMeshActor_12",
              "impact_point": {"x": 1.0, "y": 2.0, "z": -1.5}, "normal": {"x": -1.0, "y": 0.0, "z": 0.0}}
TOPIC_MSG = {"time_stamp": 400, "object_name": "StaticMeshActor_12", "segmentation_id": 3,
             "position": {"x": 0.9, "y": 2.0, "z": -1.5}, "impact_point": {"x": 1.0, "y": 2.0, "z": -1.5},
             "normal": {"x": -1.0, "y": 0.0, "z": 0.0}, "penetration_depth": 0.01}


def test_event_parsing():
    expected = CollisionEvent(400, "StaticMeshActor_12", (1.0, 2.0, -1.5), (-1.0, 0.0, 0.0))
    assert event_from_step(STEP_EVENT) == expected
    assert event_from_topic(TOPIC_MSG) == expected


def test_log_merges_and_deduplicates_step_and_topic_events():
    log = CollisionLog()
    log.topic_callback(None, TOPIC_MSG)
    log.topic_callback(None, dict(TOPIC_MSG, time_stamp=600))  # belongs to a later step
    first = log.collect([STEP_EVENT, {"type": "gate_pass", "sim_time_ns": 400}], up_to_ns=400)
    assert [e.sim_time_ns for e in first] == [400]
    second = log.collect([STEP_EVENT], up_to_ns=600)  # late duplicate of the 400 hit plus the 600 hit
    assert [e.sim_time_ns for e in second] == [600]
    log.clear()
    assert log.collect([STEP_EVENT], up_to_ns=600)[0].sim_time_ns == 400


def test_collisions_after():
    events = [CollisionEvent(t, "a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) for t in (200, 400, 600)]
    assert [e.sim_time_ns for e in collisions_after(events, 400)] == [600]
