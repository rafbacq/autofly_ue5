"""Collision events from Step() results and the collision_info topic.

A hit on a step's final pose can be reported in the next Step() result; the topic
message usually arrives first. Events are merged, de-duplicated by
(sim_time_ns, object_name) and assigned by their own timestamps.
"""

import threading
from typing import Iterable

from autofly_ue5.sim.types import CollisionEvent


def _vec(d: dict) -> tuple[float, float, float]:
    return (float(d["x"]), float(d["y"]), float(d["z"]))


def event_from_step(event: dict) -> CollisionEvent:
    return CollisionEvent(int(event["sim_time_ns"]), str(event["object_name"]), _vec(event["impact_point"]), _vec(event["normal"]))


def event_from_topic(msg: dict) -> CollisionEvent:
    return CollisionEvent(int(msg["time_stamp"]), str(msg["object_name"]), _vec(msg["impact_point"]), _vec(msg["normal"]))


def collisions_after(events: Iterable[CollisionEvent], t_ns: int) -> tuple[CollisionEvent, ...]:
    return tuple(e for e in events if e.sim_time_ns > t_ns)


class CollisionLog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[CollisionEvent] = []
        self._seen: set[tuple[int, str]] = set()

    def topic_callback(self, _topic: object, msg: dict) -> None:
        event = event_from_topic(msg)
        with self._lock:
            self._pending.append(event)

    def collect(self, step_events: list[dict], up_to_ns: int) -> tuple[CollisionEvent, ...]:
        with self._lock:
            candidates = [event_from_step(e) for e in step_events if e.get("type") == "collision"]
            candidates += [e for e in self._pending if e.sim_time_ns <= up_to_ns]
            self._pending = [e for e in self._pending if e.sim_time_ns > up_to_ns]
            fresh = []
            for event in sorted(candidates, key=lambda e: e.sim_time_ns):
                key = (event.sim_time_ns, event.object_name)
                if key not in self._seen:
                    self._seen.add(key)
                    fresh.append(event)
            return tuple(fresh)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._seen.clear()
