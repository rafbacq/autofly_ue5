"""Wait for camera frames whose timestamp equals the simulator time a step ended at."""

import threading
import time
from typing import Callable


class FrameTimeoutError(RuntimeError):
    """A stream delivered no frame at or after the target time within the timeout."""


class FrameTimestampError(RuntimeError):
    """A frame arrived with a timestamp other than the target time."""


class FrameCollector:
    def __init__(self, keys: tuple[str, ...] = ("rgb", "depth")) -> None:
        self._keys = tuple(keys)
        self._cond = threading.Condition()
        self._target: int | None = None
        self._slots: dict[str, dict | None] = {k: None for k in self._keys}
        self.received = 0

    def callback(self, key: str) -> Callable[[object, dict], None]:
        if key not in self._keys:
            raise KeyError(key)

        def _on_message(_topic: object, msg: dict) -> None:
            with self._cond:
                self.received += 1
                current = self._slots.get(key)
                if current is not None and int(current["time_stamp"]) == self._target:
                    return  # slot already holds the exact-target frame; a later arrival cannot displace it
                if self._target is not None and int(msg["time_stamp"]) >= self._target:
                    self._slots[key] = msg
                    self._cond.notify_all()

        return _on_message

    def arm(self, target_ns: int) -> None:
        with self._cond:
            self._target = int(target_ns)
            self._slots = {k: None for k in self._keys}

    def wait(self, timeout_s: float) -> dict[str, dict]:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            if self._target is None:
                raise RuntimeError("arm() must be called before wait()")
            while any(v is None for v in self._slots.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [k for k, v in self._slots.items() if v is None]
                    raise FrameTimeoutError(f"no {missing} frame with time_stamp >= {self._target} within {timeout_s} s")
                self._cond.wait(remaining)
            late = {k: int(v["time_stamp"]) for k, v in self._slots.items() if int(v["time_stamp"]) != self._target}
            if late:
                raise FrameTimestampError(f"frames {late} do not match the step time {self._target}")
            return dict(self._slots)
