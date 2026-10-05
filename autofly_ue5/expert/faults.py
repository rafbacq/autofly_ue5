"""The backend hazards the expert's training and evaluation recover from, and how their counts are summarised.

See `autofly_ue5/expert/train.py`'s module docstring for how each hazard was found and measured live.
"""

from __future__ import annotations

from collections import Counter

# pynng.exceptions.Timeout: a raw NNG transport-level timeout, found live during Task 8's shakedown propagating
# straight out of the third-party `projectairsim` client (not through any of airsim_backend.py's own named
# exceptions). Not a new project dependency: pynng is installed transitively (projectairsim depends on it).
from pynng.exceptions import Timeout as NngTimeout

# pynng.exceptions.ConnectionReset: never observed in any run log (the killed-simulator case surfaced as Timeout),
# but it is the same kind of transport-level failure and ending a 12-hour run on it would be the costlier mistake.
# The rest of NNGException stays uncaught on purpose: NotSupported, InvalidOperation, ... are configuration or
# protocol bugs that retrying cannot fix.
from pynng.exceptions import ConnectionReset as NngConnectionReset

from autofly_ue5.expert.episode import EpisodeSetupError
from autofly_ue5.sim.airsim_backend import CommandTimeoutError, StaleStateError, StepTimingError
from autofly_ue5.sim.process import SimExitedError, SimReadyTimeout
from autofly_ue5.sim.sync import FrameTimeoutError, FrameTimestampError
from autofly_ue5.sim.types import (
    CameraPoseError,
    KinematicsJumpError,
    ObjectMoveRefusedError,
    ResetPoseError,
    SetPoseError,
    SimConnectionLostError,
    SimRequestTimeoutError,
    StartCollisionError,
)

# SimRequestTimeoutError (2026-10-02, moving obstacles): a mover request got no reply. It ends the episode like any step
# fault; its connection is gone, so the next reset() raises SimConnectionLostError and relaunches (below).
# ObjectMoveRefusedError (2026-10-03): Unreal refused a movable pillar's move twice; the next reset re-places every mover.
# FrameTimeoutError (2026-10-05 08:15, run 5): a step's clock and command both succeeded, but no rgb/depth frame for it
# arrived within 5 s. Never seen before in any run, and unknown, so it ended a 12-hour run at 25k steps. Like a stuck
# camera, the next reset retries in place and then relaunches. FrameTimestampError is the same wait's other failure (a
# frame stamped at another time), not yet seen.
FAULT_ERRORS_STEP = (CameraPoseError, StepTimingError, StaleStateError, CommandTimeoutError, NngTimeout, NngConnectionReset,
                     KinematicsJumpError, SimRequestTimeoutError, ObjectMoveRefusedError, FrameTimeoutError,
                     FrameTimestampError)
# An episode that did not start where it should (C9): retried like any reset fault, so the gate replays the seed and
# training never sees it. Raised only during reset(), so not step faults.
FAULT_ERRORS_START = (ResetPoseError, StartCollisionError, SetPoseError)
# A launch that failed, or a connection that is gone (SimConnectionLostError: projectairsim's client disconnected itself
# after a request timeout): retrying reset() on the same slot cannot help, so the wrapper goes straight to its next
# relaunch round instead of spending max_reset_attempts attempts on it.
FAULT_ERRORS_LAUNCH = (SimExitedError, SimReadyTimeout, SimConnectionLostError)
FAULT_ERRORS_RESET = FAULT_ERRORS_STEP + FAULT_ERRORS_START + (EpisodeSetupError,) + FAULT_ERRORS_LAUNCH
# Every fault name the resilient wrapper knows how to recover from. Seeded into every fault/recovery counter dict
# (see ResilientAutoFlyEnv.__init__, combine_fault_summaries) so the run record always shows an explicit 0 for a
# hazard that never fired, rather than omitting the key -- a 12-hour run that never faults must be distinguishable
# from one whose counting is silently broken.
KNOWN_FAULT_NAMES = tuple(err.__name__ for err in FAULT_ERRORS_RESET)


def combine_fault_summaries(summaries: list[dict]) -> dict:
    """Sums fault_counts/recovered_counts/relaunch_count across every worker's get_fault_summary(). Always
    seeded with an explicit 0 for every KNOWN_FAULT_NAMES entry -- even if `summaries` is empty (e.g. an
    early failure before any env was built) -- so the run record never shows an ambiguous {} that could
    mean either "nothing faulted" or "this was never computed"."""
    fault_counts: Counter[str] = Counter({name: 0 for name in KNOWN_FAULT_NAMES})
    recovered_counts: Counter[str] = Counter({name: 0 for name in KNOWN_FAULT_NAMES})
    relaunch_count = 0
    for s in summaries:
        if not isinstance(s, dict):  # never let a malformed reply cost a run its record (2026-10-03)
            continue
        fault_counts.update(s.get("fault_counts", {}))
        recovered_counts.update(s.get("recovered_counts", {}))
        relaunch_count += s.get("relaunch_count", 0)
    return {"fault_counts": dict(fault_counts), "recovered_counts": dict(recovered_counts), "relaunch_count": relaunch_count}
