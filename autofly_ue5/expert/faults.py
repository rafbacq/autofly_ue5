"""The backend hazards the expert's training and evaluation recover from, and how their counts are summarised.

See `autofly_ue5/expert/train.py`'s module docstring for how each hazard was found and measured live.
"""

from __future__ import annotations

from collections import Counter

# pynng.exceptions.Timeout: a raw NNG transport-level timeout, found live during Task 8's shakedown propagating
# straight out of the third-party `projectairsim` client (not through any of airsim_backend.py's own named
# exceptions). Not a new project dependency: pynng is installed transitively (projectairsim depends on it).
from pynng.exceptions import Timeout as NngTimeout

from autofly_ue5.expert.episode import EpisodeSetupError
from autofly_ue5.sim.airsim_backend import CameraPoseError, CommandTimeoutError, StaleStateError, StepTimingError

FAULT_ERRORS_STEP = (CameraPoseError, StepTimingError, StaleStateError, CommandTimeoutError, NngTimeout)
FAULT_ERRORS_RESET = FAULT_ERRORS_STEP + (EpisodeSetupError,)
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
        fault_counts.update(s.get("fault_counts", {}))
        recovered_counts.update(s.get("recovered_counts", {}))
        relaunch_count += s.get("relaunch_count", 0)
    return {"fault_counts": dict(fault_counts), "recovered_counts": dict(recovered_counts), "relaunch_count": relaunch_count}
