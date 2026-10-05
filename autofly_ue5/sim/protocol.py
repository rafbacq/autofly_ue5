"""The simulator interface every backend implements (spec §7)."""

from typing import Mapping, Protocol, runtime_checkable

from autofly_ue5.sim.types import CONTROL_DT_S, Observation, Pose


@runtime_checkable
class Simulator(Protocol):
    @property
    def steps_taken(self) -> int:
        """Number of simulator clock steps issued since construction (reset steps included)."""

    def launch(self, map_path: str, instance: int) -> None:
        """Start (or attach to) the simulator process for `map_path` as instance `instance`."""

    def close(self) -> None:
        """Disconnect and stop the process this object launched."""

    def reset(self, pose: Pose) -> Observation:
        """Teleport the vehicle to `pose` with zero velocity, advancing the clock by whole control steps (counted in
        steps_taken); returns the settled observation with collided=False.

        Raises ResetPoseError if the vehicle settled somewhere other than `pose`, and SetPoseError if the simulator
        refused the teleport (autofly_ue5.sim.types). Both are recoverable: reset again, or relaunch."""

    def spawn(self, name: str, asset: str, pose: Pose, scale: tuple[float, float, float], material: str | None = None) -> str:
        """Spawn a static object; returns the actual (possibly uniquified) name.

        `asset` is the short AssetRegistry name of a cooked static mesh (e.g. "1M_Cube"), not an object path: Project
        AirSim keys its spawn table by short name and a later duplicate silently replaces an earlier one
        (WorldSimApi.cpp:464-475), so spawnable assets need globally unique names. `material` is a package path of a
        base UMaterial (e.g. "/Game/Geometry/Materials/M_Orange")."""

    def destroy(self, name: str) -> None:
        """Destroy a spawned object; raises ObjectNotFoundError if it does not exist."""

    def set_object_poses(self, poses: Mapping[str, Pose]) -> None:
        """Teleport named, already-placed movable scene objects (no sweep), one request per name in order (spec §6.5,
        §7). Allowed any time after launch(), before the first reset() too.

        Raises ObjectPoseError for a name the implementation does not allow or the server cannot find or move (not
        recoverable), SimRequestTimeoutError when the server stops answering (the connection is then lost; later calls
        raise SimConnectionLostError until a relaunch). A failure partway leaves the earlier names moved."""

    def command_velocity(self, v_forward: float, yaw_rate: float, v_z: float) -> None:
        """Set the command held for the next step: body forward m/s, yaw rate rad/s, vertical m/s (positive up)."""

    def step(self, dt: float = CONTROL_DT_S) -> int:
        """Advance the simulator clock by exactly dt; returns the simulator time in ns after the step.

        Requires reset() to have already run at least once on this connection: frame 0 of a session is
        corrupt (spec §7.1) and reset()'s own steps are the only thing allowed to consume it.

        Raises KinematicsJumpError if the vehicle moved farther than it can fly in dt since the previous observation
        (a teleport, not flight) -- checked on collision steps too. The episode cannot continue; reset. A backend may
        also raise FrameTimeoutError when the step's camera frames do not arrive (sim/sync.py; seen once, 2026-10-05),
        or FrameTimestampError when they arrive stamped at another time: both are recoverable step faults."""

    def observe(self) -> Observation:
        """Observation at the end of the last step or reset. Also requires reset() to have already run
        at least once on this connection, for the same reason as step()."""
