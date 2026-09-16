"""The simulator interface every backend implements (spec §7)."""

from typing import Protocol, runtime_checkable

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
        steps_taken); returns the settled observation with collided=False."""

    def spawn(self, name: str, asset: str, pose: Pose, scale: tuple[float, float, float], material: str | None = None) -> str:
        """Spawn a static object; returns the actual (possibly uniquified) name.

        `asset` is the short AssetRegistry name of a cooked static mesh (e.g. "1M_Cube"), not an object path: Project
        AirSim keys its spawn table by short name and a later duplicate silently replaces an earlier one
        (WorldSimApi.cpp:464-475), so spawnable assets need globally unique names. `material` is a package path of a
        base UMaterial (e.g. "/Game/Geometry/Materials/M_Orange")."""

    def destroy(self, name: str) -> None:
        """Destroy a spawned object; raises ObjectNotFoundError if it does not exist."""

    def command_velocity(self, v_forward: float, yaw_rate: float, v_z: float) -> None:
        """Set the command held for the next step: body forward m/s, yaw rate rad/s, vertical m/s (positive up)."""

    def step(self, dt: float = CONTROL_DT_S) -> int:
        """Advance the simulator clock by exactly dt; returns the simulator time in ns after the step.

        Requires reset() to have already run at least once on this connection: frame 0 of a session is
        corrupt (spec §7.1) and reset()'s own steps are the only thing allowed to consume it."""

    def observe(self) -> Observation:
        """Observation at the end of the last step or reset. Also requires reset() to have already run
        at least once on this connection, for the same reason as step()."""
