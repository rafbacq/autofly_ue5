"""Coordinate conventions.

Project AirSim world: NED, metres (x north, y east, z down); yaw rotates north toward east, radians.
Unreal Engine world: X forward (= north), Y right (= east), Z up, centimetres; there is no origin
offset between the UE world origin and the NED origin, and a pure yaw is the same angle in both.
AutoFly state: z is up, metres, so z_up = -z_ned.
"""

import math

CM_PER_M = 100.0


def ned_to_ue_cm(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (x * CM_PER_M, y * CM_PER_M, -z * CM_PER_M)


def ue_cm_to_ned(x_cm: float, y_cm: float, z_cm: float) -> tuple[float, float, float]:
    return (x_cm / CM_PER_M, y_cm / CM_PER_M, -z_cm / CM_PER_M)


def ned_yaw_deg_to_ue_yaw_deg(yaw_deg: float) -> float:
    return yaw_deg


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def z_up_from_ned(z_ned: float) -> float:
    return -z_ned


def heading_unit(yaw: float) -> tuple[float, float]:
    return (math.cos(yaw), math.sin(yaw))


def body_to_ned(w: float, x: float, y: float, z: float, v: tuple[float, float, float]) -> tuple[float, float, float]:
    vx, vy, vz = v
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx + 2.0 * (x * y - w * z) * vy + 2.0 * (x * z + w * y) * vz,
        2.0 * (x * y + w * z) * vx + (1.0 - 2.0 * (x * x + z * z)) * vy + 2.0 * (y * z - w * x) * vz,
        2.0 * (x * z - w * y) * vx + 2.0 * (y * z + w * x) * vy + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )
