"""Scene file and asset registry loading; typed layout objects. Coordinates are NED metres."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonschema

from autofly_ue5.paths import ASSET_REGISTRY
from autofly_ue5.sim.types import CONTROL_DT_S

SCHEMA_PATH = Path(__file__).with_name("scene.schema.json")

# How far the drone's rotor tips reach from its centre: props at +-0.253 m with radius 0.1143 m
# (configs/robot_autofly_quadrotor.jsonc) give 0.472 m. A moving obstacle's contact distance must keep one teleport
# (max speed x one step) short of this, so no mover is ever placed inside the drone (spec §6.5).
DRONE_HALF_SPAN_M = 0.48


class SceneFileError(ValueError):
    """A scene file failed schema or consistency validation."""


@dataclass(frozen=True)
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class ObstacleGroup:
    asset: str
    count: int
    scale_xy: tuple[float, float]
    scale_z: tuple[float, float]
    palette: tuple[str, ...]
    placement: dict


@dataclass(frozen=True)
class DynamicSpec:
    """A scene's moving-obstacle block (spec §6.5): which of the layout's obstacles move, how, and the rules that keep
    every episode fair. Ranges are (low, high); counts are inclusive."""
    count: tuple[int, int]
    path_movers: tuple[int, int]
    path_corridor_m: float
    route_kinds: tuple[str, ...]
    speed_m_s: tuple[float, float]
    pingpong_half_length_m: tuple[float, float]
    orbit_radius_m: tuple[float, float]
    min_gap_m: float
    contact_m: float
    yield_margin_m: float
    start_keepout_m: float
    target_keepout_m: float
    max_path_ratio: float
    source: str = "layout"


@dataclass(frozen=True)
class SceneFile:
    id: str
    split: str
    seed: int
    bounds: Bounds
    ground: str
    obstacle_groups: tuple[ObstacleGroup, ...]
    start_band: tuple[float, float]
    target_band: tuple[float, float]
    altitude_band: tuple[float, float]
    instruction_obstacle: str
    sha256: str
    path: str
    level: str | None = None  # the scene whose built level this one reuses (spec §6.1); None: its own
    dynamic: DynamicSpec | None = None  # spec §6.5; None: nothing moves
    # Reward coefficients this scene sets for itself, as sorted (name, value) pairs; () keeps spec §8's defaults
    # (expert.reward.reward_config_for_scene). s01d prices leaving the bounds like a collision (2026-10-03).
    reward: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class AssetEntry:
    name: str
    ue_path: str
    base_size_m: tuple[float, float, float]
    pivot: str
    footprint: str
    category: str
    role: str
    seen: bool | None
    measured_extent_cm_at_unit_scale: tuple[float, float, float] | None


@dataclass(frozen=True)
class MaterialEntry:
    name: str
    kind: str
    ue_path: str
    parent: str | None
    parameter: str | None
    rgba: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class AssetRegistry:
    assets: dict[str, AssetEntry]
    materials: dict[str, MaterialEntry]


@dataclass(frozen=True)
class Instance:
    tag: str
    asset: str
    x: float
    y: float
    z_center: float
    yaw: float
    scale: tuple[float, float, float]
    material: str
    radius_m: float
    height_m: float


@dataclass(frozen=True)
class Layout:
    scene_id: str
    seed: int
    bounds: Bounds
    instances: tuple[Instance, ...]

    def to_json(self) -> dict:
        return {"scene_id": self.scene_id, "seed": self.seed, "bounds": asdict(self.bounds),
                "instances": [asdict(i) for i in self.instances]}


def _range(values: list, name: str) -> tuple[float, float]:
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise SceneFileError(f"{name}: minimum {low} is greater than maximum {high}")
    return (low, high)


def _int_range(values: list, name: str) -> tuple[int, int]:
    low, high = int(values[0]), int(values[1])
    if low > high:
        raise SceneFileError(f"{name}: minimum {low} is greater than maximum {high}")
    return (low, high)


def _dynamic_spec(raw: dict, path: Path) -> DynamicSpec:
    movers = raw["movers"]
    spec = DynamicSpec(
        count=_int_range(movers["count"], "dynamic.movers.count"),
        path_movers=_int_range(movers["path_movers"], "dynamic.movers.path_movers"),
        path_corridor_m=float(movers["path_corridor_m"]),
        route_kinds=tuple(raw["route_kinds"]),
        speed_m_s=_range(raw["speed_m_s"], "dynamic.speed_m_s"),
        pingpong_half_length_m=_range(raw["pingpong_half_length_m"], "dynamic.pingpong_half_length_m"),
        orbit_radius_m=_range(raw["orbit_radius_m"], "dynamic.orbit_radius_m"),
        min_gap_m=float(raw["min_gap_m"]),
        contact_m=float(raw["contact_m"]),
        yield_margin_m=float(raw["yield_margin_m"]),
        start_keepout_m=float(raw["start_keepout_m"]),
        target_keepout_m=float(raw["target_keepout_m"]),
        max_path_ratio=float(raw["max_path_ratio"]),
        source=movers["source"],
    )
    if spec.path_movers[1] > spec.count[0]:
        raise SceneFileError(f"{path}: dynamic.movers.path_movers {spec.path_movers} can exceed the smallest mover count "
                             f"{spec.count[0]}")
    reach = spec.contact_m - spec.speed_m_s[1] * CONTROL_DT_S
    if reach <= DRONE_HALF_SPAN_M:
        raise SceneFileError(
            f"{path}: dynamic invariant contact_m - max_speed * dt > {DRONE_HALF_SPAN_M} m fails "
            f"({spec.contact_m} - {spec.speed_m_s[1]} * {CONTROL_DT_S} = {reach:.3f} m): one teleport of the fastest "
            f"mover could land inside the drone's rotor span (spec §6.5)")
    return spec


def load_scene_file(path: Path) -> SceneFile:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    schema = json.loads(SCHEMA_PATH.read_text())
    try:
        jsonschema.validate(data, schema, cls=jsonschema.Draft202012Validator)
    except jsonschema.ValidationError as err:
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        raise SceneFileError(f"{path}: {location}: {err.message}") from err
    b = data["bounds"]
    bounds = Bounds(float(b["x_min"]), float(b["x_max"]), float(b["y_min"]), float(b["y_max"]))
    if bounds.width <= 0 or bounds.height <= 0:
        raise SceneFileError(f"{path}: bounds must have x_min < x_max and y_min < y_max")
    if bounds.x_min != -bounds.x_max or bounds.y_min != -bounds.y_max or bounds.width != 70.0 or bounds.height != 70.0:
        raise SceneFileError(f"{path}: bounds must be 70 x 70 m centred on the origin (spec 6.1), got {b}")
    groups = tuple(
        ObstacleGroup(
            asset=g["asset"], count=int(g["count"]),
            scale_xy=_range(g["scale_range"]["xy"], f"obstacle_groups[{i}].scale_range.xy"),
            scale_z=_range(g["scale_range"]["z"], f"obstacle_groups[{i}].scale_range.z"),
            palette=tuple(g["palette"]), placement=dict(g["placement"]),
        )
        for i, g in enumerate(data["obstacle_groups"])
    )
    return SceneFile(
        id=data["id"], split=data["split"], seed=int(data["seed"]), bounds=bounds, ground=data["ground"],
        obstacle_groups=groups,
        start_band=_range(data["start_band"], "start_band"),
        target_band=_range(data["target_band"], "target_band"),
        altitude_band=_range(data["altitude_band"], "altitude_band"),
        instruction_obstacle=data["instruction_obstacle"],
        sha256=hashlib.sha256(raw).hexdigest(), path=str(path),
        level=data.get("level"),
        dynamic=_dynamic_spec(data["dynamic"], path) if "dynamic" in data else None,
        reward=tuple(sorted((name, float(value)) for name, value in data.get("reward", {}).items())),
    )


def load_registry(path: Path = ASSET_REGISTRY) -> AssetRegistry:
    data = json.loads(Path(path).read_text())
    assets = {
        name: AssetEntry(
            name=name, ue_path=a["ue_path"], base_size_m=tuple(float(v) for v in a["base_size_m"]),
            pivot=a["pivot"], footprint=a["footprint"], category=a["category"], role=a["role"], seen=a["seen"],
            measured_extent_cm_at_unit_scale=(tuple(float(v) for v in a["measured_extent_cm_at_unit_scale"])
                                              if a["measured_extent_cm_at_unit_scale"] is not None else None),
        )
        for name, a in data["assets"].items()
    }
    materials = {
        name: MaterialEntry(name=name, kind=m["kind"], ue_path=m["ue_path"], parent=m.get("parent"),
                            parameter=m.get("parameter"),
                            rgba=tuple(float(v) for v in m["rgba"]) if "rgba" in m else None)
        for name, m in data["materials"].items()
    }
    return AssetRegistry(assets=assets, materials=materials)
