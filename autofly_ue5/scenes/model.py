"""Scene file and asset registry loading; typed layout objects. Coordinates are NED metres."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonschema

from autofly_ue5.paths import ASSET_REGISTRY

SCHEMA_PATH = Path(__file__).with_name("scene.schema.json")


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
