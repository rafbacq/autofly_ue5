"""One answer to "what does scene id X fly on?" (spec §6.1, §6.5).

Training, the gate, the renderer, the probes and the throughput script each used to guess: the file from
`scenes/<id>_*.json`, the layout from `runs/levels/<id>.layout.json`, the map from `/Game/AutoFly/Maps/<ID>` and the
config from `scene_autofly_<id>.jsonc`. A scene that reuses another's level (s01d flies s01's map and layout) breaks
every one of those guesses, so they are made here, once, and checked:

- exactly one scene file claims the id, and its file name starts with `<id>_`;
- a scene with `level` must match that base scene in every static field, and the base must have its own level;
- the stored layout must have been built from the base scene file as it is now (its recorded `scene_sha256`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from autofly_ue5.paths import RUNS_DIR, SCENES_DIR
from autofly_ue5.scenes.level_spec import map_path_for
from autofly_ue5.scenes.model import Bounds, Instance, Layout, SceneFile, SceneFileError, load_scene_file

# What a scene that reuses a level must share with its base: everything the built level or the episode geometry
# depends on. id, split, instruction_obstacle and the dynamic block may differ.
STATIC_FIELDS = ("seed", "bounds", "ground", "obstacle_groups", "start_band", "target_band", "altitude_band")


@dataclass(frozen=True)
class ResolvedScene:
    scene: SceneFile          # the scene asked for (e.g. s01d)
    base_scene: SceneFile     # the scene whose level it flies (s01; the scene itself when it has no `level`)
    layout: Layout
    layout_path: Path
    layout_sha256: str        # of the stored layout file, for run records
    map_path: str
    default_scene_config: str

    @property
    def base_id(self) -> str:
        return self.base_scene.id

    @property
    def movable_objects(self) -> tuple[str, ...]:
        """The tags a simulator flying this scene may move (its allow-list, spec §6.5): every layout obstacle of a
        dynamic scene, nothing for a static one."""
        if self.scene.dynamic is None:
            return ()
        return tuple(sorted(inst.tag for inst in self.layout.instances))


def _scene_file(scene_id: str, scenes_dir: Path) -> SceneFile:
    matches = sorted(Path(scenes_dir).glob(f"{scene_id}_*.json"))
    claimed = []
    for path in sorted(Path(scenes_dir).glob("*.json")):
        try:
            claimed_id = json.loads(path.read_text()).get("id")
        except (json.JSONDecodeError, AttributeError):
            continue
        if claimed_id == scene_id:
            claimed.append(path)
    if not matches and not claimed:
        raise FileNotFoundError(f"no scene file for scene {scene_id!r} (expected {scenes_dir}/{scene_id}_*.json)")
    if len(claimed) > 1 or claimed != matches:
        raise SceneFileError(
            f"scene {scene_id!r} must be claimed by exactly one file named {scene_id}_*.json: files named for it "
            f"{[p.name for p in matches]}, files whose id is {scene_id!r} {[p.name for p in claimed]}")
    scene = load_scene_file(matches[0])
    return scene


def _load_layout(base: SceneFile, levels_dir: Path) -> tuple[Layout, Path, str]:
    path = Path(levels_dir) / f"{base.id}.layout.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist -- build scene {base.id}'s level (scripts/build_scenes.py) first")
    blob = path.read_bytes()
    stored = json.loads(blob)
    if stored.get("scene_sha256") != base.sha256:
        raise SceneFileError(
            f"{path} was built from a different version of {base.path} (layout records scene_sha256 "
            f"{stored.get('scene_sha256')}, the file is {base.sha256}); rebuild the level, or restore the scene file")
    raw = stored["layout"]
    layout = Layout(scene_id=raw["scene_id"], seed=raw["seed"], bounds=Bounds(**raw["bounds"]),
                    instances=tuple(Instance(**i) for i in raw["instances"]))
    if layout.scene_id != base.id:
        raise SceneFileError(f"{path} holds scene {layout.scene_id!r}'s layout, not {base.id!r}'s")
    return layout, path, hashlib.sha256(blob).hexdigest()


def resolve_scene(scene_id: str, *, scenes_dir: Path = SCENES_DIR, levels_dir: Path = RUNS_DIR / "levels") -> ResolvedScene:
    scene = _scene_file(scene_id, scenes_dir)
    base = scene
    if scene.level is not None:
        if scene.level == scene.id:
            raise SceneFileError(f"{scene.path}: level {scene.level!r} names the scene itself")
        base = _scene_file(scene.level, scenes_dir)
        if base.level is not None:
            raise SceneFileError(f"{scene.path}: its level {base.id!r} itself reuses {base.level!r}; levels do not chain")
        differ = [f for f in STATIC_FIELDS if getattr(scene, f) != getattr(base, f)]
        if differ:
            raise SceneFileError(f"{scene.path} reuses {base.id}'s level but differs from {base.path} in {differ}: a "
                                 f"scene that flies another's level must match it in {list(STATIC_FIELDS)}")
    layout, layout_path, layout_sha = _load_layout(base, levels_dir)
    return ResolvedScene(scene=scene, base_scene=base, layout=layout, layout_path=layout_path, layout_sha256=layout_sha,
                         map_path=map_path_for(base.id), default_scene_config=f"scene_autofly_{base.id}.jsonc")
