"""The target pool (spec §6.4, §9 step 1; plan 5 C5, "Design (2026-10-08)"): which objects an episode's target and
distractors are, and the name its instruction gives the target.

The pool is a collection-time choice, not a scene field. The expert flies on depth to a privileged target position, so
training and the gate keep the orange cylinder: no scene file changes, no layout re-hash, and s01's golden episodes
(tests/test_static_s01_golden.py) stay as M2 flew them. `scripts/collect_dataset.py --targets assets/targets.json`
(and M6's harness) hand a pool to the env, which hands it to `expert.episode.sample_setup`; without one every draw is
as before.

    {"version": 1, "targets": [{"name": "blue hatchback", "category": "vehicle", "seen": true,
      "asset": "<registry key, null until imported>", "scale": [sx, sy, sz], "material": "<registry material or null>",
      "status": "candidate|placeholder|manual|imported|primitive", "source": {"source", "id", "licence", "page"} or null,
      "height_m": ...}]}

An optional `"stand_in": true` marks a low-poly stand-in (the Kenney models drawn for the vehicle and human categories
until a Fab session replaces them, plan 6 step 2b): it is drawn like any imported entry, and every episode's record of
its target and distractors carries the mark, so a dataset can filter them out later.

Every entry carries all nine keys, so a misspelt key is a missing one and is refused; other keys are ignored. Only
`primitive` (engine content) and `imported` entries are drawn, and each must be spawnable as it stands: its asset in the
registry, one mesh (a runtime spawn places one: a model with a no-collision `cards` part would arrive without its
leaves), a scale, and a material that is either none (the mesh's own) or a base `UMaterial` the runtime can apply. A
palette colour is a material instance, which Project AirSim's `setMaterial` cannot load (expert/episode.py
`_runtime_material_path`): spawned, a "red cube" would be the mesh's own grey under a name that says red, so such an
entry is refused here rather than recorded as a mislabelled episode.

Per-scene exclusions (plan 6 step 2c, revision 1 item 7): `target_exclusions.json` beside the pool file names, per scene,
the registry assets whose entries are never drawn there, because they look like that scene's own obstacles (a cylinder
among pillars would make the instruction's target one of them). See `TargetExclusions`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autofly_ue5.expert.episode import RUNTIME_SPAWNABLE_MATERIAL_KINDS
from autofly_ue5.scenes.model import AssetRegistry, load_registry

VERSION = 1
CATEGORIES = ("vehicle", "geometry", "furniture", "equipment", "human", "other")  # the Fig. 3c mix (plan 5 C5)
# "manual": a target with no scripted source, waiting for a hand download (a Fab session with the user; plan 5 C5).
STATUSES = ("candidate", "placeholder", "manual", "imported", "primitive")
USABLE_STATUSES = ("primitive", "imported")  # in the registry and spawnable now; the rest wait for an import
SPLITS = ("train", "test_seen", "test_unseen")
# An evaluation's override (plan 6 revision 1 item 5): an unseen target among seen distractors, in any scene. Not a
# scene's split: test_unseen scenes (s11/s12) are out of scope, and a seen scene is where unseen targets are evaluated.
UNSEEN_TARGETS = "unseen"
TARGET_SPLITS = SPLITS + (UNSEEN_TARGETS,)
# One target and up to five distractors (spec §9 step 1: 3-5), all different entries: fewer cannot fill an episode
# without drawing an object twice.
MIN_USABLE = 6
ENTRY_KEYS = ("name", "category", "seen", "asset", "scale", "material", "status", "source", "height_m")
EXCLUSIONS_FILE = "target_exclusions.json"  # beside the pool file: the pool's own data, loaded with it
EXCLUSIONS_VERSION = 1


class TargetPoolError(ValueError):
    """A pool file that cannot be drawn from as it stands."""


@dataclass(frozen=True)
class TargetEntry:
    name: str
    category: str
    seen: bool
    asset: str | None  # a registry key; None until imported
    scale: tuple[float, float, float] | None
    material: str | None  # a registry material key; None keeps the mesh's own
    status: str
    source: dict | None = field(default=None, hash=False)
    height_m: float | None = None
    stand_in: bool = False  # a low-poly stand-in, drawn until a better model replaces it (plan 6 step 2b)

    @property
    def usable(self) -> bool:
        return self.status in USABLE_STATUSES

    def to_json(self) -> dict:
        """What an episode's provenance records of its target and each distractor (plan 5 C5, "Records"), the stand-in
        mark included (plan 6 revision 1 item 7: such episodes can be filtered out later)."""
        return {"name": self.name, "category": self.category, "seen": self.seen, "asset": self.asset,
                "scale": list(self.scale) if self.scale is not None else None, "material": self.material,
                "status": self.status, "stand_in": self.stand_in}


def _scale(value: Any, where: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in value)):
        raise TargetPoolError(f"{where}: scale must be three positive numbers [sx, sy, sz] or null, got {value!r}")
    return tuple(float(v) for v in value)


def _entry(raw: Any, where: str, registry: AssetRegistry) -> TargetEntry:
    if not isinstance(raw, dict):
        raise TargetPoolError(f"{where}: an entry must be an object, got {type(raw).__name__}")
    missing = [k for k in ENTRY_KEYS if k not in raw]
    if missing:
        raise TargetPoolError(f"{where}: missing {missing} (every entry carries {list(ENTRY_KEYS)}; null where unknown)")
    name = raw["name"]
    if not isinstance(name, str) or not name or name != name.strip():
        raise TargetPoolError(f"{where}: name must be a non-empty string without surrounding spaces, got {name!r}")
    if "." in name:
        # scripts/detect_targets.py's prompt_for refuses it too: Grounding DINO reads a period as a phrase break.
        raise TargetPoolError(f"{where}: name {name!r} contains a period, which the rebalancing detector reads as two "
                              f"phrases")
    where = f"{where} ({name!r})"
    if raw["category"] not in CATEGORIES:
        raise TargetPoolError(f"{where}: category {raw['category']!r} is not one of {CATEGORIES}")
    if not isinstance(raw["seen"], bool):
        raise TargetPoolError(f"{where}: seen must be true or false, got {raw['seen']!r}")
    if raw["status"] not in STATUSES:
        raise TargetPoolError(f"{where}: status {raw['status']!r} is not one of {STATUSES}")
    height = raw["height_m"]
    if height is not None and not (isinstance(height, (int, float)) and not isinstance(height, bool)
                                   and math.isfinite(height) and height > 0):
        raise TargetPoolError(f"{where}: height_m must be a positive number of metres or null, got {height!r}")
    if raw["source"] is not None and not isinstance(raw["source"], dict):
        raise TargetPoolError(f"{where}: source must be an object ({{source, id, licence, page}}) or null, "
                              f"got {raw['source']!r}")
    for key in ("asset", "material"):
        if raw[key] is not None and not isinstance(raw[key], str):
            raise TargetPoolError(f"{where}: {key} must be a registry key or null, got {raw[key]!r}")
    stand_in = raw.get("stand_in", False)
    if not isinstance(stand_in, bool):
        raise TargetPoolError(f"{where}: stand_in must be true or false where given, got {stand_in!r}")
    entry = TargetEntry(name=name, category=raw["category"], seen=raw["seen"], asset=raw["asset"],
                        scale=_scale(raw["scale"], where), material=raw["material"], status=raw["status"],
                        source=raw["source"], height_m=float(height) if height is not None else None,
                        stand_in=stand_in)
    if entry.usable:
        _check_spawnable(entry, where, registry)
    return entry


def _check_spawnable(entry: TargetEntry, where: str, registry: AssetRegistry) -> None:
    """A drawn entry must spawn as it stands (see the module docstring): checked at load, not in the 4th hour of a
    collection."""
    if entry.asset is None:
        raise TargetPoolError(f"{where}: a {entry.status} entry needs its registry asset")
    asset = registry.assets.get(entry.asset)
    if asset is None:
        raise TargetPoolError(f"{where}: asset {entry.asset!r} is not in the registry (assets/registry.json); a "
                              f"{entry.status} entry must be spawnable now")
    if asset.cards_part is not None:
        raise TargetPoolError(f"{where}: asset {entry.asset!r} has a no-collision cards part, and a runtime spawn places "
                              f"one mesh: the target would arrive without its cards")
    if asset.pivot not in ("center", "base"):
        raise TargetPoolError(f"{where}: asset {entry.asset!r} has pivot {asset.pivot!r}; standing it on the ground "
                              f"needs 'center' or 'base'")
    if entry.scale is None:
        raise TargetPoolError(f"{where}: a {entry.status} entry needs its scale")
    if entry.material is not None:
        material = registry.materials.get(entry.material)
        if material is None:
            raise TargetPoolError(f"{where}: material {entry.material!r} is not in the registry")
        if material.kind not in RUNTIME_SPAWNABLE_MATERIAL_KINDS:
            raise TargetPoolError(
                f"{where}: material {entry.material!r} is a {material.kind}, not a base UMaterial, so a runtime spawn "
                f"cannot apply it and the object would show the mesh's own material under a name that says otherwise; "
                f"give it a base UMaterial (kind {RUNTIME_SPAWNABLE_MATERIAL_KINDS}) or null")


@dataclass(frozen=True)
class TargetExclusions:
    """Per scene, the registry assets whose pool entries are never drawn there, each with the reason (plan 6 step 2c):

        {"version": 1, "scenes": {"<scene id>": {"<registry asset>": "<why it looks like this scene's obstacles>"}}}

    Keyed by asset, not by entry name: the shape is what reads as an obstacle, so an asset takes every colour of a
    primitive with it (one added later too), and a model is its own asset. Keyed by the scene's own id: s01d flies s01's
    level but draws by its own entry. A scene without an entry excludes nothing. Refused at load: a scene no scene file
    claims, an asset no entry of the pool is drawn from, an empty entry, a missing reason."""

    scenes: dict[str, dict[str, str]] = field(hash=False)
    path: str | None = None
    sha256: str | None = None

    @classmethod
    def load(cls, path: Path, entries: tuple[TargetEntry, ...]) -> "TargetExclusions":
        raw = Path(path).read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as err:
            raise TargetPoolError(f"{path}: not JSON ({err})") from err
        return cls.from_json(data, entries=entries, path=str(path), sha256=hashlib.sha256(raw).hexdigest())

    @classmethod
    def from_json(cls, data: Any, *, entries: tuple[TargetEntry, ...], path: str | None = None,
                  sha256: str | None = None, scene_ids: frozenset[str] | None = None) -> "TargetExclusions":
        from autofly_ue5.scenes.resolve import known_scene_ids

        where = path or "the exclusion table"
        if not isinstance(data, dict) or data.get("version") != EXCLUSIONS_VERSION:
            got = data.get("version") if isinstance(data, dict) else data
            raise TargetPoolError(f"{where}: version must be {EXCLUSIONS_VERSION}, got {got!r}")
        if not isinstance(data.get("scenes"), dict):
            raise TargetPoolError(f"{where}: 'scenes' must map a scene id to the assets it excludes, with a reason each")
        known = scene_ids if scene_ids is not None else known_scene_ids()
        assets = {e.asset for e in entries if e.asset is not None}
        scenes: dict[str, dict[str, str]] = {}
        for scene_id, excluded in data["scenes"].items():
            if scene_id not in known:
                raise TargetPoolError(f"{where}: scene {scene_id!r} is claimed by no scene file (known: {sorted(known)})")
            if not isinstance(excluded, dict) or not excluded:
                raise TargetPoolError(f"{where}: scene {scene_id!r} must map one or more registry assets to the reason "
                                      f"each is excluded")
            for asset, reason in excluded.items():
                if asset not in assets:
                    raise TargetPoolError(f"{where}: scene {scene_id!r} excludes {asset!r}, which no entry of the pool is "
                                          f"drawn from")
                if not isinstance(reason, str) or not reason.strip():
                    raise TargetPoolError(f"{where}: scene {scene_id!r}, asset {asset!r}: give the reason it is excluded")
            scenes[scene_id] = dict(excluded)
        return cls(scenes=scenes, path=path, sha256=sha256)

    def for_scene(self, scene_id: str | None) -> dict[str, str]:
        """{asset: reason} for the assets `scene_id` never draws; empty for a scene without an entry, or no scene."""
        return dict(self.scenes.get(scene_id, {})) if scene_id is not None else {}


@dataclass(frozen=True)
class TargetPool:
    entries: tuple[TargetEntry, ...]
    path: str | None = None
    sha256: str | None = None
    exclusions: TargetExclusions | None = None

    @classmethod
    def load(cls, path: Path, registry: AssetRegistry | None = None) -> "TargetPool":
        """The pool file, with the exclusion table beside it (EXCLUSIONS_FILE) when there is one: the table is the
        pool's own data, so every reader of the pool draws by it without being told to."""
        raw = Path(path).read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as err:
            raise TargetPoolError(f"{path}: not JSON ({err})") from err
        pool = cls.from_json(data, registry=registry, path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        table = Path(path).with_name(EXCLUSIONS_FILE)
        if table.is_file():
            pool = dataclasses.replace(pool, exclusions=TargetExclusions.load(table, pool.entries))
        return pool

    @classmethod
    def from_json(cls, data: Any, *, registry: AssetRegistry | None = None, path: str | None = None,
                  sha256: str | None = None) -> "TargetPool":
        where = path or "the target pool"
        if not isinstance(data, dict) or data.get("version") != VERSION:
            got = data.get("version") if isinstance(data, dict) else data
            raise TargetPoolError(f"{where}: version must be {VERSION}, got {got!r}")
        if not isinstance(data.get("targets"), list):
            raise TargetPoolError(f"{where}: 'targets' must be a list of entries")
        registry = registry if registry is not None else load_registry()
        entries = tuple(_entry(raw, f"{where}: targets[{i}]", registry) for i, raw in enumerate(data["targets"]))
        seen_names: dict[str, str] = {}
        for entry in entries:
            # Case-insensitively: the detector lower-cases its query, so two such names would be one phrase.
            key = entry.name.lower()
            if key in seen_names:
                raise TargetPoolError(f"{where}: names must be unique; {entry.name!r} repeats {seen_names[key]!r}")
            seen_names[key] = entry.name
        return cls(entries=entries, path=path, sha256=sha256)

    def excluded(self, scene_id: str | None) -> dict[str, str]:
        """{asset: reason} for the assets scene `scene_id` never draws (its entry of the exclusion table)."""
        return self.exclusions.for_scene(scene_id) if self.exclusions is not None else {}

    def for_split(self, split: str, *, scene_id: str | None = None) -> tuple[TargetEntry, ...]:
        """The entries a scene of `split` draws from, in file order (the sampler's draws index this tuple): seen targets
        for train and test_seen, seen and unseen for test_unseen (spec §9 step 1), less those whose asset scene
        `scene_id` excludes (None: no scene, nothing excluded). Refused under MIN_USABLE, after the exclusions."""
        if split not in SPLITS:
            raise TargetPoolError(f"unknown split {split!r}; a scene's split is one of {SPLITS}")
        excluded = self.excluded(scene_id)
        usable = tuple(e for e in self.entries
                       if e.usable and (e.seen or split == "test_unseen") and e.asset not in excluded)
        if len(usable) < MIN_USABLE:
            after = f" in scene {scene_id!r} after its exclusions ({sorted(excluded)})" if excluded else ""
            raise TargetPoolError(f"{self.path or 'the target pool'}: {len(usable)} usable entries for split {split!r}"
                                  f"{after}; an episode needs {MIN_USABLE} different ones (a target and up to five "
                                  f"distractors)")
        return usable

    def unseen_targets(self, *, scene_id: str | None = None) -> tuple[TargetEntry, ...]:
        """The usable unseen entries in file order, less those scene `scene_id` excludes: what an UNSEEN_TARGETS draw
        takes its target from. Refused when there is none."""
        excluded = self.excluded(scene_id)
        unseen = tuple(e for e in self.entries if e.usable and not e.seen and e.asset not in excluded)
        if not unseen:
            where = f" in scene {scene_id!r} after its exclusions ({sorted(excluded)})" if excluded else ""
            raise TargetPoolError(f"{self.path or 'the target pool'}: no usable unseen entry{where}; an unseen-target "
                                  f"draw needs at least one")
        return unseen

    def draw_lists(self, split: str, *, scene_id: str | None = None) -> tuple[list[TargetEntry], list[TargetEntry]]:
        """(the entries an episode's target is drawn from, the entries its distractors are drawn from), fresh lists in
        file order after scene `scene_id`'s exclusions. For a split: `for_split(split)`, one list returned twice (the
        sampler removes the target from it before drawing the distractors, as it always has). For UNSEEN_TARGETS:
        `unseen_targets()` and `for_split("train")` (the seen entries), two disjoint lists."""
        if split == UNSEEN_TARGETS:
            return list(self.unseen_targets(scene_id=scene_id)), list(self.for_split("train", scene_id=scene_id))
        if split not in SPLITS:
            raise TargetPoolError(f"unknown target split {split!r}; one of {TARGET_SPLITS}")
        rest = list(self.for_split(split, scene_id=scene_id))
        return rest, rest

    def to_json(self, split: str, *, scene_id: str | None = None) -> dict:
        """The pool a collection drew from, for its record and provenance: the file, its hash and the split's names; for
        a scene, the names it draws, what its exclusions took out and why, and the table they came from."""
        record = {"path": self.path, "sha256": self.sha256, "version": VERSION, "split": split,
                  "names": [e.name for e in self.for_split(split, scene_id=scene_id)]}
        if scene_id is not None:
            table = self.exclusions
            record.update(scene_id=scene_id, excluded=self.excluded(scene_id),
                          exclusions={"path": table.path, "sha256": table.sha256} if table is not None else None)
        return record
