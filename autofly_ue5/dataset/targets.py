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

Every entry carries all nine keys, so a misspelt key is a missing one and is refused; other keys are ignored. Only
`primitive` (engine content) and `imported` entries are drawn, and each must be spawnable as it stands: its asset in the
registry, one mesh (a runtime spawn places one: a model with a no-collision `cards` part would arrive without its
leaves), a scale, and a material that is either none (the mesh's own) or a base `UMaterial` the runtime can apply. A
palette colour is a material instance, which Project AirSim's `setMaterial` cannot load (expert/episode.py
`_runtime_material_path`): spawned, a "red cube" would be the mesh's own grey under a name that says red, so such an
entry is refused here rather than recorded as a mislabelled episode.
"""

from __future__ import annotations

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
# One target and up to five distractors (spec §9 step 1: 3-5), all different entries: fewer cannot fill an episode
# without drawing an object twice.
MIN_USABLE = 6
ENTRY_KEYS = ("name", "category", "seen", "asset", "scale", "material", "status", "source", "height_m")


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

    @property
    def usable(self) -> bool:
        return self.status in USABLE_STATUSES

    def to_json(self) -> dict:
        """What an episode's provenance records of its target and each distractor (plan 5 C5, "Records")."""
        return {"name": self.name, "category": self.category, "seen": self.seen, "asset": self.asset,
                "scale": list(self.scale) if self.scale is not None else None, "material": self.material,
                "status": self.status}


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
    entry = TargetEntry(name=name, category=raw["category"], seen=raw["seen"], asset=raw["asset"],
                        scale=_scale(raw["scale"], where), material=raw["material"], status=raw["status"],
                        source=raw["source"], height_m=float(height) if height is not None else None)
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
class TargetPool:
    entries: tuple[TargetEntry, ...]
    path: str | None = None
    sha256: str | None = None

    @classmethod
    def load(cls, path: Path, registry: AssetRegistry | None = None) -> "TargetPool":
        raw = Path(path).read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as err:
            raise TargetPoolError(f"{path}: not JSON ({err})") from err
        return cls.from_json(data, registry=registry, path=str(path), sha256=hashlib.sha256(raw).hexdigest())

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

    def for_split(self, split: str) -> tuple[TargetEntry, ...]:
        """The entries a scene of `split` draws from, in file order (the sampler's draws index this tuple): seen targets
        for train and test_seen, seen and unseen for test_unseen (spec §9 step 1). Refused under MIN_USABLE."""
        if split not in SPLITS:
            raise TargetPoolError(f"unknown split {split!r}; a scene's split is one of {SPLITS}")
        usable = tuple(e for e in self.entries if e.usable and (e.seen or split == "test_unseen"))
        if len(usable) < MIN_USABLE:
            raise TargetPoolError(f"{self.path or 'the target pool'}: {len(usable)} usable entries for split {split!r}; "
                                  f"an episode needs {MIN_USABLE} different ones (a target and up to five distractors)")
        return usable

    def to_json(self, split: str) -> dict:
        """The pool a collection drew from, for its record and provenance: the file, its hash and the split's names."""
        return {"path": self.path, "sha256": self.sha256, "version": VERSION, "split": split,
                "names": [e.name for e in self.for_split(split)]}
