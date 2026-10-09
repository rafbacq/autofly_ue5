"""What a set of raw dataset stores holds, read for reports (plan 6 step 7: the figures and videos of final_results/).

A store is what dataset/raw.py writes: data/<name>/ (manifest.json, episodes/<id>/, provenance/<id>.json) and
data/rejects/<name>/<id>.json, one file per flight not kept, with its reason. Everything here is read from those files
as they are. Two questions need more than one file, and each answer says where it came from:

- *What is each kept episode's target?* Its provenance's target record says, for an episode drawn from the target pool
  (name, category, seen, status; and `stand_in` since plan 6 step 2b marks the Kenney stand-ins). A store from before
  the pool (the M3 pilot: the orange cylinder under a placeholder name, no category) or from before the mark is
  completed from the pool file by the target's name, and `Target.from_pool` lists every field so filled; nothing else
  is inferred. A pool file with no `stand_in` key anywhere predates the mark, so it cannot say that an entry is *not* a
  stand-in: such a target's mark stays None (unknown).
- *How many flights did it take?* Kept plus rejected. The rejects directory names each reject's reason; without it the
  collector's record (`collection.outcomes`, scripts/collect_dataset.py) does; without both, the manifest's reject count
  still gives the total, the reasons then "unrecorded". Where a record exists beside the files the two are compared and
  each disagreement is reported, never resolved: a report must show a store whose files and record differ.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autofly_ue5.dataset.raw import FORMAT

UNKNOWN = "unknown"
UNRECORDED = "unrecorded"  # rejects whose reasons no file holds: only the manifest's count of them is known
AUTO = "auto"  # load_store's default rejects directory: <data root>/rejects/<name>, where RawDatasetWriter puts them
Z_95 = 1.959964  # two-sided 95 %, as the pilot screening's intervals (docs/gates/m5_pilot_screening.json)


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval of k successes in n trials: the interval of plan 6's pilot bar (a kept-rate whose
    lower bound is at least 0.25) and of the screening record, which the tests hold this to."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"a Wilson interval needs 0 <= k <= n and n > 0, got k={k}, n={n}")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Pool:
    """The target pool file (assets/targets.json), read for lookups by name only. It is not validated against the
    registry (dataset/targets.py does that for collection): a report may describe stores collected under an older pool."""

    path: str
    sha256: str
    entries: Mapping[str, dict]
    marks_stand_ins: bool  # some entry carries the `stand_in` key: the file postdates plan 6 step 2b's mark


def load_pool(path: Path) -> Pool:
    blob = Path(path).read_bytes()
    data = json.loads(blob)
    targets = data.get("targets") if isinstance(data, dict) else None
    if not isinstance(targets, list):
        raise ValueError(f"{path}: not a target pool file (no 'targets' list)")
    entries = {t["name"]: t for t in targets if isinstance(t, dict) and isinstance(t.get("name"), str)}
    return Pool(path=str(path), sha256=hashlib.sha256(blob).hexdigest(), entries=entries,
                marks_stand_ins=any("stand_in" in t for t in entries.values()))


@dataclass(frozen=True)
class Target:
    name: str
    category: str  # UNKNOWN where neither the provenance nor the pool names one
    seen: bool | None
    stand_in: bool | None  # None: nothing marks it either way
    status: str | None  # the pool entry's ("primitive", "imported", "placeholder", ...)
    name_status: str | None  # the collector's: "pool", or M3's "placeholder_until_M4"
    from_pool: tuple[str, ...] = ()  # the fields above that the pool file filled in, by name


def resolve_target(record: dict | None, name: str, pool: Pool | None) -> Target:
    """An episode's target from its provenance record, completed from the pool by name where the record lacks a field."""
    record = record if isinstance(record, dict) else {}
    entry = pool.entries.get(name) if pool is not None else None
    filled: list[str] = []

    def pick(key: str, valid):
        if valid(record.get(key)):
            return record[key]
        if entry is not None and valid(entry.get(key)):
            filled.append(key)
            return entry[key]
        return None

    category = pick("category", lambda v: isinstance(v, str) and v != "")
    seen = pick("seen", lambda v: isinstance(v, bool))
    status = pick("status", lambda v: isinstance(v, str) and v != "")
    if isinstance(record.get("stand_in"), bool):
        stand_in = record["stand_in"]
    elif entry is not None and pool.marks_stand_ins:
        stand_in = entry.get("stand_in") is True
        filled.append("stand_in")
    else:
        stand_in = None
    return Target(name=name, category=category or UNKNOWN, seen=seen, stand_in=stand_in, status=status,
                  name_status=record.get("name_status"), from_pool=tuple(filled))


@dataclass(frozen=True)
class Episode:
    """One kept episode as its store's manifest lists it; its records and provenance are read on demand."""

    store: str
    id: str
    scene: str
    seed: int
    steps: int
    instruction: str
    target: Target
    directory: Path
    provenance_path: Path
    # Whether the first recorded frame shows the target (the collector's `first_frame_visibility` block, plan 6 step 2a:
    # visible, missing, occluded, out_of_view or undetermined); None for an episode collected before the check.
    first_frame: str | None = None

    def frame(self, index: int) -> Path:
        """The PNG of record `index` (negative counts from the end: -1 is the last record)."""
        i = index + self.steps if index < 0 else index
        if not 0 <= i < self.steps:
            raise IndexError(f"{self.id} has {self.steps} records; there is no record {index}")
        return self.directory / "frames" / f"{i:06d}.png"

    def records(self) -> dict[str, np.ndarray]:
        """steps.npz: state (T, 9), action (T, 3), sim_time_ns (T,)."""
        with np.load(self.directory / "steps.npz") as data:
            return {key: data[key] for key in data.files}

    def provenance(self) -> dict:
        return json.loads(self.provenance_path.read_text())


def _pose(value) -> tuple[float, ...] | None:
    return tuple(float(v) for v in value) if isinstance(value, list) and value else None


@dataclass(frozen=True)
class Reject:
    """One flight not kept: its reason (the env's outcome, or the collector's exceeded_max_steps) and where it ended."""

    store: str
    id: str
    seed: int | None
    reason: str
    scene: str | None
    collision_source: str | None
    start_pose: tuple[float, ...] | None
    final_pose: tuple[float, ...] | None
    target_xyz: tuple[float, ...] | None


def read_rejects(directory: Path, store: str) -> tuple[Reject, ...]:
    out = []
    for path in Path(directory).glob("*.json"):
        data = json.loads(path.read_text())
        prov = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
        seed = prov.get("seed")
        out.append(Reject(store=store, id=str(data.get("id", path.stem)), seed=seed if isinstance(seed, int) else None,
                          reason=str(data.get("reason") or UNKNOWN), scene=(prov.get("scene") or {}).get("id"),
                          collision_source=prov.get("collision_source"), start_pose=_pose(prov.get("start_pose")),
                          final_pose=_pose(prov.get("final_pose")),
                          target_xyz=_pose((prov.get("target") or {}).get("xyz"))))
    return tuple(sorted(out, key=lambda r: (r.seed is None, r.seed or 0, r.id)))


@dataclass(frozen=True)
class Attempts:
    """A store's flights: kept, and rejected by reason."""

    kept: int
    flown: int
    rejected: dict[str, int]
    source: str  # where `rejected` came from: "rejects", "record" or "manifest"; "+record" when a record was compared
    collision_sources: dict[str, int]  # the rejects directory's collisions by source (footprint, sim, mover, ...)
    fault_retries: int | None  # the record's: backend faults whose seeds were flown again (never in any store)
    disagreements: tuple[str, ...]

    @property
    def kept_rate(self) -> float | None:
        return self.kept / self.flown if self.flown else None

    def wilson(self) -> tuple[float, float] | None:
        return wilson_interval(self.kept, self.flown) if self.flown else None


def _outcomes(collection: dict) -> dict[str, int]:
    return {str(k): int(v) for k, v in collection["outcomes"].items() if k != "success" and v}


def count_attempts(kept: int, rejects: tuple[Reject, ...] | None, record: dict | None,
                   manifest_rejects: int | None) -> Attempts:
    collection = record.get("collection") if isinstance(record, dict) else None
    collection = collection if isinstance(collection, dict) else None
    notes: list[str] = []
    collisions: Counter = Counter()
    if rejects is not None:
        rejected = dict(Counter(r.reason for r in rejects))
        collisions = Counter(r.collision_source or UNKNOWN for r in rejects if r.reason == "collision")
        source = "rejects"
        if isinstance(manifest_rejects, int) and manifest_rejects != len(rejects):
            notes.append(f"the manifest counts {manifest_rejects} rejects, the rejects directory holds {len(rejects)}")
    elif collection is not None and isinstance(collection.get("outcomes"), dict):
        rejected, source = _outcomes(collection), "record"
    elif isinstance(manifest_rejects, int):
        rejected, source = ({UNRECORDED: manifest_rejects} if manifest_rejects else {}), "manifest"
    else:
        rejected, source = {}, "kept only"
    flown = kept + sum(rejected.values())
    if collection is not None:
        if source != "record":
            source += "+record"
        if collection.get("kept") is not None and int(collection["kept"]) != kept:
            notes.append(f"the record says kept {collection['kept']}, the store holds {kept}")
        if collection.get("attempted") is not None and int(collection["attempted"]) != flown:
            notes.append(f"the record says attempted {collection['attempted']}, the store's files make {flown}")
        if source == "rejects+record" and isinstance(collection.get("outcomes"), dict):
            if _outcomes(collection) != rejected:
                notes.append(f"the record's outcomes {_outcomes(collection)} differ from the rejects directory's "
                             f"{rejected}")
    retries = collection.get("fault_retries") if collection is not None else None
    return Attempts(kept=kept, flown=flown, rejected=dict(sorted(rejected.items())), source=source,
                    collision_sources=dict(sorted(collisions.items())),
                    fault_retries=int(retries) if isinstance(retries, int) else None, disagreements=tuple(notes))


@dataclass(frozen=True)
class Store:
    name: str
    root: Path
    manifest: dict
    episodes: tuple[Episode, ...]
    rejects: tuple[Reject, ...] | None  # None: no rejects directory was read
    rejects_dir: Path | None
    record: dict | None
    record_path: Path | None
    attempts: Attempts

    @property
    def scenes(self) -> tuple[str, ...]:
        """Every scene the store names: its manifest's, its episodes' and its rejects'."""
        named = set(self.manifest.get("scenes", {})) | {e.scene for e in self.episodes}
        named |= {r.scene for r in self.rejects or () if r.scene}
        return tuple(sorted(named))


def load_store(root: Path, *, pool: Pool | None = None, rejects_dir: Path | str | None = AUTO,
               record_path: Path | None = None) -> Store:
    """A store as its files hold it. `rejects_dir`: AUTO reads <root's parent>/rejects/<name> when it exists (where
    RawDatasetWriter writes them), None reads none, a path must exist. `record_path`: the collector's record, refused
    when it is another dataset's."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{root} is not a raw dataset store: it has no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{manifest_path}: format {manifest.get('format')!r}, expected {FORMAT!r}")
    name = manifest.get("name") or root.name
    episodes = []
    for entry in manifest.get("episodes", []):
        prov_path = root / "provenance" / f"{entry['id']}.json"
        prov = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
        check = prov.get("first_frame_visibility")
        status = check.get("status") if isinstance(check, dict) else None
        episodes.append(Episode(store=name, id=entry["id"], scene=entry["scene"], seed=int(entry["seed"]),
                                steps=int(entry["steps"]), instruction=entry["instruction"],
                                target=resolve_target(prov.get("target"), entry["target_name"], pool),
                                directory=root / entry["path"], provenance_path=prov_path,
                                first_frame=status if isinstance(status, str) else None))
    if rejects_dir == AUTO:
        rejects_dir = root.parent / "rejects" / name
        rejects_dir = rejects_dir if rejects_dir.is_dir() else None
    elif rejects_dir is not None and not Path(rejects_dir).is_dir():
        raise ValueError(f"{rejects_dir} is not a directory of rejects")
    rejects = read_rejects(Path(rejects_dir), name) if rejects_dir is not None else None
    record = None
    if record_path is not None:
        record = json.loads(Path(record_path).read_text())
        recorded = (record.get("dataset") or {}).get("name") if isinstance(record, dict) else None
        if recorded != name:
            raise ValueError(f"{record_path} is the collector's record of dataset {recorded!r}, not of {name!r}")
    attempts = count_attempts(len(episodes), rejects, record, manifest.get("counts", {}).get("rejects"))
    return Store(name=name, root=root, manifest=manifest, episodes=tuple(episodes), rejects=rejects,
                 rejects_dir=Path(rejects_dir) if rejects_dir is not None else None, record=record,
                 record_path=Path(record_path) if record_path is not None else None, attempts=attempts)


def load_stores(roots: Iterable[Path], *, pool: Pool | None = None, read_rejects_dirs: bool = True,
                records: Mapping[str, Path] | None = None) -> list[Store]:
    """Several stores; two of one name are refused (a name is a store's identity in every report and record).
    `records` maps a store's name to its collector's record."""
    stores: list[Store] = []
    records = dict(records or {})
    for root in roots:
        manifest = Path(root) / "manifest.json"
        name = (json.loads(manifest.read_text()).get("name") if manifest.is_file() else None) or Path(root).name
        store = load_store(root, pool=pool, rejects_dir=AUTO if read_rejects_dirs else None,
                           record_path=records.get(name))
        if any(s.name == store.name for s in stores):
            raise ValueError(f"two stores are named {store.name!r} ({store.root} and "
                             f"{next(s.root for s in stores if s.name == store.name)})")
        stores.append(store)
    unused = set(records) - {s.name for s in stores}
    if unused:
        raise ValueError(f"records given for stores not listed: {sorted(unused)}")
    return stores
