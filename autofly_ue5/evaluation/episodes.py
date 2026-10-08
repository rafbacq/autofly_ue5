"""Frozen evaluation episodes (plan 6 item 6 and revision 1 item 5; spec §12 M6): per condition, which (scene, seed,
target split) a policy is scored on, written once and committed as assets/eval_episodes/<condition>.json.

Conditions:
- `seen`: the nine training scenes, targets from their own split (seen ones).
- `test_seen`: the test_seen layouts, the training scenes' elements in new layouts (spec §3.1): s05r, and s06r, listed
  but pending until a gate record of it passes (its gate waits on a probe fallback; revision 1 item 9). A pending
  scene's seeds are frozen with the rest, and the harness flies it once its gate passes, without a new file.
- `unseen_targets`: the nine training scenes with target split "unseen": the target drawn from the pool's unseen entries
  only, the distractors from seen ones. `TargetPool.for_split` draws unseen targets only for test_unseen scenes, which
  are out of scope, so the harness overrides the split (revision 1 item 5).

Seeds: scene `s` of condition `c` flies eval_harness_seed_base(c's slot, s's slot) + i (expert/seeds.py), a range no
training, gate, selection, probe or collection episode uses, and in which no two files or scenes share a seed. A seed
whose geometry the sampler cannot place is skipped at write time and named in the file: the resilient wrapper retries an
EpisodeSetupError as a reset fault, so a run would spend three simulator relaunches on it before giving up.

Each file pins every scene's file and layout by sha256. A seed is an episode only on the layout it was drawn on, so the
harness refuses a scene whose layout has changed rather than score another evaluation under the same name. Targets are
drawn at run time from the pool the run is given (the pool comes last in the sampler's draws, so it never moves a start
or a target); the run's record pins that pool by sha256, and only runs on the same pool compare target for target. A
file is never written over.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from autofly_ue5.evidence import COMMITTED_GATES_DIR
from autofly_ue5.expert.episode import EpisodeSetupError, sample_setup
from autofly_ue5.expert.seeds import EVAL_HARNESS_SCENE_STRIDE, eval_harness_seed_base
from autofly_ue5.paths import ROOT
from autofly_ue5.scenes.resolve import ResolvedScene, resolve_scene

EPISODES_DIR = ROOT / "assets" / "eval_episodes"
FILE_KIND = "autofly_ue5_eval_episodes"
FILE_VERSION = 1
DEFAULT_EPISODES_PER_SCENE = 20
A0_MODE = "sector8"  # spec §9 step 4: the collector's aligned start, so a policy starts as its training episodes did
UNSEEN = "unseen"  # AutoFlyEnv's target_split: the target from unseen entries, the distractors from seen ones
TRAINING_SCENES = ("s01", "s01d", "s02", "s03", "s04", "s05", "s06", "s07", "s09")
TEST_SEEN_SCENES = ("s05r", "s06r")
# The seed slots (expert/seeds.py eval_harness_seed_base). Append only, never reorder: a frozen file's seeds were drawn by
# them, and a reused slot would hand a new file another file's episodes.
CONDITION_SLOTS = ("seen", "test_seen", "unseen_targets")
SCENE_SLOTS = ("s01", "s01d", "s02", "s03", "s04", "s05", "s06", "s07", "s09", "s05r", "s06r")


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    scenes: tuple[str, ...]
    scene_split: str  # the split each of its scene files must declare
    target_split: str | None  # AutoFlyEnv's target_split; None: the scene's own split
    gated: bool  # each scene is flown only once a gate record of it passes (m4_gate_status)
    description: str


CONDITIONS = {
    "seen": ConditionSpec("seen", TRAINING_SCENES, "train", None, False,
                          "The nine training scenes with seen targets: the dataset's own conditions, new episodes."),
    "test_seen": ConditionSpec("test_seen", TEST_SEEN_SCENES, "test_seen", None, True,
                               "The test_seen layouts (training elements, new layouts; spec §3.1) with seen targets. "
                               "s06r is flown once a gate record of it passes."),
    "unseen_targets": ConditionSpec("unseen_targets", TRAINING_SCENES, "train", UNSEEN, False,
                                    "The nine training scenes, each target drawn from the pool's unseen entries and the "
                                    "distractors from seen ones (plan 6 revision 1 item 5)."),
}


@dataclass(frozen=True)
class EvalEpisode:
    scene: str
    seed: int
    target_split: str | None


@dataclass(frozen=True, eq=False)
class ConditionFile:
    """A loaded, checked condition file: its raw document, its episodes in file order, and where it came from."""

    path: Path | None
    sha256: str | None
    data: dict
    episodes: tuple[EvalEpisode, ...]

    @property
    def name(self) -> str:
        return self.data["condition"]

    @property
    def spec(self) -> ConditionSpec:
        return CONDITIONS[self.name]

    @property
    def scenes(self) -> dict:
        return self.data["scenes"]

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(self.data["scenes"])

    @property
    def a0(self) -> str:
        return self.data["a0"]

    @property
    def episodes_per_scene(self) -> int:
        return self.data["episodes_per_scene"]

    def episodes_for(self, scene_id: str) -> tuple[EvalEpisode, ...]:
        return tuple(e for e in self.episodes if e.scene == scene_id)


def _run_number(path: Path) -> int:
    match = re.search(r"_run(\d+)\.json$", path.name)
    return int(match.group(1)) if match else 0


def m4_gate_status(scene_id: str, gates_dir: Path = COMMITTED_GATES_DIR) -> dict:
    """Whether a live gate record of `scene_id` passes: m4_<scene>_gate.json or a re-run's m4_<scene>_gate_run<N>.json
    (s02 passed on its third, m4_s02_gate_run3.json). {"ready", "gate": the passing file's name, "reason"}."""
    gates_dir = Path(gates_dir)
    candidates = [gates_dir / f"m4_{scene_id}_gate.json",
                  *sorted(gates_dir.glob(f"m4_{scene_id}_gate_run*.json"), key=_run_number)]
    for path in candidates:
        try:
            passed = json.loads(path.read_text()).get("pass") is True
        except (OSError, ValueError, AttributeError):
            continue
        if passed:
            return {"ready": True, "gate": path.name, "reason": None}
    return {"ready": False, "gate": None,
            "reason": f"no passing m4_{scene_id}_gate.json or m4_{scene_id}_gate_run<N>.json in {gates_dir}"}


def _placeable_seeds(resolved: ResolvedScene, base: int, n: int) -> tuple[list[int], list[dict]]:
    """The first `n` seeds from `base` whose setup the sampler places, and those it could not."""
    seeds: list[int] = []
    skipped: list[dict] = []
    for seed in range(base, base + EVAL_HARNESS_SCENE_STRIDE):
        if len(seeds) == n:
            break
        try:
            sample_setup(resolved.scene, resolved.layout, np.random.default_rng(seed))
        except EpisodeSetupError as err:
            skipped.append({"seed": seed, "reason": str(err)})
            continue
        seeds.append(seed)
    if len(seeds) < n:
        raise ValueError(f"scene {resolved.scene.id}: only {len(seeds)} of the {n} seeds asked for can be placed in its "
                         f"slice from {base}")
    return seeds, skipped


def condition_document(name: str, *, episodes_per_scene: int = DEFAULT_EPISODES_PER_SCENE,
                       scenes: Iterable[str] | None = None, resolve: Callable[[str], ResolvedScene] = resolve_scene,
                       gates_dir: Path = COMMITTED_GATES_DIR) -> dict:
    """The file `write_condition` writes for condition `name`: its scenes (all of the condition's, or `scenes`), each
    pinned by its file's and its layout's sha256 (`resolve` reads them), and `episodes_per_scene` seeds per scene."""
    if name not in CONDITIONS:
        raise ValueError(f"unknown evaluation condition {name!r} (conditions: {', '.join(CONDITIONS)})")
    spec = CONDITIONS[name]
    if not 1 <= episodes_per_scene <= EVAL_HARNESS_SCENE_STRIDE:
        raise ValueError(f"episodes per scene must be in [1, {EVAL_HARNESS_SCENE_STRIDE}], got {episodes_per_scene}")
    chosen = spec.scenes if scenes is None else tuple(scenes)
    unknown = [s for s in chosen if s not in spec.scenes]
    if unknown:
        raise ValueError(f"condition {name} has no scene {', '.join(unknown)} (its scenes: {', '.join(spec.scenes)})")
    condition_slot = CONDITION_SLOTS.index(name)
    entries: dict[str, dict] = {}
    episodes: list[dict] = []
    for scene_id in chosen:
        resolved = resolve(scene_id)
        if resolved.scene.id != scene_id:
            raise ValueError(f"resolving scene {scene_id} gave scene {resolved.scene.id}")
        if resolved.scene.split != spec.scene_split:
            raise ValueError(f"scene {scene_id} is a {resolved.scene.split} scene; condition {name} flies "
                             f"{spec.scene_split} scenes (its split decides the targets it draws)")
        slot = SCENE_SLOTS.index(scene_id)
        base = eval_harness_seed_base(condition_slot, slot)
        seeds, skipped = _placeable_seeds(resolved, base, episodes_per_scene)
        entry = {"slot": slot, "seed_base": base, "split": resolved.scene.split, "target_split": spec.target_split,
                 "scene_sha256": resolved.scene.sha256, "layout_sha256": resolved.layout_sha256,
                 "requires_gate": None, "status_at_write": "ready", "gate_at_write": None, "pending_reason": None,
                 "skipped_seeds": skipped}
        if spec.gated:
            status = m4_gate_status(scene_id, gates_dir)
            entry.update(requires_gate=f"docs/gates/m4_{scene_id}_gate.json",
                         status_at_write="ready" if status["ready"] else "pending", gate_at_write=status["gate"],
                         pending_reason=None if status["ready"] else
                         f"{status['reason']}: its seeds are frozen here, and it is flown once a gate record of it passes")
        entries[scene_id] = entry
        episodes += [{"scene": scene_id, "seed": seed, "target_split": spec.target_split} for seed in seeds]
    return {"kind": FILE_KIND, "version": FILE_VERSION, "condition": name, "description": spec.description,
            "written": time.strftime("%Y-%m-%d %H:%M:%S"), "a0": A0_MODE, "episodes_per_scene": episodes_per_scene,
            "seed_base": eval_harness_seed_base(condition_slot, 0), "scenes": entries, "episodes": episodes}


def _dump(document: dict) -> str:
    """JSON with one episode per line, so a file reads (and diffs) as the list it is."""
    head = json.dumps({k: v for k, v in document.items() if k != "episodes"}, indent=1)
    rows = ",\n".join("  " + json.dumps(e) for e in document["episodes"])
    return head[:-2] + ',\n "episodes": [\n' + rows + "\n ]\n}\n"


def write_condition(name: str, out_dir: Path = EPISODES_DIR, **kwargs) -> Path:
    """Write condition `name`'s file (`condition_document`'s keyword arguments) as <out_dir>/<name>.json; an existing
    file is refused (FileExistsError), never written over."""
    path = Path(out_dir) / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"{path} is a frozen evaluation file and is never written over")
    text = _dump(condition_document(name, **kwargs))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as handle:
        handle.write(text)
    return path


def check_document(data: dict, where: str = "the condition file") -> None:
    """Refuse (ValueError) a document that is not a condition file of a known condition, whose scenes are not the
    condition's, whose seeds leave their scene's slice or repeat, or whose counts and splits contradict it."""
    try:
        if data.get("kind") != FILE_KIND or data.get("version") != FILE_VERSION:
            raise ValueError(f"{where}: not a version {FILE_VERSION} {FILE_KIND} file (kind {data.get('kind')!r}, "
                             f"version {data.get('version')!r})")
        name = data["condition"]
        if name not in CONDITIONS:
            raise ValueError(f"{where}: unknown condition {name!r}")
        spec = CONDITIONS[name]
        if data["a0"] != A0_MODE:
            raise ValueError(f"{where}: a0 must be {A0_MODE!r} (the collector's aligned start), got {data['a0']!r}")
        condition_slot = CONDITION_SLOTS.index(name)
        for scene_id, entry in data["scenes"].items():
            if scene_id not in spec.scenes:
                raise ValueError(f"{where}: condition {name} has no scene {scene_id}")
            slot = SCENE_SLOTS.index(scene_id)
            if entry["slot"] != slot or entry["seed_base"] != eval_harness_seed_base(condition_slot, slot):
                raise ValueError(f"{where}: scene {scene_id} does not start at its own seed slice "
                                 f"{eval_harness_seed_base(condition_slot, slot)}")
            if entry["target_split"] != spec.target_split:
                raise ValueError(f"{where}: scene {scene_id}'s target_split {entry['target_split']!r} is not condition "
                                 f"{name}'s {spec.target_split!r}")
        flown: set[tuple[str, int]] = set()
        counts: Counter[str] = Counter()
        for episode in data["episodes"]:
            scene_id, seed = episode["scene"], episode["seed"]
            if scene_id not in data["scenes"]:
                raise ValueError(f"{where}: an episode of scene {scene_id}, which the file does not list")
            entry = data["scenes"][scene_id]
            if not (isinstance(seed, int) and entry["seed_base"] <= seed < entry["seed_base"] + EVAL_HARNESS_SCENE_STRIDE):
                raise ValueError(f"{where}: seed {seed!r} lies outside scene {scene_id}'s slice from {entry['seed_base']}")
            if (scene_id, seed) in flown:
                raise ValueError(f"{where}: scene {scene_id} lists seed {seed} twice")
            if episode["target_split"] != entry["target_split"]:
                raise ValueError(f"{where}: an episode's target_split {episode['target_split']!r} is not its scene's "
                                 f"{entry['target_split']!r}")
            flown.add((scene_id, seed))
            counts[scene_id] += 1
        uneven = {s: counts[s] for s in data["scenes"] if counts[s] != data["episodes_per_scene"]}
        if uneven:
            raise ValueError(f"{where}: episodes_per_scene is {data['episodes_per_scene']}, but scenes hold {uneven}")
    except (KeyError, TypeError, AttributeError) as err:
        raise ValueError(f"{where}: malformed condition file ({type(err).__name__}: {err})") from err


def load_condition(path: Path) -> ConditionFile:
    """A condition file, checked (`check_document`), with its sha256 for the run's record."""
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError(f"{path}: not JSON ({err})") from err
    check_document(data, str(path))
    episodes = tuple(EvalEpisode(scene=e["scene"], seed=e["seed"], target_split=e["target_split"]) for e in data["episodes"])
    return ConditionFile(path=Path(path), sha256=hashlib.sha256(raw).hexdigest(), data=data, episodes=episodes)
