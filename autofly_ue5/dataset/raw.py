"""The canonical raw dataset store (spec §10; Plan 4 task B4). Every export (the RLDS shards) is derived from it.

    data/<name>/manifest.json                     the episodes, their split and counts; rewritten after each episode
    data/<name>/episodes/<id>/frames/000000.png   front RGB, 256 x 256, one per record
    data/<name>/episodes/<id>/steps.npz           state (T, 9) float32, action (T, 3) float32, sim_time_ns (T,) int64
    data/<name>/provenance/<id>.json              where the episode came from (spec §10.2), outside the record
    data/rejects/<name>/<id>.json                 every episode not kept, with its reason and provenance

The record itself holds AutoFly's fields only (spec §2): image, instruction, action[3], state[9]. An episode is written
into a temporary directory and renamed into place, and the manifest is replaced atomically, so a crash never leaves a
half-written episode listed. A store that already exists, or a name with rejects left over, is never written into.
Nothing is created until the first episode or reject: a launch the GPU guard refuses leaves the name free to retry
(the M3 review, 2026-10-03). The manifest is then created exclusively, so two collectors cannot share a name.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from autofly_ue5.dataset.state import STATE_FIELDS
from autofly_ue5.scenes.model import SceneFile

FORMAT = "autofly_ue5_raw/1"


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


class RawDatasetWriter:
    def __init__(self, data_root: Path, name: str, *, scene: SceneFile, provenance: dict, split: str | None = None,
                 card: dict | None = None) -> None:
        self.root = Path(data_root) / name
        self.rejects = Path(data_root) / "rejects" / name
        self._refuse_existing()
        self.claimed = False
        self.provenance = dict(provenance)
        self.manifest = {
            "name": name,
            "format": FORMAT,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "record_fields": {"observation.image_0": "RGB 256x256 PNG", "observation.state": "float32[9]",
                              "action": "float32[3]: forward m/s, yaw rate rad/s, vertical m/s (up)",
                              "language_instruction": "string"},
            "state_fields": list(STATE_FIELDS),
            "scenes": {scene.id: {"split": split or scene.split, "sha256": scene.sha256,
                                  "instruction_obstacle": scene.instruction_obstacle}},
            "card": dict(card or {}),  # the dataset card's facts, e.g. which target names are placeholders
            "episodes": [],
            "counts": {"episodes": 0, "records": 0, "rejects": 0, "by_scene": {}, "by_target": {}},
        }

    def _refuse_existing(self) -> None:
        if (self.root / "manifest.json").exists():
            raise FileExistsError(f"{self.root} already holds a dataset; collect into a new --name")
        if self.rejects.is_dir() and any(self.rejects.iterdir()):
            raise FileExistsError(f"{self.rejects} holds another collection's rejects; collect into a new --name")

    def _claim(self) -> None:
        if self.claimed:
            return
        self._refuse_existing()
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "manifest.json", "x") as handle:  # exclusive: the name is ours or this raises
            handle.write(json.dumps(self.manifest, indent=2) + "\n")
        for d in (self.root / "episodes", self.root / "provenance", self.rejects):
            d.mkdir(parents=True, exist_ok=True)
        self.claimed = True

    def _save_manifest(self) -> None:
        _write_json_atomic(self.root / "manifest.json", self.manifest)

    def write_episode(self, *, episode_id: str, scene_id: str, split: str, seed: int, instruction: str, target_name: str,
                      frames: list[np.ndarray], states: np.ndarray, actions: np.ndarray, sim_time_ns: np.ndarray,
                      provenance: dict) -> None:
        n = len(frames)
        if not (n == len(states) == len(actions) == len(sim_time_ns)) or n == 0:
            raise ValueError(f"episode {episode_id}: {n} frames, {len(states)} states, {len(actions)} actions, "
                             f"{len(sim_time_ns)} times")
        self._claim()
        final = self.root / "episodes" / episode_id
        if final.exists() or any(e["id"] == episode_id for e in self.manifest["episodes"]):
            raise FileExistsError(f"episode {episode_id} is already in {self.root}")
        tmp = self.root / "episodes" / f".{episode_id}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "frames").mkdir(parents=True)
        for i, rgb in enumerate(frames):
            if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"episode {episode_id} frame {i}: {rgb.shape} {rgb.dtype}, not 256x256x3 uint8 RGB")
            if not cv2.imwrite(str(tmp / "frames" / f"{i:06d}.png"), np.ascontiguousarray(rgb[:, :, ::-1])):
                raise OSError(f"could not write frame {i} of {episode_id}")
        np.savez(tmp / "steps.npz", state=np.asarray(states, dtype=np.float32), action=np.asarray(actions, dtype=np.float32),
                 sim_time_ns=np.asarray(sim_time_ns, dtype=np.int64))
        _write_json_atomic(self.root / "provenance" / f"{episode_id}.json", {**self.provenance, **provenance})
        os.replace(tmp, final)
        self.manifest["episodes"].append({"id": episode_id, "scene": scene_id, "split": split, "seed": seed,
                                          "steps": n, "instruction": instruction, "target_name": target_name,
                                          "path": f"episodes/{episode_id}"})
        counts = self.manifest["counts"]
        counts["episodes"] += 1
        counts["records"] += n
        counts["by_scene"][scene_id] = counts["by_scene"].get(scene_id, 0) + 1
        counts["by_target"][target_name] = counts["by_target"].get(target_name, 0) + 1
        self._save_manifest()

    def write_reject(self, *, episode_id: str, reason: str, provenance: dict) -> None:
        self._claim()
        _write_json_atomic(self.rejects / f"{episode_id}.json",
                           {"id": episode_id, "reason": reason, "provenance": {**self.provenance, **provenance}})
        self.manifest["counts"]["rejects"] += 1
        self._save_manifest()
