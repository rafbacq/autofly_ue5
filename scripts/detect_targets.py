"""Score every frame of a raw dataset store for its episode's target with Grounding DINO: stage one of the rebalancing
(spec §10.3; AutoFly App. A.2.4). `scripts/rebalance_dataset.py` turns the scores into phases and resampling weights.

    HF_HOME=runs/tools/hf_home runs/tools/gdino_venv/bin/python scripts/detect_targets.py --raw data/<name> \\
        [--out data/<name>/detections.json] [--episodes N] [--device cpu] [--threads 8] [--batch 4]

Runs in its own venv, like the TFDS read-back: the project venv has no `transformers`. Self-contained on purpose: it
imports nothing from autofly_ue5. CPU by default, because a detector on the shared GPU counts as a foreign job for the
simulator launch guard (`autofly_ue5.gpu`); on this host's CPU it scores about 0.37 frames/s with 8 threads
(grounding-dino-tiny, 2026-10-06), so the 100-episode pilot is a 13-hour job there and a quarter-hour on the GPU.

The score of a frame is the highest box confidence the model gives the target phrase: max over its 900 queries of the
max over text tokens of sigmoid(logit), which is what Grounding DINO's own inference utility thresholds (`box_threshold`)
and what the processor's `post_process_grounded_object_detection` returns as `scores` (checked equal on pilot frames).
The box of that query is kept too, normalised (cx, cy, w, h), so a later check can ask whether a detection sat where the
target really was.

The output is written after every episode and resumed: a store of 17,738 frames is hours of CPU, and this host freezes
(MEMORY.md). A file made for another store or by another detector is refused, never continued.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

FORMAT = "autofly_ue5_detections/1"
DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
PREVIEW_THRESHOLD = 0.7  # printed per episode as a progress aid only; the rebalancing stage chooses the threshold
SCORE_DOC = ("per record: the highest box confidence Grounding DINO gives the episode's target phrase (max over queries "
             "of max over text tokens of sigmoid(logit)); box: that query's box, normalised (cx, cy, w, h)")


def prompt_for(target_name: str) -> str:
    """Grounding DINO's text convention: lower case, one phrase, closed by a period. A period inside the name would split
    it into two phrases and the score would answer for either of them."""
    name = target_name.strip().lower()
    if not name:
        raise ValueError("a target name must not be empty")
    if "." in name:
        raise ValueError(f"target name {target_name!r} contains a period, which Grounding DINO reads as a phrase break")
    return name + "."


class GroundingDino:
    """The real detector. Imports torch and transformers only when constructed, so the offline tests (which use a fake)
    and the project venv never need them."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu", threads: int = 8) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        import transformers

        torch.set_num_threads(threads)
        local = Path(snapshot_download(model_id))  # the snapshot directory is named after the revision it holds
        self._processor = AutoProcessor.from_pretrained(str(local))
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(str(local)).to(device).eval()
        self._torch = torch
        self._device = device
        size = getattr(self._processor.image_processor, "size", None)
        self._identity = {"model": model_id, "revision": local.name, "transformers": transformers.__version__,
                          "image_size": {k: int(v) for k, v in dict(size or {}).items() if v is not None}}
        self._run_info = {"device": device, "threads": threads, "torch": torch.__version__}

    def identity(self) -> dict:
        """What makes two scorings comparable: a file is only ever continued by the same identity."""
        return dict(self._identity)

    def run_info(self) -> dict:
        return dict(self._run_info)

    def detect(self, images: list[Image.Image], prompt: str) -> list[tuple[float, list[float]]]:
        inputs = self._processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            out = self._model(**inputs)
        per_query = out.logits.sigmoid().max(dim=-1).values  # (batch, queries): each query's confidence for the phrase
        best = per_query.argmax(dim=-1)
        scores = per_query.gather(1, best[:, None])[:, 0]
        boxes = out.pred_boxes[self._torch.arange(len(images)), best]
        return [(float(s), [float(v) for v in b]) for s, b in zip(scores.cpu(), boxes.cpu())]


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        handle.write(json.dumps(data) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _continue_or_start(out: Path, header: dict) -> dict:
    """The document to continue: the existing file when it is this store's and this detector's, else a fresh one."""
    if not out.is_file():
        return {**header, "runs": [], "episodes": {}}
    existing = json.loads(out.read_text())
    if existing.get("format") != FORMAT or existing.get("dataset") != header["dataset"] \
            or existing.get("dataset_created") != header["dataset_created"]:
        raise FileExistsError(f"{out} holds detections for another store ({existing.get('dataset')!r} created "
                              f"{existing.get('dataset_created')!r}); pass another --out")
    if existing.get("detector") != header["detector"]:
        raise FileExistsError(f"{out} was scored by another detector ({existing.get('detector')}); this one is "
                              f"{header['detector']}: pass another --out")
    return existing


def score_frames(detector, frames: list[Path], prompt: str, batch: int) -> tuple[list[float], list[list[float]]]:
    scores: list[float] = []
    boxes: list[list[float]] = []
    for start in range(0, len(frames), batch):
        images = [Image.open(path).convert("RGB") for path in frames[start:start + batch]]
        for score, box in detector.detect(images, prompt):
            scores.append(round(score, 6))
            boxes.append([round(v, 6) for v in box])
    return scores, boxes


def run(raw: Path, out: Path, detector, *, episodes: int | None = None, batch: int = 4, log=print) -> dict:
    """Score the store's episodes (the manifest's first `episodes`, or all) into `out`, continuing a file this detector
    started on this store. Returns the document as written."""
    raw, out = Path(raw), Path(out)
    manifest = json.loads((raw / "manifest.json").read_text())
    header = {"format": FORMAT, "dataset": manifest["name"], "dataset_created": manifest["created"], "score": SCORE_DOC,
              "detector": detector.identity(), "episodes_total": len(manifest["episodes"])}
    doc = _continue_or_start(out, header)
    out.parent.mkdir(parents=True, exist_ok=True)
    wanted = manifest["episodes"][:episodes] if episodes is not None else manifest["episodes"]
    run_record = {**detector.run_info(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "episodes_scored": 0, "frames_scored": 0,
                  "seconds": 0.0}
    doc["runs"].append(run_record)
    for entry in wanted:
        if entry["id"] in doc["episodes"]:
            if len(doc["episodes"][entry["id"]]["scores"]) != entry["steps"]:
                raise ValueError(f"{out} holds {len(doc['episodes'][entry['id']]['scores'])} scores for episode "
                                 f"{entry['id']}, which has {entry['steps']} records")
            continue
        frames = sorted((raw / entry["path"] / "frames").glob("*.png"))
        if len(frames) != entry["steps"]:
            raise ValueError(f"episode {entry['id']}: {len(frames)} frames for {entry['steps']} records")
        prompt = prompt_for(entry["target_name"])
        t0 = time.time()
        scores, boxes = score_frames(detector, frames, prompt, batch)
        seconds = time.time() - t0
        doc["episodes"][entry["id"]] = {"query": entry["target_name"], "prompt": prompt, "scores": scores, "boxes": boxes}
        run_record["episodes_scored"] += 1
        run_record["frames_scored"] += len(frames)
        run_record["seconds"] = round(run_record["seconds"] + seconds, 1)
        doc["complete"] = len(doc["episodes"]) == len(manifest["episodes"])
        _write_json_atomic(out, doc)
        first = next((i for i, s in enumerate(scores) if s > PREVIEW_THRESHOLD), None)
        log(f"DETECT {entry['id']}: {len(frames)} frames in {seconds:.0f} s ({len(frames) / max(seconds, 1e-9):.2f}/s), "
            f"max score {max(scores):.3f}, first > {PREVIEW_THRESHOLD} at record {first} -- "
            f"{len(doc['episodes'])}/{len(manifest['episodes'])} episodes scored", flush=True)
    doc["complete"] = len(doc["episodes"]) == len(manifest["episodes"])
    run_record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json_atomic(out, doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True, help="the raw store, data/<name>")
    p.add_argument("--out", type=Path, default=None, help="default: <raw>/detections.json; continued if it is this run's")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu", help="'cpu' (default; hides CUDA from torch) or a torch device")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--episodes", type=int, default=None, help="score only the manifest's first N episodes")
    args = p.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported: no CUDA context, nothing on the shared GPU
    out = args.out if args.out is not None else args.raw / "detections.json"
    detector = GroundingDino(args.model, device=args.device, threads=args.threads)
    print(f"detector {detector.identity()} {detector.run_info()}", flush=True)
    doc = run(args.raw, out, detector, episodes=args.episodes, batch=args.batch)
    print(f"{'complete' if doc['complete'] else 'partial'}: {len(doc['episodes'])}/{doc['episodes_total']} episodes in {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
