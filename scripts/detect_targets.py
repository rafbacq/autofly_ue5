"""Score every frame of a raw dataset store for its episode's target with Grounding DINO: stage one of the rebalancing
(spec §10.3; AutoFly App. A.2.4). `scripts/rebalance_dataset.py` turns the scores into phases and resampling weights.

    HF_HOME=runs/tools/hf_home runs/tools/gdino_venv/bin/python scripts/detect_targets.py --raw data/<name> \\
        [--out data/<name>/detections.json] [--episodes N] [--threads 8] [--batch 4] [--device cuda --allow-gpu]

Runs in its own venv, like the TFDS read-back: the project venv has no `transformers`. Self-contained on purpose: it
imports nothing from autofly_ue5. CPU by default, because a torch process on the shared GPU is a foreign job to the
simulator launch guard (`autofly_ue5.gpu`): a live run's relaunch would be refused while it holds more than 2 GB. On
this host's CPU it scores 0.34-0.37 frames/s with 8 threads (grounding-dino-tiny, 2026-10-06), so the 100-episode
pilot is a 13-14 hour job there; `--device cuda` needs `--allow-gpu` and a host with nothing that may need to launch.

The score of a frame is the highest box confidence the model gives the target phrase: max over its 900 queries of the
max over text tokens of sigmoid(logit), which is what Grounding DINO's own inference utility thresholds (`box_threshold`)
and what the processor's `post_process_grounded_object_detection` returns as `scores` (checked equal on pilot frames).
The box of that query is kept too, normalised (cx, cy, w, h), so a later check can ask whether a detection sat where the
target really was.

The output is written after every episode and resumed: a store of 17,738 frames is hours of CPU, and this host freezes
(MEMORY.md). A file made for another store or by another detector is refused, never continued, and a lock file beside it
keeps two scorers from racing over one file. The whole document is rewritten per episode, which suits the pilot's 100
episodes; M5's ~13K episodes need an appendable cache (one line per episode) before the full collection is scored.
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
DEFAULT_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"  # the snapshot scored on 2026-10-06; pinned so a resume
# never meets a moved Hub branch, and a cached model loads without the network
FRAME_SIZE = (256, 256)  # the raw store's frames (dataset/raw.py); a batch of one size keeps the normalised boxes honest
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

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu", threads: int = 8,
                 revision: str | None = None) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        import transformers

        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")
        torch.set_num_threads(threads)
        if revision is None and model_id == DEFAULT_MODEL:
            revision = DEFAULT_REVISION
        local = Path(snapshot_download(model_id, revision=revision))  # the snapshot directory is named after its revision
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
        return reduce_outputs(out.logits, out.pred_boxes)


def reduce_outputs(logits, pred_boxes) -> list[tuple[float, list[float]]]:
    """Each image's score and box from the model's `logits` (batch, queries, text positions; -inf where the text is
    padding) and `pred_boxes` (batch, queries, 4): the query whose confidence for the phrase is highest, that
    confidence being the max over text positions of sigmoid(logit)."""
    per_query = logits.sigmoid().max(dim=-1).values  # (batch, queries)
    best = per_query.argmax(dim=-1)
    scores = per_query.gather(1, best[:, None])[:, 0]
    boxes = pred_boxes[list(range(len(best))), best]
    return [(float(s), [float(v) for v in b]) for s, b in zip(scores.cpu(), boxes.cpu())]


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
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


class _Lock:
    """`<out>.lock` holding this pid for the whole run: two scorers resuming one file would each score the same next
    episodes and the last writer would drop the other's. A lock whose pid is gone (the host froze) is taken over."""

    def __init__(self, out: Path) -> None:
        self.path = out.with_suffix(out.suffix + ".lock")

    def __enter__(self) -> "_Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
            except ValueError:
                pid = None
            if pid is not None and Path(f"/proc/{pid}").exists():
                raise FileExistsError(f"{self.path} is held by live process {pid}: another scorer is writing this file")
            self.path.unlink()
        with open(self.path, "x") as handle:
            handle.write(f"{os.getpid()}\n")
        return self

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)


def score_frames(detector, frames: list[Path], prompt: str, batch: int) -> tuple[list[float], list[list[float]]]:
    scores: list[float] = []
    boxes: list[list[float]] = []
    for start in range(0, len(frames), batch):
        paths = frames[start:start + batch]
        images = [Image.open(path).convert("RGB") for path in paths]
        for path, image in zip(paths, images):
            if image.size != FRAME_SIZE:
                raise ValueError(f"{path.name} is {image.size[0]}x{image.size[1]}, not the store's {FRAME_SIZE[0]}x{FRAME_SIZE[1]}")
        results = detector.detect(images, prompt)
        if len(results) != len(images):
            raise ValueError(f"the detector returned {len(results)} results for {len(images)} images")
        for score, box in results:
            scores.append(round(score, 6))
            boxes.append([round(v, 6) for v in box])
    return scores, boxes


def run(raw: Path, out: Path, detector, *, episodes: int | None = None, batch: int = 4, log=print) -> dict:
    """Score the store's episodes (the manifest's first `episodes`, or all) into `out`, continuing a file this detector
    started on this store. Returns the document as written."""
    raw, out = Path(raw), Path(out)
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if episodes is not None and episodes < 0:
        raise ValueError(f"episodes must be >= 0, got {episodes}")
    manifest = json.loads((raw / "manifest.json").read_text())
    header = {"format": FORMAT, "dataset": manifest["name"], "dataset_created": manifest["created"], "score": SCORE_DOC,
              "detector": detector.identity(), "episodes_total": len(manifest["episodes"])}
    with _Lock(out):
        doc = _continue_or_start(out, header)
        doc.update(score=header["score"], episodes_total=header["episodes_total"])  # the identity matched; the rest follows the store
        return _score_into(doc, manifest, raw, out, detector, episodes=episodes, batch=batch, log=log)


def _score_into(doc: dict, manifest: dict, raw: Path, out: Path, detector, *, episodes: int | None, batch: int, log) -> dict:
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
    p.add_argument("--revision", default=None, help=f"Hub revision; the default model is pinned to {DEFAULT_REVISION[:8]}")
    p.add_argument("--device", default="cpu", help="'cpu' (default; hides CUDA from torch) or a torch device, with --allow-gpu")
    p.add_argument("--allow-gpu", action="store_true",
                   help="required for any device but cpu: a torch process on the shared GPU makes the simulator launch "
                        "guard refuse every launch, so only when nothing may need one")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--episodes", type=int, default=None, help="score only the manifest's first N episodes")
    args = p.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported: no CUDA context, nothing on the shared GPU
    elif not args.allow_gpu:
        p.error(f"--device {args.device} needs --allow-gpu (see its help)")
    out = args.out if args.out is not None else args.raw / "detections.json"
    detector = GroundingDino(args.model, device=args.device, threads=args.threads, revision=args.revision)
    print(f"detector {detector.identity()} {detector.run_info()}", flush=True)
    doc = run(args.raw, out, detector, episodes=args.episodes, batch=args.batch)
    print(f"{'complete' if doc['complete'] else 'partial'}: {len(doc['episodes'])}/{doc['episodes_total']} episodes in {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
