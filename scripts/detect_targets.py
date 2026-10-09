"""Score a raw dataset store's frames for its episode's target with Grounding DINO: stage one of the rebalancing (spec
§10.3; AutoFly App. A.2.4). `scripts/rebalance_dataset.py` turns the scores into phases and resampling weights.

    HF_HOME=runs/tools/hf_home runs/tools/gdino_venv/bin/python scripts/detect_targets.py --raw data/<name> \\
        [--out data/<name>/detections.jsonl] [--episodes N] [--threads 8] [--batch 4] [--device cuda --allow-gpu] \\
        [--sparse-stride 8 [--sparse-alert 0.5] [--sparse-threshold 0.7]]

Runs in its own venv, like the TFDS read-back: the project venv has no `transformers`. Of autofly_ue5 it imports only
the cache format (`autofly_ue5/dataset/detections.py`) and the sparse search (`autofly_ue5/dataset/sparse.py`), two
standard-library modules that the rebalancing stage reads and verifies with, so the two stages run one piece of code
(tests/test_sparse_search.py holds them to the standard library). CPU by default, because a torch process on the shared
GPU is a foreign job to the simulator launch guard (`autofly_ue5.gpu`): a live run's relaunch would be refused while it
holds more than 2 GB. On this host's CPU it scores 0.34-0.37 frames/s with 8 threads (grounding-dino-tiny,
2026-10-06), so the 100-episode pilot was a 13-14 hour job there; `--device cuda` needs `--allow-gpu` and a host with
nothing that may need to launch.

Two modes. Dense (the default) scores every record, as the pilot was scored. Sparse (`--sparse-stride S`) scores only
the records the sparse search visits on its way to the episode's first record above `--sparse-threshold`, which is all
the one-way phase split needs (the search, its one bet and its validation on the pilot: `dataset/sparse.py` and
`scripts/sparse_search_study.py`); the rebalancing stage then accepts only that threshold.

The score of a frame is the highest box confidence the model gives the target phrase: max over its 900 queries of the
max over text tokens of sigmoid(logit), which is what Grounding DINO's own inference utility thresholds (`box_threshold`)
and what the processor's `post_process_grounded_object_detection` returns as `scores` (checked equal on pilot frames).
The box of that query is kept too, normalised (cx, cy, w, h), so a later check can ask whether a detection sat where the
target really was. The phrase is the target name the episode's instruction uses (a pool entry's name, e.g. "blue
hatchback"), and an episode whose target name its instruction does not contain (a registry key, say) is refused.

The output is the appendable cache of `dataset/detections.py`: a header line, then one line per finished episode, each
synced as it is written, so a host freeze (MEMORY.md) costs at most the episode being scored, and a torn last line is
cut off on resume. A file made for another store, by another detector or by another search is refused, never continued,
and a lock file beside it keeps two scorers from racing over one file. The pilot's whole-document file
(`data/s01_pilot/detections.json`, FORMAT_V1) is read by the rebalancing stage and never continued here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

from PIL import Image  # noqa: E402

from autofly_ue5.dataset import detections as cache  # noqa: E402  (standard library only)
from autofly_ue5.dataset import sparse as sparse_search  # noqa: E402  (standard library only)

FORMAT = cache.FORMAT
DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"  # the snapshot scored on 2026-10-06; pinned so a resume
# never meets a moved Hub branch, and a cached model loads without the network
FRAME_SIZE = (256, 256)  # the raw store's frames (dataset/raw.py); a batch of one size keeps the normalised boxes honest
PREVIEW_THRESHOLD = 0.7  # printed per episode as a progress aid only (dense mode); the rebalancing stage chooses the threshold
DEFAULT_SPARSE_THRESHOLD = 0.7  # spec §10.3's threshold; a sparse file answers for this one only
DEFAULT_SPARSE_ALERT = 0.5  # docs/runbook-rebalance.md §4: at stride 8 it changed none of the pilot's transitions
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


def check_target_phrase(entry: dict) -> None:
    """The detector looks for what the episode's instruction asks for. The collector writes the drawn pool entry's
    `name` into both the instruction and the manifest's `target_name` (collect/collector.py; the validator holds the two
    equal), so a target name the instruction does not contain -- a registry key, a placeholder -- is a store this
    stage would score for the wrong words."""
    name, instruction = entry.get("target_name"), entry.get("instruction")
    if not isinstance(name, str) or not isinstance(instruction, str) \
            or not re.search(rf"(?<!\w){re.escape(name.strip())}(?!\w)", instruction):
        raise ValueError(f"episode {entry.get('id')}: its target name {name!r} is not a phrase of its instruction "
                         f"{instruction!r}, so the detector would not look for what the instruction names")


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


class _Lock:
    """`<out>.lock` holding this pid for the whole run: two scorers resuming one file would each score the same next
    episodes and both append them. A lock whose pid is gone (the host froze) is taken over."""

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


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _create(out: Path, header: dict) -> None:
    """The cache appears with its header already on disk: written and synced under a private name, linked into place
    (`os.link` fails on an existing name), then the directory synced. A freeze leaves no file or a whole header."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            handle.write(cache.encode(header))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_directory(out.parent)


def _append(handle, line: dict) -> None:
    handle.write(cache.encode(line))
    handle.flush()
    os.fsync(handle.fileno())


def _open_cache(out: Path, header: dict, manifest: dict, log) -> dict:
    """The episodes `out` already holds, once it is known to be this store's, this detector's and this search's (any
    other is refused, never continued). A torn last line -- the host stopped mid-append -- is cut off, so the next
    line starts on a clean one; its episode is scored again."""
    if not out.exists():
        _create(out, header)
        return {}
    blob = out.read_bytes()
    try:
        parsed = cache.parse(blob)
    except ValueError as err:
        raise ValueError(f"{out}: {err}") from err
    existing = parsed.doc
    if existing["format"] != FORMAT:
        raise FileExistsError(f"{out} is a {existing['format']} file, the pilot's whole-document form: the rebalancing "
                              f"stage reads it, but it is never continued; pass another --out")
    if existing.get("dataset") != header["dataset"] or existing.get("dataset_created") != header["dataset_created"]:
        raise FileExistsError(f"{out} holds detections for another store ({existing.get('dataset')!r} created "
                              f"{existing.get('dataset_created')!r}); pass another --out")
    if existing.get("detector") != header["detector"]:
        raise FileExistsError(f"{out} was scored by another detector ({existing.get('detector')}); this one is "
                              f"{header['detector']}: pass another --out")
    if existing.get("sparse") != header["sparse"]:
        raise FileExistsError(f"{out} was scored by another search ({existing.get('sparse') or 'dense'}); this run's is "
                              f"{header['sparse'] or 'dense'}: one file holds one kind, so pass another --out")
    steps = {entry["id"]: entry["steps"] for entry in manifest["episodes"]}
    for eid, ep in existing["episodes"].items():
        held = len(ep["scores"]) if header["sparse"] is None else ep.get("records")
        if eid in steps and held != steps[eid]:
            raise ValueError(f"{out} holds {held} records' scores for episode {eid}, which has {steps[eid]} records")
    if parsed.torn:
        with open(out, "r+b") as handle:
            handle.truncate(parsed.complete_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        log(f"DETECT {out}: cut off a torn last line of {len(blob) - parsed.complete_bytes} bytes (the host stopped "
            f"mid-append); its episode is scored again", flush=True)
    return existing["episodes"]


def _load_frames(paths: list[Path]) -> list[Image.Image]:
    images = [Image.open(path).convert("RGB") for path in paths]
    for path, image in zip(paths, images):
        if image.size != FRAME_SIZE:
            raise ValueError(f"{path.name} is {image.size[0]}x{image.size[1]}, not the store's {FRAME_SIZE[0]}x{FRAME_SIZE[1]}")
    return images


def _detect(detector, paths: list[Path], prompt: str) -> list[tuple[float, list[float]]]:
    images = _load_frames(paths)
    results = detector.detect(images, prompt)
    if len(results) != len(images):
        raise ValueError(f"the detector returned {len(results)} results for {len(images)} images")
    return [(round(score, 6), [round(v, 6) for v in box]) for score, box in results]


def score_frames(detector, frames: list[Path], prompt: str, batch: int) -> tuple[list[float], list[list[float]]]:
    """Dense mode: every record's score and box."""
    scores: list[float] = []
    boxes: list[list[float]] = []
    for start in range(0, len(frames), batch):
        for score, box in _detect(detector, frames[start:start + batch], prompt):
            scores.append(score)
            boxes.append(box)
    return scores, boxes


def search_frames(detector, frames: list[Path], prompt: str, search: dict, batch: int) -> dict:
    """Sparse mode: the records the search visits, their scores and boxes, and the first detection it found. The search
    decides on the rounded scores the file stores, so the rebalancing stage's replay of it sees what it saw."""
    boxes: dict[int, list[float]] = {}

    def score(indices: list[int]) -> list[float]:
        results = _detect(detector, [frames[i] for i in indices], prompt)
        for i, (_score, box) in zip(indices, results):
            boxes[i] = box
        return [s for s, _box in results]

    first, visited = sparse_search.search(len(frames), stride=search["stride"], threshold=search["threshold"],
                                   alert=search["alert"], score=score, batch=batch)
    order = sorted(visited)
    return {"visited": order, "scores": [visited[i] for i in order], "boxes": [boxes[i] for i in order],
            "first_detection": first}


def run(raw: Path, out: Path, detector, *, episodes: int | None = None, batch: int = 4, sparse: dict | None = None,
        log=print) -> dict:
    """Score the store's episodes (the manifest's first `episodes`, or all) into `out`, continuing a file this detector
    and search started on this store. `sparse`: None for dense, else the search's {stride, threshold, alert}. Returns
    the file as written, read back (`dataset/detections.parse`), with `episodes_total` and `complete` for the store's
    manifest as it is now."""
    raw, out = Path(raw), Path(out)
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if episodes is not None and episodes < 0:
        raise ValueError(f"episodes must be >= 0, got {episodes}")
    search = None if sparse is None else sparse_search.describe(**sparse)
    manifest = json.loads((raw / "manifest.json").read_text())
    wanted = manifest["episodes"][:episodes] if episodes is not None else manifest["episodes"]
    for entry in wanted:  # before anything is scored or written
        check_target_phrase(entry)
    header = {"kind": "header", "format": FORMAT, "dataset": manifest["name"], "dataset_created": manifest["created"],
              "score": SCORE_DOC, "detector": detector.identity(), "sparse": search}
    with _Lock(out):
        done = set(_open_cache(out, header, manifest, log))
        with open(out, "ab") as handle:
            _append(handle, {"kind": "run", **detector.run_info(), "batch": batch,
                             "started": time.strftime("%Y-%m-%d %H:%M:%S"), "episodes_total": len(manifest["episodes"])})
            for entry in wanted:
                if entry["id"] in done:
                    continue
                message = _score_episode(handle, raw, entry, detector, search, batch)
                done.add(entry["id"])
                log(f"{message} -- {len(done)}/{len(manifest['episodes'])} episodes scored", flush=True)
            _append(handle, {"kind": "run_end", "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    doc = cache.parse(out.read_bytes()).doc
    doc.update(episodes_total=len(manifest["episodes"]),
               complete=all(entry["id"] in doc["episodes"] for entry in manifest["episodes"]))
    return doc


def _score_episode(handle, raw: Path, entry: dict, detector, search: dict | None, batch: int) -> str:
    """Score one episode and append its line (synced); returns the progress line for the log."""
    frames = sorted((raw / entry["path"] / "frames").glob("*.png"))
    if len(frames) != entry["steps"]:
        raise ValueError(f"episode {entry['id']}: {len(frames)} frames for {entry['steps']} records")
    prompt = prompt_for(entry["target_name"])
    t0 = time.time()
    line = {"kind": "episode", "id": entry["id"], "query": entry["target_name"], "prompt": prompt, "records": len(frames)}
    if search is None:
        scores, boxes = score_frames(detector, frames, prompt, batch)
        line.update(scores=scores, boxes=boxes, frames_scored=len(frames))
        threshold = PREVIEW_THRESHOLD
        first = next((i for i, s in enumerate(scores) if s > threshold), None)
        summary = f"max score {max(scores):.3f}, "
    else:
        found = search_frames(detector, frames, prompt, search, batch)
        line.update(**found, frames_scored=len(found["visited"]))
        threshold, first, summary = search["threshold"], found["first_detection"], ""
    seconds = time.time() - t0
    line["seconds"] = round(seconds, 1)
    _append(handle, line)
    return (f"DETECT {entry['id']}: {line['frames_scored']}/{len(frames)} frames in {seconds:.0f} s "
            f"({line['frames_scored'] / max(seconds, 1e-9):.2f}/s), {summary}first > {threshold} at record {first}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True, help="the raw store, data/<name>")
    p.add_argument("--out", type=Path, default=None, help="default: <raw>/detections.jsonl; continued if it is this run's")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None, help=f"Hub revision; the default model is pinned to {DEFAULT_REVISION[:8]}")
    p.add_argument("--device", default="cpu", help="'cpu' (default; hides CUDA from torch) or a torch device, with --allow-gpu")
    p.add_argument("--allow-gpu", action="store_true",
                   help="required for any device but cpu: a torch process on the shared GPU makes the simulator launch "
                        "guard refuse every launch, so only when nothing may need one")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch", type=int, default=4, help="frames per detector call (sparse mode: per coarse look-ahead "
                                                         "and per window, at most batch - 1 wasted at each)")
    p.add_argument("--episodes", type=int, default=None, help="score only the manifest's first N episodes")
    p.add_argument("--sparse-stride", type=int, default=None,
                   help="sparse mode: score only what the search of autofly_ue5/dataset/sparse.py visits, with a coarse "
                        "probe every S records")
    p.add_argument("--sparse-alert", type=float, default=None,
                   help=f"sparse mode: scan the records between two probes when either scored above this "
                        f"(default {DEFAULT_SPARSE_ALERT})")
    p.add_argument("--sparse-threshold", type=float, default=None,
                   help=f"sparse mode: the confidence a first detection must exceed (default {DEFAULT_SPARSE_THRESHOLD}); "
                        f"the rebalancing stage accepts this threshold only")
    args = p.parse_args(argv)
    search = None
    if args.sparse_stride is None:
        if args.sparse_alert is not None or args.sparse_threshold is not None:
            p.error("--sparse-alert and --sparse-threshold belong to sparse mode: give --sparse-stride too")
    else:
        search = {"stride": args.sparse_stride,
                  "threshold": DEFAULT_SPARSE_THRESHOLD if args.sparse_threshold is None else args.sparse_threshold,
                  "alert": DEFAULT_SPARSE_ALERT if args.sparse_alert is None else args.sparse_alert}
        try:
            sparse_search.describe(**search)
        except ValueError as err:
            p.error(str(err))
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported: no CUDA context, nothing on the shared GPU
    elif not args.allow_gpu:
        p.error(f"--device {args.device} needs --allow-gpu (see its help)")
    out = args.out
    if out is None:
        out = args.raw / cache.FILE_NAME
        if (args.raw / cache.FILE_NAME_V1).is_file():
            print(f"refused: {args.raw / cache.FILE_NAME_V1} exists, so this store is scored already (the pilot's "
                  f"whole-document form); pass --out to score it again into another file", file=sys.stderr)
            return 2
    detector = GroundingDino(args.model, device=args.device, threads=args.threads, revision=args.revision)
    print(f"detector {detector.identity()} {detector.run_info()} search {search or 'dense'}", flush=True)
    doc = run(args.raw, out, detector, episodes=args.episodes, batch=args.batch, sparse=search)
    print(f"{'complete' if doc['complete'] else 'partial'}: {len(doc['episodes'])}/{doc['episodes_total']} episodes in {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
