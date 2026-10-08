"""Load a merged AutoFly export the way OpenVLA's RLDS pipeline does, on the CPU, and check every frame that comes out
(plan 6, revision 1, item 1: done means the merged dataset loads through an OpenVLA-style RLDS pipeline).

    runs/tools/vla_loader_venv/bin/python scripts/check_vla_loader.py --data-dir <out> --name autofly_ue5_proto \\
        [--frames 300] [--out <json>]

Runs in its own venv with OpenVLA's pins (openvla/openvla@c8f03f4 pyproject.toml): tensorflow 2.15.0 (as tensorflow-cpu),
tensorflow_datasets 4.9.3 and dlimp from moojink/dlimp_openvla. Self-contained: it imports nothing from autofly_ue5, nor
OpenVLA's `prismatic` (which needs torch and transformers); the transforms below are OpenVLA's, copied from
openvla/openvla@c8f03f4 (prismatic/vla/datasets/rlds/{dataset.py, utils/data_utils.py, traj_transforms.py,
obs_transforms.py}), and the frames pass through them unchanged. Per split (train, val):

1. `tfds.builder(name, data_dir=...)` and `dlimp.DLataset.from_rlds(builder, split=...)` (make_dataset_from_rlds);
2. restructure: image_obs_keys {"primary": "image_0"}, state_obs_keys ["state"] (the 9-dim state as proprio),
   language_key "language_instruction", the 3-dim action (no standardize_fn is needed: the steps are already AutoFly's
   observation / action / language_instruction);
3. normalize_action_and_proprio with BOUNDS_Q99 (what OpenVLA trains with) from dataset_statistics.json[name];
4. add_pad_mask_dict and chunk_act_obs(window_size=1) (apply_trajectory_transforms with OpenVLA's settings);
5. flatten, then decode_and_resize to 224 x 224 (apply_frame_transforms; OpenVLA's resize_resolution).

Every frame read (--frames per split) is checked: image uint8 (1, 224, 224, 3) and not padding; action float32 (1, 3)
and proprio float32 (1, 9), both within [-1, 1] and equal to the raw store's record normalised in numpy; the
instruction is the episode's; the episode belongs to the split merge.json gives it. Also: "val" is a split of its own
(otherwise OpenVLA would carve train[95%:]), the split counts are merge.json's, and OpenVLA's get_dataset_statistics
recomputed over the train split matches dataset_statistics.json (quantiles, min, max and counts to float32 precision;
mean and std are reported as drift, since OpenVLA accumulates them in float32). Frames carry a `_check` passenger (the
file path, the raw action, state and image bytes) that no OpenVLA transform touches. Writes one JSON result; exit code 0
only if everything held.

The venv (2026-10-08; tensorflow-metadata is not pinned by OpenVLA, and its newest needs protobuf >= 5.26, which TF 2.15
refuses; dlimp_openvla's setup.py asks for "tensorflow==2.15.0", which tensorflow-cpu provides as the same module):

    uv venv runs/tools/vla_loader_venv --python 3.10
    uv pip install --python runs/tools/vla_loader_venv/bin/python "tensorflow-cpu==2.15.0" \\
        "tensorflow-datasets==4.9.3" "tensorflow-metadata==1.15.0" pillow
    uv pip install --python runs/tools/vla_loader_venv/bin/python --no-deps \\
        "dlimp @ git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np

RESIZE = (224, 224)  # OpenVLA's resize_resolution for its 224 px backbones
IMAGE_OBS_KEYS = {"primary": "image_0"}  # RLDSDataset loads load_camera_views=("primary",)
STATE_OBS_KEYS = ["state"]
LANGUAGE_KEY = "language_instruction"
OPENVLA_SOURCE = "openvla/openvla@c8f03f48af692657d3060c19588038c7220e9af9"


class NormalizationType(str, Enum):  # prismatic/vla/datasets/rlds/utils/data_utils.py
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"


def tree_map(fn, tree):  # data_utils.py
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def normalize_action_and_proprio(traj, metadata, normalization_type, *, tf, dl):  # data_utils.py, verbatim
    keys_to_normalize = {"action": "action", "proprio": "observation/proprio"}
    if normalization_type == NormalizationType.NORMAL:
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == traj_key,  # noqa: B023
                map_fn=lambda x: tf.where(mask, (x - metadata[key]["mean"])  # noqa: B023
                                          / (metadata[key]["std"] + 1e-8), x))  # noqa: B023
        return traj
    elif normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:
        for key, traj_key in keys_to_normalize.items():
            if normalization_type == NormalizationType.BOUNDS:
                low, high = metadata[key]["min"], metadata[key]["max"]
            else:
                low, high = metadata[key]["q01"], metadata[key]["q99"]
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == traj_key,  # noqa: B023
                map_fn=lambda x: tf.where(mask, tf.clip_by_value(  # noqa: B023
                    2 * (x - low) / (high - low + 1e-8) - 1, -1, 1), x))  # noqa: B023
            zeros_mask = metadata[key]["min"] == metadata[key]["max"]
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == traj_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x))  # noqa: B023
        return traj
    raise ValueError(f"Unknown Normalization Type {normalization_type}")


def make_restructure(name, *, tf, image_obs_keys, state_obs_keys, language_key, keep_check: bool):
    """make_dataset_from_rlds's `restructure` (dataset.py), standardize_fn None, no depth, no absolute_action_mask; with
    `keep_check` the frame also carries `_check`, which no later transform reads or changes."""
    required = {"observation", "action", language_key}

    def restructure(traj):
        if not all(k in traj for k in required):
            raise ValueError(f"Trajectory is missing keys: {required - set(traj.keys())}.")
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            new_obs[f"image_{new}"] = tf.repeat("", traj_len) if old is None else old_obs[old]
        if state_obs_keys:
            new_obs["proprio"] = tf.concat([tf.zeros((traj_len, 1), dtype=tf.float32) if key is None
                                            else tf.cast(old_obs[key], tf.float32) for key in state_obs_keys], axis=1)
        new_obs["timestep"] = tf.range(traj_len)
        task = {}
        if traj[language_key].dtype != tf.string:
            raise ValueError(f"Language key {language_key} has dtype {traj[language_key].dtype}, but it must be tf.string.")
        check = None
        if keep_check:
            check = {"file_path": traj["traj_metadata"]["episode_metadata"]["file_path"],
                     "episode_id": traj["traj_metadata"]["episode_metadata"]["episode_id"],
                     "raw_action": traj["action"], "raw_state": tf.cast(old_obs["state"], tf.float32),
                     "image_bytes": old_obs["image_0"], "frame": traj["_frame_index"]}
        task["language_instruction"] = traj.pop(language_key)
        out = {"observation": new_obs, "task": task, "action": tf.cast(traj["action"], tf.float32),
               "dataset_name": tf.repeat(name, traj_len)}
        if check is not None:
            out["_check"] = check
        return out

    return restructure


def add_pad_mask_dict(traj, *, tf):  # traj_transforms.py, verbatim
    traj_len = tf.shape(traj["action"])[0]
    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)
        traj[key]["pad_mask_dict"] = pad_mask_dict
    return traj


def chunk_act_obs(traj, window_size, future_action_window_size=0, *, tf):  # traj_transforms.py, verbatim
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size])
    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size), [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, window_size + future_action_window_size])
    floored_chunk_indices = tf.maximum(chunk_indices, 0)
    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)
    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])
    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)
    traj["observation"]["pad_mask"] = chunk_indices >= 0
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(absolute_action_mask[:, None, :], traj["action"], tf.zeros_like(traj["action"]))
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])
    return traj


def decode_and_resize(obs, resize_size, *, tf, dl):  # obs_transforms.py (images only: this dataset has no depth)
    image_names = {key[6:] for key in obs if key.startswith("image_")}
    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    for name in image_names:
        image = obs[f"image_{name}"]
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
            else:
                image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
        elif image.dtype != tf.uint8:
            raise ValueError(f"Unsupported image dtype: found image_{name} with dtype {image.dtype}")
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f"image_{name}"] = image
    return obs


def openvla_frames(builder, name, split, stats, *, tf, dl, keep_check=True):
    """make_dataset_from_rlds + apply_trajectory_transforms + apply_frame_transforms for one split, unshuffled."""
    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=False, num_parallel_reads=1)
    dataset = dataset.traj_map(make_restructure(name, tf=tf, image_obs_keys=IMAGE_OBS_KEYS, state_obs_keys=STATE_OBS_KEYS,
                                                language_key=LANGUAGE_KEY, keep_check=keep_check), 1)
    dataset = dataset.traj_map(partial(normalize_action_and_proprio, metadata=stats,
                                       normalization_type=NormalizationType.BOUNDS_Q99, tf=tf, dl=dl), 1)
    dataset = dataset.traj_map(partial(add_pad_mask_dict, tf=tf), 1)
    dataset = dataset.traj_map(partial(chunk_act_obs, window_size=1, future_action_window_size=0, tf=tf), 1)
    dataset = dataset.flatten(num_parallel_calls=8)  # interleaves 8 trajectories (deterministically): wider coverage

    def apply_obs_transform(fn, frame):
        frame["task"] = fn(frame["task"])
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    return dataset.frame_map(partial(apply_obs_transform, partial(decode_and_resize, resize_size=RESIZE, tf=tf, dl=dl)), 4)


def _normalise(x, block):
    low, high = np.asarray(block["q01"]), np.asarray(block["q99"])
    out = np.clip(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1)
    return np.where(np.asarray(block["min"]) == np.asarray(block["max"]), 0.0, out)


def check_split(frames, split, expected, stats_np, raw_cache, n_frames):
    failures, episodes = [], set()
    lo = {"action": np.full(3, np.inf), "proprio": np.full(9, np.inf)}
    hi = {"action": np.full(3, -np.inf), "proprio": np.full(9, -np.inf)}
    count, instructions = 0, set()
    for frame in frames.take(n_frames).as_numpy_iterator():
        count += 1
        check = frame["_check"]
        key = check["file_path"].decode()
        entry = expected.get(key)
        where = f"{split} {key} frame {int(check['frame'])}"
        if entry is None:
            failures.append(f"{where}: not in merge.json")
            continue
        episodes.add(key)
        if entry["split"] != split:
            failures.append(f"{where}: merge.json puts it in {entry['split']}")
        image, action, proprio = frame["observation"]["image_primary"], frame["action"], frame["observation"]["proprio"]
        real_image = frame["observation"]["pad_mask_dict"]["image_primary"][0]  # not a padding image
        if image.dtype != np.uint8 or image.shape != (1, *RESIZE, 3) or not real_image:
            failures.append(f"{where}: image {image.dtype} {image.shape}")
        if action.dtype != np.float32 or action.shape != (1, 3) or proprio.dtype != np.float32 or proprio.shape != (1, 9):
            failures.append(f"{where}: action {action.dtype} {action.shape}, proprio {proprio.dtype} {proprio.shape}")
            continue
        if not (np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1) and np.all(np.abs(proprio) <= 1)):
            failures.append(f"{where}: normalised values outside [-1, 1]")
        t = int(frame["observation"]["timestep"][0])
        if t != int(check["frame"]):
            failures.append(f"{where}: timestep {t}")
        if key not in raw_cache:
            ep = Path(entry["store_root"]) / entry["path"]
            with np.load(ep / "steps.npz") as steps:
                raw_cache[key] = (steps["action"], steps["state"], sorted((ep / "frames").glob("*.png")))
        raw_action, raw_state, frame_files = raw_cache[key]
        if not (np.array_equal(check["raw_action"], raw_action[t]) and np.array_equal(check["raw_state"], raw_state[t])):
            failures.append(f"{where}: the pipeline's record is not the raw store's record {t}")
        if check["image_bytes"] != frame_files[t].read_bytes():
            failures.append(f"{where}: image bytes differ from {frame_files[t]}")
        if not np.allclose(action[0], _normalise(raw_action[t].astype(np.float64), stats_np["action"]), atol=1e-5):
            failures.append(f"{where}: action {action[0]} is not the raw action normalised")
        if not np.allclose(proprio[0], _normalise(raw_state[t].astype(np.float64), stats_np["proprio"]), atol=1e-5):
            failures.append(f"{where}: proprio is not the raw state normalised")
        text = frame["task"]["language_instruction"].decode()
        instructions.add(text)
        if text != entry["instruction"]:
            failures.append(f"{where}: instruction {text!r}")
        if frame["dataset_name"].decode() != stats_np["_name"]:
            failures.append(f"{where}: dataset_name {frame['dataset_name']!r}")
        for k in ("action", "proprio"):
            v = action[0] if k == "action" else proprio[0]
            lo[k], hi[k] = np.minimum(lo[k], v), np.maximum(hi[k], v)
    return {"frames": count, "episodes": len(episodes), "distinct_instructions": len(instructions),
            "image": {"dtype": "uint8", "shape": [1, *RESIZE, 3]},
            "action_normalised": {"dtype": "float32", "shape": [1, 3], "min": lo["action"].tolist(),
                                  "max": hi["action"].tolist()},
            "proprio_normalised": {"dtype": "float32", "shape": [1, 9], "min": lo["proprio"].tolist(),
                                   "max": hi["proprio"].tolist()},
            "failures": failures[:20], "failure_count": len(failures)}


def recompute_statistics(builder, name, split, *, tf, dl):
    """get_dataset_statistics (data_utils.py) over one split, as OpenVLA computes it (float32)."""
    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=False, num_parallel_reads=1)
    dataset = dataset.traj_map(make_restructure(name, tf=tf, image_obs_keys=IMAGE_OBS_KEYS, state_obs_keys=STATE_OBS_KEYS,
                                                language_key=LANGUAGE_KEY, keep_check=False), 1)
    dataset = dataset.traj_map(lambda traj: {"action": traj["action"], "proprio": traj["observation"]["proprio"]})
    actions, proprios, n_traj = [], [], 0
    for traj in dataset.iterator():
        actions.append(traj["action"])
        proprios.append(traj["proprio"])
        n_traj += 1
    actions, proprios = np.concatenate(actions), np.concatenate(proprios)

    def block(v):
        return {"mean": v.mean(0).tolist(), "std": v.std(0).tolist(), "max": v.max(0).tolist(), "min": v.min(0).tolist(),
                "q01": np.quantile(v, 0.01, axis=0).tolist(), "q99": np.quantile(v, 0.99, axis=0).tolist()}

    return {"action": block(actions), "proprio": block(proprios), "num_transitions": int(len(actions)),
            "num_trajectories": n_traj}


def compare_statistics(ours, theirs) -> dict:
    """Quantiles, min, max and the counts must agree (to float32 precision: OpenVLA's are float32). Mean and std are
    reported only: OpenVLA sums float32 along axis 0 without pairwise summation, so its own drift with the data's size
    (dataset_statistics.json accumulates in float64)."""
    out, ok = {"max_abs_diff": {}, "mean_std_drift": {}}, True
    for key in ("action", "proprio"):
        for stat in ("q01", "q99", "min", "max", "mean", "std"):
            a, b = np.asarray(ours[key][stat]), np.asarray(theirs[key][stat])
            diff = np.abs(a - b)
            out["max_abs_diff"][f"{key}.{stat}"] = float(diff.max())
            if stat in ("mean", "std"):
                out["mean_std_drift"][f"{key}.{stat}"] = float(np.max(diff / (1 + np.abs(a))))
            else:
                ok &= bool(np.all(diff <= 1e-6 * (1 + np.abs(a))))
    for key in ("num_transitions", "num_trajectories"):
        out[key] = [ours[key], theirs[key]]
        ok &= ours[key] == theirs[key]
    out["pass"] = bool(ok)
    return out


def _versions(tf, tfds) -> dict:
    from importlib import metadata

    dlimp = {"version": metadata.version("dlimp")}
    try:
        direct = json.loads(metadata.distribution("dlimp").read_text("direct_url.json") or "{}")
        dlimp.update(url=direct.get("url"), commit=(direct.get("vcs_info") or {}).get("commit_id"))
    except (OSError, ValueError):
        pass
    return {"python": platform.python_version(), "tensorflow": tf.__version__, "tensorflow_datasets": tfds.__version__,
            "numpy": np.__version__, "dlimp": dlimp,
            "tensorflow_distribution": next((d for d in ("tensorflow-cpu", "tensorflow") if _has(d)), None),
            "protobuf": metadata.version("protobuf"), "tensorflow_metadata": metadata.version("tensorflow-metadata")}


def _has(dist: str) -> bool:
    from importlib import metadata

    try:
        metadata.version(dist)
        return True
    except metadata.PackageNotFoundError:
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="the export's --out: <data-dir>/<name>/1.0.0/")
    p.add_argument("--name", required=True)
    p.add_argument("--statistics", type=Path, default=None, help="default <data-dir>/<name>/dataset_statistics.json")
    p.add_argument("--merged", type=Path, default=None, help="default <data-dir>/<name>/merge.json")
    p.add_argument("--frames", type=int, default=300, help="frames to read and check per split")
    p.add_argument("--out", type=Path, default=None, help="default <data-dir>/<name>/vla_loader_check.json")
    p.add_argument("--no-statistics-recompute", action="store_true")
    args = p.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    t0 = time.monotonic()
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")  # as OpenVLA's dataset.py: TF never takes the GPU
    root = args.data_dir / args.name
    stats_path = args.statistics or root / "dataset_statistics.json"
    merged_path = args.merged or root / "merge.json"
    out_path = args.out or root / "vla_loader_check.json"
    record = json.loads(merged_path.read_text())
    expected = {e["metadata"]["file_path"]: e for e in record["episodes"]}
    stats_file = json.loads(stats_path.read_text())
    result = {"dataset": args.name, "data_dir": str(args.data_dir), "versions": _versions(tf, tfds),
              "openvla_source": OPENVLA_SOURCE,
              "pipeline": {"image_obs_keys": IMAGE_OBS_KEYS, "state_obs_keys": STATE_OBS_KEYS, "language_key": LANGUAGE_KEY,
                           "normalization": NormalizationType.BOUNDS_Q99.value, "window_size": 1,
                           "future_action_window_size": 0, "resize": list(RESIZE)},
              "failures": []}
    schema_ok = (list(stats_file) == [args.name] and set(stats_file[args.name]) == {"action", "proprio", "num_transitions",
                                                                                     "num_trajectories"}
                 and all(set(stats_file[args.name][k]) == {"mean", "std", "max", "min", "q01", "q99", "mask"}
                         for k in ("action", "proprio"))
                 and len(stats_file[args.name]["action"]["q01"]) == 3 and len(stats_file[args.name]["proprio"]["q01"]) == 9
                 and all(m is True for k in ("action", "proprio") for m in stats_file[args.name][k]["mask"]))
    result["statistics_schema_ok"] = schema_ok
    if not schema_ok:
        result["failures"].append(f"{stats_path} does not hold OpenVLA's schema for {args.name}")
    stats = tree_map(np.array, stats_file[args.name])  # as make_dataset_from_rlds does with a loaded file
    stats_np = {**stats_file[args.name], "_name": args.name}

    builder = tfds.builder(args.name, data_dir=str(args.data_dir))  # OpenVLA's call
    splits = {k: v.num_examples for k, v in builder.info.splits.items()}
    result["builder"] = {"splits": splits, "expected_splits": {k: v["episodes"] for k, v in record["splits"].items()},
                         "val_is_a_split": "val" in builder.info.splits, "data_dir": builder.data_dir}
    if not result["builder"]["val_is_a_split"]:
        result["failures"].append("no val split: OpenVLA would carve train[95%:] instead")
    if splits != result["builder"]["expected_splits"]:
        result["failures"].append(f"split counts {splits}, merge.json {result['builder']['expected_splits']}")
    raw_cache: dict = {}
    result["splits"] = {}
    for split in ("train", "val"):
        frames = openvla_frames(builder, args.name, split, stats, tf=tf, dl=dl)
        report = check_split(frames, split, expected, stats_np, raw_cache, args.frames)
        result["splits"][split] = report
        if report["failure_count"] or report["frames"] == 0:
            result["failures"].append(f"{split}: {report['failure_count']} frame failures over {report['frames']} frames")
    if not args.no_statistics_recompute:
        theirs = recompute_statistics(builder, args.name, "train", tf=tf, dl=dl)
        result["statistics_recomputed"] = {"split": "train", **compare_statistics(stats_file[args.name], theirs)}
        if not result["statistics_recomputed"]["pass"]:
            result["failures"].append("OpenVLA's get_dataset_statistics over train differs from dataset_statistics.json")
    result["seconds"] = round(time.monotonic() - t0, 1)
    result["pass"] = not result["failures"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
