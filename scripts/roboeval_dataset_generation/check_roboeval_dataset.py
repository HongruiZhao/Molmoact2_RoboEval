"""Sanity check that the converted RoboEval LeRobot dataset loads and looks right.

Checks metadata (fps, features, episode / frame counts, tasks), per-frame values (shapes,
finite values, image range), decodes frames from several episodes into a PNG for visual review,
and loads batches with MolmoAct2-style action chunks through a DataLoader.

Usage:
    python scripts/roboeval_dataset_generation/check_roboeval_dataset.py --root /home/hongrui/nas/dataset/roboeval_lerobot
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

CAMERAS = ("head", "left_wrist", "right_wrist")
IMAGE_KEYS = [f"observation.images.{c}" for c in CAMERAS]
FPS = 20
CHUNK = 10  # MolmoAct2 action chunk used for LIBERO fine-tuning


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        raise SystemExit(f"Check failed: {msg}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/home/hongrui/nas/dataset/roboeval_lerobot"))
    p.add_argument("--repo-id", default="roboeval_lerobot")
    p.add_argument("--expected-episodes", type=int, default=None, help="Defaults to the sidecar row count")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/roboeval_lerobot/check")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(0)

    print("== Metadata")
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    sources = [json.loads(l) for l in (args.root / "meta/roboeval_source.jsonl").read_text().splitlines()]
    expected = args.expected_episodes or len(sources)
    print(f"  fps={meta.fps} episodes={meta.total_episodes} frames={meta.total_frames} robot={meta.robot_type}")
    check(meta.fps == FPS, f"fps == {FPS}")
    check(meta.total_episodes == expected, f"total_episodes == {expected}")
    check(meta.total_frames == sum(s["length"] for s in sources), "total_frames == sum of rendered episode lengths")
    check(sorted(meta.video_keys) == sorted(IMAGE_KEYS), f"video keys == {IMAGE_KEYS}")
    for key in ("observation.state", "action"):
        check(tuple(meta.features[key]["shape"]) == (16,), f"{key} is 16-D")
    for key in IMAGE_KEYS:
        check(tuple(meta.features[key]["shape"]) == (256, 256, 3), f"{key} is 256x256x3")

    tasks = list(meta.tasks.index)
    print(f"  {len(tasks)} tasks:")
    per_task = Counter(s["task"] for s in sources)
    for t in tasks:
        print(f"    {per_task[t]:5d} episodes  {t}")
    check(set(per_task) == set(tasks), "sidecar tasks match meta/tasks")

    print("== Frames")
    dataset = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
    check(len(dataset) == meta.total_frames, "len(dataset) == total_frames")
    ep_index = dataset.hf_dataset["episode_index"]
    starts = {}
    for i, e in enumerate(ep_index):
        e = int(e)
        starts.setdefault(e, [i, i])[1] = i
    check(len(starts) == expected, "every episode has frames")

    # First, middle and last frame of the first, a middle and the last episode of every task.
    by_task: dict[str, list[int]] = {}
    for s in sources:
        by_task.setdefault(s["task"], []).append(s["episode_index"])
    rows = []
    for task, episodes in sorted(by_task.items()):
        for e in (episodes[0], episodes[len(episodes) // 2], episodes[-1]):
            first, last = starts[e]
            for idx in (first, (first + last) // 2, last):
                item = dataset[idx]
                check_item(item, task, sources[e])
            item = dataset[(first + last) // 2]
        rows.append(np.concatenate([to_uint8(item[k]) for k in IMAGE_KEYS], axis=1))
    grid = args.out / "frames_per_task.png"
    Image.fromarray(np.concatenate(rows, axis=0)).save(grid)
    print(f"  saved {grid} (one row per task: head | left_wrist | right_wrist, mid-episode)")

    print("== Action chunks + DataLoader")
    delta = {"action": [t / FPS for t in range(CHUNK)]}
    chunked = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav", delta_timestamps=delta)
    loader = torch.utils.data.DataLoader(chunked, batch_size=8, shuffle=True, num_workers=2)
    batch = next(iter(loader))
    for k in [*IMAGE_KEYS, "observation.state", "action", "action_is_pad"]:
        print(f"  {k:32s} {tuple(batch[k].shape)} {batch[k].dtype}")
    check(tuple(batch["action"].shape) == (8, CHUNK, 16), f"action chunk shape == (8, {CHUNK}, 16)")
    check(tuple(batch[IMAGE_KEYS[0]].shape) == (8, 3, 256, 256), "image batch shape == (8, 3, 256, 256)")
    check(len(batch["task"]) == 8 and all(isinstance(t, str) for t in batch["task"]), "batch carries task strings")
    print("\nAll checks passed.")


def check_item(item: dict, task: str, source: dict) -> None:
    for k in IMAGE_KEYS:
        img = item[k]
        assert img.shape == (3, 256, 256), (k, img.shape)
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0, (k, img.min(), img.max())
    for k in ("observation.state", "action"):
        v = item[k]
        assert v.shape == (16,) and torch.isfinite(v).all(), (k, v)
    assert item["task"] == task == source["task"], (item["task"], task)


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)


if __name__ == "__main__":
    main()
