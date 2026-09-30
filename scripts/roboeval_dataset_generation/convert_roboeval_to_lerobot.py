"""Write rendered RoboEval episodes as a LeRobotDataset v3.0 (stage 2 of the conversion).

Stage 1 (`RoboEval/data_cleaning/render_demos.py`, `roboeval` env) replays the successful
RoboEval demos and writes one .npz per episode (head / left_wrist / right_wrist RGB at 256x256,
16-D joint state and absolute joint-position actions at 20 fps). This script (`molmoact2` env)
splits those episodes into contiguous shards, writes each shard as its own LeRobot dataset in
parallel, then aggregates the shards in order into the final dataset.

Usage:
    python scripts/roboeval_dataset_generation/convert_roboeval_to_lerobot.py \
        --staging ~/.cache/roboeval_lerobot_staging \
        --output /home/hongrui/nas/dataset/roboeval_lerobot
"""

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ID = "roboeval_lerobot"
FPS = 20
CAMERAS = ("head", "left_wrist", "right_wrist")
JOINT_NAMES = (
    [f"left_joint{i}" for i in range(1, 8)]
    + [f"right_joint{i}" for i in range(1, 8)]
    + ["left_gripper", "right_gripper"]
)

# Each variant is its own LeRobot task, "<family instruction>; <randomization>", so variants can be
# trained separately while `task.split("; ")[0]` still gives the instruction shared by the family.
INSTRUCTIONS = {
    "LiftPot": "lift the pot off the stove with both hands",
    "LiftTray": "lift the tray with both hands",
    "StackTwoBlocks": "stack one block on top of the other block",
    "VerticalCubeHandover": "pick up the upright cube with the left gripper and hand it over to the right gripper",
    "CubeHandover": "pick up the cube with the left gripper and hand it over to the right gripper",
    "PackBox": "close the lid of the box",
    "PickSingleBookFromTable": "pick up the book from the table",
    "StackSingleBookShelf": "move the book from the counter onto the shelf",
    "RotateValve": "rotate both valves counterclockwise",
}


RANDOMIZATION = {
    "": "no randomization",
    "Position": "position randomization",
    "Orientation": "orientation randomization",
    "PositionAndOrientation": "position and orientation randomization",
    "Obstacle": "obstacle, position and orientation randomization",
}
# Variants whose name has no suffix but which still randomize on reset.
RANDOMIZATION_OVERRIDES = {"VerticalCubeHandover": "position and orientation randomization"}
SEPARATOR = "; "


def instruction_for(variant: str) -> str:
    # Longest family prefix wins, so "VerticalCubeHandover" is not matched as "CubeHandover".
    family = max((f for f in INSTRUCTIONS if variant.startswith(f)), key=len, default=None)
    suffix = variant[len(family) :] if family else None
    if family is None or suffix not in RANDOMIZATION:
        raise KeyError(f"No instruction for task variant {variant!r}")
    return INSTRUCTIONS[family] + SEPARATOR + RANDOMIZATION_OVERRIDES.get(variant, RANDOMIZATION[suffix])


def features(resolution: int) -> dict:
    feats = {
        f"observation.images.{c}": {
            "dtype": "video",
            "shape": (resolution, resolution, 3),
            "names": ["height", "width", "channels"],
        }
        for c in CAMERAS
    }
    for key in ("observation.state", "action"):
        feats[key] = {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": list(JOINT_NAMES)}
    return feats


def shard_worker(shard_idx: int, files: list[str], shard_root: str, resolution: int, progress, results) -> None:
    # Runs in a non-daemonic process: save_episode() encodes the cameras in its own process pool,
    # which daemonic multiprocessing.Pool workers are not allowed to start.
    try:
        results.put((shard_idx, write_shard(shard_idx, files, shard_root, resolution, progress), None))
    except BaseException as e:
        import traceback

        results.put((shard_idx, None, traceback.format_exc()))
        raise SystemExit(1) from e


def write_shard(shard_idx: int, files: list[str], shard_root: str, resolution: int, progress) -> list[dict]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"{REPO_ID}_shard{shard_idx}",
        root=shard_root,
        fps=FPS,
        features=features(resolution),
        robot_type="bimanual_panda",
        use_videos=True,
        image_writer_threads=4,
    )
    sources = []
    for f in files:
        with np.load(f) as z:
            variant = str(z["variant"])
            task = instruction_for(variant)
            images = {c: z[f"image_{c}"] for c in CAMERAS}
            state, action = z["state"], z["action"]
            for t in range(len(state)):
                frame = {f"observation.images.{c}": images[c][t] for c in CAMERAS}
                frame["observation.state"] = state[t]
                frame["action"] = action[t]
                frame["task"] = task
                dataset.add_frame(frame)
            dataset.save_episode()
            sources.append(
                {
                    "variant": variant,
                    "uuid": str(z["uuid"]),
                    "seed": int(z["seed"]),
                    "source": str(z["source"]),
                    "task": task,
                    "length": int(len(state)),
                }
            )
        progress.put(1)
    dataset.finalize()
    return sources


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--staging", type=Path, default=Path.home() / ".cache/roboeval_lerobot_staging")
    p.add_argument("--output", type=Path, default=Path("/home/hongrui/nas/dataset/roboeval_lerobot"))
    p.add_argument("--shards-dir", type=Path, default=Path.home() / ".cache/roboeval_lerobot_shards")
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256)
    args = p.parse_args()

    files = sorted(str(f) for f in args.staging.glob("*.npz") if not f.name.endswith(".tmp.npz"))
    if not files:
        raise SystemExit(f"No rendered episodes in {args.staging}")
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; remove it first")
    if args.shards_dir.exists():
        shutil.rmtree(args.shards_dir)
    args.shards_dir.mkdir(parents=True)

    # Contiguous blocks keep the aggregated episode order equal to the sorted file order.
    n = min(args.shards, len(files))
    bounds = np.linspace(0, len(files), n + 1).astype(int)
    blocks = [files[bounds[i] : bounds[i + 1]] for i in range(n)]
    shard_roots = [args.shards_dir / f"shard_{i}" for i in range(n)]
    print(f"{len(files)} episodes -> {n} shards -> {args.output}", flush=True)

    ctx = mp.get_context("spawn")
    progress, results = ctx.Queue(), ctx.Queue()
    workers = [
        ctx.Process(
            target=shard_worker, args=(i, blocks[i], str(shard_roots[i]), args.resolution, progress, results)
        )
        for i in range(n)
    ]
    for w in workers:
        w.start()

    shard_sources: dict[int, list[dict]] = {}
    with tqdm(total=len(files), desc="Writing episodes", unit="ep", dynamic_ncols=True) as bar:
        while len(shard_sources) < n:
            while not progress.empty():
                progress.get()
                bar.update()
            while not results.empty():
                idx, sources, error = results.get()
                if error:
                    for w in workers:
                        w.terminate()
                    raise SystemExit(f"Shard {idx} failed:\n{error}")
                shard_sources[idx] = sources
            dead = [i for i, w in enumerate(workers) if w.exitcode not in (None, 0) and i not in shard_sources]
            if dead and results.empty():
                raise SystemExit(f"Shard(s) {dead} exited without reporting a result")
            time.sleep(0.5)
        while not progress.empty():
            progress.get()
            bar.update()
    for w in workers:
        w.join()
    sources = [row for i in range(n) for row in shard_sources[i]]

    from lerobot.datasets.aggregate import aggregate_datasets

    print("Aggregating shards ...", flush=True)
    aggregate_datasets(
        repo_ids=[f"{REPO_ID}_shard{i}" for i in range(n)],
        aggr_repo_id=REPO_ID,
        roots=shard_roots,
        aggr_root=args.output,
    )

    with (args.output / "meta" / "roboeval_source.jsonl").open("w") as f:
        for episode_index, row in enumerate(sources):
            f.write(json.dumps({"episode_index": episode_index, **row}) + "\n")
    print(f"Done: {len(sources)} episodes, {sum(r['length'] for r in sources)} frames at {args.output}", flush=True)


if __name__ == "__main__":
    main()
