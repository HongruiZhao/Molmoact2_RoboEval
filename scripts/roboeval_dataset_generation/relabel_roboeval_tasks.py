"""Re-apply the task instructions of `convert_roboeval_to_lerobot.py` to an existing dataset.

Rewrites only task metadata in place (meta/tasks.parquet, the task_index column, per-episode
task lists, meta/info.json and the `task` field of meta/roboeval_source.jsonl); videos and
state/action data are untouched. Use it after editing INSTRUCTIONS / RANDOMIZATION instead of
rebuilding the dataset.

Usage:
    python scripts/roboeval_dataset_generation/relabel_roboeval_tasks.py --root /home/hongrui/nas/dataset/roboeval_lerobot
"""

import argparse
import json
from pathlib import Path

from convert_roboeval_to_lerobot import REPO_ID, instruction_for
from lerobot.datasets.dataset_tools import modify_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/home/hongrui/nas/dataset/roboeval_lerobot"))
    args = p.parse_args()

    sidecar = args.root / "meta" / "roboeval_source.jsonl"
    sources = [json.loads(l) for l in sidecar.read_text().splitlines() if l.strip()]
    episode_tasks = {s["episode_index"]: instruction_for(s["variant"]) for s in sources}

    dataset = LeRobotDataset(REPO_ID, root=args.root)
    if len(sources) != dataset.meta.total_episodes:
        raise SystemExit(f"Sidecar has {len(sources)} rows but the dataset has {dataset.meta.total_episodes} episodes")
    modify_tasks(dataset, episode_tasks=episode_tasks)

    for s in sources:
        s["task"] = episode_tasks[s["episode_index"]]
    tmp = sidecar.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(s) + "\n" for s in sources))
    tmp.replace(sidecar)
    print(f"Relabelled {len(sources)} episodes into {len(set(episode_tasks.values()))} tasks")


if __name__ == "__main__":
    main()
