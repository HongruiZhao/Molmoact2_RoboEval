"""Write the RoboEval LeRobot dataset with absolute EE-pose state/action (EE conversion stage 3).

Runs in the `molmoact2` env. Takes the .npz from `compute_ee_poses.py` and, with LeRobot's
`dataset_tools.modify_features`, writes a copy of the joint-space dataset in which
`observation.state` / `action` are the 14-D EE poses; the original 16-D joint arrays are kept as
`joint_state` / `joint_action` (no `observation.` prefix, so policies do not take them as inputs).
Videos are copied unchanged. Stats for the numeric features are recomputed with one running histogram
over the whole dataset (LeRobot's `augment_dataset_quantile_stats`), because per-episode aggregation
only keeps a min/max envelope of the episode quantiles, which MolmoAct2's q01/q99 normalization uses.

Usage:
    python scripts/roboeval_dataset_generation/write_ee_dataset.py SRC_ROOT EE_NPZ OUT_ROOT
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from lerobot.datasets import write_stats
from lerobot.datasets.dataset_tools import modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

SRC_REPO_ID = "roboeval_lerobot"
REPO_ID = "roboeval_lerobot_ee"


def write_global_stats(dataset: LeRobotDataset) -> None:
    """Replace numeric-feature stats with whole-dataset histogram stats; keep the image stats."""
    stats = compute_quantile_stats_for_dataset(dataset, skip_images=True)
    for key, feature_stats in dataset.meta.stats.items():
        stats.setdefault(key, feature_stats)
    write_stats(stats, dataset.meta.root)
    dataset.meta.stats = stats


def main():
    src_root, ee_npz, out_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if out_root.exists():
        raise SystemExit(f"{out_root} already exists; remove it first")
    ee = np.load(ee_npz)
    names = [str(n) for n in ee["names"]]

    src = LeRobotDataset(SRC_REPO_ID, root=src_root)
    assert len(ee["action"]) == src.meta.total_frames, "EE arrays do not match the source dataset"
    joint_state = np.stack(src.hf_dataset.data.table.column("observation.state").to_pylist())
    joint_action = np.stack(src.hf_dataset.data.table.column("action").to_pylist())
    joint_info = {k: dict(src.meta.features[k]) for k in ("observation.state", "action")}

    # modify_features cannot add a key that already exists, so move the joint arrays first.
    with tempfile.TemporaryDirectory(dir=out_root.parent) as tmp:
        staged = modify_features(
            src,
            add_features={
                "joint_state": (joint_state, joint_info["observation.state"]),
                "joint_action": (joint_action, joint_info["action"]),
            },
            remove_features=["observation.state", "action"],
            output_dir=Path(tmp) / "joint_renamed",
            repo_id=f"{REPO_ID}_tmp",
        )
        ee_info = {"dtype": "float32", "shape": (len(names),), "names": names}
        dataset = modify_features(
            staged,
            add_features={"observation.state": (ee["state"], ee_info), "action": (ee["action"], ee_info)},
            output_dir=out_root,
            repo_id=REPO_ID,
        )
    write_global_stats(dataset)
    shutil.copy(src_root / "meta/roboeval_source.jsonl", out_root / "meta/roboeval_source.jsonl")
    print(f"Done: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames at {out_root}")


if __name__ == "__main__":
    main()
