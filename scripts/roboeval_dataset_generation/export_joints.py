"""Dump the joint state/action of the RoboEval LeRobot dataset to a .npz (EE conversion stage 1).

Runs in the `molmoact2` env. `compute_ee_poses.py` (`roboeval` env, which has no pandas/pyarrow)
reads this file and applies RoboEval forward kinematics.

Usage:
    python scripts/roboeval_dataset_generation/export_joints.py SRC_ROOT OUT_NPZ
"""

import json
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "roboeval_lerobot"


def main():
    src_root, out = Path(sys.argv[1]), Path(sys.argv[2])
    table = LeRobotDataset(REPO_ID, root=src_root).hf_dataset.data.table

    def column(name: str) -> np.ndarray:
        col = table.column(name).combine_chunks()
        if hasattr(col.type, "value_type"):  # list column -> (N, D)
            return col.flatten().to_numpy().reshape(len(col), -1)
        return col.to_numpy()

    index = column("index")
    assert (index == np.arange(len(index))).all(), "expected frames stored in global index order"
    sources = [json.loads(l) for l in (src_root / "meta/roboeval_source.jsonl").read_text().splitlines()]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        state=column("observation.state").astype(np.float32),
        action=column("action").astype(np.float32),
        episode_index=column("episode_index").astype(np.int64),
        episode_variant=np.array([s["variant"] for s in sources]),
        episode_source=np.array([s["source"] for s in sources]),
        episode_seed=np.array([s["seed"] for s in sources], dtype=np.int64),
    )
    print(f"{len(index)} frames, {len(sources)} episodes -> {out}")


if __name__ == "__main__":
    main()
