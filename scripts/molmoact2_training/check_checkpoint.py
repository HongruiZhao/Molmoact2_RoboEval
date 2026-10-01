"""Reload a MolmoAct2 checkpoint saved by LeRobot training and predict actions on dataset frames.

Loads `<checkpoint>/pretrained_model` the same way `lerobot-eval --policy.path=...` does (config,
weights and saved pre/post-processors), runs `predict_action_chunk` on a few random frames of the
training dataset and prints the error against the ground-truth action chunks.

Usage:
    python scripts/molmoact2_training/check_checkpoint.py outputs/<run>/checkpoints/last/pretrained_model
"""

import sys

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors

N_FRAMES = 4
DATASET_ROOT = "/home/hongrui/nas/dataset/roboeval_lerobot_ee"
REPO_ID = "roboeval_lerobot_ee"


def main() -> None:
    pretrained = sys.argv[1]
    cfg = PreTrainedConfig.from_pretrained(pretrained)
    cfg.pretrained_path = pretrained
    cfg.inference_action_mode = "continuous"
    meta = LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)
    policy = make_policy(cfg=cfg, ds_meta=meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=pretrained,
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )

    chunk = cfg.chunk_size
    dataset = LeRobotDataset(
        REPO_ID,
        root=DATASET_ROOT,
        video_backend="pyav",
        delta_timestamps={"action": [t / meta.fps for t in range(chunk)]},
    )
    rng = np.random.default_rng(0)
    errors = []
    for idx in rng.choice(len(dataset), N_FRAMES, replace=False):
        item = dataset[int(idx)]
        batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k, v in item.items()}
        with torch.inference_mode():
            pred = postprocessor(policy.predict_action_chunk(preprocessor(batch)))
        pred = pred.float().cpu()[0]
        gt = item["action"]
        errors.append((pred - gt).abs().mean(0))
        print(f"frame {int(idx)} '{item['task']}': pred chunk {tuple(pred.shape)}, "
              f"mean |pred - gt| = {(pred - gt).abs().mean():.4f}")
    print("mean |pred - gt| per action dim:", np.round(torch.stack(errors).mean(0).numpy(), 4))


if __name__ == "__main__":
    main()
