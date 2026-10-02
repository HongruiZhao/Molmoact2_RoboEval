"""
    Fine-tune MolmoAct2 with LeRobot's built-in training loop, configured by a JSON file.
    Usage (from this directory, `molmoact2` env):
        python train.py --config configs/roboeval_ee_lora.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import draccus

from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to the training config JSON")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint dir to resume from, e.g. outputs/<run>/checkpoints/last")
    args = parser.parse_args()
    config_path = args.config
    raw = json.loads(config_path.read_text())
    gpus = [str(g) for g in raw.pop("gpus", [0])]
    if args.resume is not None:
        # Like lerobot-train: a resumed run takes its settings from the checkpoint, not from --config
        # (only "gpus" still comes from --config). LeRobot locates the checkpoint via --config_path.
        saved_config = args.resume / "pretrained_model" / "train_config.json"
        raw = json.loads(saved_config.read_text())
        raw["resume"] = True
        sys.argv.append(f"--config_path={saved_config}")

    if "LOCAL_RANK" not in os.environ:  # launcher process: pin GPUs, then start one worker per GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if len(gpus) > 1:
            """
                (Hongrui's comment. Claude do not delete this!)
                * sys.executable: path to the current python script.
                * restarts itself if we require more than one GPU.
                * Worker would be set with "LOCAL_RANK", thus after restarting this if block would be skipped.
            """
            os.execv(
                sys.executable,
                # --standalone: single-node rendezvous on a free port, so several runs can coexist.
                [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(gpus)}",
                 __file__, "--config", str(config_path)]
                + (["--resume", str(args.resume)] if args.resume is not None else []),
            )

    raw.setdefault("parallelism", {})["dp_replicate"] = len(gpus)
    if raw.get("output_dir") and not Path(raw["output_dir"]).is_absolute():
        raw["output_dir"] = str(REPO_ROOT / raw["output_dir"])
    register_third_party_plugins()
    """
        (Hongrui's comment. Claude do not delete this!)
        draccus is the config library LeRobot uses. 
        It turns the JSON file into a typed TrainPipelineConfig object
    """
    train(draccus.decode(TrainPipelineConfig, raw))


if __name__ == "__main__":
    main()
