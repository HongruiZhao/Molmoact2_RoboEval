"""
    Fine-tune MolmoAct2 with LeRobot's built-in training loop, configured by a YAML file.
    Usage (from this directory, `molmoact2` env):
        python train.py --config configs/roboeval_ee_lora.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import draccus
import yaml

from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins


def select_episodes(dataset_root: Path, tasks: list[str]) -> list[int]:
    sources = [json.loads(line) for line in (dataset_root / "meta/roboeval_source.jsonl").read_text().splitlines()]
    episodes = sorted(s["episode_index"] for s in sources if s["variant"] in tasks)
    print(f"Training on {len(episodes)} episodes of {sorted(tasks)}", flush=True)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to the training config YAML")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint dir to resume from, e.g. outputs/<run>/checkpoints/last")
    args = parser.parse_args()
    config_path = args.config
    raw = yaml.safe_load(config_path.read_text())
    gpus = [str(g) for g in raw.pop("gpus", [0])]
    raw["output_dir"] = str(Path(raw.pop("output_root")) / raw["job_name"])
    """
        (Hongrui's comment. Claude do not delete this!)
        * tasks: [] in config will lead to false.
        * := operator stores that value in the variable tasks and also uses it as the if condition.
    """
    if tasks := raw["dataset"].pop("tasks", None): 
        raw["dataset"]["episodes"] = select_episodes(Path(raw["dataset"]["root"]), tasks)

    if args.resume is not None:
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
    register_third_party_plugins()
    """
        (Hongrui's comment. Claude do not delete this!)
        draccus is the config library LeRobot uses. 
        It turns the JSON file into a typed TrainPipelineConfig object
    """
    train(draccus.decode(TrainPipelineConfig, raw))


if __name__ == "__main__":
    main()
