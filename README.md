# MolmoAct2 × RoboEval

Fine-tune [MolmoAct2](https://github.com/allenai/molmoact2) (via LeRobot) on RoboEval demonstrations.

## Environment setup

```bash
# LeRobot source (pinned commit e624f3f7; LeRobot needs Python >= 3.12)
git clone https://github.com/huggingface/lerobot third_party/lerobot
git -C third_party/lerobot checkout e624f3f7

conda create -y -n molmoact2 python=3.12 "ffmpeg>=7,<8" -c conda-forge
conda activate molmoact2
pip install "torch==2.11.0" "torchvision==0.26.0"          # same versions as LeRobot's uv.lock
pip install -e "third_party/lerobot[molmoact2,libero,training]"

hf download allenai/MolmoAct2-LIBERO                       # ~19 GB, into ~/.cache/huggingface
```

## Sanity check: MolmoAct2-LIBERO on LIBERO

```bash
python scripts/sanity_check_libero.py --task libero_goal --task_ids 0 --n_episodes 1
```

- `--task`: LIBERO suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`)
- `--task_ids`: one or more task indices within the suite
- `--dtype bfloat16` for faster/lighter inference (default `float32` matches the documented replication setting)

Outputs go to `outputs/sanity_check/<suite>/task<ID>/ep<N>/`: one video per camera fed to MolmoAct2
(`image.mp4` = agentview, `wrist_image.mp4` = eye-in-hand) plus `all_cameras.mp4` side by side, and
`results.json` with per-episode success. Frames are captured after LeRobot's LIBERO env processor
(180° flip), i.e. exactly what enters the MolmoAct2 preprocessor.

Notes:
- The script waits `num_steps_wait=50` no-op steps after reset, as the LeRobot MolmoAct2 docs
  recommend; LeRobot's `lerobot-eval` config does not expose this and uses 10.
- The docs' `--policy.model_dtype` flag is called `dtype` in the current `MolmoAct2Config`.
