#!/usr/bin/env python
"""Sanity check: roll out the original MolmoAct2-LIBERO checkpoint on LIBERO task(s).

Mirrors the "Evaluation With Original MolmoAct2 Weight" recipe from
https://huggingface.co/docs/lerobot/molmoact2 but runs a plain single-env loop so the
camera images handed to MolmoAct2 can be saved as videos.

Example:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/sanity_check_libero.py \
        --task libero_goal --task_ids 0 --n_episodes 1
"""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gymnasium as gym
import imageio
import numpy as np
import torch

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import create_libero_envs
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.random_utils import set_seed

# LIBERO camera -> image key expected by MolmoAct2-LIBERO (see norm_stats.json "camera_keys").
CAMERA_NAME_MAPPING = {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="LIBERO suite, e.g. libero_goal, libero_10, libero_spatial")
    p.add_argument("--task_ids", type=int, nargs="+", required=True, help="Task index/indices within the suite")
    p.add_argument("--n_episodes", type=int, default=1, help="Episodes per task (consecutive init states)")
    p.add_argument("--checkpoint", default="allenai/MolmoAct2-LIBERO")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="float32 matches the documented replication setting; bfloat16 enables AMP")
    p.add_argument("--num_steps_wait", type=int, default=50,
                   help="No-op steps after reset to let the scene settle (docs: 10 is not enough)")
    p.add_argument("--resolution", type=int, default=360, help="LIBERO render height/width")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--output_dir", default="outputs/sanity_check")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) float in [0, 1] -> (H, W, C) uint8."""
    return (img[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = args.dtype == "bfloat16"

    env_cfg = LiberoEnvConfig(
        task=args.task,
        task_ids=args.task_ids,
        camera_name_mapping=CAMERA_NAME_MAPPING,
        observation_height=args.resolution,
        observation_width=args.resolution,
    )
    policy_cfg = MolmoAct2Config(
        checkpoint_path=args.checkpoint,
        norm_tag="libero",
        inference_action_mode="continuous",
        dtype=args.dtype,
        use_amp=use_amp,
        enable_inference_cuda_graph=True,
        device=args.device,
        per_episode_seed=True,
        eval_seed=args.seed,
    )

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    print(f"image keys fed to MolmoAct2: {policy_cfg.image_keys or 'from norm_stats camera_keys'}")
    print(f"chunk_size={policy.config.chunk_size} n_action_steps={policy.config.n_action_steps} "
          f"setup_type={policy.config.setup_type!r} control_mode={policy.config.control_mode!r}")

    envs = create_libero_envs(
        task=args.task,
        n_envs=1,
        camera_name=env_cfg.camera_name,
        init_states=env_cfg.init_states,
        gym_kwargs={**env_cfg.gym_kwargs, "num_steps_wait": args.num_steps_wait},
        env_cls=gym.vector.SyncVectorEnv,
        control_mode=env_cfg.control_mode,
        camera_name_mapping=CAMERA_NAME_MAPPING,
    )

    out_root = Path(args.output_dir) / args.task
    results = []
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else nullcontext()
    for task_id, env in envs[args.task].items():
        task_desc = env.call("task_description")[0]
        max_steps = env.call("_max_episode_steps")[0]
        for ep in range(args.n_episodes):
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            observation, _ = env.reset(seed=[args.seed + ep])
            frames: dict[str, list[np.ndarray]] = {}
            success = False
            step = 0
            for step in range(1, max_steps + 1):
                obs = preprocess_observation(observation)
                obs["task"] = [task_desc]
                obs = env_preprocessor(obs)  # 180-degree flip + flat 8-D state
                # These are exactly the images passed on to the MolmoAct2 processor.
                for key in sorted(k for k in obs if k.startswith(f"{OBS_IMAGES}.")):
                    frames.setdefault(key.removeprefix(f"{OBS_IMAGES}."), []).append(to_uint8_hwc(obs[key]))
                obs = preprocessor(obs)
                with torch.inference_mode(), amp_ctx:
                    action = policy.select_action(obs)
                action = postprocessor(action)
                action = env_postprocessor({ACTION: action})[ACTION]
                observation, _, terminated, truncated, info = env.step(action.to("cpu").numpy())
                success = bool(info["is_success"][0])
                if terminated[0] or truncated[0]:
                    break

            ep_dir = out_root / f"task{task_id:02d}" / f"ep{ep:02d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            for cam, cam_frames in frames.items():
                imageio.mimsave(ep_dir / f"{cam}.mp4", cam_frames, fps=args.fps, macro_block_size=1)
            combined = [np.concatenate(views, axis=1) for views in zip(*frames.values(), strict=True)]
            imageio.mimsave(ep_dir / "all_cameras.mp4", combined, fps=args.fps, macro_block_size=1)

            result = {"task_id": task_id, "episode": ep, "task": task_desc, "success": success, "steps": step}
            results.append(result)
            print(f"[{args.task} task {task_id} ep {ep}] success={success} steps={step} "
                  f"'{task_desc}' -> {ep_dir}")
        env.close()

    (out_root / "results.json").write_text(json.dumps(results, indent=2))
    n_success = sum(r["success"] for r in results)
    print(f"Success: {n_success}/{len(results)}")


if __name__ == "__main__":
    main()
