"""Convert joint state/action to absolute end-effector poses with RoboEval FK (EE conversion stage 2).

Runs in the `roboeval` env. Reads the .npz written by `export_joints.py` and writes the same frames
as 14-D absolute EE poses in RoboEval's absolute-EE action layout
    [left xyz, left euler-xyz, right xyz, right euler-xyz, left gripper, right gripper]
where each arm pose is `robot.forward_kinematics` (pelvis-relative position, extrinsic xyz euler),
i.e. exactly what `JointPositionActionMode(absolute=True, ee=True)` feeds to RoboEval's IK.

Euler angles from FK are wrapped to [-pi, pi), which splits the rolls (around +-pi for a
downward-facing Panda gripper) and the yaws (often crossing +-pi) with 2pi jumps inside episodes.
Each is re-wrapped to `[low, low + 2pi)` from `WRAP_LOW`, chosen to minimize within-episode jumps
over the dataset; this is lossless for RoboEval, whose IK goes through `Rotation.from_euler`. A few
jumps from gimbal flips (|pitch| near pi/2) remain in ~1.5% of episodes; no wrap choice removes them.

Usage (from the RoboEval repo root, `roboeval` env):
    python /path/to/compute_ee_poses.py JOINTS_NPZ OUT_NPZ [WORKERS]
"""

import sys
import warnings
from multiprocessing import Pool

import numpy as np

N_ARM_JOINTS = 14
# Index into the 14-D EE vector -> lower bound of its re-wrapped euler range [low, low + 2pi).
WRAP_LOW = {3: 0.0, 9: 0.0, 5: -5.1, 11: -5.1}  # rolls, yaws; pitch stays in [-pi/2, pi/2]
EE_NAMES = [
    f"{side}_{k}" for side in ("left", "right") for k in ("x", "y", "z", "roll", "pitch", "yaw")
] + ["left_gripper", "right_gripper"]


def fk_variant(job):
    """FK for all frames of one task variant; one env per job (the base pose comes from the preset)."""
    warnings.filterwarnings("ignore")
    from roboeval.demonstrations.demo import Demo
    from roboeval.utils.observation_config import ObservationConfig

    source, joints = job
    md = Demo.from_safetensors(source).metadata
    env = md.env_cls(
        action_mode=md.get_action_mode(),
        observation_config=ObservationConfig(cameras=[]),
        render_mode=None,
        control_frequency=500,
        robot_cls=md.robot_cls,
    )
    env.reset(seed=0)
    ee = np.stack([env.robot.forward_kinematics(q) for q in joints[:, :N_ARM_JOINTS]])
    env.close()
    return np.concatenate([ee, joints[:, N_ARM_JOINTS:]], axis=1).astype(np.float32)


def main():
    src, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    z = np.load(src)
    state, action, episode_index = z["state"], z["action"], z["episode_index"]
    variant_of_frame = z["episode_variant"][episode_index]

    variants = sorted(set(z["episode_variant"]))
    jobs, masks = [], []
    for v in variants:
        mask = variant_of_frame == v
        source = z["episode_source"][int(np.where(z["episode_variant"] == v)[0][0])]
        # State and action frames of a variant go through FK in one job.
        jobs.append((source, np.concatenate([state[mask], action[mask]])))
        masks.append(mask)

    ee_state = np.zeros((len(state), len(EE_NAMES)), np.float32)
    ee_action = np.zeros((len(action), len(EE_NAMES)), np.float32)
    with Pool(workers) as pool:
        for i, (v, mask, ee) in enumerate(zip(variants, masks, pool.imap(fk_variant, jobs), strict=True)):
            n = int(mask.sum())
            ee_state[mask], ee_action[mask] = ee[:n], ee[n:]
            print(f"[{i + 1}/{len(variants)}] {v}: {n} frames", flush=True)

    for arr in (ee_state, ee_action):
        for d, low in WRAP_LOW.items():
            arr[:, d] = np.mod(arr[:, d] - low, 2 * np.pi) + low
    np.savez(out, state=ee_state, action=ee_action, names=np.array(EE_NAMES))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
