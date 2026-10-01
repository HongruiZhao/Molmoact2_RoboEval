"""Check the FK-derived EE actions by replaying them through RoboEval's absolute-EE (IK) action mode.

Runs in the `roboeval` env. For a few episodes per task variant, replays the dataset's 20 Hz actions
twice from the same seed: once as absolute joint targets (reference) and once as absolute EE poses
(`JointPositionActionMode(absolute=True, ee=True)`). Reports success of both and the mean arm-joint
tracking gap between the two rollouts.

Usage (from the RoboEval repo root):
    python /path/to/verify_ee_replay.py JOINTS_NPZ EE_NPZ [EPISODES_PER_VARIANT] [WORKERS]
"""

import sys
import warnings
from multiprocessing import Pool

import numpy as np

FPS = 20
# robot.qpos = [left arm 7, left fingers 2, right arm 7, right fingers 2]
ARM_QPOS = np.r_[0:7, 9:16]


def replay(job):
    warnings.filterwarnings("ignore")
    from roboeval.demonstrations.demo import Demo
    from roboeval.utils.observation_config import ObservationConfig

    source, seed, variant, joint_actions, ee_actions = job
    md = Demo.from_safetensors(source).metadata
    result = {"variant": variant, "source": source}
    qpos = {}
    for mode, actions in (("joint", joint_actions), ("ee", ee_actions)):
        action_mode = md.get_action_mode()
        action_mode.ee = mode == "ee"
        env = md.env_cls(
            action_mode=action_mode,
            observation_config=ObservationConfig(cameras=[]),
            render_mode=None,
            control_frequency=FPS,
            robot_cls=md.robot_cls,
        )
        env.reset(seed=seed)
        success, trace = False, []
        for a in actions:
            env.step(a)
            trace.append(env.robot.qpos[ARM_QPOS].copy())
            success = success or bool(env.success)
        env.close()
        result[f"{mode}_success"] = success
        qpos[mode] = np.stack(trace)
    result["joint_gap"] = float(np.abs(qpos["joint"] - qpos["ee"]).mean())
    return result


def main():
    joints_npz, ee_npz = sys.argv[1], sys.argv[2]
    per_variant = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    j, ee = np.load(joints_npz), np.load(ee_npz)
    ep = j["episode_index"]
    ee_actions = ee["action"].copy()
    ee_actions[:, [3, 4, 5, 9, 10, 11]] = np.mod(ee_actions[:, [3, 4, 5, 9, 10, 11]] + np.pi, 2 * np.pi) - np.pi

    rng = np.random.default_rng(0)
    jobs = []
    for v in sorted(set(j["episode_variant"])):
        for e in rng.choice(np.where(j["episode_variant"] == v)[0], per_variant, replace=False):
            m = ep == e
            jobs.append((str(j["episode_source"][e]), int(j["episode_seed"][e]), v, j["action"][m], ee_actions[m]))

    with Pool(workers) as pool:
        results = pool.map(replay, jobs)
    for r in results:
        print(f"{r['variant']:45s} joint_success={r['joint_success']!s:5s} ee_success={r['ee_success']!s:5s} "
              f"joint_gap={r['joint_gap']:.4f} rad")
    n = len(results)
    js = sum(r["joint_success"] for r in results)
    es = sum(r["ee_success"] for r in results)
    both = sum(r["joint_success"] and r["ee_success"] for r in results)
    print(f"\njoint replay success {js}/{n}, EE replay success {es}/{n}, "
          f"EE success among joint successes {both}/{js}, "
          f"median joint gap {np.median([r['joint_gap'] for r in results]):.4f} rad")


if __name__ == "__main__":
    main()
