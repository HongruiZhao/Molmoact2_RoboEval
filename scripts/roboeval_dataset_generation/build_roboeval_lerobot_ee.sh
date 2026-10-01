#!/usr/bin/env bash
# Derive the absolute end-effector-pose RoboEval dataset from the joint-space one
# (built by build_roboeval_lerobot.sh).
#   Stage 1 (molmoact2 env): dump joint state/action -> joints.npz
#   Stage 2 (roboeval env):  RoboEval FK -> 14-D EE poses -> ee.npz
#   Check   (roboeval env):  replay a few episodes per variant with EE actions through RoboEval IK
#   Stage 3 (molmoact2 env): write LeRobotDataset with EE state/action + whole-dataset stats
set -euo pipefail

ROBOEVAL_REPO=${ROBOEVAL_REPO:-/home/hongrui/codes/RoboEval}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
SRC=${SRC:-/home/hongrui/nas/dataset/roboeval_lerobot}
OUTPUT=${OUTPUT:-/home/hongrui/nas/dataset/roboeval_lerobot_ee}
WORK=${WORK:-$HOME/.cache/roboeval_lerobot_ee}
WORKERS=${WORKERS:-32}
LOG_DIR="$REPO/outputs/roboeval_lerobot_ee"
mkdir -p "$LOG_DIR" "$WORK"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[stage 1] export joints -> $WORK/joints.npz" | tee -a "$LOG_DIR/build.log"
conda activate molmoact2
python "$SCRIPT_DIR/export_joints.py" "$SRC" "$WORK/joints.npz" 2>&1 | tee -a "$LOG_DIR/build.log"

echo "[stage 2] forward kinematics -> $WORK/ee.npz" | tee -a "$LOG_DIR/build.log"
conda activate roboeval
(cd "$ROBOEVAL_REPO" && MUJOCO_GL=egl python -W ignore "$SCRIPT_DIR/compute_ee_poses.py" \
    "$WORK/joints.npz" "$WORK/ee.npz" "$WORKERS") 2>&1 | tee -a "$LOG_DIR/build.log"

echo "[check] EE replay through RoboEval IK" | tee -a "$LOG_DIR/build.log"
(cd "$ROBOEVAL_REPO" && MUJOCO_GL=egl python -W ignore "$SCRIPT_DIR/verify_ee_replay.py" \
    "$WORK/joints.npz" "$WORK/ee.npz" 2 "$WORKERS") 2>&1 | grep -E "success" | tee -a "$LOG_DIR/build.log"

echo "[stage 3] writing LeRobot dataset -> $OUTPUT" | tee -a "$LOG_DIR/build.log"
conda activate molmoact2
python "$SCRIPT_DIR/write_ee_dataset.py" "$SRC" "$WORK/ee.npz" "$OUTPUT" 2>&1 | tee -a "$LOG_DIR/build.log"
