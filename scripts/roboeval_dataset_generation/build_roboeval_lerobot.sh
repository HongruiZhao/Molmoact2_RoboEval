#!/usr/bin/env bash
# Build the RoboEval LeRobot dataset end to end.
#   Stage 1 (roboeval env):  replay successful demos, render head + wrist cameras -> per-episode .npz
#   Stage 2 (molmoact2 env): write LeRobotDataset v3.0 shards in parallel, aggregate into $OUTPUT
#   Stage 3 (molmoact2 env): sanity check the result
# Both stages are resumable/idempotent up to the aggregate step. Progress bars go to the terminal
# and to $LOG_DIR/*.log (follow with: tail -f outputs/roboeval_lerobot/build.log).
set -euo pipefail

ROBOEVAL_REPO=${ROBOEVAL_REPO:-/home/hongrui/codes/RoboEval}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
STAGING=${STAGING:-$HOME/.cache/roboeval_lerobot_staging}
SHARDS_DIR=${SHARDS_DIR:-$HOME/.cache/roboeval_lerobot_shards}
OUTPUT=${OUTPUT:-/home/hongrui/nas/dataset/roboeval_lerobot}
RENDER_WORKERS=${RENDER_WORKERS:-64}
WRITE_SHARDS=${WRITE_SHARDS:-16}
LOG_DIR="$REPO/outputs/roboeval_lerobot"
mkdir -p "$LOG_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[stage 1] rendering -> $STAGING" | tee -a "$LOG_DIR/build.log"
conda activate roboeval
(cd "$ROBOEVAL_REPO" && MUJOCO_GL=egl python -W ignore data_cleaning/render_demos.py \
    --out "$STAGING" --workers "$RENDER_WORKERS") 2>&1 | tee -a "$LOG_DIR/build.log"

echo "[stage 2] writing LeRobot dataset -> $OUTPUT" | tee -a "$LOG_DIR/build.log"
conda activate molmoact2
python "$SCRIPT_DIR/convert_roboeval_to_lerobot.py" \
    --staging "$STAGING" --shards-dir "$SHARDS_DIR" --output "$OUTPUT" --shards "$WRITE_SHARDS" \
    2>&1 | tee -a "$LOG_DIR/build.log"

echo "[stage 3] sanity check" | tee -a "$LOG_DIR/build.log"
python "$SCRIPT_DIR/check_roboeval_dataset.py" --root "$OUTPUT" 2>&1 | tee -a "$LOG_DIR/build.log"
