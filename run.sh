#!/usr/bin/env bash
set -euo pipefail

cd "/home/yhzhu/Industrial Robot/robot_regulator"

TASK="Mjlab-JointGain-ABB-IRB2400"
TASK_1="Mjlab-JointGainFF-ABB-IRB2400"
EXP_DIR="logs/rsl_rl/irb2400_joint_gain"
EXP_DIR_1="logs/rsl_rl/irb2400_joint_gain_ff"

RUN_DIR="$(ls -dt "$EXP_DIR"/* | head -n 1)"
RUN_DIR_1="$(ls -dt "$EXP_DIR_1"/* | head -n 1)"
CKPT="$(ls -t "$RUN_DIR"/model_*.pt | head -n 1)"
CKPT_1="$(ls -t "$RUN_DIR_1"/model_*.pt | head -n 1)"

echo "[INFO] run_dir: $RUN_DIR"
echo "[INFO] run_dir_1: $RUN_DIR_1"
echo "[INFO] checkpoint: $CKPT"
echo "[INFO] checkpoint_1: $CKPT_1"

# 有桌面显示（弹 MuJoCo viewer 窗口）
# MUJOCO_GL=glfw CUDA_VISIBLE_DEVICES=0 python -m scripts.mjlab_play "$TASK" --checkpoint-file "$CKPT" --num-envs 1 --viewer native

#CPU viewer
# MUJOCO_GL=glfw CUDA_VISIBLE_DEVICES=0 python -m scripts.mjlab_play "$TASK" --checkpoint-file "$CKPT" --num-envs 1 --viewer native

MUJOCO_GL=glfw CUDA_VISIBLE_DEVICES=0 python -m scripts.mjlab_play "$TASK_1" --checkpoint-file "$CKPT_1" --num-envs 1 --viewer native

#CPU headless
# MUJOCO_GL=glfw CUDA_VISIBLE_DEVICES=0 python -m scripts.mjlab_play "$TASK" --checkpoint-file "$CKPT" --num-envs 1 --device cpu --headless

# 无桌面/远程（用 viser）
# MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 play "$TASK" --checkpoint-file "$CKPT" --num-envs 1 --viewer viser

