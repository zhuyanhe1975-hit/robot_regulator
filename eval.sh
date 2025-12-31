# python -m scripts.mjlab_compare \
#   --checkpoint logs/rsl_rl/irb2400_joint_gain/2025-12-26_22-33-13/model_499.pt \
#   --seeds 0,1,2,3,4 \
#   --num-envs 1024 \
#   --episodes 5 \
#   --out compare/irb2400_gain_compare.json

EXP_DIR_1="logs/rsl_rl/irb2400_joint_gain"
EXP_DIR_2="logs/rsl_rl/irb2400_joint_gain_ff"
RUN_DIR_1="$(ls -dt "$EXP_DIR_1"/* | head -n 1)"
RUN_DIR_2="$(ls -dt "$EXP_DIR_2"/* | head -n 1)"
CKPT_1="$(ls -t "$RUN_DIR_1"/model_*.pt | head -n 1)"
CKPT_2="$(ls -t "$RUN_DIR_2"/model_*.pt | head -n 1)"

echo "[INFO] irb2400_joint_gain run_dir: $RUN_DIR"
echo "[INFO] irb2400_joint_gain checkpoint: $CKPT_1"

echo "[INFO] irb2400_joint_gain_ff run_dir: $RUN_DIR_2"
echo "[INFO] irb2400_joint_gain_ff checkpoint: $CKPT_2"

# python -m scripts.mjlab_compare --checkpoint "$CKPT" --seeds 0,1,2,3,4 --num-envs 1024 --episodes 5

# python -m scripts.mjlab_compare --checkpoint "$CKPT" --baseline trained_mean --seeds 0,1,2,3,4 --episodes 5 --num-envs 1024

python -m scripts.mjlab_compare --skip-steps 500 --task Mjlab-JointGain-ABB-IRB2400 --checkpoint "$CKPT_1" --baseline trained_mean --ff-task Mjlab-JointGainFF-ABB-IRB2400 --ff-checkpoint "$CKPT_2" --seeds 0,1,2,3,4 --episodes 5 --num-envs 1024