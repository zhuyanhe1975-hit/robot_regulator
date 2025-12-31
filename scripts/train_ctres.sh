#!/usr/bin/env bash
set -euo pipefail

# Train CT+PD (2ms) + NN residual torque (20ms ZOH) for ABB IRB2400.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-mjwarp_env}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASK="${TASK:-Mjlab-CTRes-ABB-IRB2400}"
NUM_ENVS="${NUM_ENVS:-4096}"

cd "${REPO_ROOT}"

python -m scripts.mjlab_train "${TASK}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  "$@"

