from __future__ import annotations

import os
import sys

# Hard-disable wandb by default for this repo.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")

# Default to headless EGL for training throughput (override by exporting MUJOCO_GL yourself).
os.environ.setdefault("MUJOCO_GL", "egl")

# Register IRB2400 task(s) into mjlab's registry.
import mjlab_irb2400_v1  # noqa: F401

from mjlab.scripts.train import main


DEFAULT_TASK = "MjlabV1-JointGainFF-ABB-IRB2400"
DEFAULT_NUM_ENVS = "4096"


def _ensure_defaults(argv: list[str]) -> list[str]:
    if argv:
        return argv
    return [DEFAULT_TASK, "--env.scene.num-envs", DEFAULT_NUM_ENVS]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_ensure_defaults(sys.argv[1:])]
    main()

