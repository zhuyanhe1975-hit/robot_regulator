from __future__ import annotations

import os
import sys

# Convenience wrapper: provide defaults and allow overriding max_iterations.
#
# Usage:
#   python -m scripts.train_gainff
#   python -m scripts.train_gainff --max-iterations 2000
#   python -m scripts.train_gainff MjlabV1-JointGainFFI-ABB-IRB2400 --max-iterations 2000
# Any additional args are forwarded to mjlab's train CLI.

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
DEFAULT_MAX_ITERS = "1000"

def _split_opt_eq(arg: str) -> tuple[str, str | None]:
    """Split tyro-style `--opt=value` into (`--opt`, `value`)."""
    if not arg.startswith("--") or "=" not in arg:
        return arg, None
    opt, value = arg.split("=", 1)
    return opt, value


def _ensure_defaults(argv: list[str]) -> list[str]:
    task = DEFAULT_TASK
    num_envs: str | None = None
    forwarded: list[str] = []
    max_iters: str | None = None

    i = 0
    while i < len(argv):
        a_raw = argv[i]
        a, a_eq = _split_opt_eq(a_raw)

        if a in ("--max-iterations", "--max_iterations"):
            if a_eq is not None:
                max_iters = a_eq
                i += 1
            else:
                if i + 1 >= len(argv):
                    raise SystemExit("--max-iterations requires a value")
                max_iters = argv[i + 1]
                i += 2
            continue

        # Allow passing through tyro-native flag names too.
        if a in ("--agent.max-iterations", "--agent.max_iterations"):
            if a_eq is not None:
                max_iters = a_eq
                i += 1
            else:
                if i + 1 >= len(argv):
                    raise SystemExit(f"{a} requires a value")
                max_iters = argv[i + 1]
                i += 2
            continue

        # Allow overriding num envs; don't duplicate if user supplies it.
        if a in ("--env.scene.num-envs", "--env.scene.num_envs"):
            if a_eq is not None:
                num_envs = a_eq
                i += 1
            else:
                if i + 1 >= len(argv):
                    raise SystemExit(f"{a} requires a value")
                num_envs = argv[i + 1]
                i += 2
            continue

        # First positional argument is task id (if it doesn't look like an option).
        if a_raw and not a_raw.startswith("-") and task == DEFAULT_TASK and not forwarded:
            task = a_raw
            i += 1
            continue

        forwarded.append(a_raw)
        i += 1

    out = [task, "--env.scene.num-envs", str(num_envs or DEFAULT_NUM_ENVS)]
    # mjlab uses tyro: nested config field is agent.max_iterations.
    out.extend(["--agent.max-iterations", str(max_iters or DEFAULT_MAX_ITERS)])
    out.extend(forwarded)
    return out


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_ensure_defaults(sys.argv[1:])]
    main()
