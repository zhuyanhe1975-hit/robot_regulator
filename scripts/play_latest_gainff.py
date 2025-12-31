from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _latest_run_dir(exp_dir: Path) -> Path:
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)
    runs = [p for p in exp_dir.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories under {exp_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def _latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matching model_*.pt under {run_dir}")
    return ckpts[-1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="MjlabV1-JointGainFF-ABB-IRB2400")
    p.add_argument("--exp-dir", type=str, default="logs/rsl_rl/irb2400_joint_gain_ff")
    p.add_argument("--checkpoint", type=str, default="", help="Optional explicit checkpoint path")
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0 or cpu")
    p.add_argument(
        "--viewer",
        type=str,
        default="native",
        choices=["native", "viser"],
        help="Viewer backend",
    )
    p.add_argument("--num-envs", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    exp_dir = Path(args.exp_dir).expanduser().resolve()

    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser().resolve()
    else:
        run_dir = _latest_run_dir(exp_dir)
        ckpt = _latest_checkpoint(run_dir)

    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("WANDB_CONSOLE", "off")
    if args.viewer == "native":
        env.setdefault("MUJOCO_GL", "glfw")

    cmd = [
        "python",
        "-m",
        "scripts.mjlab_play",
        str(args.task),
        "--checkpoint-file",
        str(ckpt),
        "--num-envs",
        str(int(args.num_envs)),
        "--viewer",
        str(args.viewer),
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])

    print(f"[OK] checkpoint: {ckpt}", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    main()
