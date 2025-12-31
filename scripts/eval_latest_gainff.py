from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

# Reuse the evaluation/metric logic from mjlab_compare (but skip baseline).
from scripts import mjlab_compare as mc


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
    p.add_argument(
        "--task",
        type=str,
        default="MjlabV1-JointGainFF-ABB-IRB2400",
        help="mjlab task id to evaluate",
    )
    p.add_argument(
        "--exp-dir",
        type=str,
        default="logs/rsl_rl/irb2400_joint_gain_ff",
        help="Experiment directory containing timestamped run folders",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional explicit checkpoint path; when omitted, uses newest model_*.pt in newest run dir",
    )
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0 or cpu")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds (overrides --seed)",
    )
    p.add_argument("--skip-steps", type=int, default=50)
    p.add_argument("--out", type=str, default="compare/latest_gainff_eval.json")
    p.add_argument("--video", action="store_true", help="Record video (forces num_envs=1, first seed only)")
    p.add_argument(
        "--viewer",
        type=str,
        default="",
        choices=["", "native", "viser"],
        help="Launch a viewer (forces num_envs=1, first seed only).",
    )
    p.add_argument("--video-length", type=int, default=600)
    p.add_argument("--video-height", type=int, default=480)
    p.add_argument("--video-width", type=int, default=640)
    return p.parse_args()


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return [int(args.seed)]


def main() -> None:
    args = _parse_args()
    mc.configure_torch_backends()

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = mc._resolve_device(args.device)
    seeds = _parse_seeds(args)
    if not seeds:
        raise SystemExit("No seeds provided.")

    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser().resolve()
    else:
        run_dir = _latest_run_dir(exp_dir)
        ckpt = _latest_checkpoint(run_dir)

    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    if args.video:
        args.num_envs = 1
        if len(seeds) > 1:
            seeds = seeds[:1]

    if args.viewer:
        args.num_envs = 1
        if len(seeds) > 1:
            seeds = seeds[:1]

    all_logs: list[dict[str, float]] = []
    per_seed: list[dict[str, Any]] = []

    video_dir = out_path.parent / "videos" / "trained" if args.video else None

    for seed in seeds:
        mc.torch.manual_seed(int(seed))
        env = mc._make_env(
            task_id=args.task,
            device=device,
            num_envs=int(args.num_envs),
            seed=int(seed),
            video=bool(args.video),
            video_folder=video_dir,
            video_length=int(args.video_length),
            video_height=int(args.video_height),
            video_width=int(args.video_width),
        )
        policy = mc._load_trained_policy(task_id=args.task, checkpoint=ckpt, env=env, device=device)
        logs = mc._run_eval(
            env=env,
            policy=policy,
            episodes=int(args.episodes),
            device=device,
            skip_steps=int(args.skip_steps),
        )
        env.close()

        all_logs.extend(logs)
        per_seed.append(
            {"seed": int(seed), "episode_logs": logs, "summary": mc._summarize(logs)}
        )

    payload = {
        "task": str(args.task),
        "checkpoint": str(ckpt),
        "exp_dir": str(exp_dir),
        "seeds": [int(s) for s in seeds],
        "num_envs": int(args.num_envs),
        "episodes": int(args.episodes),
        "skip_steps": int(args.skip_steps),
        "video": bool(args.video),
        "summary": mc._summarize(all_logs),
        "per_seed": per_seed,
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK] checkpoint: {ckpt}")
    print(f"[OK] wrote: {out_path}")
    print("\nSummary (selected):")
    for k in sorted(payload["summary"].keys()):
        if k.endswith("/mean") and any(s in k for s in ("JointError", "EEError", "Episode_Reward", "Gain")):
            print(f"  {k}: {payload['summary'][k]:.4f}")

    if args.viewer:
        env = os.environ.copy()
        # For the native MuJoCo viewer, glfw is typically required.
        env.setdefault("MUJOCO_GL", "glfw")
        cmd = [
            "python",
            "-m",
            "scripts.mjlab_play",
            str(args.task),
            "--checkpoint-file",
            str(ckpt),
            "--num-envs",
            "1",
            "--viewer",
            str(args.viewer),
        ]
        print("\n[INFO] launching viewer:")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    main()
