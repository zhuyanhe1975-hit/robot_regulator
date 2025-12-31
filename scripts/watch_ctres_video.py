from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _latest_run_dir(exp_dir: Path) -> Path | None:
    if not exp_dir.exists():
        return None
    runs = [p for p in exp_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="Mjlab-CTRes-ABB-IRB2400")
    p.add_argument("--exp-dir", type=str, default="logs/rsl_rl/irb2400_ctres")
    p.add_argument("--interval-s", type=float, default=60.0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--video-length", type=int, default=600)
    p.add_argument("--video-height", type=int, default=480)
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument("--skip-steps", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--out-dir", type=str, default="compare/ctres_watch")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    exp_dir = Path(args.exp_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt: Path | None = None
    print(f"[INFO] watching {exp_dir} for new checkpoints...", flush=True)
    while True:
        run_dir = _latest_run_dir(exp_dir)
        if run_dir is None:
            print(f"[WARN] no run dir under {exp_dir}; waiting...", flush=True)
            time.sleep(float(args.interval_s))
            continue

        ckpt = _latest_checkpoint(run_dir)
        if ckpt is None:
            print(f"[WARN] no model_*.pt under {run_dir}; waiting...", flush=True)
            time.sleep(float(args.interval_s))
            continue

        if last_ckpt is not None and ckpt.samefile(last_ckpt):
            time.sleep(float(args.interval_s))
            continue

        last_ckpt = ckpt
        step_tag = ckpt.stem  # model_123
        out_json = out_dir / f"{run_dir.name}.{step_tag}.json"

        cmd = [
            "python",
            "-m",
            "scripts.mjlab_compare",
            "--task",
            str(args.task),
            "--checkpoint",
            str(ckpt),
            "--episodes",
            str(int(args.episodes)),
            "--video",
            "--video-length",
            str(int(args.video_length)),
            "--video-height",
            str(int(args.video_height)),
            "--video-width",
            str(int(args.video_width)),
            "--skip-steps",
            str(int(args.skip_steps)),
            "--out",
            str(out_json),
        ]
        if args.device:
            cmd.extend(["--device", str(args.device)])

        print(f"[INFO] new checkpoint: {ckpt}", flush=True)
        subprocess.run(cmd, check=False)
        time.sleep(float(args.interval_s))


if __name__ == "__main__":
    main()

