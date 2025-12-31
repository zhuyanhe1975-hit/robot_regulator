from __future__ import annotations

import json
from pathlib import Path


def _pick(d: dict, keys: list[str]) -> dict:
    out = {}
    for k in keys:
        if k in d:
            out[k] = d[k]
    return out


def main() -> None:
    snapshot_path = Path("reproduce/v1_irb2400_threeway.json")
    if not snapshot_path.exists():
        raise SystemExit(f"Missing snapshot: {snapshot_path}")
    snap = json.loads(snapshot_path.read_text())

    args = [
        "python",
        "-m",
        "scripts.mjlab_compare",
        "--task",
        snap["task"],
        "--checkpoint",
        snap["checkpoint"],
        "--ff-task",
        snap["ff_task"],
        "--ff-checkpoint",
        snap["ff_checkpoint"],
        "--num-envs",
        str(snap["num_envs"]),
        "--episodes",
        str(snap["episodes"]),
        "--seeds",
        ",".join(map(str, snap["seeds"])),
        "--baseline",
        snap["baseline"]["mode"],
        "--skip-steps",
        str(snap.get("skip_steps", 20)),
        "--out",
        "compare/v1_irb2400_threeway.json",
    ]

    print(" ".join(args))
    print("\nThen open: compare/v1_irb2400_threeway.json")

    # Also print the previously captured summary if available (no recompute).
    prev = Path("compare/irb2400_gain_compare.json")
    if prev.exists():
        obj = json.loads(prev.read_text())
        keys = [
            "JointError/mae_mean_rad/mean",
            "JointError/rmse_mean_rad/mean",
            "JointDyn/vel_rms_mean_rad_s/mean",
            "JointDyn/acc_rms_mean_rad_s2/mean",
            "Episode_Reward/track_joint_pos/mean",
        ]
        print("\nCaptured (previous) summary:")
        for group in ("baseline", "trained", "ff"):
            if group in obj and "summary" in obj[group]:
                print(f"- {group}: {_pick(obj[group]['summary'], keys)}")


if __name__ == "__main__":
    main()

