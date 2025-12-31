from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    snapshot_path = Path("reproduce/v1_frozen_irb2400_threeway.json")
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
        snap["out"],
    ]

    print(" ".join(args))
    print(f"\nThen open: {snap['out']}")


if __name__ == "__main__":
    main()

