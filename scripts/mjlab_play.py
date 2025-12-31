from __future__ import annotations

import os

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")

# Register IRB2400 task(s) into mjlab's registry.
import mjlab_irb2400  # noqa: F401
import mjlab_irb2400_v1  # noqa: F401
import mjlab_irb2400_ctres  # noqa: F401

from mjlab.scripts.play import main


if __name__ == "__main__":
    main()
