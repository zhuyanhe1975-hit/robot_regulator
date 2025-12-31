from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from mjlab_irb2400_ctres.irb2400_ctres.env_cfgs import irb2400_ctres_env_cfg
from mjlab_irb2400_ctres.irb2400_ctres.rl_cfg import irb2400_ctres_ppo_runner_cfg

TASK_ID_CTRES = "Mjlab-CTRes-ABB-IRB2400"


def register() -> None:
    tasks = set(list_tasks())
    if TASK_ID_CTRES not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_CTRES,
            env_cfg=irb2400_ctres_env_cfg(),
            play_env_cfg=irb2400_ctres_env_cfg(play=True),
            rl_cfg=irb2400_ctres_ppo_runner_cfg(),
            runner_cls=None,
        )


register()

