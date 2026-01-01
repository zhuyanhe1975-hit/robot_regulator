from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from mjlab_irb2400_v1.irb2400_joint_gain.env_cfgs import irb2400_joint_gain_env_cfg_v1
from mjlab_irb2400_v1.irb2400_joint_gain.rl_cfg import irb2400_joint_gain_ppo_runner_cfg_v1
from mjlab_irb2400_v1.irb2400_joint_gain_ff.env_cfgs import irb2400_joint_gain_ff_env_cfg_v1
from mjlab_irb2400_v1.irb2400_joint_gain_ff.rl_cfg import irb2400_joint_gain_ff_ppo_runner_cfg_v1

TASK_ID_GAIN_V1 = "MjlabV1-JointGain-ABB-IRB2400"
TASK_ID_GAIN_FF_V1 = "MjlabV1-JointGainFF-ABB-IRB2400"
TASK_ID_GAIN_FF_I_V1 = "MjlabV1-JointGainFFI-ABB-IRB2400"


def register() -> None:
    tasks = set(list_tasks())
    if TASK_ID_GAIN_V1 not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_GAIN_V1,
            env_cfg=irb2400_joint_gain_env_cfg_v1(),
            play_env_cfg=irb2400_joint_gain_env_cfg_v1(play=True),
            rl_cfg=irb2400_joint_gain_ppo_runner_cfg_v1(),
            runner_cls=None,
        )
    if TASK_ID_GAIN_FF_V1 not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_GAIN_FF_V1,
            env_cfg=irb2400_joint_gain_ff_env_cfg_v1(),
            play_env_cfg=irb2400_joint_gain_ff_env_cfg_v1(play=True),
            rl_cfg=irb2400_joint_gain_ff_ppo_runner_cfg_v1(),
            runner_cls=None,
        )
    if TASK_ID_GAIN_FF_I_V1 not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_GAIN_FF_I_V1,
            env_cfg=irb2400_joint_gain_ff_env_cfg_v1(use_integral=True),
            play_env_cfg=irb2400_joint_gain_ff_env_cfg_v1(play=True, use_integral=True),
            rl_cfg=irb2400_joint_gain_ff_ppo_runner_cfg_v1(),
            runner_cls=None,
        )


register()
