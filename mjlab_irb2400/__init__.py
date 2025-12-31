from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from mjlab_irb2400.irb2400_joint_gain.env_cfgs import irb2400_joint_gain_env_cfg
from mjlab_irb2400.irb2400_joint_gain.rl_cfg import irb2400_joint_gain_ppo_runner_cfg
from mjlab_irb2400.irb2400_joint_gain_ff.env_cfgs import irb2400_joint_gain_ff_env_cfg
from mjlab_irb2400.irb2400_joint_gain_ff.rl_cfg import irb2400_joint_gain_ff_ppo_runner_cfg
from mjlab_irb2400.irb2400_pdff.env_cfgs import irb2400_pdff_env_cfg
from mjlab_irb2400.irb2400_pdff.rl_cfg import irb2400_pdff_ppo_runner_cfg
from mjlab_irb2400.irb2400_pdff.env_cfgs import irb2400_pdff_coarse_dt_env_cfg
from mjlab_irb2400.irb2400_pdff.rl_cfg_coarse import irb2400_pdff_coarse_dt_ppo_runner_cfg
from mjlab_irb2400.irb2400_pdidff.env_cfgs import irb2400_pdidff_env_cfg
from mjlab_irb2400.irb2400_pdidff.rl_cfg import irb2400_pdidff_ppo_runner_cfg
from mjlab_irb2400.irb2400_torque.env_cfgs import irb2400_torque_env_cfg
from mjlab_irb2400.irb2400_torque.rl_cfg import irb2400_torque_ppo_runner_cfg

TASK_ID_GAIN = "Mjlab-JointGain-ABB-IRB2400"
TASK_ID_GAIN_FF = "Mjlab-JointGainFF-ABB-IRB2400"
TASK_ID_PDFF = "Mjlab-PDFF-ABB-IRB2400"
TASK_ID_PDFF_COARSE = "Mjlab-PDFF-CoarseDt-ABB-IRB2400"
TASK_ID_PDIDFF = "Mjlab-PDIDFF-ABB-IRB2400"
TASK_ID_TORQUE = "Mjlab-Torque-ABB-IRB2400"


def register() -> None:
    tasks = set(list_tasks())
    if TASK_ID_GAIN not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_GAIN,
            env_cfg=irb2400_joint_gain_env_cfg(),
            play_env_cfg=irb2400_joint_gain_env_cfg(play=True),
            rl_cfg=irb2400_joint_gain_ppo_runner_cfg(),
            runner_cls=None,
        )
    if TASK_ID_GAIN_FF not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_GAIN_FF,
            env_cfg=irb2400_joint_gain_ff_env_cfg(),
            play_env_cfg=irb2400_joint_gain_ff_env_cfg(play=True),
            rl_cfg=irb2400_joint_gain_ff_ppo_runner_cfg(),
            runner_cls=None,
        )
    if TASK_ID_PDFF not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_PDFF,
            env_cfg=irb2400_pdff_env_cfg(),
            play_env_cfg=irb2400_pdff_env_cfg(play=True),
            rl_cfg=irb2400_pdff_ppo_runner_cfg(),
            runner_cls=None,
        )
    if TASK_ID_PDFF_COARSE not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_PDFF_COARSE,
            env_cfg=irb2400_pdff_coarse_dt_env_cfg(),
            play_env_cfg=irb2400_pdff_coarse_dt_env_cfg(play=True),
            rl_cfg=irb2400_pdff_coarse_dt_ppo_runner_cfg(),
            runner_cls=None,
        )
    if TASK_ID_PDIDFF not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_PDIDFF,
            env_cfg=irb2400_pdidff_env_cfg(),
            play_env_cfg=irb2400_pdidff_env_cfg(play=True),
            rl_cfg=irb2400_pdidff_ppo_runner_cfg(),
            runner_cls=None,
        )
    if TASK_ID_TORQUE not in tasks:
        register_mjlab_task(
            task_id=TASK_ID_TORQUE,
            env_cfg=irb2400_torque_env_cfg(),
            play_env_cfg=irb2400_torque_env_cfg(play=True),
            rl_cfg=irb2400_torque_ppo_runner_cfg(),
            runner_cls=None,
        )


register()
