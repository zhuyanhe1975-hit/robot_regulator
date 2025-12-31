from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.actuator.pd_actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def irb2400_entity_cfg(
    *,
    get_spec,
    stiffness: float = 250.0,
    damping: float = 40.0,
    effort_limit: float = 800.0,
    frictionloss: float = 0.0,
) -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                IdealPdActuatorCfg(
                    joint_names_expr=(
                        "joint_1",
                        "joint_2",
                        "joint_3",
                        "joint_4",
                        "joint_5",
                        "joint_6",
                    ),
                    stiffness=float(stiffness),
                    damping=float(damping),
                    effort_limit=float(effort_limit),
                    armature=0.05,
                    frictionloss=float(frictionloss),
                ),
            ),
            soft_joint_pos_limit_factor=0.98,
        ),
    )


def track_joint_pos(
    env: "ManagerBasedRlEnv",
    *,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    j = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, j]
    q0 = asset.data.default_joint_pos[:, j]
    e = (q - q0) - cmd[:, j]
    return torch.exp(-torch.mean(e * e, dim=-1) / (std * std))


def joint_pos_error_l2(
    env: "ManagerBasedRlEnv",
    *,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    j = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, j]
    q0 = asset.data.default_joint_pos[:, j]
    e = (q - q0) - cmd[:, j]
    return torch.mean(e * e, dim=-1)


def joint_acc_l2_clipped(
    env: "ManagerBasedRlEnv",
    *,
    asset_cfg: SceneEntityCfg,
    clip: float = 50.0,
) -> torch.Tensor:
    """Penalize joint accelerations with clipping to avoid reward-scale explosion.

    MJWarp joint_acc can include large spikes due to constraints; clipping makes PPO stable.
    """
    asset = env.scene[asset_cfg.name]
    j = asset_cfg.joint_ids
    acc = torch.clamp(asset.data.joint_acc[:, j], -float(clip), float(clip))
    return torch.sum(acc * acc, dim=-1)


def held_action_rate_l2(
    env: "ManagerBasedRlEnv",
    *,
    term_name: str,
) -> torch.Tensor:
    """Penalize rate of change of the *applied* (held) action.

    Useful when the policy outputs actions at env.step rate but the controller
    holds/updates them at a lower rate (e.g. 100Hz).
    """
    term = env.action_manager.get_term(term_name)
    held = getattr(term, "held_action", None)
    prev = getattr(term, "prev_held_action", None)
    updated = getattr(term, "updated_in_env_step", None)
    if held is None or prev is None:
        return torch.zeros((env.num_envs,), device=env.device)
    penalty = torch.sum(torch.square(held - prev), dim=1)
    # Only penalize when the held action actually updated during this env step.
    if updated is not None:
        penalty = penalty * updated
    return penalty


def time_out(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length
