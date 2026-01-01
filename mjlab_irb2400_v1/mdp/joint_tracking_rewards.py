from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import warnings

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
    frictionloss: float = 2.0,
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
    asset = env.scene[asset_cfg.name]
    j = asset_cfg.joint_ids
    acc = torch.clamp(asset.data.joint_acc[:, j], -float(clip), float(clip))
    return torch.sum(acc * acc, dim=-1)


def ee_pos_error_l2_from_joint_command(
    env: "ManagerBasedRlEnv",
    *,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    site_name: str = "ee",
) -> torch.Tensor:
    """End-effector position MSE (m^2) w.r.t. the reference joint-position command."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    q = asset.data.joint_pos
    q0 = asset.data.default_joint_pos
    if q0 is None:
        raise RuntimeError("default_joint_pos is required for EE tracking reward.")

    cache = getattr(ee_pos_error_l2_from_joint_command, "_cache", None)
    if cache is None:
        cache = {}
        setattr(ee_pos_error_l2_from_joint_command, "_cache", cache)

    cache_key = (id(asset), str(site_name))
    ee_site_id = cache.get(cache_key)
    if ee_site_id is None:
        site_ids, _ = asset.find_sites(str(site_name), preserve_order=True)
        if not site_ids:
            raise RuntimeError(f"Could not find site named '{site_name}' on entity '{asset_cfg.name}'.")
        if len(site_ids) != 1:
            raise RuntimeError(f"Expected exactly one '{site_name}' site, got {site_ids}")
        ee_site_id = int(site_ids[0])
        cache[cache_key] = ee_site_id

    data = asset.data.data
    model = asset.data.model
    joint_q_adr = asset.data.indexing.joint_q_adr

    try:
        import mujoco_warp as mjwarp  # type: ignore
    except Exception:
        warnings.warn(
            "mujoco_warp not available; EE reward falls back to joint_pos_error_l2.",
            stacklevel=2,
        )
        return joint_pos_error_l2(env, command_name=command_name, asset_cfg=asset_cfg)

    model_struct = getattr(model, "struct", model)
    data_struct = getattr(data, "struct", data)

    def _mjwarp_update_kinematics() -> None:
        if hasattr(mjwarp, "kinematics"):
            mjwarp.kinematics(model_struct, data_struct)
        else:  # pragma: no cover
            mjwarp.fwd_position(model_struct, data_struct)

    # Current EE position (world).
    ee_pos = asset.data.site_pos_w[:, ee_site_id, :]

    # Compute reference FK at q_ref = q0 + cmd by temporarily overwriting qpos.
    q_ref = q0 + cmd
    qpos0 = data.qpos[:, joint_q_adr].clone()
    data.qpos[:, joint_q_adr] = q_ref
    _mjwarp_update_kinematics()
    ee_ref = asset.data.site_pos_w[:, ee_site_id, :].clone()

    # Restore actual qpos and kinematics (avoid leaving derived state inconsistent).
    data.qpos[:, joint_q_adr] = q
    _mjwarp_update_kinematics()

    err = ee_pos - ee_ref
    return torch.mean(err * err, dim=-1)


def track_ee_pos_from_joint_command(
    env: "ManagerBasedRlEnv",
    *,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
    site_name: str = "ee",
) -> torch.Tensor:
    """Exponential EE tracking reward in (0,1], more sensitive near zero."""
    mse = ee_pos_error_l2_from_joint_command(
        env, command_name=command_name, asset_cfg=asset_cfg, site_name=site_name
    )
    return torch.exp(-mse / (float(std) * float(std)))


def time_out(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length
