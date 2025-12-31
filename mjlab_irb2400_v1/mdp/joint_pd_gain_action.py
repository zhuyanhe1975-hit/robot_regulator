from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.pd_actuator import IdealPdActuator
from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class JointPdGainAction(ActionTerm):
    """V1 variable-gain PD action term (policy outputs kp/kd per joint)."""

    cfg: "JointPdGainActionCfg"

    def __init__(self, cfg: "JointPdGainActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        self._asset: Entity
        joint_ids, joint_names = self._asset.find_joints_by_actuator_names(cfg.actuator_names)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._joint_names = joint_names

        self._num_joints = len(joint_ids)
        self._action_dim = 2 * self._num_joints
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_cmd = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._pd_act: IdealPdActuator | None = None
        for act in self._asset.actuators:
            if isinstance(act, IdealPdActuator):
                self._pd_act = act
                break
        if self._pd_act is None:
            raise ValueError("JointPdGainAction requires IdealPdActuator on the entity.")

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        cmd = self._env.command_manager.get_command(self.cfg.command_name)
        if isinstance(env_ids, slice) and env_ids == slice(None):
            self._prev_cmd[:] = cmd[:, self._joint_ids]
        else:
            self._prev_cmd[env_ids] = cmd[env_ids][:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        # Set targets.
        cmd_term = self._env.command_manager.get_term(self.cfg.command_name)
        cmd = cmd_term.command
        default_joint_pos = self._asset.data.default_joint_pos
        assert default_joint_pos is not None
        pos_target = default_joint_pos[:, self._joint_ids] + cmd[:, self._joint_ids]
        self._asset.set_joint_position_target(pos_target, joint_ids=self._joint_ids)

        if hasattr(cmd_term, "command_vel"):
            vel_target = getattr(cmd_term, "command_vel")[:, self._joint_ids]
        else:
            dt = float(self._env.step_dt)
            cmd_sel = cmd[:, self._joint_ids]
            vel_target = (cmd_sel - self._prev_cmd) / max(dt, 1e-6)
            self._prev_cmd = cmd_sel.detach()
        self._asset.set_joint_velocity_target(vel_target, joint_ids=self._joint_ids)

        # No feedforward.
        self._asset.set_joint_effort_target(torch.zeros_like(pos_target), joint_ids=self._joint_ids)

        # Apply gains.
        a = torch.clamp(self._raw_actions, -1.0, 1.0)
        kp_raw = a[:, : self._num_joints]
        kd_raw = a[:, self._num_joints :]
        kp = torch.clamp(kp_raw * self.cfg.kp_scale + self.cfg.kp_offset, *self.cfg.clip_kp)
        kd = torch.clamp(kd_raw * self.cfg.kd_scale + self.cfg.kd_offset, *self.cfg.clip_kd)
        assert self._pd_act is not None
        self._pd_act.set_gains(slice(None), kp=kp, kd=kd)


@dataclass(kw_only=True)
class JointPdGainActionCfg(ActionTermCfg):
    asset_name: str
    actuator_names: tuple[str, ...]
    command_name: str

    kp_scale: float
    kp_offset: float
    kd_scale: float
    kd_offset: float
    clip_kp: tuple[float, float] = (0.0, 1000.0)
    clip_kd: tuple[float, float] = (0.0, 200.0)

    class_type: type[ActionTerm] = JointPdGainAction

