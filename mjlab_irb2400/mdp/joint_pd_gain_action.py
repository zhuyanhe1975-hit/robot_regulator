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
    cfg: "JointPdGainActionCfg"

    def __init__(self, cfg: "JointPdGainActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        self._asset: Entity
        joint_ids, joint_names = self._asset.find_joints_by_actuator_names(cfg.actuator_names)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._joint_names = joint_names

        self._num_joints = len(joint_ids)
        self._action_dim = 2 * self._num_joints  # kp,kd per joint

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_cmd = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._held_actions = torch.zeros_like(self._raw_actions)
        self._prev_held_actions = torch.zeros_like(self._raw_actions)
        self._physics_step = 0
        self._updated_in_env_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # Find PD actuators controlling these joints.
        self._pd_acts: list[IdealPdActuator] = []
        for act in self._asset.actuators:
            if isinstance(act, IdealPdActuator):
                self._pd_acts.append(act)
        if not self._pd_acts:
            raise ValueError(
                "JointPdGainAction requires IdealPdActuator on the entity, but none were found."
            )
        if len(self._pd_acts) != 1:
            raise ValueError(
                "JointPdGainAction expects exactly one IdealPdActuator controlling all joints."
            )
        if self._pd_acts[0].joint_ids.numel() != self._num_joints:
            raise ValueError(
                "IdealPdActuator joint count does not match action joint count: "
                f"{self._pd_acts[0].joint_ids.numel()} vs {self._num_joints}"
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def held_action(self) -> torch.Tensor:
        return self._held_actions

    @property
    def prev_held_action(self) -> torch.Tensor:
        return self._prev_held_actions

    @property
    def updated_in_env_step(self) -> torch.Tensor:
        return self._updated_in_env_step

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._held_actions[env_ids] = 0.0
        self._prev_held_actions[env_ids] = 0.0
        self._updated_in_env_step[env_ids] = 0.0
        cmd = self._env.command_manager.get_command(self.cfg.command_name)
        if isinstance(env_ids, slice) and env_ids == slice(None):
            self._prev_cmd[:] = cmd[:, self._joint_ids]
        else:
            self._prev_cmd[env_ids] = cmd[env_ids][:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        self._physics_step += 1
        # Reset per-env-step flag at the start of each env step (first physics substep).
        if ((self._physics_step - 1) % int(self._env.cfg.decimation)) == 0:
            self._updated_in_env_step.zero_()
        update_every = max(1, int(round(float(self.cfg.update_period_s) / float(self._env.physics_dt))))
        if (self._physics_step % update_every) == 0:
            self._prev_held_actions[:] = self._held_actions
            self._held_actions[:] = self._raw_actions
            self._updated_in_env_step.fill_(1.0)

        # 1) Set joint position targets from command (reference trajectory).
        cmd_term = self._env.command_manager.get_term(self.cfg.command_name)
        cmd = cmd_term.command
        default_joint_pos = self._asset.data.default_joint_pos
        assert default_joint_pos is not None
        pos_target = default_joint_pos[:, self._joint_ids] + cmd[:, self._joint_ids]
        self._asset.set_joint_position_target(pos_target, joint_ids=self._joint_ids)

        # Prefer command-provided velocity if available (avoids numerical noise).
        if hasattr(cmd_term, "command_vel"):
            vel_cmd = getattr(cmd_term, "command_vel")
            vel_target = vel_cmd[:, self._joint_ids]
        else:
            dt = float(self._env.step_dt)
            cmd_sel = cmd[:, self._joint_ids]
            vel_target = (cmd_sel - self._prev_cmd) / max(dt, 1e-6)
            self._prev_cmd = cmd_sel.detach()
        self._asset.set_joint_velocity_target(vel_target, joint_ids=self._joint_ids)

        # No dynamics feedforward in this task: pure PD with variable gains.
        self._asset.set_joint_effort_target(torch.zeros_like(pos_target), joint_ids=self._joint_ids)

        # 2) Map actions -> gains and set actuator gains.
        a = torch.clamp(self._held_actions, -1.0, 1.0)
        kp_raw = a[:, : self._num_joints]
        kd_raw = a[:, self._num_joints :]

        kp = kp_raw * self.cfg.kp_scale + self.cfg.kp_offset
        kd = kd_raw * self.cfg.kd_scale + self.cfg.kd_offset
        kp = torch.clamp(kp, *self.cfg.clip_kp)
        kd = torch.clamp(kd, *self.cfg.clip_kd)

        self._pd_acts[0].set_gains(slice(None), kp=kp, kd=kd)


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
    update_period_s: float = 0.01  # NN update period (s); PD runs every physics step.

    class_type: type[ActionTerm] = JointPdGainAction
