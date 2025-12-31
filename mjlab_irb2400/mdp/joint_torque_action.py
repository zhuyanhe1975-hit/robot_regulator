from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class JointTorqueAction(ActionTerm):
    cfg: "JointTorqueActionCfg"

    def __init__(self, cfg: "JointTorqueActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        self._asset: Entity
        joint_ids, _joint_names = self._asset.find_joints_by_actuator_names(cfg.actuator_names)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._num_joints = len(joint_ids)
        self._action_dim = self._num_joints
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        # Normalized torque action in [-1, 1] -> tau in [-limit, limit].
        a = torch.clamp(self._raw_actions, -1.0, 1.0)
        tau = a * float(self.cfg.torque_limit)
        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)


@dataclass(kw_only=True)
class JointTorqueActionCfg(ActionTermCfg):
    asset_name: str
    actuator_names: tuple[str, ...]
    torque_limit: float = 200.0

    class_type: type[ActionTerm] = JointTorqueAction
