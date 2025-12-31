from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.pd_actuator import IdealPdActuator
from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class JointCtResAction(ActionTerm):
    """Computed-torque (M*qdd_ref + bias) + fixed-gain PD + residual torque action.

    - Physics/control step: 2ms (env.physics_dt).
    - Residual action is held with ZOH at update_period_s (e.g. 20ms).
    - qdd_ref is taken from the command term (analytic for sine command).
    """

    cfg: "JointCtResActionCfg"

    def __init__(self, cfg: "JointCtResActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        self._asset: Entity
        joint_ids, joint_names = self._asset.find_joints_by_actuator_names(cfg.actuator_names)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._joint_names = joint_names
        self._num_joints = len(joint_ids)

        self._action_dim = self._num_joints
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        self._held_raw = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._prev_tau_res = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._step_counter = 0
        self._warned_id = False

        self._update_steps = 1
        if float(cfg.update_period_s) > 0.0:
            self._update_steps = max(1, int(round(float(cfg.update_period_s) / float(env.physics_dt))))

        self._pd_act: IdealPdActuator | None = None
        for act in self._asset.actuators:
            if isinstance(act, IdealPdActuator):
                self._pd_act = act
                break
        if self._pd_act is None:
            raise ValueError("JointCtResAction requires IdealPdActuator on the entity.")

        kp = torch.full((self.num_envs, self._num_joints), float(cfg.kp), device=self.device)
        kd = torch.full((self.num_envs, self._num_joints), float(cfg.kd), device=self.device)
        self._pd_act.set_gains(slice(None), kp=kp, kd=kd)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        if isinstance(env_ids, slice) and env_ids == slice(None):
            self._held_raw.zero_()
            self._prev_tau_res.zero_()
        else:
            self._held_raw[env_ids] = 0.0
            self._prev_tau_res[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        cmd_term = self._env.command_manager.get_term(self.cfg.command_name)
        cmd = cmd_term.command
        default_joint_pos = self._asset.data.default_joint_pos
        assert default_joint_pos is not None

        pos_target = default_joint_pos[:, self._joint_ids] + cmd[:, self._joint_ids]
        self._asset.set_joint_position_target(pos_target, joint_ids=self._joint_ids)

        if hasattr(cmd_term, "command_vel"):
            vel_target = getattr(cmd_term, "command_vel")[:, self._joint_ids]
        else:  # pragma: no cover
            vel_target = torch.zeros_like(pos_target)
        self._asset.set_joint_velocity_target(vel_target, joint_ids=self._joint_ids)

        qdd_ref = None
        if hasattr(cmd_term, "command_acc"):
            qdd_ref = getattr(cmd_term, "command_acc")[:, self._joint_ids]

        # Hold residual action at update_period_s.
        if (self._step_counter % self._update_steps) == 0:
            self._held_raw = torch.clamp(self._raw_actions, -1.0, 1.0)
        self._step_counter += 1

        tau_res = self._held_raw * float(self.cfg.tau_res_limit)
        max_delta = float(self.cfg.tau_res_slew_rate) * float(self._env.physics_dt)
        if max_delta > 0.0:
            delta = torch.clamp(tau_res - self._prev_tau_res, -max_delta, max_delta)
            tau_res = self._prev_tau_res + delta
        self._prev_tau_res = tau_res.detach()

        tau_ct = self._computed_torque(qdd_ref)
        self._asset.set_joint_effort_target(tau_ct + tau_res, joint_ids=self._joint_ids)

    def _computed_torque(self, qdd_ref: torch.Tensor | None) -> torch.Tensor:
        id_mode = str(getattr(self.cfg, "id_mode", "full")).lower()
        if id_mode not in ("full", "bias", "disabled"):
            raise ValueError(f"Unsupported id_mode='{id_mode}'. Use 'full', 'bias', or 'disabled'.")
        if id_mode == "disabled":
            return torch.zeros((self.num_envs, self._num_joints), device=self.device)

        data = self._asset.data.data
        model = self._asset.data.model
        idx_v = self._asset.data.indexing.joint_v_adr[self._joint_ids]

        qfrc_bias = getattr(data, "qfrc_bias", None)
        if id_mode == "bias" and qfrc_bias is not None:
            tau = qfrc_bias[:, idx_v]
            tau = tau * float(self.cfg.id_scale)
            tau = torch.clamp(tau, -float(self.cfg.id_limit), float(self.cfg.id_limit))
            return tau

        if qdd_ref is None:
            return torch.zeros((self.num_envs, self._num_joints), device=self.device)

        tau = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        try:
            import mujoco_warp as mjwarp  # type: ignore

            # mjlab wraps mujoco_warp Model/Data in WarpBridge; mujoco_warp APIs expect raw structs.
            model_struct = getattr(model, "struct", model)
            data_struct = getattr(data, "struct", data)

            qacc0 = data.qacc[:, idx_v].clone()
            try:
                data.qacc[:, idx_v] = qdd_ref

                if hasattr(mjwarp, "kinematics"):
                    mjwarp.kinematics(model_struct, data_struct)
                if hasattr(mjwarp, "com_pos"):
                    mjwarp.com_pos(model_struct, data_struct)
                if hasattr(mjwarp, "com_vel"):
                    mjwarp.com_vel(model_struct, data_struct)
                # Prefer inverse() if present; rne() signature varies across builds.
                if hasattr(mjwarp, "inverse"):
                    mjwarp.inverse(model_struct, data_struct)
                elif hasattr(mjwarp, "rne"):
                    try:
                        mjwarp.rne(model_struct, data_struct, flg_acc=True)
                    except TypeError:
                        mjwarp.rne(model_struct, data_struct)

                qfrc_inverse = getattr(data, "qfrc_inverse", None)
                if qfrc_inverse is not None:
                    tau = qfrc_inverse[:, idx_v]
                elif qfrc_bias is not None:
                    tau = qfrc_bias[:, idx_v]
                else:
                    raise AttributeError("mujoco_warp data has no qfrc_inverse/qfrc_bias field")
            finally:
                data.qacc[:, idx_v] = qacc0
        except ModuleNotFoundError:
            if not self._warned_id:
                warnings.warn(
                    "JointCtResAction: mujoco_warp not available; computed torque disabled.",
                    stacklevel=2,
                )
                self._warned_id = True
            tau.zero_()
        except Exception as e:
            if not self._warned_id:
                warnings.warn(
                    f"JointCtResAction: computed torque failed ({type(e).__name__}: {e}); disabled.",
                    stacklevel=2,
                )
                self._warned_id = True
            tau.zero_()

        tau = tau * float(self.cfg.id_scale)
        tau = torch.clamp(tau, -float(self.cfg.id_limit), float(self.cfg.id_limit))
        return tau


@dataclass(kw_only=True)
class JointCtResActionCfg(ActionTermCfg):
    asset_name: str
    actuator_names: tuple[str, ...]
    command_name: str

    kp: float
    kd: float

    tau_res_limit: float = 10.0
    tau_res_slew_rate: float = 300.0
    update_period_s: float = 0.02

    # Feedforward mode:
    # - "bias": use qfrc_bias only (cheap, good for 2ms controller)
    # - "full": try full inverse dynamics (qfrc_inverse for qacc=qdd_ref)
    # - "disabled": no computed torque
    id_mode: str = "full"
    id_scale: float = 1.0
    id_limit: float = 800.0

    class_type: type[ActionTerm] = JointCtResAction
