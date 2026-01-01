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


class JointPdGainFfActionV1(ActionTerm):
    """V1: variable-gain PD + inverse-dynamics bias + direct torque feedforward.

    Policy action:
    - default: [kp_raw (N), kd_raw (N), tau_ff_raw (N)] in [-1,1].
    - with integral enabled: [kp_raw (N), kd_raw (N), ki_raw (N), tau_ff_raw (N)] in [-1,1].
    """

    cfg: "JointPdGainFfActionV1Cfg"

    def __init__(self, cfg: "JointPdGainFfActionV1Cfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        self._asset: Entity
        joint_ids, joint_names = self._asset.find_joints_by_actuator_names(cfg.actuator_names)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._joint_names = joint_names
        self._num_joints = len(joint_ids)

        self._use_integral = bool(getattr(cfg, "use_integral", False))
        self._action_dim = (4 if self._use_integral else 3) * self._num_joints
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_cmd = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._prev_tau_ff = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._tau_acc_hold = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._physics_step = 0
        self._warned_acc = False
        self._i_state = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._prev_tau_i = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._acc_update_steps = 1
        if float(getattr(cfg, "acc_update_period_s", 0.0)) > 0.0:
            self._acc_update_steps = max(
                1, int(round(float(cfg.acc_update_period_s) / float(env.physics_dt)))
            )

        self._pd_act: IdealPdActuator | None = None
        for act in self._asset.actuators:
            if isinstance(act, IdealPdActuator):
                self._pd_act = act
                break
        if self._pd_act is None:
            raise ValueError("JointPdGainFfActionV1 requires IdealPdActuator on the entity.")

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
            self._prev_tau_ff.zero_()
            self._i_state.zero_()
            self._prev_tau_i.zero_()
        else:
            self._prev_cmd[env_ids] = cmd[env_ids][:, self._joint_ids]
            self._prev_tau_ff[env_ids] = 0.0
            self._tau_acc_hold[env_ids] = 0.0
            self._i_state[env_ids] = 0.0
            self._prev_tau_i[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        self._physics_step += 1
        # Targets from command.
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

        a = torch.clamp(self._raw_actions, -1.0, 1.0)
        kp_raw = a[:, : self._num_joints]
        kd_raw = a[:, self._num_joints : 2 * self._num_joints]
        if self._use_integral:
            ki_raw = a[:, 2 * self._num_joints : 3 * self._num_joints]
            tau_raw = a[:, 3 * self._num_joints :]
        else:
            ki_raw = None
            tau_raw = a[:, 2 * self._num_joints :]

        kp = torch.clamp(kp_raw * self.cfg.kp_scale + self.cfg.kp_offset, *self.cfg.clip_kp)
        kd = torch.clamp(kd_raw * self.cfg.kd_scale + self.cfg.kd_offset, *self.cfg.clip_kd)
        assert self._pd_act is not None
        self._pd_act.set_gains(slice(None), kp=kp, kd=kd)

        tau_i = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        if self._use_integral and ki_raw is not None:
            ki = torch.clamp(
                ki_raw * float(getattr(self.cfg, "ki_scale")) + float(getattr(self.cfg, "ki_offset")),
                *getattr(self.cfg, "clip_ki"),
            )
            q = self._asset.data.joint_pos[:, self._joint_ids]
            e = pos_target - q
            dt_i = float(self._env.physics_dt)
            leak = float(getattr(self.cfg, "i_leak", 0.0))
            if leak > 0.0:
                self._i_state = self._i_state * max(0.0, 1.0 - leak * dt_i)
            self._i_state = self._i_state + e * dt_i
            i_lim = float(getattr(self.cfg, "i_limit", 0.0))
            if i_lim > 0.0:
                self._i_state = torch.clamp(self._i_state, -i_lim, i_lim)

            tau_i = ki * self._i_state
            tau_i_limit = float(getattr(self.cfg, "tau_i_limit", 0.0))
            if tau_i_limit > 0.0:
                tau_i = torch.clamp(tau_i, -tau_i_limit, tau_i_limit)

            # Optional slew-rate limiting for integral torque to avoid sudden kick.
            tau_i_slew = float(getattr(self.cfg, "tau_i_slew_rate", 0.0))
            if tau_i_slew > 0.0:
                max_delta_i = tau_i_slew * float(self._env.physics_dt)
                delta_i = torch.clamp(tau_i - self._prev_tau_i, -max_delta_i, max_delta_i)
                tau_i = self._prev_tau_i + delta_i
            self._prev_tau_i = tau_i.detach()

        # Direct torque feedforward from policy (slew-rate limited).
        tau_ff = tau_raw * float(self.cfg.tau_scale)
        tau_ff = torch.clamp(tau_ff, -float(self.cfg.tau_limit), float(self.cfg.tau_limit))
        max_delta = float(self.cfg.tau_slew_rate) * float(self._env.physics_dt)
        if max_delta > 0.0:
            delta = torch.clamp(tau_ff - self._prev_tau_ff, -max_delta, max_delta)
            tau_ff = self._prev_tau_ff + delta
        self._prev_tau_ff = tau_ff.detach()

        tau_id = self._bias_feedforward()

        # Acceleration feedforward (M(q) qdd_ref) computed at a lower rate and held (e.g. 10ms).
        tau_acc = torch.zeros_like(tau_id)
        use_acc_ff = bool(getattr(self.cfg, "use_acc_feedforward", False))
        acc_scale = float(getattr(self.cfg, "acc_scale", 0.0))
        if use_acc_ff and abs(acc_scale) > 0.0:
            if (self._physics_step % self._acc_update_steps) == 0:
                qdd_ref = None
                if hasattr(cmd_term, "command_acc"):
                    qdd_ref = getattr(cmd_term, "command_acc")[:, self._joint_ids]
                self._tau_acc_hold = self._acc_feedforward(qdd_ref).detach()
            tau_acc = self._tau_acc_hold

        self._asset.set_joint_effort_target(tau_id + tau_acc + tau_i + tau_ff, joint_ids=self._joint_ids)

    def _bias_feedforward(self) -> torch.Tensor:
        if not self.cfg.use_inverse_dynamics:
            return torch.zeros((self.num_envs, self._num_joints), device=self.device)

        data = self._asset.data.data
        idx_v = self._asset.data.indexing.joint_v_adr[self._joint_ids]

        # Fast path: MuJoCo forward pass already updates bias forces (qfrc_bias) every physics step,
        # so we can use it directly and keep inverse dynamics in the 2ms loop without extra kernels.
        qfrc_bias = getattr(data, "qfrc_bias", None)
        if qfrc_bias is not None:
            tau = qfrc_bias[:, idx_v]
        else:
            # Fallback for older/odd builds where qfrc_bias isn't exposed.
            tau = torch.zeros((self.num_envs, self._num_joints), device=self.device)

        tau = tau * float(self.cfg.id_scale)
        tau = torch.clamp(tau, -float(self.cfg.id_limit), float(self.cfg.id_limit))
        return tau

    def _acc_feedforward(self, qdd_ref: torch.Tensor | None) -> torch.Tensor:
        if qdd_ref is None:
            return torch.zeros((self.num_envs, self._num_joints), device=self.device)

        data = self._asset.data.data
        model = self._asset.data.model
        idx_v = self._asset.data.indexing.joint_v_adr[self._joint_ids]

        tau_acc = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        try:
            import mujoco_warp as mjwarp  # type: ignore

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

                if hasattr(mjwarp, "inverse"):
                    mjwarp.inverse(model_struct, data_struct)
                elif hasattr(mjwarp, "rne"):
                    try:
                        mjwarp.rne(model_struct, data_struct, flg_acc=True)
                    except TypeError:
                        mjwarp.rne(model_struct, data_struct)

                qfrc_inverse = getattr(data, "qfrc_inverse", None)
                qfrc_bias = getattr(data, "qfrc_bias", None)
                if qfrc_inverse is not None and qfrc_bias is not None:
                    tau_acc = qfrc_inverse[:, idx_v] - qfrc_bias[:, idx_v]
            finally:
                data.qacc[:, idx_v] = qacc0
        except ModuleNotFoundError:
            if not self._warned_acc:
                warnings.warn(
                    "JointPdGainFfActionV1: mujoco_warp not available; acc feedforward disabled.",
                    stacklevel=2,
                )
                self._warned_acc = True
            tau_acc.zero_()
        except Exception as e:
            if not self._warned_acc:
                warnings.warn(
                    f"JointPdGainFfActionV1: acc feedforward failed ({type(e).__name__}: {e}); disabled.",
                    stacklevel=2,
                )
                self._warned_acc = True
            tau_acc.zero_()

        tau_acc = tau_acc * float(getattr(self.cfg, "acc_scale", 1.0))
        tau_acc = torch.clamp(tau_acc, -float(getattr(self.cfg, "acc_limit", 0.0)), float(getattr(self.cfg, "acc_limit", 0.0)))
        return tau_acc


@dataclass(kw_only=True)
class JointPdGainFfActionV1Cfg(ActionTermCfg):
    asset_name: str
    actuator_names: tuple[str, ...]
    command_name: str

    kp_scale: float
    kp_offset: float
    kd_scale: float
    kd_offset: float
    clip_kp: tuple[float, float]
    clip_kd: tuple[float, float]

    tau_scale: float
    tau_limit: float
    tau_slew_rate: float

    use_inverse_dynamics: bool = True
    id_scale: float = 1.0
    id_limit: float = 800.0

    # Optional integral action (policy outputs ki per joint).
    use_integral: bool = False
    ki_scale: float = 200.0
    ki_offset: float = 0.0
    clip_ki: tuple[float, float] = (0.0, 200.0)
    i_limit: float = 0.2  # clamp on integral state (rad*s)
    i_leak: float = 0.0   # leaky integrator (1/s)
    tau_i_limit: float = 200.0  # clamp on integral torque (Nm)
    tau_i_slew_rate: float = 0.0  # Nm/s, 0 disables slew limiting

    # Acceleration feedforward (M(q) qdd_ref) computed via inverse dynamics and held.
    use_acc_feedforward: bool = False
    acc_update_period_s: float = 0.01
    acc_scale: float = 1.0
    acc_limit: float = 800.0

    class_type: type[ActionTerm] = JointPdGainFfActionV1
