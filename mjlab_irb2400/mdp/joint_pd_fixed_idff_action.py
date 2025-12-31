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


class JointPdFixedIdFfAction(ActionTerm):
    """Fixed-gain PD + inverse dynamics + learned residual torque.

    - PD runs at physics rate (via IdealPdActuator).
    - Inverse dynamics feedforward is computed/held at `update_period_s`.
    - Policy outputs residual parameters (a,b) per joint:
        tau_res = a*tanh(qd/sign_eps) + b*qd
    """

    cfg: "JointPdFixedIdFfActionCfg"

    def __init__(self, cfg: "JointPdFixedIdFfActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        self._asset: Entity
        self._pd_acts: list[IdealPdActuator] = [
            act for act in self._asset.actuators if isinstance(act, IdealPdActuator)
        ]
        if not self._pd_acts:
            raise ValueError(
                "JointPdFixedIdFfAction requires IdealPdActuator on the entity, but none were found."
            )
        if len(self._pd_acts) != 1:
            raise ValueError(
                "JointPdFixedIdFfAction expects exactly one IdealPdActuator controlling all joints."
            )

        # Use the actuator's joint order as the single source of truth.
        self._joint_ids = self._pd_acts[0].joint_ids.to(dtype=torch.long)
        self._joint_names = list(self._pd_acts[0].joint_names)
        self._num_joints = int(self._joint_ids.numel())
        self._action_dim = 2 * self._num_joints
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._held_actions = torch.zeros_like(self._raw_actions)
        self._prev_held_actions = torch.zeros_like(self._raw_actions)

        self._prev_cmd = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._prev_tau_res = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._tau_model_hold = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._physics_step = 0
        self._updated_in_env_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._id_exception_count = 0
        self._id_exception_printed = False

        # Set fixed PD gains once.
        kp = torch.full((self.num_envs, self._num_joints), float(cfg.kp), device=self.device)
        kd = torch.full((self.num_envs, self._num_joints), float(cfg.kd), device=self.device)
        self._pd_acts[0].set_gains(slice(None), kp=kp, kd=kd)

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

    def _log_metric(self, key: str, value: float | torch.Tensor) -> None:
        log = getattr(self._env, "extras", {}).get("log", None)
        if isinstance(log, dict):
            log[key] = value

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._held_actions[env_ids] = 0.0
        self._prev_held_actions[env_ids] = 0.0
        self._updated_in_env_step[env_ids] = 0.0
        cmd = self._env.command_manager.get_command(self.cfg.command_name)
        if isinstance(env_ids, slice) and env_ids == slice(None):
            self._prev_cmd[:] = cmd[:, self._joint_ids]
            self._prev_tau_res[:] = 0.0
            self._tau_model_hold[:] = 0.0
        else:
            self._prev_cmd[env_ids] = cmd[env_ids][:, self._joint_ids]
            self._prev_tau_res[env_ids] = 0.0
            self._tau_model_hold[env_ids] = 0.0

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
            self._tau_model_hold = self._inverse_dynamics_feedforward(
                self._env.command_manager.get_term(self.cfg.command_name), dt=float(self._env.physics_dt)
            )
            self._updated_in_env_step.fill_(1.0)

        # 1) Set joint position targets from command (reference trajectory).
        cmd_term = self._env.command_manager.get_term(self.cfg.command_name)
        cmd = cmd_term.command
        default_joint_pos = self._asset.data.default_joint_pos
        assert default_joint_pos is not None
        pos_target = default_joint_pos[:, self._joint_ids] + cmd[:, self._joint_ids]
        self._asset.set_joint_position_target(pos_target, joint_ids=self._joint_ids)

        dt = float(self._env.step_dt)
        if hasattr(cmd_term, "command_vel"):
            vel_cmd = getattr(cmd_term, "command_vel")
            vel_target = vel_cmd[:, self._joint_ids]
        else:
            cmd_sel = cmd[:, self._joint_ids]
            vel_target = (cmd_sel - self._prev_cmd) / max(dt, 1e-6)
            self._prev_cmd = cmd_sel.detach()
        self._asset.set_joint_velocity_target(vel_target, joint_ids=self._joint_ids)

        # 2) Map actions -> residual friction compensation parameters.
        a = torch.clamp(self._held_actions, -1.0, 1.0)
        a_raw = a[:, : self._num_joints]
        b_raw = a[:, self._num_joints :]

        # IMPORTANT: make "action=0" correspond to zero residual, so the initial
        # (near-zero) policy does not inject large torques and destabilize training.
        a_mag = torch.clamp(a_raw, 0.0, 1.0) * float(self.cfg.a_max)
        b_mag = torch.clamp(b_raw, 0.0, 1.0) * float(self.cfg.b_max)

        qd = self._asset.data.joint_vel[:, self._joint_ids]
        sign = torch.tanh(qd / float(self.cfg.sign_eps))
        tau_res = a_mag * sign + b_mag * qd
        tau_res = torch.clamp(tau_res, -float(self.cfg.tau_limit), float(self.cfg.tau_limit))

        # Limit how fast residual torques can change.
        max_delta = float(self.cfg.tau_slew_rate) * float(self._env.physics_dt)
        if max_delta > 0.0:
            delta = torch.clamp(tau_res - self._prev_tau_res, -max_delta, max_delta)
            tau_res = self._prev_tau_res + delta
        self._prev_tau_res = tau_res.detach()

        self._asset.set_joint_effort_target(self._tau_model_hold + tau_res, joint_ids=self._joint_ids)

        # Optional debug metrics.
        if ((self._physics_step - 1) % int(self._env.cfg.decimation)) == 0:
            self._log_metric("Metrics/tau_res_rms", torch.sqrt(torch.mean(tau_res * tau_res)))
            self._log_metric("Metrics/tau_id_rms", torch.sqrt(torch.mean(self._tau_model_hold * self._tau_model_hold)))

    def _inverse_dynamics_feedforward(self, cmd_term, *, dt: float) -> torch.Tensor:
        if not self.cfg.use_inverse_dynamics:
            return torch.zeros((self.num_envs, self._num_joints), device=self.device)

        if hasattr(cmd_term, "command_acc"):
            qdd_ref = getattr(cmd_term, "command_acc")[:, self._joint_ids]
        else:
            qdd_ref = torch.zeros((self.num_envs, self._num_joints), device=self.device)

        data = self._asset.data.data
        model = self._asset.data.model
        idx_v = self._asset.data.indexing.joint_v_adr[self._joint_ids]
        idx_q = self._asset.data.indexing.joint_q_adr[self._joint_ids]

        if hasattr(cmd_term, "command_vel"):
            qd_ref = getattr(cmd_term, "command_vel")[:, self._joint_ids]
        else:
            qd_ref = torch.zeros((self.num_envs, self._num_joints), device=self.device)

        default_joint_pos = self._asset.data.default_joint_pos
        assert default_joint_pos is not None
        q_ref = default_joint_pos[:, self._joint_ids] + cmd_term.command[:, self._joint_ids]

        tau_id = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        id_exception = False
        try:
            import mujoco_warp as mjwarp  # type: ignore

            # mjlab wraps mujoco_warp Model/Data in WarpBridge; mujoco_warp APIs expect raw structs.
            model_struct = getattr(model, "struct", model)
            data_struct = getattr(data, "struct", data)

            id_mode = str(getattr(self.cfg, "id_mode", "full")).lower()
            if id_mode not in ("full", "gravity"):
                raise ValueError(f"Unsupported id_mode='{id_mode}'. Use 'full' or 'gravity'.")

            qvel0 = data.qvel[:, idx_v].clone()
            qacc0 = data.qacc[:, idx_v].clone()

            if id_mode == "gravity":
                # Gravity-only: evaluate bias with qvel=0, qacc=0 so Coriolis/centrifugal vanish.
                data.qvel[:, idx_v] = 0.0
                data.qacc[:, idx_v] = 0.0
                if hasattr(mjwarp, "kinematics"):
                    mjwarp.kinematics(model_struct, data_struct)
                if hasattr(mjwarp, "com_pos"):
                    mjwarp.com_pos(model_struct, data_struct)
                if hasattr(mjwarp, "rne"):
                    try:
                        mjwarp.rne(model_struct, data_struct, flg_acc=True)
                    except TypeError:
                        mjwarp.rne(model_struct, data_struct)
                qfrc_bias = getattr(data, "qfrc_bias", None)
                if qfrc_bias is not None:
                    tau_id = qfrc_bias[:, idx_v]
            else:
                # Full inverse dynamics at the *current* state (q,qd) with desired acceleration.
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
                if qfrc_inverse is not None:
                    tau_id = qfrc_inverse[:, idx_v]
                else:
                    qfrc_bias = getattr(data, "qfrc_bias", None)
                    if qfrc_bias is not None:
                        tau_id = qfrc_bias[:, idx_v]

            data.qvel[:, idx_v] = qvel0
            data.qacc[:, idx_v] = qacc0
        except Exception as e:
            tau_id.zero_()
            id_exception = True
            self._id_exception_count += 1
            if not self._id_exception_printed:
                self._id_exception_printed = True
                print(f"[WARN] ID feedforward exception (first): {type(e).__name__}: {e}")

        tau = tau_id * float(self.cfg.id_scale)
        tau = torch.clamp(tau, -float(self.cfg.id_limit), float(self.cfg.id_limit))
        # Log whether ID succeeded (1) or hit an exception (0).
        if ((self._physics_step - 1) % int(self._env.cfg.decimation)) == 0:
            self._log_metric(
                "Metrics/id_ok",
                torch.tensor(0.0 if id_exception else 1.0, device=self.device, dtype=torch.float32),
            )
            self._log_metric(
                "Metrics/id_exception_count",
                torch.tensor(float(self._id_exception_count), device=self.device, dtype=torch.float32),
            )
        return tau


@dataclass(kw_only=True)
class JointPdFixedIdFfActionCfg(ActionTermCfg):
    asset_name: str
    actuator_names: tuple[str, ...]
    command_name: str

    kp: float
    kd: float

    a_max: float = 3.0
    b_max: float = 0.5
    sign_eps: float = 0.05
    tau_limit: float = 200.0
    tau_slew_rate: float = 800.0  # Nm/s
    update_period_s: float = 0.02  # controller/NN update period (s)

    use_inverse_dynamics: bool = True
    # ID mode:
    # - "gravity": gravity compensation only (bias at q with qvel=0, qacc=0)
    # - "full": full inverse dynamics (default)
    id_mode: str = "full"
    id_scale: float = 1.0
    id_limit: float = 800.0

    class_type: type[ActionTerm] = JointPdFixedIdFfAction
