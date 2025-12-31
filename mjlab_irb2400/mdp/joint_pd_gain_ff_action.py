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


class JointPdGainFfAction(ActionTerm):
    cfg: "JointPdGainFfActionCfg"

    def __init__(self, cfg: "JointPdGainFfActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        self._asset: Entity
        self._pd_acts: list[IdealPdActuator] = [
            act for act in self._asset.actuators if isinstance(act, IdealPdActuator)
        ]
        if not self._pd_acts:
            raise ValueError(
                "JointPdGainFfAction requires IdealPdActuator on the entity, but none were found."
            )
        if len(self._pd_acts) != 1:
            raise ValueError(
                "JointPdGainFfAction expects exactly one IdealPdActuator controlling all joints."
            )

        # Use actuator joint order as the source of truth. This keeps action/targets/reward aligned.
        self._joint_ids = self._pd_acts[0].joint_ids.to(dtype=torch.long)
        self._joint_names = list(self._pd_acts[0].joint_names)
        self._num_joints = int(self._joint_ids.numel())
        # kp,kd and friction-compensation parameters (a,b) per joint.
        # Residual torque is computed as: tau_res = a*sign(qd) + b*qd.
        self._action_dim = 4 * self._num_joints
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._held_actions = torch.zeros_like(self._raw_actions)
        self._prev_held_actions = torch.zeros_like(self._raw_actions)
        self._prev_cmd = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._prev_tau_res = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._tau_model_hold = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._kp_hold = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._kd_hold = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._physics_step = 0
        self._updated_in_env_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._id_exception_count = 0
        self._id_exception_printed = False

    def _log_metric(self, key: str, value: float | torch.Tensor) -> None:
        log = getattr(self._env, "extras", {}).get("log", None)
        if isinstance(log, dict):
            log[key] = value

    def _action_to_kp_kd(self, held_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.clamp(held_actions, -1.0, 1.0)
        kp_raw = a[:, : self._num_joints]
        kd_raw = a[:, self._num_joints : 2 * self._num_joints]

        # Curriculum: start from baseline gains and gradually allow NN to modulate them.
        gain_scale = self._gain_scale()

        kp = kp_raw * (float(self.cfg.kp_scale) * gain_scale) + float(self.cfg.kp_offset)
        kd = kd_raw * (float(self.cfg.kd_scale) * gain_scale) + float(self.cfg.kd_offset)
        kp = torch.clamp(kp, *self.cfg.clip_kp)
        kd = torch.clamp(kd, *self.cfg.clip_kd)
        return kp, kd

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

    @property
    def friction_scale(self) -> float:
        return self._friction_scale()

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
            self._kp_hold[:] = float(self.cfg.kp_offset)
            self._kd_hold[:] = float(self.cfg.kd_offset)
        else:
            self._prev_cmd[env_ids] = cmd[env_ids][:, self._joint_ids]
            self._prev_tau_res[env_ids] = 0.0
            self._tau_model_hold[env_ids] = 0.0
            self._kp_hold[env_ids] = float(self.cfg.kp_offset)
            self._kd_hold[env_ids] = float(self.cfg.kd_offset)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        self._physics_step += 1
        # Reset per-env-step flag at the start of each env step (first physics substep).
        if ((self._physics_step - 1) % int(self._env.cfg.decimation)) == 0:
            self._updated_in_env_step.zero_()
            self._log_metric(
                "Metrics/gain_curriculum_scale",
                torch.tensor(self._gain_scale(), device=self.device, dtype=torch.float32),
            )
            self._log_metric(
                "Metrics/friction_curriculum_scale",
                torch.tensor(self._friction_scale(), device=self.device, dtype=torch.float32),
            )
            self._log_metric("Metrics/kp_mean", torch.mean(self._kp_hold))
            self._log_metric("Metrics/kp_min", torch.amin(self._kp_hold))
            self._log_metric("Metrics/kp_max", torch.amax(self._kp_hold))
            self._log_metric("Metrics/kd_mean", torch.mean(self._kd_hold))
            self._log_metric("Metrics/kd_min", torch.amin(self._kd_hold))
            self._log_metric("Metrics/kd_max", torch.amax(self._kd_hold))
        update_every = max(1, int(round(float(self.cfg.update_period_s) / float(self._env.physics_dt))))
        if (self._physics_step % update_every) == 0:
            self._prev_held_actions[:] = self._held_actions
            self._held_actions[:] = self._raw_actions
            # Throttle inverse dynamics feedforward to the same update rate (default 100Hz).
            self._tau_model_hold = self._inverse_dynamics_feedforward(
                self._env.command_manager.get_term(self.cfg.command_name), dt=float(self._env.physics_dt)
            )
            self._updated_in_env_step.fill_(1.0)

            # Update gains at NN rate with slew-rate limiting to avoid destabilizing jumps.
            kp_target, kd_target = self._action_to_kp_kd(self._held_actions)
            dt_u = float(self.cfg.update_period_s)
            kp_step = float(self.cfg.kp_slew_rate) * dt_u
            kd_step = float(self.cfg.kd_slew_rate) * dt_u
            if kp_step > 0.0:
                self._kp_hold = self._kp_hold + torch.clamp(kp_target - self._kp_hold, -kp_step, kp_step)
            else:
                self._kp_hold = kp_target
            if kd_step > 0.0:
                self._kd_hold = self._kd_hold + torch.clamp(kd_target - self._kd_hold, -kd_step, kd_step)
            else:
                self._kd_hold = kd_target

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

        # 2) Map actions -> gains and residual friction compensation parameters.
        a = torch.clamp(self._held_actions, -1.0, 1.0)
        a_raw = a[:, 2 * self._num_joints : 3 * self._num_joints]
        b_raw = a[:, 3 * self._num_joints :]

        # Gains are held at the NN update rate (100Hz) to match real controller scheduling.
        kp, kd = self._kp_hold, self._kd_hold

        # Curriculum: ramp in friction compensation so early training learns stable PD+ID tracking.
        scale = self._friction_scale()

        # IMPORTANT: make "action=0" correspond to zero residual, so the initial
        # (near-zero) policy does not inject large torques and destabilize training.
        a_mag = torch.clamp(a_raw, 0.0, 1.0) * (float(self.cfg.a_max) * scale)
        b_mag = torch.clamp(b_raw, 0.0, 1.0) * (float(self.cfg.b_max) * scale)

        qd = self._asset.data.joint_vel[:, self._joint_ids]
        sign = torch.tanh(qd / float(self.cfg.sign_eps))
        tau_res = a_mag * sign + b_mag * qd
        tau_res = torch.clamp(tau_res, -float(self.cfg.tau_limit), float(self.cfg.tau_limit))

        # Limit how fast residual torques can change (prevents high-frequency jitter).
        max_delta = float(self.cfg.tau_slew_rate) * float(self._env.physics_dt)
        if max_delta > 0.0:
            delta = torch.clamp(tau_res - self._prev_tau_res, -max_delta, max_delta)
            tau_res = self._prev_tau_res + delta
        self._prev_tau_res = tau_res.detach()

        self._asset.set_joint_effort_target(self._tau_model_hold + tau_res, joint_ids=self._joint_ids)

        self._pd_acts[0].set_gains(slice(None), kp=kp, kd=kd)

    def _friction_scale(self) -> float:
        # Approximate global timesteps = env_steps * num_envs.
        # This matches the PPO "Total timesteps" scale used in logs.
        global_steps = int(getattr(self._env, "common_step_counter", 0)) * int(self.num_envs)
        warmup = int(self.cfg.friction_warmup_steps)
        ramp = int(self.cfg.friction_ramp_steps)
        if global_steps < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        t = (global_steps - warmup) / float(ramp)
        return float(max(0.0, min(1.0, t)))

    def _gain_scale(self) -> float:
        global_steps = int(getattr(self._env, "common_step_counter", 0)) * int(self.num_envs)
        warmup = int(self.cfg.gain_warmup_steps)
        ramp = int(self.cfg.gain_ramp_steps)
        if global_steps < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        t = (global_steps - warmup) / float(ramp)
        return float(max(0.0, min(1.0, t)))

    def _inverse_dynamics_feedforward(self, cmd_term, *, dt: float) -> torch.Tensor:
        if not self.cfg.use_inverse_dynamics:
            return torch.zeros((self.num_envs, self._num_joints), device=self.device)

        # Desired joint accelerations from command (if available), else numerical diff of vel target.
        if hasattr(cmd_term, "command_acc"):
            qdd_ref = getattr(cmd_term, "command_acc")[:, self._joint_ids]
        else:
            if hasattr(cmd_term, "command_vel"):
                qd = getattr(cmd_term, "command_vel")[:, self._joint_ids]
            else:
                qd = torch.zeros((self.num_envs, self._num_joints), device=self.device)
            qdd_ref = (qd - torch.zeros_like(qd)) / max(dt, 1e-6)

        # If available, compute inverse dynamics at the reference state (q_ref, qd_ref, qdd_ref).
        # This better matches tracking than using current-state bias + M(q)*qdd_ref.
        data = self._asset.data.data
        model = self._asset.data.model
        idx_v = self._asset.data.indexing.joint_v_adr[self._joint_ids]
        idx_q = self._asset.data.indexing.joint_q_adr[self._joint_ids]

        if hasattr(cmd_term, "command_vel"):
            qd_ref = getattr(cmd_term, "command_vel")[:, self._joint_ids]
        else:
            qd_ref = torch.zeros((self.num_envs, self._num_joints), device=self.device)

        tau_id = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        id_exception = False
        try:
            import mujoco_warp as mjwarp  # type: ignore
            # mjlab wraps mujoco_warp Model/Data in WarpBridge; mujoco_warp APIs expect raw structs.
            model_struct = getattr(model, "struct", model)
            data_struct = getattr(data, "struct", data)

            # Snapshot current generalized coordinates/vel/acc and restore after computing ID.
            qacc0 = data.qacc[:, idx_v].clone()
            # Compute inverse dynamics at the *current* state (q,qd) with desired acceleration.
            # This is more robust than evaluating at (q_ref,qd_ref) when tracking error exists.
            data.qacc[:, idx_v] = qdd_ref

            # Update kinematics/COM for ID (avoid fwd_position() which calls camlight()).
            if hasattr(mjwarp, "kinematics"):
                mjwarp.kinematics(model_struct, data_struct)
            if hasattr(mjwarp, "com_pos"):
                mjwarp.com_pos(model_struct, data_struct)
            if hasattr(mjwarp, "com_vel"):
                mjwarp.com_vel(model_struct, data_struct)

            # Prefer full inverse dynamics if available; fall back to RNE.
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

            # Restore state (simulation step will recompute derived fields as needed).
            data.qacc[:, idx_v] = qacc0
        except Exception as e:
            # Fallback to zeros if mujoco_warp is unavailable or any API mismatch occurs.
            tau_id.zero_()
            id_exception = True
            self._id_exception_count += 1
            if not self._id_exception_printed:
                self._id_exception_printed = True
                print(f"[WARN] ID feedforward exception (first): {type(e).__name__}: {e}")

        tau = tau_id * float(self.cfg.id_scale)
        tau = torch.clamp(tau, -float(self.cfg.id_limit), float(self.cfg.id_limit))
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
class JointPdGainFfActionCfg(ActionTermCfg):
    asset_name: str
    actuator_names: tuple[str, ...]
    command_name: str

    kp_scale: float
    kp_offset: float
    kd_scale: float
    kd_offset: float
    clip_kp: tuple[float, float] = (0.0, 1000.0)
    clip_kd: tuple[float, float] = (0.0, 200.0)

    # Residual friction compensation model:
    #   tau_res = a*sign(qd) + b*qd
    # NOTE: These should be on the same order as the plant friction parameters
    # (actuator frictionloss and joint damping), otherwise the policy can easily
    # over-compensate and turn friction into anti-damping.
    a_max: float = 3.0
    b_max: float = 0.5
    sign_eps: float = 0.05
    tau_limit: float = 300.0
    tau_slew_rate: float = 800.0  # Nm/s
    update_period_s: float = 0.01  # NN & inverse dynamics update period (s)
    # With large num_envs, timesteps accumulate very fast; use conservative defaults.
    friction_warmup_steps: int = 200_000_000
    friction_ramp_steps: int = 300_000_000
    gain_warmup_steps: int = 50_000_000
    gain_ramp_steps: int = 100_000_000
    # Slew-rate limits for gain changes at NN update rate (units: gain per second).
    # Setting these >0 prevents abrupt gain jumps that can destabilize the plant.
    kp_slew_rate: float = 20_000.0
    kd_slew_rate: float = 2_000.0
    use_inverse_dynamics: bool = True
    id_scale: float = 1.0
    id_limit: float = 800.0

    class_type: type[ActionTerm] = JointPdGainFfAction
