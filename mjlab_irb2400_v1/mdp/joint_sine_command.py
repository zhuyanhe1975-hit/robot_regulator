from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class RandomSineJointPositionCommand(CommandTerm):
    cfg: "RandomSineJointPositionCommandCfg"

    def __init__(self, cfg: "RandomSineJointPositionCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]

        joint_ids, _ = self.robot.find_joints(cfg.joint_names)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._num_joints = len(joint_ids)

        self._amp = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._freq = torch.zeros_like(self._amp)
        self._phase = torch.zeros_like(self._amp)
        self._t = torch.zeros(self.num_envs, device=self.device)
        self._cmd = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self._cmd_vel = torch.zeros_like(self._cmd)
        self._cmd_acc = torch.zeros_like(self._cmd)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    @property
    def command_vel(self) -> torch.Tensor:
        return self._cmd_vel

    @property
    def command_acc(self) -> torch.Tensor:
        return self._cmd_acc

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        # Base class does metrics reset + resample.
        extras = super().reset(env_ids)
        assert isinstance(env_ids, torch.Tensor)
        # Initialize command buffers at t=0 so the first step doesn't briefly see stale commands.
        self._write_command(env_ids)
        # Optional warm start: align robot joint state to the command start.
        if bool(getattr(self.cfg, "warm_start_reset", False)):
            self._warm_start_robot(
                env_ids, use_command_vel=bool(getattr(self.cfg, "warm_start_use_command_vel", True))
            )
            # For interactive viewing, keep derived quantities in sync immediately (avoids 1-frame transient).
            if bool(getattr(self.cfg, "warm_start_forward", False)):
                self._env.scene.write_data_to_sim()
                self._env.sim.forward()
        return extras

    def _update_metrics(self) -> None:
        return

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty((len(env_ids), self._num_joints), device=self.device)
        self._amp[env_ids] = r.uniform_(*self.cfg.amp_range)
        self._freq[env_ids] = r.uniform_(*self.cfg.freq_range)
        self._phase[env_ids] = r.uniform_(*self.cfg.phase_range)
        self._t[env_ids] = 0.0

    def _warm_start_robot(self, env_ids: torch.Tensor, *, use_command_vel: bool) -> None:
        default_joint_pos = self.robot.data.default_joint_pos
        assert default_joint_pos is not None
        default_joint_vel = self.robot.data.default_joint_vel
        assert default_joint_vel is not None
        soft_limits = self.robot.data.soft_joint_pos_limits
        assert soft_limits is not None

        j = self._joint_ids
        q = default_joint_pos[env_ids][:, j] + self._cmd[env_ids][:, j]
        lim = soft_limits[env_ids][:, j]
        q = q.clamp_(lim[..., 0], lim[..., 1])

        if use_command_vel:
            qd = self._cmd_vel[env_ids][:, j]
        else:
            qd = default_joint_vel[env_ids][:, j].clone()
            qd.zero_()

        self.robot.write_joint_state_to_sim(q, qd, joint_ids=j, env_ids=env_ids)

    def _write_command(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        t = self._t[env_ids]
        phi = 2.0 * torch.pi * self._freq[env_ids] * t[:, None] + self._phase[env_ids]
        w = 2.0 * torch.pi * self._freq[env_ids]
        q_sin = self._amp[env_ids] * torch.sin(phi)
        qd_sin = self._amp[env_ids] * w * torch.cos(phi)
        qdd_sin = -self._amp[env_ids] * (w * w) * torch.sin(phi)

        ramp_time = float(self.cfg.ramp_time)
        if ramp_time > 0.0:
            t_clamp = torch.clamp(t, 0.0, ramp_time)
            s = t_clamp / ramp_time
            r = 0.5 * (1.0 - torch.cos(torch.pi * s))
            rdot = 0.5 * (torch.pi / ramp_time) * torch.sin(torch.pi * s)
            rddot = 0.5 * (torch.pi * torch.pi / (ramp_time * ramp_time)) * torch.cos(torch.pi * s)
        else:
            r = torch.ones_like(t)
            rdot = torch.zeros_like(t)
            rddot = torch.zeros_like(t)

        r = r[:, None]
        rdot = rdot[:, None]
        rddot = rddot[:, None]

        q_cmd = r * q_sin
        qd_cmd = r * qd_sin + rdot * q_sin
        qdd_cmd = r * qdd_sin + 2.0 * rdot * qd_sin + rddot * q_sin

        self._cmd[env_ids, :] = 0.0
        self._cmd[env_ids][:, self._joint_ids] = q_cmd
        self._cmd_vel[env_ids, :] = 0.0
        self._cmd_vel[env_ids][:, self._joint_ids] = qd_cmd
        self._cmd_acc[env_ids, :] = 0.0
        self._cmd_acc[env_ids][:, self._joint_ids] = qdd_cmd

    def _update_command(self) -> None:
        self._t += self._env.step_dt
        self._write_command()


@dataclass(kw_only=True)
class RandomSineJointPositionCommandCfg(CommandTermCfg):
    asset_name: str
    joint_names: tuple[str, ...] = (".*",)
    amp_range: tuple[float, float] = (0.15, 0.6)
    freq_range: tuple[float, float] = (0.05, 0.35)
    phase_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    ramp_time: float = 0.5
    resampling_time_range: tuple[float, float] = (4.0, 4.0)
    warm_start_reset: bool = False
    warm_start_use_command_vel: bool = True
    warm_start_forward: bool = False
    class_type: type[CommandTerm] = RandomSineJointPositionCommand
    debug_vis: bool = False
