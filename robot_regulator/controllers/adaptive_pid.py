from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_regulator.controllers.gain_policy import GainLimits, GainPolicy


@dataclass(frozen=True)
class AdaptivePidState:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    i_term: np.ndarray


class AdaptivePidTorqueController:
    def __init__(
        self,
        *,
        policy: GainPolicy,
        limits: GainLimits,
        dt: float,
        dof: int,
        i_limit: np.ndarray,
        init_kp: np.ndarray,
        init_ki: np.ndarray,
        init_kd: np.ndarray,
    ):
        if dt <= 0:
            raise ValueError("dt must be > 0")
        self.policy = policy
        self.limits = limits
        self.dt = float(dt)
        self.dof = int(dof)
        self.i_limit = np.asarray(i_limit, dtype=float)

        self._kp = np.asarray(init_kp, dtype=float).copy()
        self._ki = np.asarray(init_ki, dtype=float).copy()
        self._kd = np.asarray(init_kd, dtype=float).copy()
        self._i = np.zeros(self.dof, dtype=float)

    def reset(self) -> None:
        self._i.fill(0.0)
        self.policy.reset()

    def state(self) -> AdaptivePidState:
        return AdaptivePidState(
            kp=self._kp.copy(),
            ki=self._ki.copy(),
            kd=self._kd.copy(),
            i_term=self._i.copy(),
        )

    def __call__(
        self,
        *,
        obs: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        tau_ff: np.ndarray,
    ) -> np.ndarray:
        kp_t, ki_t, kd_t = self.policy(obs)
        self._kp = _rate_limit(_clip(kp_t, self.limits.kp), self._kp, self.limits.dk_dt[0], self.dt)
        self._ki = _rate_limit(_clip(ki_t, self.limits.ki), self._ki, self.limits.dk_dt[1], self.dt)
        self._kd = _rate_limit(_clip(kd_t, self.limits.kd), self._kd, self.limits.dk_dt[2], self.dt)

        e = q_des - q
        ed = qd_des - qd

        self._i = self._i + e * self.dt
        self._i = np.clip(self._i, -self.i_limit, self.i_limit)

        return tau_ff + self._kp * e + self._kd * ed + self._ki * self._i


def _clip(x: np.ndarray, lim: tuple[float, float]) -> np.ndarray:
    return np.clip(x, float(lim[0]), float(lim[1]))


def _rate_limit(target: np.ndarray, prev: np.ndarray, max_rate: float, dt: float) -> np.ndarray:
    r = float(max_rate)
    if r <= 0:
        return target
    delta = np.clip(target - prev, -r * dt, r * dt)
    return prev + delta

