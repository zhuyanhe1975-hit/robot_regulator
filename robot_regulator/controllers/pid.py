from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PidGains:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    i_limit: np.ndarray


class PidTorqueController:
    def __init__(self, gains: PidGains, dt: float):
        if dt <= 0:
            raise ValueError("dt must be > 0")
        self.gains = gains
        self.dt = float(dt)
        self._i = np.zeros_like(gains.kp, dtype=float)

    def reset(self) -> None:
        self._i.fill(0.0)

    def __call__(
        self,
        *,
        q: np.ndarray,
        qd: np.ndarray,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        tau_ff: np.ndarray,
    ) -> np.ndarray:
        e = q_des - q
        ed = qd_des - qd

        self._i = self._i + e * self.dt
        self._i = np.clip(self._i, -self.gains.i_limit, self.gains.i_limit)

        return (
            tau_ff
            + self.gains.kp * e
            + self.gains.kd * ed
            + self.gains.ki * self._i
        )

