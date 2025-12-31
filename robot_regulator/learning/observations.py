from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObsConfig:
    use_sin_cos_q: bool = True
    use_bias: bool = True
    use_prev_gains: bool = True


def build_obs(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    tau_bias: np.ndarray | None,
    prev_kp: np.ndarray | None,
    prev_ki: np.ndarray | None,
    prev_kd: np.ndarray | None,
    cfg: ObsConfig,
) -> np.ndarray:
    parts: list[np.ndarray] = []

    if cfg.use_sin_cos_q:
        parts.append(np.sin(q))
        parts.append(np.cos(q))
    else:
        parts.append(q)

    parts.append(qd)
    parts.append(q_des)
    parts.append(qd_des)
    parts.append(q_des - q)
    parts.append(qd_des - qd)

    if cfg.use_bias:
        parts.append(np.zeros_like(q) if tau_bias is None else tau_bias)

    if cfg.use_prev_gains:
        parts.append(np.zeros_like(q) if prev_kp is None else prev_kp)
        parts.append(np.zeros_like(q) if prev_ki is None else prev_ki)
        parts.append(np.zeros_like(q) if prev_kd is None else prev_kd)

    return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts], axis=0)

