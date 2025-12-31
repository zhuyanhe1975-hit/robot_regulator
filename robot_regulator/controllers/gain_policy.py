from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GainLimits:
    kp: tuple[float, float]
    ki: tuple[float, float]
    kd: tuple[float, float]
    dk_dt: tuple[float, float]  # rate limit (per second) for (kp,ki,kd) applied elementwise


class GainPolicy:
    def reset(self) -> None:  # noqa: D401 - simple interface
        """Reset internal state (if any)."""

    def __call__(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class FixedGainPolicy(GainPolicy):
    def __init__(self, *, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray):
        self._kp = np.asarray(kp, dtype=float)
        self._ki = np.asarray(ki, dtype=float)
        self._kd = np.asarray(kd, dtype=float)

    def __call__(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._kp, self._ki, self._kd


class NumpyMlpGainPolicy(GainPolicy):
    """
    A tiny MLP inference-only policy (no torch dependency).
    Loads weights from an .npz with arrays: w0,b0,w1,b1,...,wN,bN.

    Output shape: [dof, 3] representing (kp, ki, kd) in log-space; converted via softplus.
    """

    def __init__(self, *, weights_npz: Path, dof: int):
        weights_npz = weights_npz.expanduser().resolve()
        blob = np.load(weights_npz)

        layers: list[tuple[np.ndarray, np.ndarray]] = []
        i = 0
        while True:
            w_key = f"w{i}"
            b_key = f"b{i}"
            if w_key not in blob or b_key not in blob:
                break
            w = np.asarray(blob[w_key], dtype=float)
            b = np.asarray(blob[b_key], dtype=float)
            layers.append((w, b))
            i += 1

        if not layers:
            raise ValueError(f"No layers found in {weights_npz} (expected w0,b0,...)")

        self._layers = layers
        self._dof = int(dof)

    def __call__(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(obs, dtype=float)
        for li, (w, b) in enumerate(self._layers):
            x = x @ w + b
            if li != (len(self._layers) - 1):
                x = np.tanh(x)

        # x: [3*dof]
        x = x.reshape(self._dof, 3)
        kp = _softplus(x[:, 0])
        ki = _softplus(x[:, 1])
        kd = _softplus(x[:, 2])
        return kp, ki, kd


def _softplus(x: np.ndarray) -> np.ndarray:
    # Stable softplus: log(1+exp(x))
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

