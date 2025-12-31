from __future__ import annotations

import time

import mujoco
import mujoco.viewer

class MujocoRunner:
    def __init__(self, *, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._viewer = None
        self._last_sync = None

    def is_running(self) -> bool:
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._last_sync = time.perf_counter()
        return self._viewer.is_running()

    def step_and_render(self) -> None:
        mujoco.mj_step(self.model, self.data)

        now = time.perf_counter()
        if self._last_sync is None:
            self._last_sync = now

        # Keep the viewer responsive; sync at ~60Hz.
        if (now - self._last_sync) >= (1.0 / 60.0):
            self._viewer.sync()
            self._last_sync = now

