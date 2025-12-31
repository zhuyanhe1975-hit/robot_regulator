from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robot_regulator.controllers.adaptive_pid import AdaptivePidTorqueController
from robot_regulator.controllers.gain_policy import FixedGainPolicy, GainLimits, NumpyMlpGainPolicy
from robot_regulator.learning.observations import ObsConfig, build_obs
from robot_regulator.sim.model_loader import ModelLoadOptions, load_model


@dataclass(frozen=True)
class EpisodeConfig:
    hz: float
    horizon_steps: int
    amp: float
    freq: float
    payload_mass: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out", type=str, default="rollouts/rollout.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--hz", type=float, default=500.0)
    p.add_argument("--horizon", type=float, default=4.0, help="seconds per episode")

    p.add_argument("--payload-min", type=float, default=0.0)
    p.add_argument("--payload-max", type=float, default=10.0)
    p.add_argument("--amp-min", type=float, default=0.15)
    p.add_argument("--amp-max", type=float, default=0.6)
    p.add_argument("--freq-min", type=float, default=0.05)
    p.add_argument("--freq-max", type=float, default=0.4)

    p.add_argument("--gain-policy", type=str, default="fixed", choices=["fixed", "mlp"])
    p.add_argument("--gain-npz", type=str, default="")

    p.add_argument("--kp", type=float, default=120.0)
    p.add_argument("--ki", type=float, default=0.0)
    p.add_argument("--kd", type=float, default=20.0)

    p.add_argument("--kp-range", type=str, default="0,400")
    p.add_argument("--ki-range", type=str, default="0,50")
    p.add_argument("--kd-range", type=str, default="0,80")
    p.add_argument("--dkdt", type=str, default="800,80,120")
    p.add_argument("--i-limit", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    model_path = Path(args.model).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-load to get dof and obs dim.
    loaded0 = load_model(model_path)
    dof = loaded0.model.nu
    obs_cfg = ObsConfig()
    obs_dim = build_obs(
        q=np.zeros(dof),
        qd=np.zeros(dof),
        q_des=np.zeros(dof),
        qd_des=np.zeros(dof),
        tau_bias=np.zeros(dof),
        prev_kp=np.zeros(dof),
        prev_ki=np.zeros(dof),
        prev_kd=np.zeros(dof),
        cfg=obs_cfg,
    ).shape[0]

    if args.gain_policy == "mlp":
        if not args.gain_npz:
            raise SystemExit("--gain-npz is required when --gain-policy mlp")
        policy = NumpyMlpGainPolicy(weights_npz=Path(args.gain_npz), dof=dof)
    else:
        policy = FixedGainPolicy(
            kp=np.full(dof, float(args.kp)),
            ki=np.full(dof, float(args.ki)),
            kd=np.full(dof, float(args.kd)),
        )

    kp_min, kp_max = (float(x) for x in args.kp_range.split(","))
    ki_min, ki_max = (float(x) for x in args.ki_range.split(","))
    kd_min, kd_max = (float(x) for x in args.kd_range.split(","))
    dkdt = np.array([float(x) for x in args.dkdt.split(",")], dtype=float)
    limits = GainLimits(
        kp=(kp_min, kp_max),
        ki=(ki_min, ki_max),
        kd=(kd_min, kd_max),
        dk_dt=(float(dkdt[0]), float(dkdt[1]), float(dkdt[2])),
    )

    controller = AdaptivePidTorqueController(
        policy=policy,
        limits=limits,
        dt=1.0 / args.hz,
        dof=dof,
        i_limit=np.full(dof, float(args.i_limit)),
        init_kp=np.full(dof, float(args.kp)),
        init_ki=np.full(dof, max(float(args.ki), 0.0)),
        init_kd=np.full(dof, float(args.kd)),
    )

    horizon_steps = int(round(float(args.horizon) * float(args.hz)))

    obs_buf = np.zeros((args.episodes, horizon_steps, obs_dim), dtype=np.float32)
    act_buf = np.zeros((args.episodes, horizon_steps, dof * 3), dtype=np.float32)  # (kp,ki,kd)
    rew_buf = np.zeros((args.episodes, horizon_steps), dtype=np.float32)
    info_buf = np.zeros((args.episodes, 4), dtype=np.float32)  # amp,freq,payload,seed

    for ep in range(args.episodes):
        cfg = EpisodeConfig(
            hz=float(args.hz),
            horizon_steps=horizon_steps,
            amp=float(rng.uniform(args.amp_min, args.amp_max)),
            freq=float(rng.uniform(args.freq_min, args.freq_max)),
            payload_mass=float(rng.uniform(args.payload_min, args.payload_max)),
        )
        info_buf[ep] = np.array([cfg.amp, cfg.freq, cfg.payload_mass, float(args.seed)], dtype=np.float32)

        loaded = (
            load_model(model_path, options=ModelLoadOptions(payload_mass=cfg.payload_mass))
            if (model_path.suffix.lower() == ".urdf" and cfg.payload_mass > 0)
            else load_model(model_path)
        )
        model, data = loaded.model, loaded.data
        data_ff = mujoco.MjData(model)
        ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

        # Randomize initial pose slightly around zero.
        data.qpos[loaded.qpos_idx] = rng.normal(0.0, 0.05, size=dof)
        data.qvel[loaded.qvel_idx] = rng.normal(0.0, 0.02, size=dof)
        mujoco.mj_forward(model, data)
        controller.reset()

        for k in range(horizon_steps):
            t = k / cfg.hz
            q_des = cfg.amp * np.sin(2.0 * np.pi * cfg.freq * t) * np.ones(dof)
            qd_des = (cfg.amp * 2.0 * np.pi * cfg.freq) * np.cos(2.0 * np.pi * cfg.freq * t) * np.ones(dof)

            mujoco.mj_forward(model, data)
            q = data.qpos[loaded.qpos_idx].copy()
            qd = data.qvel[loaded.qvel_idx].copy()

            # Current-state bias feedforward.
            tau_bias = data.qfrc_bias[loaded.qvel_idx].copy()

            st = controller.state()
            obs = build_obs(
                q=q,
                qd=qd,
                q_des=q_des,
                qd_des=qd_des,
                tau_bias=tau_bias,
                prev_kp=st.kp,
                prev_ki=st.ki,
                prev_kd=st.kd,
                cfg=obs_cfg,
            )

            tau = controller(obs=obs, q=q, qd=qd, q_des=q_des, qd_des=qd_des, tau_ff=tau_bias)
            data.ctrl[:] = np.clip(tau, ctrl_lo, ctrl_hi)
            mujoco.mj_step(model, data)

            # Reward: tracking - effort (simple baseline, tune later).
            e = q_des - q
            ed = qd_des - qd
            r = -(
                2.0 * float(np.mean(e**2))
                + 0.2 * float(np.mean(ed**2))
                + 1e-5 * float(np.mean(tau**2))
            )

            obs_buf[ep, k] = obs.astype(np.float32)
            act_buf[ep, k] = np.concatenate([st.kp, st.ki, st.kd], axis=0).astype(np.float32)
            rew_buf[ep, k] = np.float32(r)

    np.savez_compressed(out_path, obs=obs_buf, action=act_buf, reward=rew_buf, info=info_buf)
    print(f"wrote {out_path} obs={obs_buf.shape} action={act_buf.shape} reward={rew_buf.shape}")


if __name__ == "__main__":
    main()

