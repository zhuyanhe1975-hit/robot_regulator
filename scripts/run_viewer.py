from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_regulator.controllers.adaptive_pid import AdaptivePidTorqueController
from robot_regulator.controllers.gain_policy import (
    FixedGainPolicy,
    GainLimits,
    NumpyMlpGainPolicy,
)
from robot_regulator.controllers.pid import PidGains, PidTorqueController
from robot_regulator.learning.observations import ObsConfig, build_obs
from robot_regulator.sim.model_loader import ModelLoadOptions, load_model
from robot_regulator.sim.mujoco_runner import MujocoRunner


def _parse_vec(s: str, *, n: int) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1:
        return np.full(n, float(parts[0]))
    if len(parts) != n:
        raise argparse.ArgumentTypeError(f"Expected 1 or {n} values, got {len(parts)}: {s}")
    return np.array([float(p) for p in parts], dtype=float)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to MJCF .xml or URDF .urdf",
    )
    parser.add_argument(
        "--payload-mass",
        type=float,
        default=0.0,
        help="Attach a spherical payload (kg) at the end-effector (URDF-imported models only)",
    )
    parser.add_argument("--hz", type=float, default=500.0, help="Control loop frequency")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a viewer window",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of simulation steps in headless mode",
    )
    parser.add_argument(
        "--bias-ff",
        type=str,
        default="current",
        choices=["none", "current", "desired"],
        help="Feedforward torque: none/current-state/desired-state bias forces",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="sine",
        choices=["hold", "sine", "step"],
        help="Simple demo trajectory",
    )
    parser.add_argument("--amp", type=float, default=0.35, help="Sine amplitude (rad)")
    parser.add_argument("--freq", type=float, default=0.2, help="Sine frequency (Hz)")
    parser.add_argument("--step-joint", type=int, default=2, help="Step joint index (1..N)")
    parser.add_argument("--step-size", type=float, default=0.5, help="Step size (rad)")
    parser.add_argument("--step-time", type=float, default=1.0, help="Step start time (s)")

    parser.add_argument("--kp", type=str, default="60", help="Kp (scalar or csv)")
    parser.add_argument("--ki", type=str, default="0", help="Ki (scalar or csv)")
    parser.add_argument("--kd", type=str, default="6", help="Kd (scalar or csv)")
    parser.add_argument("--i-limit", type=str, default="2", help="Integral clamp (scalar or csv)")
    parser.add_argument(
        "--gain-policy",
        type=str,
        default="fixed",
        choices=["fixed", "mlp"],
        help="PID gains source: fixed or NN (numpy MLP from .npz)",
    )
    parser.add_argument(
        "--gain-npz",
        type=str,
        default="",
        help="Path to .npz weights for --gain-policy mlp",
    )
    parser.add_argument("--kp-range", type=str, default="0,400", help="kp min,max")
    parser.add_argument("--ki-range", type=str, default="0,50", help="ki min,max")
    parser.add_argument("--kd-range", type=str, default="0,80", help="kd min,max")
    parser.add_argument("--dkdt", type=str, default="800,80,120", help="rate limit kp,ki,kd per second")
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Print tracking metrics (useful in headless tuning)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    loaded = load_model(model_path)
    if model_path.suffix.lower() == ".urdf" and args.payload_mass > 0:
        loaded = load_model(
            model_path, options=ModelLoadOptions(payload_mass=float(args.payload_mass))
        )
    model, data = loaded.model, loaded.data
    dof = model.nu

    kp = _parse_vec(args.kp, n=dof)
    ki = _parse_vec(args.ki, n=dof)
    kd = _parse_vec(args.kd, n=dof)
    i_limit = _parse_vec(args.i_limit, n=dof)

    fixed_gains = PidGains(kp=kp, ki=ki, kd=kd, i_limit=i_limit)
    fixed_controller = PidTorqueController(gains=fixed_gains, dt=1.0 / args.hz)

    if args.gain_policy == "mlp":
        if not args.gain_npz:
            raise SystemExit("--gain-npz is required when --gain-policy mlp")
        pol = NumpyMlpGainPolicy(weights_npz=Path(args.gain_npz), dof=dof)
    else:
        pol = FixedGainPolicy(kp=kp, ki=ki, kd=kd)

    kp_min, kp_max = (float(x) for x in args.kp_range.split(","))
    ki_min, ki_max = (float(x) for x in args.ki_range.split(","))
    kd_min, kd_max = (float(x) for x in args.kd_range.split(","))
    dkdt = _parse_vec(args.dkdt, n=3)
    limits = GainLimits(
        kp=(kp_min, kp_max),
        ki=(ki_min, ki_max),
        kd=(kd_min, kd_max),
        dk_dt=(float(dkdt[0]), float(dkdt[1]), float(dkdt[2])),
    )
    adaptive_controller = AdaptivePidTorqueController(
        policy=pol,
        limits=limits,
        dt=1.0 / args.hz,
        dof=dof,
        i_limit=i_limit,
        init_kp=kp,
        init_ki=ki,
        init_kd=kd,
    )

    data_ff = mujoco.MjData(model) if args.bias_ff == "desired" else None
    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    runner = MujocoRunner(model=model, data=data)

    err_hist = []
    start = time.perf_counter()
    if args.headless:
        for k in range(args.steps):
            t = k / args.hz

            q_des, qd_des = _desired_trajectory(args=args, t=t, dof=dof)

            if args.bias_ff != "none":
                mujoco.mj_forward(model, data)

            q = data.qpos[loaded.qpos_idx].copy()
            qd = data.qvel[loaded.qvel_idx].copy()

            tau_ff = _bias_ff(args=args, model=model, data=data, data_ff=data_ff, q_des=q_des, qd_des=qd_des, qpos_idx=loaded.qpos_idx, qvel_idx=loaded.qvel_idx)
            if args.gain_policy == "fixed":
                tau = fixed_controller(q=q, qd=qd, q_des=q_des, qd_des=qd_des, tau_ff=tau_ff)
            else:
                st = adaptive_controller.state()
                obs = build_obs(
                    q=q,
                    qd=qd,
                    q_des=q_des,
                    qd_des=qd_des,
                    tau_bias=tau_ff,
                    prev_kp=st.kp,
                    prev_ki=st.ki,
                    prev_kd=st.kd,
                    cfg=ObsConfig(),
                )
                tau = adaptive_controller(
                    obs=obs,
                    q=q,
                    qd=qd,
                    q_des=q_des,
                    qd_des=qd_des,
                    tau_ff=tau_ff,
                )

            data.ctrl[:] = np.clip(tau, ctrl_lo, ctrl_hi)
            mujoco.mj_step(model, data)
            err_hist.append(q_des - q)

        if args.print_metrics:
            _print_metrics(np.asarray(err_hist))
        return

    while runner.is_running():
        t = time.perf_counter() - start

        q_des, qd_des = _desired_trajectory(args=args, t=t, dof=dof)

        if args.bias_ff != "none":
            mujoco.mj_forward(model, data)

        q = data.qpos[loaded.qpos_idx].copy()
        qd = data.qvel[loaded.qvel_idx].copy()
        tau_ff = _bias_ff(args=args, model=model, data=data, data_ff=data_ff, q_des=q_des, qd_des=qd_des, qpos_idx=loaded.qpos_idx, qvel_idx=loaded.qvel_idx)

        if args.gain_policy == "fixed":
            tau = fixed_controller(q=q, qd=qd, q_des=q_des, qd_des=qd_des, tau_ff=tau_ff)
        else:
            st = adaptive_controller.state()
            obs = build_obs(
                q=q,
                qd=qd,
                q_des=q_des,
                qd_des=qd_des,
                tau_bias=tau_ff,
                prev_kp=st.kp,
                prev_ki=st.ki,
                prev_kd=st.kd,
                cfg=ObsConfig(),
            )
            tau = adaptive_controller(
                obs=obs,
                q=q,
                qd=qd,
                q_des=q_des,
                qd_des=qd_des,
                tau_ff=tau_ff,
            )

        data.ctrl[:] = np.clip(tau, ctrl_lo, ctrl_hi)
        runner.step_and_render()


def _desired_trajectory(*, args: argparse.Namespace, t: float, dof: int) -> tuple[np.ndarray, np.ndarray]:
    if args.trajectory == "hold":
        return np.zeros(dof), np.zeros(dof)

    if args.trajectory == "sine":
        q_des = args.amp * np.sin(2.0 * np.pi * args.freq * t) * np.ones(dof)
        qd_des = (args.amp * 2.0 * np.pi * args.freq) * np.cos(
            2.0 * np.pi * args.freq * t
        ) * np.ones(dof)
        return q_des, qd_des

    # step
    q_des = np.zeros(dof)
    qd_des = np.zeros(dof)
    j = int(args.step_joint) - 1
    if not (0 <= j < dof):
        raise SystemExit(f"--step-joint must be in 1..{dof}, got {args.step_joint}")
    if t >= float(args.step_time):
        q_des[j] = float(args.step_size)
    return q_des, qd_des


def _bias_ff(
    *,
    args: argparse.Namespace,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    data_ff: mujoco.MjData | None,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
) -> np.ndarray:
    if args.bias_ff == "none":
        return np.zeros(model.nu)

    if args.bias_ff == "current":
        return data.qfrc_bias[qvel_idx].copy()

    if data_ff is None:
        raise RuntimeError("Internal error: data_ff is None for desired bias feedforward")

    data_ff.qpos[:] = data.qpos
    data_ff.qvel[:] = data.qvel
    data_ff.qpos[qpos_idx] = q_des
    data_ff.qvel[qvel_idx] = qd_des
    mujoco.mj_forward(model, data_ff)
    return data_ff.qfrc_bias[qvel_idx].copy()


def _print_metrics(err: np.ndarray) -> None:
    # err shape: [T, dof]
    rmse = np.sqrt(np.mean(err**2, axis=0))
    max_abs = np.max(np.abs(err), axis=0)
    final = err[-1]
    tail = err[int(0.8 * len(err)) :]
    tail_rmse = np.sqrt(np.mean(tail**2, axis=0)) if len(tail) > 0 else rmse
    print("rmse(rad):", ",".join(f"{x:.4f}" for x in rmse))
    print("tail_rmse(rad):", ",".join(f"{x:.4f}" for x in tail_rmse))
    print("max_abs(rad):", ",".join(f"{x:.4f}" for x in max_abs))
    print("final_err(rad):", ",".join(f"{x:.4f}" for x in final))


if __name__ == "__main__":
    main()
