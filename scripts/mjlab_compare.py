from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner

# Register IRB2400 task(s) into mjlab's registry.
import mjlab_irb2400  # noqa: F401
import mjlab_irb2400_v1  # noqa: F401
import mjlab_irb2400_ctres  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="Mjlab-JointGain-ABB-IRB2400")
    p.add_argument("--checkpoint", type=str, default="", help="Path to model_*.pt for --task")
    p.add_argument("--ff-task", type=str, default="Mjlab-JointGainFF-ABB-IRB2400")
    p.add_argument("--ff-checkpoint", type=str, default="", help="Optional model_*.pt for --ff-task")
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0 or cpu")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds (overrides --seed), e.g. '0,1,2,3,4'",
    )

    p.add_argument(
        "--baseline",
        type=str,
        default="offset",
        choices=["offset", "custom", "trained_mean"],
        help=(
            "Baseline fixed gains: 'offset' uses action offsets; 'custom' uses --baseline-kp/--baseline-kd; "
            "'trained_mean' uses per-joint mean gains observed from the trained policy."
        ),
    )
    p.add_argument("--baseline-kp", type=float, default=220.0)
    p.add_argument("--baseline-kd", type=float, default=40.0)

    p.add_argument("--out", type=str, default="compare/irb2400_gain_compare.json")
    p.add_argument("--video", action="store_true", help="Record videos (num_envs forced to 1)")
    p.add_argument("--video-length", type=int, default=200)
    p.add_argument("--video-height", type=int, default=480)
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument(
        "--skip-steps",
        type=int,
        default=20,
        help="Skip the first N steps of each episode when computing error metrics (removes reset transient).",
    )
    return p.parse_args()


class ConstantActionPolicy:
    def __init__(self, action: torch.Tensor):
        self._action = action

    def __call__(self, obs) -> torch.Tensor:  # noqa: ANN001 - rsl_rl style
        del obs
        return self._action


def _resolve_device(cli_device: str) -> str:
    if cli_device:
        return cli_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_trained_policy(
    *, task_id: str, checkpoint: Path, env: RslRlVecEnvWrapper, device: str
):
    agent_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), map_location=device)
    return runner.get_inference_policy(device=device)


def _make_env(
    *,
    task_id: str,
    device: str,
    num_envs: int,
    seed: int,
    video: bool,
    video_folder: Path | None,
    video_length: int,
    video_height: int,
    video_width: int,
    play: bool = True,
) -> RslRlVecEnvWrapper:
    env_cfg = load_env_cfg(task_id, play=bool(play))
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.viewer.height = video_height
    env_cfg.viewer.width = video_width

    render_mode = "rgb_array" if video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if video:
        assert video_folder is not None
        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )

    agent_cfg = load_rl_cfg(task_id)
    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def _resolve_gain_action_layout(env: RslRlVecEnvWrapper) -> tuple[str, int, int]:
    """Return (term_key, dof, action_dim) for gain-scheduling tasks.

    Supported action layouts:
    - "pd_gains":    [kp_raw (N), kd_raw (N)]
    - "pd_gains_ff": [kp_raw (N), kd_raw (N), tau_ff_raw (N)]
    """
    cfg_actions = getattr(env.unwrapped.cfg, "actions", {}) or {}
    if "pd_gains" in cfg_actions:
        term_key = "pd_gains"
        blocks = 2
    elif "pd_gains_ff" in cfg_actions:
        term_key = "pd_gains_ff"
        term_cfg = cfg_actions[term_key]
        use_integral = bool(getattr(term_cfg, "use_integral", False))
        blocks = 4 if use_integral else 3
    else:
        raise KeyError(
            f"Expected action term 'pd_gains' or 'pd_gains_ff', got: {sorted(cfg_actions.keys())}"
        )

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    if action_dim % blocks != 0:
        raise RuntimeError(
            f"Invalid action_dim={action_dim} for blocks={blocks} (term={term_key})"
        )
    dof = action_dim // blocks
    return term_key, dof, action_dim


def _episode_logs_from_extras(extras: dict[str, Any]) -> dict[str, float]:
    log = extras.get("log", {})
    if not isinstance(log, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in log.items():
        if isinstance(v, (float, int)):
            out[k] = float(v)
        elif isinstance(v, torch.Tensor):
            # RewardManager returns scalar tensors; TerminationManager returns ints.
            if v.numel() == 1:
                out[k] = float(v.item())
    return out


def _try_resolve_effort_limit_per_joint(robot, dof: int) -> torch.Tensor | None:  # noqa: ANN001
    """Best-effort resolve joint torque/effort limit per joint for saturation checks."""
    # Preferred: read from IdealPdActuator instance if present.
    try:
        from mjlab.actuator.pd_actuator import IdealPdActuator  # type: ignore

        for act in getattr(robot, "actuators", []):
            if isinstance(act, IdealPdActuator):
                lim = getattr(act, "effort_limit", None)
                if lim is None:
                    continue
                if isinstance(lim, (float, int)):
                    return torch.full((dof,), float(lim), device=robot.data.joint_pos.device)
                if isinstance(lim, torch.Tensor):
                    if lim.numel() == 1:
                        return torch.full((dof,), float(lim.item()), device=lim.device)
                    if lim.numel() >= dof:
                        return lim[:dof].to(device=robot.data.joint_pos.device)
    except Exception:
        pass

    # Fallback: see if the robot data exposes a limit tensor.
    for name in ("joint_effort_limit", "joint_torque_limit", "effort_limit"):
        lim = getattr(getattr(robot, "data", None), name, None)
        if isinstance(lim, (float, int)):
            return torch.full((dof,), float(lim), device=robot.data.joint_pos.device)
        if isinstance(lim, torch.Tensor):
            if lim.numel() == 1:
                return torch.full((dof,), float(lim.item()), device=lim.device)
            if lim.numel() >= dof:
                return lim[:dof].to(device=robot.data.joint_pos.device)

    return None


def _to_torch_tensor(x: Any) -> torch.Tensor | None:
    """Best-effort convert mujoco_warp TorchArray or other array-likes into torch.Tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("tensor", "to_torch", "torch"):
        v = getattr(x, attr, None)
        if v is None:
            continue
        try:
            y = v() if callable(v) else v
        except Exception:
            continue
        if isinstance(y, torch.Tensor):
            return y
    try:
        y = torch.as_tensor(x)
        if isinstance(y, torch.Tensor):
            return y
    except Exception:
        return None
    return None


def _try_resolve_effort_limit_from_model(robot, dof: int) -> torch.Tensor | None:  # noqa: ANN001
    """Fallback: resolve per-joint effort limits from MuJoCo model actuator force ranges."""
    model = getattr(getattr(robot, "data", None), "model", None)
    if model is None:
        return None
    fr = getattr(model, "actuator_forcerange", None)
    fr_t = _to_torch_tensor(fr)
    if fr_t is None:
        return None
    # Expect shape (nu, 2) with min/max. Some wrappers may store (2, nu).
    if fr_t.ndim != 2 or 2 not in fr_t.shape:
        return None
    if fr_t.shape[-1] == 2:
        lim = torch.max(fr_t.abs(), dim=-1).values
    else:  # (2, nu)
        lim = torch.max(fr_t.abs(), dim=0).values

    if lim.numel() == 1:
        return torch.full((dof,), float(lim.item()), device=robot.data.joint_pos.device)
    if lim.numel() >= dof:
        return lim[:dof].to(device=robot.data.joint_pos.device)
    return None


def _try_resolve_effort_limit_from_env_cfg(env: RslRlVecEnvWrapper, dof: int) -> torch.Tensor | None:
    """Fallback: read actuator effort_limit from the env config (ManagerBasedRlEnvCfg)."""
    try:
        entities = getattr(getattr(env.unwrapped.cfg, "scene", None), "entities", None) or {}
        robot_cfg = entities.get("robot", None)
        articulation = getattr(robot_cfg, "articulation", None)
        actuators = getattr(articulation, "actuators", None)
        if not actuators:
            return None
        # Typical IRB2400 config uses one IdealPdActuatorCfg covering all joints.
        act0 = actuators[0]
        lim = getattr(act0, "effort_limit", None)
        if lim is None:
            return None
        if isinstance(lim, (float, int)):
            return torch.full((dof,), float(lim), device=env.unwrapped.scene["robot"].data.joint_pos.device)
        t = _to_torch_tensor(lim)
        if t is None:
            return None
        if t.numel() == 1:
            return torch.full((dof,), float(t.item()), device=t.device)
        if t.numel() >= dof:
            return t[:dof].to(device=env.unwrapped.scene["robot"].data.joint_pos.device)
    except Exception:
        return None
    return None


def debug_torque_fields(env: RslRlVecEnvWrapper) -> None:
    """Print which torque-related fields are available for saturation diagnostics."""
    robot = env.unwrapped.scene["robot"]
    dof = int(robot.data.joint_pos.shape[1])
    data = robot.data.data
    idx_v = robot.data.indexing.joint_v_adr[:dof]

    def _describe(name: str) -> str:
        v = getattr(data, name, None)
        if v is None:
            return "missing"
        t = _to_torch_tensor(v)
        if t is None:
            return f"type={type(v).__name__} (no torch conversion)"
        shape = tuple(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        try:
            sample = t[:, idx_v]
            sample_shape = tuple(sample.shape)
        except Exception:
            sample_shape = None
        return f"type={type(v).__name__} torch_shape={shape} dtype={dtype} sel_shape={sample_shape}"

    print("[DEBUG] torque fields on robot.data.data:", flush=True)
    for name in (
        "qfrc_actuator",
        "qfrc_bias",
        "qfrc_applied",
        "qfrc_passive",
        "qfrc_constraint",
        "qfrc_inverse",
        "actuator_force",
        "ctrl",
    ):
        print(f"  - {name}: {_describe(name)}", flush=True)

    lim = (
        _try_resolve_effort_limit_per_joint(robot, dof)
        or _try_resolve_effort_limit_from_model(robot, dof)
        or _try_resolve_effort_limit_from_env_cfg(env, dof)
    )
    if lim is None:
        print("  - effort_limit: unresolved (actuator + model + env_cfg)", flush=True)
    else:
        print(
            f"  - effort_limit: tensor shape={tuple(lim.shape)} min={float(lim.min().item()):.3f} max={float(lim.max().item()):.3f}",
            flush=True,
        )

    cfg_actions = getattr(env.unwrapped.cfg, "actions", {}) or {}
    if "pd_gains_ff" in cfg_actions:
        cfg = cfg_actions["pd_gains_ff"]
        tau_limit = float(getattr(cfg, "tau_limit", float("inf")))
        id_limit = float(getattr(cfg, "id_limit", float("inf")))
        print(f"  - cfg.tau_limit: {tau_limit}", flush=True)
        print(f"  - cfg.id_limit: {id_limit}", flush=True)


def _run_eval(
    *,
    env: RslRlVecEnvWrapper,
    policy,
    episodes: int,
    device: str,
    skip_steps: int = 0,
) -> list[dict[str, float]]:
    logs: list[dict[str, float]] = []

    # Guard against skipping the whole episode (common when switching tasks with different episode length).
    # Ensure we still accumulate at least 1 step of metrics per episode.
    if env.max_episode_length <= 0:
        raise RuntimeError(f"Invalid env.max_episode_length={env.max_episode_length}")
    if skip_steps >= env.max_episode_length:
        new_skip = max(0, env.max_episode_length - 1)
        print(
            f"[WARN] skip_steps ({skip_steps}) >= episode length ({env.max_episode_length}); "
            f"clamping skip_steps to {new_skip}",
            flush=True,
        )
        skip_steps = new_skip

    obs = env.get_observations()
    env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)

    robot = env.unwrapped.scene["robot"]
    if robot.data.default_joint_pos is None:
        raise RuntimeError("robot.data.default_joint_pos is required for joint error metrics.")

    # End-effector site (entity-local) for EE error evaluation.
    ee_site_ids, _ = robot.find_sites("ee", preserve_order=True)
    if not ee_site_ids:
        raise RuntimeError("Could not find site named 'ee' on robot entity.")
    if len(ee_site_ids) != 1:
        raise RuntimeError(f"Expected exactly one 'ee' site, got {ee_site_ids}")
    ee_site_id = int(ee_site_ids[0])

    try:
        import mujoco_warp as mjwarp  # type: ignore
    except Exception:  # pragma: no cover - depends on runtime environment
        mjwarp = None

    def _mjwarp_update_kinematics() -> None:
        """Update positions needed for site_xpos without triggering camlight."""
        if mjwarp is None:
            return
        # Prefer kinematics() since fwd_position() calls camlight() and can break on some builds.
        if hasattr(mjwarp, "kinematics"):
            mjwarp.kinematics(model, data)
        else:  # pragma: no cover
            # Fall back to fwd_position if kinematics is unavailable.
            mjwarp.fwd_position(model, data)

    # Per-step accumulation for "true" joint tracking error.
    # Each episode contains `env.num_envs` parallel rollouts; we report mean over envs.
    num_joints = int(robot.data.joint_pos.shape[1])
    sum_abs = torch.zeros((env.num_envs, num_joints), device=device)
    sum_sq = torch.zeros_like(sum_abs)
    max_abs = torch.zeros_like(sum_abs)
    steps = torch.zeros((env.num_envs, 1), device=device)

    # Per-step accumulation for end-effector position tracking error (meters).
    ee_sum = torch.zeros((env.num_envs, 1), device=device)
    ee_sum_sq = torch.zeros_like(ee_sum)
    ee_max = torch.zeros_like(ee_sum)

    # Physical smoothness metrics (independent of reward weights).
    sum_vel_sq = torch.zeros_like(sum_abs)
    sum_acc_sq = torch.zeros_like(sum_abs)
    prev_vel = robot.data.joint_vel.clone()
    dt = float(env.unwrapped.step_dt)

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    cfg_actions = getattr(env.unwrapped.cfg, "actions", {})
    has_pd_gains = "pd_gains" in cfg_actions
    has_pd_gains_ff = "pd_gains_ff" in cfg_actions
    has_torque_action = "joint_torque" in cfg_actions

    update_every_env_steps = 1
    if has_pd_gains or has_pd_gains_ff:
        term_cfg = env.unwrapped.cfg.actions["pd_gains"] if has_pd_gains else env.unwrapped.cfg.actions["pd_gains_ff"]
        if hasattr(term_cfg, "update_period_s"):
            update_every_env_steps = max(
                1, int(round(float(getattr(term_cfg, "update_period_s")) / float(env.unwrapped.step_dt)))
            )

    held_action_for_stats = None

    # Always define these to avoid UnboundLocalError in mixed task comparisons.
    ff_tau_steps = 0
    ff_tau_abs_sum = None
    ff_tau_abs_max = None

    # Gain stats (what the actuator actually sees after clamping/mapping) for the gain-scheduling task.
    if has_pd_gains or has_pd_gains_ff:
        pd_cfg = env.unwrapped.cfg.actions["pd_gains"] if has_pd_gains else env.unwrapped.cfg.actions["pd_gains_ff"]
        kp_scale = float(getattr(pd_cfg, "kp_scale"))
        kp_offset = float(getattr(pd_cfg, "kp_offset"))
        kd_scale = float(getattr(pd_cfg, "kd_scale"))
        kd_offset = float(getattr(pd_cfg, "kd_offset"))
        clip_kp = tuple(getattr(pd_cfg, "clip_kp"))
        clip_kd = tuple(getattr(pd_cfg, "clip_kd"))

        dof = int(env.unwrapped.scene["robot"].data.joint_pos.shape[1])
        gain_steps = 0
        kp_sum = torch.zeros((env.num_envs, dof), device=device)
        kd_sum = torch.zeros_like(kp_sum)
        kp_min = torch.full_like(kp_sum, float("inf"))
        kd_min = torch.full_like(kd_sum, float("inf"))
        kp_max = torch.full_like(kp_sum, float("-inf"))
        kd_max = torch.full_like(kd_sum, float("-inf"))
        if has_pd_gains_ff:
            ff_tau_steps = 0
            ff_tau_abs_sum = torch.zeros((env.num_envs, dof), device=device)
            ff_tau_abs_max = torch.zeros_like(ff_tau_abs_sum)
    else:
        kp_scale = kp_offset = kd_scale = kd_offset = None
        clip_kp = clip_kd = None
        dof = None
        gain_steps = 0
        kp_sum = kd_sum = kp_min = kd_min = kp_max = kd_max = None

    # Torque stats for the torque-control task.
    if has_torque_action:
        torque_steps = 0
        torque_abs_sum = torch.zeros((env.num_envs, action_dim), device=device)
        torque_abs_max = torch.zeros_like(torque_abs_sum)
    else:
        torque_steps = 0
        torque_abs_sum = torque_abs_max = None

    # Torque stats for GainFF tasks: total applied torque (qfrc_actuator) + ID component saturation.
    total_tau_steps = 0
    total_tau_abs_sum = total_tau_abs_max = None
    total_tau_sq_sum = None
    total_tau_sat_cnt = total_tau_sat_per_joint_cnt = None
    total_tau_sat_denom = None
    id_tau_steps = 0
    id_tau_abs_sum = id_tau_abs_max = None
    id_tau_sq_sum = None
    id_tau_sat_cnt = id_tau_sat_per_joint_cnt = None
    id_tau_sat_denom = None
    ff_tau_sq_sum = None

    # Alignment stats between components and total torque.
    ff_align_steps = 0
    ff_total_sum = ff_total_sq_sum = ff_comp_sum = ff_comp_sq_sum = ff_dot_sum = None
    id_align_steps = 0
    id_total_sum = id_total_sq_sum = id_comp_sum = id_comp_sq_sum = id_dot_sum = None
    if has_pd_gains_ff:
        assert dof is not None
        total_tau_abs_sum = torch.zeros((env.num_envs, dof), device=device)
        total_tau_abs_max = torch.zeros_like(total_tau_abs_sum)
        total_tau_sq_sum = torch.zeros_like(total_tau_abs_sum)
        total_tau_sat_cnt = torch.zeros((env.num_envs, 1), device=device)
        total_tau_sat_per_joint_cnt = torch.zeros((env.num_envs, dof), device=device)
        total_tau_sat_denom = torch.zeros((env.num_envs, 1), device=device)

        id_tau_abs_sum = torch.zeros((env.num_envs, dof), device=device)
        id_tau_abs_max = torch.zeros_like(id_tau_abs_sum)
        id_tau_sq_sum = torch.zeros_like(id_tau_abs_sum)
        id_tau_sat_cnt = torch.zeros((env.num_envs, 1), device=device)
        id_tau_sat_per_joint_cnt = torch.zeros((env.num_envs, dof), device=device)
        id_tau_sat_denom = torch.zeros((env.num_envs, 1), device=device)

        ff_tau_sq_sum = torch.zeros((env.num_envs, dof), device=device)

        ff_total_sum = torch.zeros((env.num_envs, dof), device=device)
        ff_total_sq_sum = torch.zeros_like(ff_total_sum)
        ff_comp_sum = torch.zeros_like(ff_total_sum)
        ff_comp_sq_sum = torch.zeros_like(ff_total_sum)
        ff_dot_sum = torch.zeros_like(ff_total_sum)

        id_total_sum = torch.zeros((env.num_envs, dof), device=device)
        id_total_sq_sum = torch.zeros_like(id_total_sum)
        id_comp_sum = torch.zeros_like(id_total_sum)
        id_comp_sq_sum = torch.zeros_like(id_total_sum)
        id_dot_sum = torch.zeros_like(id_total_sum)

    for _ in range(episodes):
        # Run exactly one full episode worth of steps; env auto-resets on time_out.
        for _step in range(env.max_episode_length):
            with torch.no_grad():
                actions = policy(obs)
            obs, _rew, dones, extras = env.step(actions)

            # "True" tracking error metrics (skip early transient right after reset).
            cmd = env.unwrapped.command_manager.get_command("joint_pos")
            q = robot.data.joint_pos
            q0 = robot.data.default_joint_pos
            if _step >= skip_steps:
                err = (q - q0) - cmd
                abs_err = torch.abs(err)
                sum_abs += abs_err
                sum_sq += err * err
                max_abs = torch.maximum(max_abs, abs_err)
                steps += 1.0

            # End-effector position error w.r.t. reference joint command.
            if mjwarp is not None and _step >= skip_steps:
                ee_pos = robot.data.site_pos_w[:, ee_site_id, :]
                q_ref = q0 + cmd
                joint_q_adr = robot.data.indexing.joint_q_adr
                data = robot.data.data
                model = robot.data.model
                # Compute reference FK at q_ref.
                data.qpos[:, joint_q_adr] = q_ref
                _mjwarp_update_kinematics()
                ee_ref = robot.data.site_pos_w[:, ee_site_id, :]
                # Restore FK at actual q for subsequent dynamics.
                data.qpos[:, joint_q_adr] = q
                _mjwarp_update_kinematics()

                ee_err = torch.linalg.norm(ee_pos - ee_ref, dim=-1, keepdim=True)
                ee_sum += ee_err
                ee_sum_sq += ee_err * ee_err
                ee_max = torch.maximum(ee_max, ee_err)

            # Smoothness metrics from state.
            vel = robot.data.joint_vel
            acc = (vel - prev_vel) / dt
            prev_vel = vel.clone()
            if _step >= skip_steps:
                sum_vel_sq += vel * vel
                sum_acc_sq += acc * acc

            # Gain stats from actions (mirror of JointPdGainAction mapping).
            if held_action_for_stats is None:
                held_action_for_stats = actions.clone()
            if (_step % update_every_env_steps) == 0:
                held_action_for_stats = actions.clone()
            a = torch.clamp(held_action_for_stats, -1.0, 1.0)
            ff_tau_for_align = None
            if has_pd_gains:
                assert dof is not None
                kp_raw = a[:, :dof]
                kd_raw = a[:, dof:]
                kp = torch.clamp(kp_raw * kp_scale + kp_offset, clip_kp[0], clip_kp[1])
                kd = torch.clamp(kd_raw * kd_scale + kd_offset, clip_kd[0], clip_kd[1])
                kp_sum += kp
                kd_sum += kd
                kp_min = torch.minimum(kp_min, kp)
                kd_min = torch.minimum(kd_min, kd)
                kp_max = torch.maximum(kp_max, kp)
                kd_max = torch.maximum(kd_max, kd)
                gain_steps += 1

            if has_pd_gains_ff:
                assert dof is not None
                # Support multiple "pd_gains_ff" action layouts:
                # - Legacy (4*dof): [kp, kd, a, b] used for simple friction residual tau_res
                # - v1 (3*dof): [kp, kd, tau_ff] direct feedforward torque
                kp_raw = a[:, :dof]
                kd_raw = a[:, dof : 2 * dof]
                kp = torch.clamp(kp_raw * kp_scale + kp_offset, clip_kp[0], clip_kp[1])
                kd = torch.clamp(kd_raw * kd_scale + kd_offset, clip_kd[0], clip_kd[1])
                kp_sum += kp
                kd_sum += kd
                kp_min = torch.minimum(kp_min, kp)
                kd_min = torch.minimum(kd_min, kd)
                kp_max = torch.maximum(kp_max, kp)
                kd_max = torch.maximum(kd_max, kd)
                gain_steps += 1

                assert ff_tau_abs_sum is not None and ff_tau_abs_max is not None
                if a.shape[1] == 3 * dof:
                    tau_raw = a[:, 2 * dof : 3 * dof]
                    tau_scale = float(getattr(pd_cfg, "tau_scale", 1.0))
                    tau_limit = float(getattr(pd_cfg, "tau_limit", float("inf")))
                    tau_ff = torch.clamp(tau_raw * tau_scale, -tau_limit, tau_limit)
                    ff_tau_for_align = tau_ff
                    abs_tau = torch.abs(tau_ff)
                    ff_tau_abs_sum += abs_tau
                    ff_tau_abs_max = torch.maximum(ff_tau_abs_max, abs_tau)
                    assert ff_tau_sq_sum is not None
                    ff_tau_sq_sum += tau_ff * tau_ff
                    ff_tau_steps += 1
                elif a.shape[1] == 4 * dof:
                    a_raw = a[:, 2 * dof : 3 * dof]
                    b_raw = a[:, 3 * dof : 4 * dof]
                    tau_limit = float(getattr(pd_cfg, "tau_limit"))
                    a_max = float(getattr(pd_cfg, "a_max"))
                    b_max = float(getattr(pd_cfg, "b_max"))
                    sign_eps = float(getattr(pd_cfg, "sign_eps", 0.05))
                    a_mag = 0.5 * (a_raw + 1.0) * a_max
                    b_mag = 0.5 * (b_raw + 1.0) * b_max
                    qd = robot.data.joint_vel[:, :dof]
                    sign = torch.tanh(qd / sign_eps)
                    tau_res = a_mag * sign + b_mag * qd
                    tau_res = torch.clamp(tau_res, -tau_limit, tau_limit)
                    ff_tau_for_align = tau_res
                    abs_tau = torch.abs(tau_res)
                    ff_tau_abs_sum += abs_tau
                    ff_tau_abs_max = torch.maximum(ff_tau_abs_max, abs_tau)
                    assert ff_tau_sq_sum is not None
                    ff_tau_sq_sum += tau_res * tau_res
                    ff_tau_steps += 1
                else:
                    # Unknown layout; skip FF magnitude metrics (keep gain stats).
                    pass

            if has_torque_action:
                tau = a
                abs_tau = torch.abs(tau)
                assert torque_abs_sum is not None and torque_abs_max is not None
                torque_abs_sum += abs_tau
                torque_abs_max = torch.maximum(torque_abs_max, abs_tau)
                torque_steps += 1

            # Total torque + ID component (GainFF tasks).
            if has_pd_gains_ff and _step >= skip_steps:
                assert dof is not None
                pd_cfg = env.unwrapped.cfg.actions["pd_gains_ff"]

                data = robot.data.data
                idx_v = robot.data.indexing.joint_v_adr[:dof]

                # Total actuator generalized forces applied by MuJoCo.
                qfrc_actuator = getattr(data, "qfrc_actuator", None)
                qfrc_actuator_t = _to_torch_tensor(qfrc_actuator)
                if qfrc_actuator_t is not None:
                    tau_total = qfrc_actuator_t[:, idx_v]
                    abs_total = torch.abs(tau_total)
                    assert total_tau_abs_sum is not None and total_tau_abs_max is not None
                    total_tau_abs_sum += abs_total
                    total_tau_abs_max = torch.maximum(total_tau_abs_max, abs_total)
                    assert total_tau_sq_sum is not None
                    total_tau_sq_sum += tau_total * tau_total
                    total_tau_steps += 1

                    lim = (
                        _try_resolve_effort_limit_per_joint(robot, dof)
                        or _try_resolve_effort_limit_from_model(robot, dof)
                        or _try_resolve_effort_limit_from_env_cfg(env, dof)
                    )
                    if lim is not None:
                        th = 0.98 * lim.view(1, -1)
                        sat = abs_total >= th
                        assert total_tau_sat_cnt is not None
                        assert total_tau_sat_per_joint_cnt is not None
                        assert total_tau_sat_denom is not None
                        total_tau_sat_per_joint_cnt += sat.to(dtype=abs_total.dtype)
                        total_tau_sat_cnt += sat.any(dim=-1, keepdim=True).to(dtype=abs_total.dtype)
                        total_tau_sat_denom += 1.0

                # ID component from qfrc_bias (matches action term).
                qfrc_bias = getattr(data, "qfrc_bias", None)
                qfrc_bias_t = _to_torch_tensor(qfrc_bias)
                tau_id_for_align = None
                if qfrc_bias_t is not None:
                    tau_id = qfrc_bias_t[:, idx_v] * float(getattr(pd_cfg, "id_scale", 1.0))
                    id_limit = float(getattr(pd_cfg, "id_limit", float("inf")))
                    tau_id = torch.clamp(tau_id, -id_limit, id_limit)
                    tau_id_for_align = tau_id
                    abs_id = torch.abs(tau_id)
                    assert id_tau_abs_sum is not None and id_tau_abs_max is not None
                    id_tau_abs_sum += abs_id
                    id_tau_abs_max = torch.maximum(id_tau_abs_max, abs_id)
                    assert id_tau_sq_sum is not None
                    id_tau_sq_sum += tau_id * tau_id
                    id_tau_steps += 1

                    if id_limit < float("inf"):
                        sat = abs_id >= (0.98 * id_limit)
                        assert id_tau_sat_cnt is not None
                        assert id_tau_sat_per_joint_cnt is not None
                        assert id_tau_sat_denom is not None
                        id_tau_sat_per_joint_cnt += sat.to(dtype=abs_id.dtype)
                        id_tau_sat_cnt += sat.any(dim=-1, keepdim=True).to(dtype=abs_id.dtype)
                        id_tau_sat_denom += 1.0

                # Alignment: projection/correlation of components onto the actual total torque.
                if qfrc_actuator_t is not None:
                    if ff_tau_for_align is not None:
                        assert ff_total_sum is not None and ff_total_sq_sum is not None
                        assert ff_comp_sum is not None and ff_comp_sq_sum is not None
                        assert ff_dot_sum is not None
                        ff_total_sum += tau_total
                        ff_total_sq_sum += tau_total * tau_total
                        ff_comp_sum += ff_tau_for_align
                        ff_comp_sq_sum += ff_tau_for_align * ff_tau_for_align
                        ff_dot_sum += tau_total * ff_tau_for_align
                        ff_align_steps += 1
                    if tau_id_for_align is not None:
                        assert id_total_sum is not None and id_total_sq_sum is not None
                        assert id_comp_sum is not None and id_comp_sq_sum is not None
                        assert id_dot_sum is not None
                        id_total_sum += tau_total
                        id_total_sq_sum += tau_total * tau_total
                        id_comp_sum += tau_id_for_align
                        id_comp_sq_sum += tau_id_for_align * tau_id_for_align
                        id_dot_sum += tau_total * tau_id_for_align
                        id_align_steps += 1

            # All envs time out together in this task; treat as episode boundary.
            if int(dones.sum().item()) == env.num_envs:
                ep = _episode_logs_from_extras(extras)

                if float(steps.max().item()) <= 0.0:
                    raise RuntimeError(
                        f"No samples accumulated for metrics (skip_steps={skip_steps} >= episode length?)"
                    )
                mae_per_joint = (sum_abs / steps).mean(dim=0)
                rmse_per_joint = torch.sqrt(sum_sq / steps).mean(dim=0)
                max_abs_per_joint = max_abs.mean(dim=0)

                ep["JointError/mae_mean_rad"] = float(mae_per_joint.mean().item())
                ep["JointError/rmse_mean_rad"] = float(rmse_per_joint.mean().item())
                ep["JointError/max_abs_mean_rad"] = float(max_abs_per_joint.mean().item())

                ep["JointError/per_joint_mae_rad"] = [float(x) for x in mae_per_joint.tolist()]
                ep["JointError/per_joint_rmse_rad"] = [float(x) for x in rmse_per_joint.tolist()]
                ep["JointError/per_joint_max_abs_rad"] = [
                    float(x) for x in max_abs_per_joint.tolist()
                ]

                if mjwarp is not None:
                    ee_mae_m = (ee_sum / steps).mean()
                    ee_rmse_m = torch.sqrt(ee_sum_sq / steps).mean()
                    ee_max_m = ee_max.mean()
                    ep["EEError/mae_mm"] = float(ee_mae_m.item() * 1000.0)
                    ep["EEError/rmse_mm"] = float(ee_rmse_m.item() * 1000.0)
                    ep["EEError/max_mm"] = float(ee_max_m.item() * 1000.0)

                vel_rms_per_joint = torch.sqrt((sum_vel_sq / steps).mean(dim=0))
                acc_rms_per_joint = torch.sqrt((sum_acc_sq / steps).mean(dim=0))
                ep["JointDyn/vel_rms_mean_rad_s"] = float(vel_rms_per_joint.mean().item())
                ep["JointDyn/acc_rms_mean_rad_s2"] = float(acc_rms_per_joint.mean().item())

                if gain_steps > 0:
                    kp_mean = (kp_sum / float(gain_steps)).mean(dim=0)
                    kd_mean = (kd_sum / float(gain_steps)).mean(dim=0)
                    ep["Gain/kp_mean"] = float(kp_mean.mean().item())
                    ep["Gain/kp_min"] = float(kp_min.mean().item())
                    ep["Gain/kp_max"] = float(kp_max.mean().item())
                    ep["Gain/kd_mean"] = float(kd_mean.mean().item())
                    ep["Gain/kd_min"] = float(kd_min.mean().item())
                    ep["Gain/kd_max"] = float(kd_max.mean().item())
                    ep["Gain/per_joint_kp_mean"] = [float(x) for x in kp_mean.tolist()]
                    ep["Gain/per_joint_kd_mean"] = [float(x) for x in kd_mean.tolist()]

                if ff_tau_steps > 0:
                    assert ff_tau_abs_sum is not None and ff_tau_abs_max is not None
                    tau_abs_mean = (ff_tau_abs_sum / float(ff_tau_steps)).mean(dim=0)
                    tau_abs_max_mean = ff_tau_abs_max.mean(dim=0)
                    ep["FF/abs_mean"] = float(tau_abs_mean.mean().item())
                    ep["FF/abs_max_mean"] = float(tau_abs_max_mean.mean().item())
                    ep["FF/per_joint_abs_mean"] = [float(x) for x in tau_abs_mean.tolist()]
                    ep["FF/per_joint_abs_max"] = [float(x) for x in tau_abs_max_mean.tolist()]
                    assert ff_tau_sq_sum is not None
                    ff_rms = torch.sqrt((ff_tau_sq_sum / float(ff_tau_steps)).mean(dim=0))
                    ep["FF/rms_mean"] = float(ff_rms.mean().item())
                    ep["FF/per_joint_rms"] = [float(x) for x in ff_rms.tolist()]
                    try:
                        tau_limit = float(getattr(pd_cfg, "tau_limit", float("inf")))
                        if tau_limit < float("inf"):
                            sat = tau_abs_max_mean >= (0.98 * tau_limit)
                            ep["FF/sat_any_frac"] = float(sat.to(dtype=torch.float32).mean().item())
                    except Exception:
                        pass

                if id_tau_steps > 0:
                    assert id_tau_abs_sum is not None and id_tau_abs_max is not None
                    tau_abs_mean = (id_tau_abs_sum / float(id_tau_steps)).mean(dim=0)
                    tau_abs_max_mean = id_tau_abs_max.mean(dim=0)
                    ep["ID/abs_mean"] = float(tau_abs_mean.mean().item())
                    ep["ID/abs_max_mean"] = float(tau_abs_max_mean.mean().item())
                    ep["ID/per_joint_abs_mean"] = [float(x) for x in tau_abs_mean.tolist()]
                    ep["ID/per_joint_abs_max"] = [float(x) for x in tau_abs_max_mean.tolist()]
                    assert id_tau_sq_sum is not None
                    id_rms = torch.sqrt((id_tau_sq_sum / float(id_tau_steps)).mean(dim=0))
                    ep["ID/rms_mean"] = float(id_rms.mean().item())
                    ep["ID/per_joint_rms"] = [float(x) for x in id_rms.tolist()]
                    if id_tau_sat_denom is not None and float(id_tau_sat_denom.mean().item()) > 0.0:
                        assert id_tau_sat_cnt is not None and id_tau_sat_per_joint_cnt is not None
                        denom = id_tau_sat_denom
                        ep["ID/sat_frac"] = float((id_tau_sat_cnt / denom).mean().item())
                        per_joint = (id_tau_sat_per_joint_cnt / denom).mean(dim=0)
                        ep["ID/per_joint_sat_frac"] = [float(x) for x in per_joint.tolist()]

                if total_tau_steps > 0:
                    assert total_tau_abs_sum is not None and total_tau_abs_max is not None
                    tau_abs_mean = (total_tau_abs_sum / float(total_tau_steps)).mean(dim=0)
                    tau_abs_max_mean = total_tau_abs_max.mean(dim=0)
                    ep["TorqueTotal/abs_mean"] = float(tau_abs_mean.mean().item())
                    ep["TorqueTotal/abs_max_mean"] = float(tau_abs_max_mean.mean().item())
                    ep["TorqueTotal/per_joint_abs_mean"] = [float(x) for x in tau_abs_mean.tolist()]
                    ep["TorqueTotal/per_joint_abs_max"] = [float(x) for x in tau_abs_max_mean.tolist()]
                    assert total_tau_sq_sum is not None
                    total_rms = torch.sqrt((total_tau_sq_sum / float(total_tau_steps)).mean(dim=0))
                    ep["TorqueTotal/rms_mean"] = float(total_rms.mean().item())
                    ep["TorqueTotal/per_joint_rms"] = [float(x) for x in total_rms.tolist()]
                    if total_tau_sat_denom is not None and float(total_tau_sat_denom.mean().item()) > 0.0:
                        assert total_tau_sat_cnt is not None and total_tau_sat_per_joint_cnt is not None
                        denom = total_tau_sat_denom
                        ep["TorqueTotal/sat_frac"] = float((total_tau_sat_cnt / denom).mean().item())
                        per_joint = (total_tau_sat_per_joint_cnt / denom).mean(dim=0)
                        ep["TorqueTotal/per_joint_sat_frac"] = [float(x) for x in per_joint.tolist()]

                # Alignment stats: projection and correlation between components and total torque.
                eps = 1e-12
                if ff_align_steps > 0 and ff_total_sum is not None and ff_total_sq_sum is not None:
                    assert ff_comp_sum is not None and ff_comp_sq_sum is not None and ff_dot_sum is not None
                    t_mean = (ff_total_sum / float(ff_align_steps)).mean(dim=0)
                    t2_mean = (ff_total_sq_sum / float(ff_align_steps)).mean(dim=0)
                    f_mean = (ff_comp_sum / float(ff_align_steps)).mean(dim=0)
                    f2_mean = (ff_comp_sq_sum / float(ff_align_steps)).mean(dim=0)
                    tf_mean = (ff_dot_sum / float(ff_align_steps)).mean(dim=0)
                    t_var = torch.clamp(t2_mean - t_mean * t_mean, min=0.0)
                    f_var = torch.clamp(f2_mean - f_mean * f_mean, min=0.0)
                    cov = tf_mean - t_mean * f_mean
                    corr = cov / torch.sqrt(t_var * f_var + eps)
                    proj = tf_mean / (t2_mean + eps)
                    ep["FF/proj_to_total_mean"] = float(proj.mean().item())
                    ep["FF/per_joint_proj_to_total"] = [float(x) for x in proj.tolist()]
                    ep["FF/corr_with_total_mean"] = float(corr.mean().item())
                    ep["FF/per_joint_corr_with_total"] = [float(x) for x in corr.tolist()]

                if id_align_steps > 0 and id_total_sum is not None and id_total_sq_sum is not None:
                    assert id_comp_sum is not None and id_comp_sq_sum is not None and id_dot_sum is not None
                    t_mean = (id_total_sum / float(id_align_steps)).mean(dim=0)
                    t2_mean = (id_total_sq_sum / float(id_align_steps)).mean(dim=0)
                    i_mean = (id_comp_sum / float(id_align_steps)).mean(dim=0)
                    i2_mean = (id_comp_sq_sum / float(id_align_steps)).mean(dim=0)
                    ti_mean = (id_dot_sum / float(id_align_steps)).mean(dim=0)
                    t_var = torch.clamp(t2_mean - t_mean * t_mean, min=0.0)
                    i_var = torch.clamp(i2_mean - i_mean * i_mean, min=0.0)
                    cov = ti_mean - t_mean * i_mean
                    corr = cov / torch.sqrt(t_var * i_var + eps)
                    proj = ti_mean / (t2_mean + eps)
                    ep["ID/proj_to_total_mean"] = float(proj.mean().item())
                    ep["ID/per_joint_proj_to_total"] = [float(x) for x in proj.tolist()]
                    ep["ID/corr_with_total_mean"] = float(corr.mean().item())
                    ep["ID/per_joint_corr_with_total"] = [float(x) for x in corr.tolist()]

                # RMS ratios vs total RMS.
                if total_tau_steps > 0 and total_tau_sq_sum is not None:
                    total_rms = torch.sqrt((total_tau_sq_sum / float(total_tau_steps)).mean(dim=0))
                    denom = float(total_rms.mean().item()) + 1e-12
                    if ff_tau_steps > 0 and ff_tau_sq_sum is not None:
                        ff_rms = torch.sqrt((ff_tau_sq_sum / float(ff_tau_steps)).mean(dim=0))
                        ep["FF/rms_ratio_to_total"] = float(ff_rms.mean().item() / denom)
                    if id_tau_steps > 0 and id_tau_sq_sum is not None:
                        id_rms = torch.sqrt((id_tau_sq_sum / float(id_tau_steps)).mean(dim=0))
                        ep["ID/rms_ratio_to_total"] = float(id_rms.mean().item() / denom)

                if torque_steps > 0:
                    assert torque_abs_sum is not None and torque_abs_max is not None
                    tau_abs_mean = (torque_abs_sum / float(torque_steps)).mean(dim=0)
                    tau_abs_max_mean = torque_abs_max.mean(dim=0)
                    ep["Torque/abs_mean"] = float(tau_abs_mean.mean().item())
                    ep["Torque/abs_max_mean"] = float(tau_abs_max_mean.mean().item())
                    ep["Torque/per_joint_abs_mean"] = [float(x) for x in tau_abs_mean.tolist()]
                    ep["Torque/per_joint_abs_max"] = [float(x) for x in tau_abs_max_mean.tolist()]

                logs.append(ep)

                # Reset accumulators for next episode.
                sum_abs.zero_()
                sum_sq.zero_()
                max_abs.zero_()
                steps.zero_()
                ee_sum.zero_()
                ee_sum_sq.zero_()
                ee_max.zero_()
                sum_vel_sq.zero_()
                sum_acc_sq.zero_()
                prev_vel = robot.data.joint_vel.clone()

                gain_steps = 0
                if has_pd_gains or has_pd_gains_ff:
                    kp_sum.zero_()
                    kd_sum.zero_()
                    kp_min.fill_(float("inf"))
                    kd_min.fill_(float("inf"))
                    kp_max.fill_(float("-inf"))
                    kd_max.fill_(float("-inf"))
                if has_pd_gains_ff:
                    ff_tau_steps = 0
                    assert ff_tau_abs_sum is not None and ff_tau_abs_max is not None
                    ff_tau_abs_sum.zero_()
                    ff_tau_abs_max.zero_()
                    if ff_tau_sq_sum is not None:
                        ff_tau_sq_sum.zero_()
                    ff_align_steps = 0
                    if ff_total_sum is not None:
                        ff_total_sum.zero_()
                    if ff_total_sq_sum is not None:
                        ff_total_sq_sum.zero_()
                    if ff_comp_sum is not None:
                        ff_comp_sum.zero_()
                    if ff_comp_sq_sum is not None:
                        ff_comp_sq_sum.zero_()
                    if ff_dot_sum is not None:
                        ff_dot_sum.zero_()

                    id_align_steps = 0
                    if id_total_sum is not None:
                        id_total_sum.zero_()
                    if id_total_sq_sum is not None:
                        id_total_sq_sum.zero_()
                    if id_comp_sum is not None:
                        id_comp_sum.zero_()
                    if id_comp_sq_sum is not None:
                        id_comp_sq_sum.zero_()
                    if id_dot_sum is not None:
                        id_dot_sum.zero_()
                    total_tau_steps = 0
                    id_tau_steps = 0
                    if total_tau_abs_sum is not None:
                        total_tau_abs_sum.zero_()
                    if total_tau_abs_max is not None:
                        total_tau_abs_max.zero_()
                    if total_tau_sq_sum is not None:
                        total_tau_sq_sum.zero_()
                    if total_tau_sat_cnt is not None:
                        total_tau_sat_cnt.zero_()
                    if total_tau_sat_per_joint_cnt is not None:
                        total_tau_sat_per_joint_cnt.zero_()
                    if total_tau_sat_denom is not None:
                        total_tau_sat_denom.zero_()
                    if id_tau_abs_sum is not None:
                        id_tau_abs_sum.zero_()
                    if id_tau_abs_max is not None:
                        id_tau_abs_max.zero_()
                    if id_tau_sq_sum is not None:
                        id_tau_sq_sum.zero_()
                    if id_tau_sat_cnt is not None:
                        id_tau_sat_cnt.zero_()
                    if id_tau_sat_per_joint_cnt is not None:
                        id_tau_sat_per_joint_cnt.zero_()
                    if id_tau_sat_denom is not None:
                        id_tau_sat_denom.zero_()

                torque_steps = 0
                if has_torque_action:
                    assert torque_abs_sum is not None and torque_abs_max is not None
                    torque_abs_sum.zero_()
                    torque_abs_max.zero_()
                break

    return logs


def _summarize(episode_logs: list[dict[str, float]]) -> dict[str, float]:
    if not episode_logs:
        return {}
    keys = sorted({k for d in episode_logs for k in d.keys()})
    summary: dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in episode_logs if k in d]
        if not vals:
            continue
        # Skip non-scalar metrics (e.g. per-joint lists).
        if not all(isinstance(v, (float, int)) for v in vals):
            continue
        summary[f"{k}/mean"] = float(sum(vals) / len(vals))
        summary[f"{k}/min"] = float(min(vals))
        summary[f"{k}/max"] = float(max(vals))
    return summary


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return [int(args.seed)]


def main() -> None:
    args = _parse_args()
    configure_torch_backends()

    task_id = args.task
    device = _resolve_device(args.device)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args)
    if not seeds:
        raise SystemExit("No seeds provided.")

    if args.video:
        args.num_envs = 1
        if len(seeds) > 1:
            print("[WARN] --video with multiple seeds: recording only the first seed.")
            seeds = seeds[:1]

    ckpt = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if ckpt is None or not ckpt.exists():
        raise SystemExit("--checkpoint is required and must exist (e.g. .../model_999.pt)")
    ff_ckpt = Path(args.ff_checkpoint).expanduser().resolve() if args.ff_checkpoint else None
    if ff_ckpt is not None and not ff_ckpt.exists():
        raise SystemExit("--ff-checkpoint must exist when provided.")

    per_seed: list[dict[str, Any]] = []
    all_baseline_logs: list[dict[str, float]] = []
    all_trained_logs: list[dict[str, float]] = []
    all_ff_logs: list[dict[str, float]] = []

    # Cache baseline mapping parameters from the first env (they are task config constants).
    kp_scale = kp_offset = kd_scale = kd_offset = None
    baseline_per_joint_kp: list[float] | None = None
    baseline_per_joint_kd: list[float] | None = None

    for seed in seeds:
        # Ensure determinism of command sampling as much as possible.
        torch.manual_seed(seed)

        trained_logs: list[dict[str, float]] = []
        baseline_logs: list[dict[str, float]] = []

        if args.baseline == "trained_mean":
            # Run trained policy first to estimate per-joint mean gains for a fair fixed-gain baseline.
            trained_video_dir = (
                out_path.parent / "videos" / "trained" if args.video else None
            )
            trained_env = _make_env(
                task_id=task_id,
                device=device,
                num_envs=args.num_envs,
                seed=seed,
                video=args.video,
                video_folder=trained_video_dir,
                video_length=args.video_length,
                video_height=args.video_height,
                video_width=args.video_width,
            )
            trained_policy = _load_trained_policy(
                task_id=task_id, checkpoint=ckpt, env=trained_env, device=device
            )
            trained_logs = _run_eval(
                env=trained_env,
                policy=trained_policy,
                episodes=args.episodes,
                device=device,
            )

            term_key, _, _ = _resolve_gain_action_layout(trained_env)
            pd_cfg = trained_env.unwrapped.cfg.actions[term_key]
            kp_scale = float(getattr(pd_cfg, "kp_scale"))
            kp_offset = float(getattr(pd_cfg, "kp_offset"))
            kd_scale = float(getattr(pd_cfg, "kd_scale"))
            kd_offset = float(getattr(pd_cfg, "kd_offset"))

            kp_means = [
                d.get("Gain/per_joint_kp_mean")
                for d in trained_logs
                if isinstance(d.get("Gain/per_joint_kp_mean"), list)
            ]
            kd_means = [
                d.get("Gain/per_joint_kd_mean")
                for d in trained_logs
                if isinstance(d.get("Gain/per_joint_kd_mean"), list)
            ]
            if not kp_means or not kd_means:
                raise RuntimeError("Expected Gain/per_joint_*_mean in trained logs.")
            dof = len(kp_means[0])
            baseline_per_joint_kp = [
                float(sum(v[i] for v in kp_means) / len(kp_means)) for i in range(dof)
            ]
            baseline_per_joint_kd = [
                float(sum(v[i] for v in kd_means) / len(kd_means)) for i in range(dof)
            ]
            trained_env.close()

            # Re-seed so baseline command sampling is comparable.
            torch.manual_seed(seed)

            baseline_video_dir = (
                out_path.parent / "videos" / "baseline" if args.video else None
            )
            baseline_env = _make_env(
                task_id=task_id,
                device=device,
                num_envs=args.num_envs,
                seed=seed,
                video=args.video,
                video_folder=baseline_video_dir,
                video_length=args.video_length,
                video_height=args.video_height,
                video_width=args.video_width,
            )
            _, dof, action_dim = _resolve_gain_action_layout(baseline_env)
            if baseline_per_joint_kp is None or baseline_per_joint_kd is None:
                raise RuntimeError("Failed to compute trained_mean baseline gains.")
            if len(baseline_per_joint_kp) != dof or len(baseline_per_joint_kd) != dof:
                raise RuntimeError(
                    f"Baseline gain DOF mismatch: got {len(baseline_per_joint_kp)} expected {dof}"
                )

            kp_raw_vec = torch.tensor(
                [(k - kp_offset) / kp_scale for k in baseline_per_joint_kp],
                device=device,
                dtype=torch.float32,
            ).clamp(-1.0, 1.0)
            kd_raw_vec = torch.tensor(
                [(k - kd_offset) / kd_scale for k in baseline_per_joint_kd],
                device=device,
                dtype=torch.float32,
            ).clamp(-1.0, 1.0)

            baseline_action = torch.zeros((baseline_env.num_envs, action_dim), device=device)
            baseline_action[:, :dof] = kp_raw_vec[None, :]
            baseline_action[:, dof : 2 * dof] = kd_raw_vec[None, :]
            baseline_policy = ConstantActionPolicy(baseline_action)
            baseline_logs = _run_eval(
                env=baseline_env,
                policy=baseline_policy,
                episodes=args.episodes,
                device=device,
                skip_steps=int(args.skip_steps),
            )
            baseline_env.close()

        else:
            baseline_video_dir = (
                out_path.parent / "videos" / "baseline" if args.video else None
            )
            baseline_env = _make_env(
                task_id=task_id,
                device=device,
                num_envs=args.num_envs,
                seed=seed,
                video=args.video,
                video_folder=baseline_video_dir,
                video_length=args.video_length,
                video_height=args.video_height,
                video_width=args.video_width,
            )

            term_key, dof, action_dim = _resolve_gain_action_layout(baseline_env)
            pd_cfg = baseline_env.unwrapped.cfg.actions[term_key]
            kp_scale = float(getattr(pd_cfg, "kp_scale"))
            kp_offset = float(getattr(pd_cfg, "kp_offset"))
            kd_scale = float(getattr(pd_cfg, "kd_scale"))
            kd_offset = float(getattr(pd_cfg, "kd_offset"))

            if args.baseline == "custom":
                kp_raw = (float(args.baseline_kp) - kp_offset) / kp_scale
                kd_raw = (float(args.baseline_kd) - kd_offset) / kd_scale
                kp_raw = float(max(-1.0, min(1.0, kp_raw)))
                kd_raw = float(max(-1.0, min(1.0, kd_raw)))
            else:
                kp_raw, kd_raw = 0.0, 0.0

            baseline_action = torch.zeros((baseline_env.num_envs, action_dim), device=device)
            baseline_action[:, :dof] = kp_raw
            baseline_action[:, dof : 2 * dof] = kd_raw
            baseline_policy = ConstantActionPolicy(baseline_action)

            baseline_logs = _run_eval(
                env=baseline_env,
                policy=baseline_policy,
                episodes=args.episodes,
                device=device,
            )
            baseline_env.close()

            # Trained env/policy (re-seed so command sampling matches baseline)
            torch.manual_seed(seed)

            trained_video_dir = out_path.parent / "videos" / "trained" if args.video else None
            trained_env = _make_env(
                task_id=task_id,
                device=device,
                num_envs=args.num_envs,
                seed=seed,
                video=args.video,
                video_folder=trained_video_dir,
                video_length=args.video_length,
                video_height=args.video_height,
                video_width=args.video_width,
            )

            trained_policy = _load_trained_policy(
                task_id=task_id, checkpoint=ckpt, env=trained_env, device=device
            )
            trained_logs = _run_eval(
                env=trained_env,
                policy=trained_policy,
                episodes=args.episodes,
                device=device,
                skip_steps=int(args.skip_steps),
            )
            trained_env.close()

        ff_logs: list[dict[str, float]] = []
        if ff_ckpt is not None:
            torch.manual_seed(seed)
            ff_env = _make_env(
                task_id=args.ff_task,
                device=device,
                num_envs=args.num_envs,
                seed=seed,
                video=False,
                video_folder=None,
                video_length=args.video_length,
                video_height=args.video_height,
                video_width=args.video_width,
            )
            ff_policy = _load_trained_policy(
                task_id=args.ff_task, checkpoint=ff_ckpt, env=ff_env, device=device
            )
            ff_logs = _run_eval(
                env=ff_env,
                policy=ff_policy,
                episodes=args.episodes,
                device=device,
                skip_steps=int(args.skip_steps),
            )
            ff_env.close()

        all_baseline_logs.extend(baseline_logs)
        all_trained_logs.extend(trained_logs)
        all_ff_logs.extend(ff_logs)
        entry: dict[str, Any] = {
            "seed": int(seed),
            "baseline": {"episode_logs": baseline_logs, "summary": _summarize(baseline_logs)},
            "trained": {"episode_logs": trained_logs, "summary": _summarize(trained_logs)},
        }
        if ff_ckpt is not None:
            entry["ff"] = {"episode_logs": ff_logs, "summary": _summarize(ff_logs)}
        per_seed.append(entry)

    payload = {
        "task": task_id,
        "checkpoint": str(ckpt),
        "ff_task": args.ff_task,
        "ff_checkpoint": str(ff_ckpt) if ff_ckpt is not None else "",
        "seeds": [int(s) for s in seeds],
        "num_envs": int(args.num_envs),
        "episodes": int(args.episodes),
        "baseline": {
            "mode": args.baseline,
            "kp": float(args.baseline_kp),
            "kd": float(args.baseline_kd),
            "kp_scale": float(kp_scale) if kp_scale is not None else None,
            "kp_offset": float(kp_offset) if kp_offset is not None else None,
            "kd_scale": float(kd_scale) if kd_scale is not None else None,
            "kd_offset": float(kd_offset) if kd_offset is not None else None,
            "per_joint_kp": baseline_per_joint_kp,
            "per_joint_kd": baseline_per_joint_kd,
            "episode_logs": all_baseline_logs,
            "summary": _summarize(all_baseline_logs),
        },
        "trained": {
            "episode_logs": all_trained_logs,
            "summary": _summarize(all_trained_logs),
        },
        "ff": {"episode_logs": all_ff_logs, "summary": _summarize(all_ff_logs)}
        if ff_ckpt is not None
        else {},
        "per_seed": per_seed,
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK] wrote {out_path}")
    print("\nBaseline summary (selected):")
    for k in sorted(payload["baseline"]["summary"].keys()):
        if k.endswith("/mean") and any(
            s in k for s in ("Episode_Reward", "Episode_Termination", "JointError", "JointDyn", "Gain", "EEError")
        ):
            print(f"  {k}: {payload['baseline']['summary'][k]:.4f}")
    print("\nTrained summary (selected):")
    for k in sorted(payload["trained"]["summary"].keys()):
        if k.endswith("/mean") and any(
            s in k
            for s in ("Episode_Reward", "Episode_Termination", "JointError", "JointDyn", "Gain", "Torque", "EEError")
        ):
            print(f"  {k}: {payload['trained']['summary'][k]:.4f}")
    if ff_ckpt is not None:
        print("\nGain+FF summary (selected):")
        for k in sorted(payload["ff"]["summary"].keys()):
            if k.endswith("/mean") and any(
                s in k
                for s in (
                    "Episode_Reward",
                    "Episode_Termination",
                    "JointError",
                    "JointDyn",
                    "Gain",
                    "Torque",
                    "EEError",
                )
            ):
                print(f"  {k}: {payload['ff']['summary'][k]:.4f}")


if __name__ == "__main__":
    main()
