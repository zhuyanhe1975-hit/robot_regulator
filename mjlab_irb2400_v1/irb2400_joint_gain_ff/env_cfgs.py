import math
from pathlib import Path

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.observations import generated_commands, joint_pos_rel, joint_vel_rel, last_action
from mjlab.envs.mdp.rewards import action_rate_l2, joint_pos_limits, joint_torques_l2, joint_vel_l2
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import reset_joints_by_offset
from mjlab.viewer import ViewerConfig

from mjlab_irb2400_v1.mdp import joint_pd_gain_ff_action, joint_sine_command, joint_tracking_rewards

COULOMB_FRICTION = 2.0
VISCOUS_DAMPING = 0.2


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "models" / "abb_irb2400").exists():
            return p
    raise FileNotFoundError("Could not locate repo root containing models/abb_irb2400")


def _irb2400_mjcf_path() -> Path:
    return _repo_root() / "models" / "abb_irb2400" / "mjcf" / "irb2400_mjlab.xml"


def irb2400_joint_gain_ff_env_cfg_v1(*, play: bool = False, use_integral: bool = False) -> ManagerBasedRlEnvCfg:
    mjcf_path = _irb2400_mjcf_path()
    if not mjcf_path.exists():
        raise FileNotFoundError(mjcf_path)

    def get_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(mjcf_path))
        for name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"):
            spec.joint(name).damping = float(VISCOUS_DAMPING)
        return spec

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "joint_pos_rel": ObservationTermCfg(
                    func=joint_pos_rel,
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
                ),
                "joint_vel_rel": ObservationTermCfg(
                    func=joint_vel_rel,
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
                ),
                "joint_cmd": ObservationTermCfg(
                    func=generated_commands,
                    params={"command_name": "joint_pos"},
                ),
                "actions": ObservationTermCfg(func=last_action),
            },
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "joint_pos_rel": ObservationTermCfg(
                    func=joint_pos_rel,
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
                ),
                "joint_vel_rel": ObservationTermCfg(
                    func=joint_vel_rel,
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
                ),
                "joint_cmd": ObservationTermCfg(
                    func=generated_commands,
                    params={"command_name": "joint_pos"},
                ),
                "actions": ObservationTermCfg(func=last_action),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "pd_gains_ff": joint_pd_gain_ff_action.JointPdGainFfActionV1Cfg(
            asset_name="robot",
            actuator_names=(".*",),
            command_name="joint_pos",
            # Action mapping: raw in [-1,1] -> physical gains.
            # kp_raw=1 => kp=8000; kd_raw=1 => kd=2000 (then clipped).
            kp_scale=7000.0,
            kp_offset=1000.0,
            kd_scale=1880.0,
            kd_offset=120.0,
            clip_kp=(200.0, 8000.0),
            clip_kd=(10.0, 2000.0),
            tau_scale=160.0,
            tau_limit=1200.0,
            tau_slew_rate=6000.0,
            use_inverse_dynamics=True,
            id_scale=1.0,
            id_limit=8000.0,
            use_integral=use_integral,
            # Acceleration feedforward (M(q) qdd_ref) computed every 10ms and held (ZOH).
            use_acc_feedforward=True,
            # With command_acc available at 2ms, compute acc FF every physics step.
            acc_update_period_s=0.0,
            acc_scale=0.2,
            acc_limit=8000.0,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "joint_pos": joint_sine_command.RandomSineJointPositionCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            joint_names=(".*",),
            amp_range=(0.15, 0.6),
            freq_range=(0.05, 0.35),
            phase_range=(-math.pi, math.pi),
            ramp_time=0.5,
            debug_vis=play,
        )
    }

    events = {
        "reset_robot_joints": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    rewards = {
        # Optimize end-effector tracking accuracy (world position) w.r.t. the reference joint command.
        "ee_pos_err_l2": RewardTermCfg(
            func=joint_tracking_rewards.ee_pos_error_l2_from_joint_command,
            weight=-500.0,  # m^2 scale
            params={"command_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "site_name": "ee"},
        ),
        "track_ee_pos": RewardTermCfg(
            func=joint_tracking_rewards.track_ee_pos_from_joint_command,
            weight=1.0,
            params={
                "command_name": "joint_pos",
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "site_name": "ee",
                "std": 0.005,  # 5mm
            },
        ),
        # Keep joint tracking terms for logging only.
        "joint_pos_err_l2": RewardTermCfg(
            func=joint_tracking_rewards.joint_pos_error_l2,
            weight=-100.0,
            params={"command_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "track_joint_pos": RewardTermCfg(
            func=joint_tracking_rewards.track_joint_pos,
            weight=1.0,
            params={"command_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "std": 0.01},
        ),
        # De-prioritize regularization terms; keep them only for logging.
        "torque_l2": RewardTermCfg(func=joint_torques_l2, weight=0.0),
        "joint_vel_l2": RewardTermCfg(
            func=joint_vel_l2,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=joint_tracking_rewards.joint_acc_l2_clipped,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "clip": 50.0},
        ),
        "joint_pos_limits": RewardTermCfg(
            func=joint_pos_limits,
            weight=-20.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=0.0),
    }

    terminations = {"time_out": TerminationTermCfg(func=joint_tracking_rewards.time_out, time_out=True)}

    sim = SimulationCfg(
        njmax=64,
        nconmax=64,
        mujoco=MujocoCfg(timestep=0.002, integrator="implicitfast", gravity=(0.0, 0.0, -9.81), iterations=100),
    )

    scene = SceneCfg(
        num_envs=1 if play else 4096,
        env_spacing=2.5,
        entities={
            "robot": joint_tracking_rewards.irb2400_entity_cfg(
                get_spec=get_spec,
                stiffness=250.0,
                damping=40.0,
                effort_limit=8000.0,
                frictionloss=COULOMB_FRICTION,
            )
        },
        extent=2.0,
    )

    return ManagerBasedRlEnvCfg(
        # Keep physics timestep at 2ms while updating the policy at 10ms.
        decimation=5,
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        episode_length_s=4.0,
        sim=sim,
        viewer=ViewerConfig(height=240, width=320),
    )
