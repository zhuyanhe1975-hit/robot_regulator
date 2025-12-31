import math
from pathlib import Path

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.observations import (
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
)
from mjlab.envs.mdp.rewards import (
    joint_pos_limits,
    joint_torques_l2,
    joint_vel_l2,
)
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

from mjlab_irb2400.mdp import (
    joint_pd_fixed_idff_action,
    joint_sine_command,
    joint_tracking_rewards,
)

# Joint friction (object model): Coulomb + viscous.
COULOMB_FRICTION = 2.0
VISCOUS_DAMPING = 0.2
JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "models" / "abb_irb2400").exists():
            return p
    raise FileNotFoundError("Could not locate repo root containing models/abb_irb2400")


def _irb2400_mjcf_path() -> Path:
    return _repo_root() / "models" / "abb_irb2400" / "mjcf" / "irb2400_mjlab.xml"


def _apply_joint_friction(spec: mujoco.MjSpec) -> None:
    for name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"):
        j = spec.joint(name)
        j.damping = float(VISCOUS_DAMPING)


def irb2400_pdidff_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Stable baseline: fixed-gain PD + inverse dynamics + NN residual FF.

    Control step is 20ms (env step), physics is 2ms (decimation=10).

    NOTE: Using a 20ms physics timestep with stiff PD/ID typically destabilizes the plant
    (integration error + discrete-time gain mismatch). We keep physics small but still
    update ID/FF at 20ms to match real controller scheduling.
    """
    mjcf_path = _irb2400_mjcf_path()
    if not mjcf_path.exists():
        raise FileNotFoundError(mjcf_path)

    def get_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(mjcf_path))
        _apply_joint_friction(spec)
        return spec

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "joint_pos_rel": ObservationTermCfg(
                    func=joint_pos_rel,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=JOINT_NAMES, preserve_order=True
                        )
                    },
                ),
                "joint_vel_rel": ObservationTermCfg(
                    func=joint_vel_rel,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=JOINT_NAMES, preserve_order=True
                        )
                    },
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
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=JOINT_NAMES, preserve_order=True
                        )
                    },
                ),
                "joint_vel_rel": ObservationTermCfg(
                    func=joint_vel_rel,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=JOINT_NAMES, preserve_order=True
                        )
                    },
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
        "pd_id_ff": joint_pd_fixed_idff_action.JointPdFixedIdFfActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            command_name="joint_pos",
            # Fixed baseline gains (start from a known stable point).
            kp=928.0,
            kd=80.0,
            # Residual authority comparable to plant friction.
            a_max=1.5 * COULOMB_FRICTION,
            b_max=1.5 * VISCOUS_DAMPING,
            sign_eps=0.05,
            tau_limit=200.0,
            tau_slew_rate=800.0,
            # Update once per env step (20ms).
            update_period_s=0.02,
            use_inverse_dynamics=True,
            id_mode="gravity",
            id_scale=1.0,
            id_limit=800.0,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "joint_pos": joint_sine_command.RandomSineJointPositionCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            joint_names=JOINT_NAMES,
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
        "joint_pos_err_l2": RewardTermCfg(
            func=joint_tracking_rewards.joint_pos_error_l2,
            weight=-180.0,
            params={
                "command_name": "joint_pos",
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                ),
            },
        ),
        "track_joint_pos": RewardTermCfg(
            func=joint_tracking_rewards.track_joint_pos,
            weight=1.0,
            params={
                "command_name": "joint_pos",
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                ),
                "std": 0.08,
            },
        ),
        "torque_l2": RewardTermCfg(func=joint_torques_l2, weight=-2e-6),
        "joint_vel_l2": RewardTermCfg(
            func=joint_vel_l2,
            weight=-5e-3,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
        ),
        "joint_acc_l2": RewardTermCfg(
            func=joint_tracking_rewards.joint_acc_l2_clipped,
            weight=-5e-5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                ),
                "clip": 50.0,
            },
        ),
        "joint_pos_limits": RewardTermCfg(
            func=joint_pos_limits,
            weight=-20.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
        ),
        "action_rate_l2": RewardTermCfg(
            func=joint_tracking_rewards.held_action_rate_l2,
            weight=-1e-2,
            params={"term_name": "pd_id_ff"},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=joint_tracking_rewards.time_out, time_out=True),
    }

    sim = SimulationCfg(
        njmax=64,
        nconmax=64,
        mujoco=MujocoCfg(
            timestep=0.002,
            integrator="implicitfast",
            gravity=(0.0, 0.0, -9.81),
        ),
    )

    scene = SceneCfg(
        num_envs=1 if play else 4096,
        env_spacing=2.5,
        entities={
            "robot": joint_tracking_rewards.irb2400_entity_cfg(
                get_spec=get_spec,
                stiffness=250.0,
                damping=40.0,
                effort_limit=800.0,
                frictionloss=COULOMB_FRICTION,
            )
        },
        extent=2.0,
    )

    return ManagerBasedRlEnvCfg(
        # Physics 2ms, control/env step 20ms.
        decimation=10,
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        episode_length_s=4.0,
        sim=sim,
        viewer=ViewerConfig(),
    )
