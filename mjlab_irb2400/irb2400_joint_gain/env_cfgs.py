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
    action_rate_l2,
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
    joint_pd_gain_action,
    joint_sine_command,
    joint_tracking_rewards,
)

# Joint friction (object model): Coulomb + viscous.
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


def irb2400_joint_gain_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    mjcf_path = _irb2400_mjcf_path()
    if not mjcf_path.exists():
        raise FileNotFoundError(mjcf_path)

    def get_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(mjcf_path))
        for name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"):
            spec.joint(name).damping = float(VISCOUS_DAMPING)
        # Some mujoco/mjwarp builds ignore <size> in XML; enforce programmatically.
        for obj, attr in ((spec, "njmax"), (spec, "nconmax")):
            if hasattr(obj, attr):
                setattr(obj, attr, 64)
        if hasattr(spec, "size"):
            for attr in ("njmax", "nconmax"):
                if hasattr(spec.size, attr):
                    setattr(spec.size, attr, 64)
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
        "pd_gains": joint_pd_gain_action.JointPdGainActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            command_name="joint_pos",
            # Wider gain ranges for pure-PD tracking (no gravity/bias feedforward).
            # action in [-1, 1] maps to: gain = action*scale + offset.
            kp_scale=800.0,
            kp_offset=1000.0,
            kd_scale=120.0,
            kd_offset=120.0,
            # Avoid degenerate low gains that let the arm "droop" into joint limits.
            clip_kp=(200.0, 2000.0),
            clip_kd=(10.0, 400.0),
            update_period_s=0.01,
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
        # Accuracy-centric: make true joint error the main objective (no saturation).
        "joint_pos_err_l2": RewardTermCfg(
            func=joint_tracking_rewards.joint_pos_error_l2,
            weight=-80.0,
            params={
                "command_name": "joint_pos",
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        # Optional shaping for logs: bounded score in (0, 1].
        "track_joint_pos": RewardTermCfg(
            func=joint_tracking_rewards.track_joint_pos,
            weight=1.0,
            params={
                "command_name": "joint_pos",
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "std": 0.08,
            },
        ),
        # Not a primary objective here; keep as a weak regularizer.
        "torque_l2": RewardTermCfg(func=joint_torques_l2, weight=-2e-6),
        "joint_vel_l2": RewardTermCfg(
            func=joint_vel_l2,
            # Smoothness: discourage oscillations / high velocities.
            weight=-5e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=joint_tracking_rewards.joint_acc_l2_clipped,
            weight=-5e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "clip": 50.0},
        ),
        "joint_pos_limits": RewardTermCfg(
            func=joint_pos_limits,
            weight=-20.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        # Smooth gain scheduling at the *applied* update rate (controller holds at 100Hz).
        "action_rate_l2": RewardTermCfg(
            func=joint_tracking_rewards.held_action_rate_l2,
            weight=-1e-2,
            params={"term_name": "pd_gains"},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=joint_tracking_rewards.time_out, time_out=True),
    }

    sim = SimulationCfg(
        # Prevent constraint buffer overflow during aggressive exploration.
        njmax=64,
        nconmax=64,
        mujoco=MujocoCfg(
            timestep=0.001,
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
        # Physics runs at 1kHz; training uses decimation=5 (5ms env step).
        # NN updates are held to 100Hz inside the action term.
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
        viewer=ViewerConfig(),
    )
