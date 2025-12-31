from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


def irb2400_torque_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    # Torque-control is harder; give the policy a bit more capacity and exploration.
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.8,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=(256, 256, 128),
            critic_hidden_dims=(256, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.002,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="irb2400_torque",
        logger="tensorboard",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=1500,
    )
