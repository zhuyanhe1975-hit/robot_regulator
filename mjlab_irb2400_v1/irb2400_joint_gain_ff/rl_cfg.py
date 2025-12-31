from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


def irb2400_joint_gain_ff_ppo_runner_cfg_v1() -> RslRlOnPolicyRunnerCfg:
    # Matches logs/rsl_rl/irb2400_joint_gain_ff/2025-12-27_23-41-53/params/agent.yaml
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
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
            entropy_coef=0.003,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="irb2400_joint_gain_ff",
        logger="tensorboard",
        save_interval=200,
        # Env step is 10ms (decimation=5, physics=2ms): keep rollout horizon ~0.48s.
        num_steps_per_env=48,
        max_iterations=500,
    )
