from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg
from .critic import MultiCriticCfg
from .multiclip_ppo import MulticlipPPOCfg


@configclass
class HumanoidStandupPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "humanoid_standup"
    clip_actions = None
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = MultiCriticCfg(
        hidden_dims=[1024, 512, 256],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = MulticlipPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )


@configclass
class HumanoidStandupLiedownPPORunnerCfg(HumanoidStandupPPORunnerCfg):
    experiment_name = "humanoid_standup_liedown"
