import torch
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO


class MulticlipPPO(PPO):
    """PPO that additionally logs episode return, value loss and surrogate loss per motion clip."""

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        alg.clip_names = env.unwrapped.command_manager.get_term("motion").clip_names
        alg.episode_return = torch.zeros(env.num_envs, device=device)
        return alg

    def process_env_step(self, obs, rewards, dones, extras):
        # Resets already switched obs to the next clip; the finished episode's clip is in the pre-step observation.
        clip = self.transition.observations["clip"].argmax(-1)
        self.episode_return += rewards
        done = dones.bool()
        if done.any():
            # Steps with resets get a fresh extras["log"], so each finished episode is logged exactly once.
            for clip_id, name in enumerate(self.clip_names):
                extras["log"][f"Metrics_{name}/episode_return"] = self.episode_return[done & (clip == clip_id)]
            self.episode_return[done] = 0.0
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        losses = super().update()
        # Re-evaluate the rollout with the updated networks; the storage is only rewound, not erased, by update().
        storage = self.storage
        with torch.no_grad():
            obs = storage.observations.flatten(0, 1)
            clip = obs["clip"].argmax(-1)
            self.actor(obs, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(storage.actions.flatten(0, 1))
            values = self.critic(obs).squeeze(-1)
            old_values = storage.values.flatten(0, 1).squeeze(-1)
            returns = storage.returns.flatten(0, 1).squeeze(-1)
            advantages = storage.advantages.flatten(0, 1).squeeze(-1)
            ratio = torch.exp(log_prob - storage.actions_log_prob.flatten(0, 1).squeeze(-1))
            clipped_ratio = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate = torch.max(-advantages * ratio, -advantages * clipped_ratio)
            value_loss = (values - returns).pow(2)
            if self.use_clipped_value_loss:
                value_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(value_loss, (value_clipped - returns).pow(2))
            counts = torch.bincount(clip, minlength=len(self.clip_names))
            value_loss = torch.bincount(clip, weights=value_loss, minlength=len(self.clip_names)) / counts
            surrogate = torch.bincount(clip, weights=surrogate, minlength=len(self.clip_names)) / counts
        for clip_id, name in enumerate(self.clip_names):
            losses[f"{name}/value"] = value_loss[clip_id].item()
            losses[f"{name}/surrogate"] = surrogate[clip_id].item()
        return losses


@configclass
class MulticlipPPOCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "humanoid_standup.tasks.manager_based.beyond_mimic.multiclip_ppo:MulticlipPPO"
