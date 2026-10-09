import torch


def encoder_bias(env, env_ids, bias_range=(-0.01, 0.01)):
    """Add observation-only encoder bias, preserving nominal action offsets."""
    command = env.command_manager.get_term("motion")
    if not hasattr(command, "encoder_bias"):
        command.encoder_bias = torch.zeros(env.num_envs, 29, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    command.encoder_bias[env_ids] = torch.empty(len(env_ids), 29, device=env.device).uniform_(*bias_range)
