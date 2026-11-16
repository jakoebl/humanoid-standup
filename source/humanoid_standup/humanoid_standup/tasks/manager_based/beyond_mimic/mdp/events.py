import torch

from isaaclab.utils.math import quat_apply_inverse


def encoder_bias(env, env_ids, bias_range=(-0.01, 0.01)):
    """Add observation-only encoder bias, preserving nominal action offsets."""
    command = env.command_manager.get_term("motion")
    if not hasattr(command, "encoder_bias"):
        command.encoder_bias = torch.zeros(env.num_envs, 29, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    command.encoder_bias[env_ids] = torch.empty(len(env_ids), 29, device=env.device).uniform_(*bias_range)


def vertical_assist_force(env, env_ids, asset_cfg, scale):
    """Push the selected bodies upward with scale * robot weight."""

    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids

    # Save the robot weight as an environment variable
    if not hasattr(env, "_assist_robot_weight"):
        total_mass = asset.root_physx_view.get_masses().sum(dim=1).to(env.device)
        env._assist_robot_weight = total_mass * abs(env.cfg.sim.gravity[2])

    # Calculate the scaled vertical force
    force_w = torch.zeros(env.num_envs, len(body_ids), 3, device=env.device)
    force_w[..., 2] = (scale * env._assist_robot_weight / len(body_ids)).unsqueeze(-1)

    # Rotate the force into body frame and apply it
    force_b = quat_apply_inverse(asset.data.body_link_quat_w[:, body_ids], force_w)
    asset.permanent_wrench_composer.set_forces_and_torques(forces=force_b, body_ids=body_ids)
