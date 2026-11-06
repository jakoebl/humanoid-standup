#!/usr/bin/env python3
"""Simulator checks for BeyondMimic dimensions, resets, and reference-free actor."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import humanoid_standup.tasks  # noqa: F401


def main():
    cfg = parse_env_cfg("BeyondMimic-Play", device=args.device, num_envs=2)
    cfg.commands.motion.debug_vis = False
    env = gym.make("BeyondMimic-Play", cfg=cfg)
    try:
        raw = env.unwrapped
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 93)
        assert obs["critic"].shape == (2, 285)
        gravity = raw.scene["imu"].data.projected_gravity_b
        torch.testing.assert_close(obs["policy"][:, :3], gravity)
        torch.testing.assert_close(torch.linalg.vector_norm(gravity, dim=-1), torch.ones(2, device=raw.device))
        assert raw.action_manager.total_action_dim == 29
        assert "joint_limit" in raw.reward_manager.active_terms
        command = raw.command_manager.get_term("motion")
        assert (command.time_steps == 0).all()
        before = raw.observation_manager.compute()["policy"].clone()
        command.time_steps[:] = 100
        after = raw.observation_manager.compute()["policy"]
        torch.testing.assert_close(before, after)
        # Partial reference reset must not alter the other environment's frame.
        command._resample_command(torch.tensor([0], device=raw.device))
        assert command.time_steps.tolist() == [0, 100]
        command.time_steps[:] = command.motion.time_step_total - 1
        command._update_command()
        assert (command.time_steps == 0).all()
        for _ in range(12):
            obs, reward, _, _, _ = env.step(torch.zeros((2, 29), device=raw.device))
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all()
        print("PASS: 93/286 observations, 29 actions, reference-free actor, partial resets, clip-end restart, finite rollouts.")
    finally:
        env.close()


try:
    main()
finally:
    app.close()
