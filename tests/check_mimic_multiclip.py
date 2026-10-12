#!/usr/bin/env python3
"""Integration checks with two distinct, unequal-length prepared motion fixtures."""
import argparse
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BeyondMimic", choices=("BeyondMimic", "Mimic-Multiclip"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
import humanoid_standup.tasks  # noqa: F401


def main():
    source = Path(__file__).resolve().parents[1] / "data/beyondmimic/stand_up_lying_V2.npz"
    with tempfile.TemporaryDirectory() as directory:
        with np.load(source, allow_pickle=False) as original:
            for index, length in enumerate((80, 130)):
                arrays = {key: original[key].copy() for key in original.files}
                for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                    arrays[key] = arrays[key][index * 50:index * 50 + length]
                np.savez(Path(directory) / f"clip_{index}.npz", **arrays)
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=4)
        cfg.commands.motion.motion_directory = directory
        cfg.commands.motion.debug_vis = False
        cfg.observations.policy.enable_corruption = False
        env = gym.make(args.task, cfg=cfg)
        try:
            raw = env.unwrapped
            obs, _ = env.reset()
            command = raw.command_manager.get_term("motion")
            assert obs["policy"].shape == (4, 93) and obs["critic"].shape == (4, 286)
            assert command.motion.lengths.tolist() == [80, 130]
            command.clip_ids[:] = torch.tensor([0, 1, 0, 1], device=raw.device)
            command.time_steps[:] = command.motion.offsets[command.clip_ids] + 10
            before = raw.observation_manager.compute()["policy"].clone()
            command.time_steps[:] += 20
            torch.testing.assert_close(before, raw.observation_manager.compute()["policy"])
            untouched = command.time_steps[1:].clone()
            command._resample_command(torch.tensor([0], device=raw.device))
            torch.testing.assert_close(untouched, command.time_steps[1:])
            command.cfg.sampling_mode = "start"
            command.cfg.play_clip_index = 1
            command.time_steps[:] = command.motion.offsets[command.clip_ids] + command.motion.lengths[command.clip_ids] - 1
            command._update_command()
            assert (command.clip_ids == 1).all() and (command.local_frames == 0).all()
            # Only clip 1's failed-frame histogram is updated.
            raw.termination_manager._terminated_buf[:] = True
            command._record_failures(torch.arange(4, device=raw.device))
            assert command.pending_failures[0].sum() == 0
            assert command.pending_failures[1].sum() == 4
            raw.termination_manager._terminated_buf[:] = False
            for _ in range(12):
                obs, reward, _, _, _ = env.step(torch.zeros((4, 29), device=raw.device))
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            print("PASS: multi-clip loading, independent frames/resets/histograms, boundaries, observations, finite rollouts.", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            raise
        finally:
            env.close()


try:
    main()
finally:
    app.close()
