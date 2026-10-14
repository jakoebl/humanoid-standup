"""Check floor penetration and reset retries in Isaac Sim's Python environment."""

import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from humanoid_standup.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand

path = Path(__file__).resolve().parents[1] / 'source/humanoid_standup/humanoid_standup/tasks/manager_based/beyond_mimic/mdp/collisionchecker.py'
spec = importlib.util.spec_from_file_location('reset_collisions', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResetCollisionTest(unittest.TestCase):
    def command(self, responses):
        # Minimal reference state; mocks record the actual command's retry sequence.
        count = 5
        command = SimpleNamespace(
            device="cpu", num_envs=count,
            cfg=SimpleNamespace(pose_range={}, velocity_range={},
                                reset_joint_noise_attempts=2, reset_root_pose_attempts=3),
            body_pos_w=torch.zeros(count, 1, 3),
            body_quat_w=torch.tensor([[[1., 0., 0., 0.]]] * count),
            body_lin_vel_w=torch.zeros(count, 1, 3), body_ang_vel_w=torch.zeros(count, 1, 3),
            motion=SimpleNamespace(joint_pos=torch.zeros(count, 1), joint_vel=torch.zeros(count, 1)),
            time_steps=torch.zeros(count, dtype=torch.long),
            _sample_frames=Mock(), _sample_and_apply_reset_pose=Mock(), update_relative_body_poses=Mock(),
            _collision_checker=SimpleNamespace(invalid_envs=Mock(side_effect=responses)),
        )
        command._sample_reset_root_pose = Mock(wraps=MethodType(MotionCommand._sample_reset_root_pose, command))
        return command

    def test_retries_only_invalid_environments_and_then_resamples_root(self):
        command = self.command([[1, 2], [2], []])
        MotionCommand._resample_command(command, torch.tensor([0, 1, 2]))
        attempts = [call.args[0] for call in command._sample_and_apply_reset_pose.call_args_list]
        self.assertEqual(attempts, [[0, 1, 2], [1, 2], [2]])
        roots = command._sample_reset_root_pose.call_args_list
        self.assertEqual(roots[0].args[0].tolist(), [0, 1, 2])
        self.assertEqual(roots[1].args[0], [2])
        command.update_relative_body_poses.assert_called_once()

    def test_exhausted_retries_retain_final_pose_and_empty_reset_is_noop(self):
        command = self.command([[3]] * 6)
        MotionCommand._resample_command(command, torch.tensor([3, 4]))
        attempts = [call.args[0] for call in command._sample_and_apply_reset_pose.call_args_list]
        self.assertEqual(attempts, [[3, 4]] + [[3]] * 5)
        roots = command._sample_reset_root_pose.call_args_list
        self.assertEqual([call.args[0] for call in roots[1:]], [[3], [3]])
        command.update_relative_body_poses.assert_called_once()
        empty = self.command([])
        MotionCommand._resample_command(empty, torch.tensor([], dtype=torch.long))
        empty._sample_and_apply_reset_pose.assert_not_called()
        empty._sample_reset_root_pose.assert_not_called()

    def test_point_floor_checks_cover_all_eight_points_and_environment_heights(self):
        # Each environment has a different point below its floor; the last one is valid.
        checker = object.__new__(module.CollisionChecker)
        floor = torch.arange(9, dtype=torch.float32)
        positions = torch.zeros(9, 8, 3)
        positions[:, :, 2] = floor[:, None]
        for index in range(8):
            positions[index, index, 2] -= .01
        checker.robot = SimpleNamespace(device="cpu", data=SimpleNamespace(body_link_pos_w=positions))
        checker.floor_heights = floor
        checker.tolerance = 1e-4
        checker.min_foot_distance = 0.
        checker.point_ids = list(range(8))
        self.assertEqual(checker.invalid_envs(list(range(9))), list(range(8)))
        self.assertEqual(checker.invalid_envs([8, 3, 1]), [3, 1])
        positions[0, 0, 2] = -.00005
        self.assertEqual(checker.invalid_envs([0, 8]), [])
        self.assertEqual(checker.invalid_envs([]), [])

    def test_foot_distance_rejects_close_feet_and_allows_threshold_and_vertical_separation(self):
        # Check horizontal and vertical separation, plus an independent floor failure.
        checker = object.__new__(module.CollisionChecker)
        positions = torch.tensor([[[0., 0., 1.], [.09, 0., 1.]],
                                  [[0., 0., 1.], [.1, 0., 1.]],
                                  [[0., 0., 1.], [.2, 0., 1.]],
                                  [[0., 0., 1.], [0., 0., 1.11]],
                                  [[0., 0., -.1], [.2, 0., -.1]]])
        checker.robot = SimpleNamespace(device="cpu", data=SimpleNamespace(
            body_link_pos_w=positions,
            body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]] * 2] * 5)))
        checker.floor_heights = torch.zeros(5)
        checker.tolerance = 1e-4
        checker.min_foot_distance = .1
        checker.point_ids = [0, 1]
        self.assertEqual(checker.invalid_envs([0, 1, 2, 3, 4]), [0, 4])
        self.assertEqual(checker.invalid_envs([3, 0, 2]), [0])
        checker.min_foot_distance = 0.
        self.assertEqual(checker.invalid_envs([0, 1, 2, 3, 4]), [4])


if __name__ == '__main__':
    unittest.main()
