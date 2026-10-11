# Adapted from HybridRobotics/whole_body_tracking; see ../PROVENANCE.md.
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def motion_command(env: ManagerBasedEnv) -> torch.Tensor:
    """Reference joint positions and velocities for the motion command."""
    return env.command_manager.get_term("motion").command


def imu_lin_vel(env):
    return env.scene["imu"].data.lin_vel_b


def imu_projected_gravity(env):
    """Unit gravity direction in the IMU frame, rather than raw specific force."""
    return env.scene["imu"].data.projected_gravity_b


def imu_ang_vel(env):
    return env.scene["imu"].data.ang_vel_b


def selected_joint_pos_rel(env, biased=False):
    command = env.command_manager.get_term("motion")
    value = command.robot_joint_pos - command.robot.data.default_joint_pos[:, command.joint_indexes]
    if biased and hasattr(command, "encoder_bias"):
        value = value + command.encoder_bias
    return value


def selected_joint_vel_rel(env):
    command = env.command_manager.get_term("motion")
    return command.robot_joint_vel - command.robot.data.default_joint_vel[:, command.joint_indexes]


def robot_anchor_ori_w(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    return command.robot_anchor_lin_vel_w.view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    return command.robot_anchor_ang_vel_w.view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
