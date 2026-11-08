# Adapted from HybridRobotics/whole_body_tracking; see ../PROVENANCE.md.
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def joint_torque_limit_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint torque ratios above a normalized soft limit threshold."""
    asset = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    excess = torch.relu((torque / limits - threshold) / (1.0 - threshold))
    return torch.mean(torch.square(excess), dim=1)


def self_collision_cost(env, sensor_names, force_threshold=10.0):
    """Count physics substeps with robot-to-robot contact, excluding the ground."""
    hits = None
    for name in sensor_names:
        forces = env.scene[name].data.force_matrix_w_history
        if forces is None:
            raise RuntimeError(f"Self-contact sensor {name} has no filtered force history.")
        contact = (torch.linalg.vector_norm(forces, dim=-1) > force_threshold).any(dim=-1).any(dim=-1)
        hits = contact if hits is None else hits | contact
    return hits.sum(dim=-1).float()
