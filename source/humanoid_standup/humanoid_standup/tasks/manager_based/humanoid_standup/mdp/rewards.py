# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, wrap_to_pi

from .reference import REFERENCE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _reference_frame_ids(env: ManagerBasedRLEnv) -> torch.Tensor:
    return REFERENCE.frame_ids_for_env(env).to(device=REFERENCE.device)


def _reference_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    selected_joint_ids = asset_cfg.joint_ids
    if isinstance(selected_joint_ids, slice):
        selected_joint_ids = range(len(asset.joint_names))[selected_joint_ids]
    selected_joint_ids = set(selected_joint_ids)
    robot_joint_ids = {
        name: index for index, name in enumerate(asset.joint_names) if index in selected_joint_ids
    }
    missing_joints = [name for name in REFERENCE.JOINT_NAMES if name not in robot_joint_ids]
    if missing_joints:
        raise ValueError(f"Reference joints missing from robot: {missing_joints}")
    return torch.tensor(
        [robot_joint_ids[name] for name in REFERENCE.JOINT_NAMES],
        device=asset.device,
        dtype=torch.long,
    )


def joint_pos_target_exp(
    env: ManagerBasedRLEnv,
    scale: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward joint-position tracking with a negative exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    frame_ids = _reference_frame_ids(env)
    joint_ids = _reference_joint_ids(asset, asset_cfg)
    error = wrap_to_pi(asset.data.joint_pos[:, joint_ids] - REFERENCE.get_joint_pos(frame_ids).to(asset.device))
    target_root = REFERENCE.get_root_state(frame_ids).to(asset.device)[:, 3:7]
    root_error = quat_error_magnitude(asset.data.root_link_quat_w, target_root)
    return torch.exp(-scale * (torch.mean(torch.square(error), dim=1) + root_error.square()))


def joint_vel_target_exp(
    env: ManagerBasedRLEnv,
    scale: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward joint-velocity tracking with a negative exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    frame_ids = _reference_frame_ids(env)
    joint_ids = _reference_joint_ids(asset, asset_cfg)
    error = asset.data.joint_vel[:, joint_ids] - REFERENCE.get_joint_vel(frame_ids).to(asset.device)
    return torch.exp(-scale * torch.mean(torch.square(error), dim=1))


def root_orientation_target_exp(
    env: ManagerBasedRLEnv,
    scale: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward root orientation tracking with a negative exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    frame_ids = _reference_frame_ids(env)
    target = REFERENCE.get_root_state(frame_ids).to(asset.device)[:, 3:7]
    orientation_error = quat_error_magnitude(asset.data.root_link_quat_w, target)
    return torch.exp(-scale * torch.square(orientation_error))


def end_effector_pos_target_exp(
    env: ManagerBasedRLEnv,
    scale: float = 40.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward wrist and foot end-effector position tracking."""
    asset: Articulation = env.scene[asset_cfg.name]
    frame_ids = _reference_frame_ids(env)
    current = asset.data.body_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins[:, None, :]
    target = REFERENCE.get_end_effector_pos(frame_ids).to(asset.device)
    error = current - target
    return torch.exp(-scale * torch.sum(torch.square(error), dim=(1, 2)))


def com_pos_target_exp(
    env: ManagerBasedRLEnv,
    scale: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward whole-body center-of-mass position tracking."""
    asset: Articulation = env.scene[asset_cfg.name]
    frame_ids = _reference_frame_ids(env)
    current = asset.data.body_com_pos_w
    masses = asset.data.default_mass.to(current.device)
    current = torch.sum(current * masses[..., None], dim=1) / torch.sum(masses, dim=1, keepdim=True)
    current -= env.scene.env_origins
    target = REFERENCE.get_com_pos(frame_ids).to(asset.device)
    return torch.exp(-scale * torch.sum(torch.square(current - target), dim=1))
