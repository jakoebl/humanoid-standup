# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, wrap_to_pi

from .reference import REFERENCE
from .position_tracking import UPRIGHT_COM_HEIGHT, _whole_body_com, _position_tracking_error, orientation_tracking_targets

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
    upright_com_height: float = UPRIGHT_COM_HEIGHT,
) -> torch.Tensor:
    """Reward joint-position tracking with a negative exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    frame_ids = _reference_frame_ids(env)
    joint_ids = _reference_joint_ids(asset, asset_cfg)
    error = wrap_to_pi(asset.data.joint_pos[:, joint_ids] - REFERENCE.get_joint_pos(frame_ids).to(asset.device))
    target_root = REFERENCE.get_root_state(frame_ids).to(asset.device)[:, 3:7]
    target_root = orientation_tracking_targets(env, asset, frame_ids, target_root, upright_com_height)
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
    upright_com_height: float = UPRIGHT_COM_HEIGHT,
) -> torch.Tensor:
    """Reward root orientation tracking with a negative exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    frame_ids = _reference_frame_ids(env)
    target = REFERENCE.get_root_state(frame_ids).to(asset.device)[:, 3:7]
    target = orientation_tracking_targets(env, asset, frame_ids, target, upright_com_height)
    orientation_error = quat_error_magnitude(asset.data.root_link_quat_w, target)
    return torch.exp(-scale * torch.square(orientation_error))


def end_effector_pos_target_exp(
    env: ManagerBasedRLEnv,
    scale: float = 40.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_com_height: float = UPRIGHT_COM_HEIGHT,
    reference_body_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Reward selected end effectors, ordered to match reference_body_indices.

    With no indices, track all reference wrists and feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    frame_ids = _reference_frame_ids(env)
    current = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins[:, None, :]
    target = REFERENCE.get_end_effector_pos(frame_ids).to(asset.device)
    if reference_body_indices is not None:
        target = target[:, list(reference_body_indices)]
    error = _position_tracking_error(env, asset, frame_ids, current, target, upright_com_height)
    return torch.exp(-scale * torch.sum(torch.square(error), dim=(1, 2)))


def com_pos_target_exp(
    env: ManagerBasedRLEnv,
    scale: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_com_height: float = UPRIGHT_COM_HEIGHT,
) -> torch.Tensor:
    """Reward whole-body center-of-mass position tracking."""
    asset: Articulation = env.scene[asset_cfg.name]
    frame_ids = _reference_frame_ids(env)
    current = _whole_body_com(asset) - env.scene.env_origins
    target = REFERENCE.get_com_pos(frame_ids).to(asset.device)
    error = _position_tracking_error(
        env, asset, frame_ids, current[:, None, :], target[:, None, :], upright_com_height
    )
    return torch.exp(-scale * torch.sum(torch.square(error), dim=(1, 2)))


class standing_foot_displacement(ManagerTermBase):
    """Sum foot distances from their first standing positions for the rest of the episode."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset = env.scene[cfg.params["asset_cfg"].name]
        self._anchors = torch.zeros((env.num_envs, 2, 3), device=asset.device,
                                    dtype=asset.data.body_link_pos_w.dtype)
        self._standing_reached = torch.zeros(env.num_envs, device=asset.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._standing_reached[env_ids] = False
        self._anchors[env_ids] = 0.

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        upright_com_height: float = UPRIGHT_COM_HEIGHT,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        feet = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
        upright = (_whole_body_com(asset) - env.scene.env_origins)[:, 2] >= upright_com_height
        first_standing = upright & ~self._standing_reached
        self._anchors[first_standing] = feet[first_standing]
        self._standing_reached |= upright
        distance = torch.linalg.vector_norm(feet - self._anchors, dim=-1).sum(dim=-1)
        return torch.where(self._standing_reached, distance, 0.)
