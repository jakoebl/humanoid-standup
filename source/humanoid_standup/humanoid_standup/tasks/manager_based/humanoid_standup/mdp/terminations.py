from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from .reference import REFERENCE
from .position_tracking import UPRIGHT_COM_HEIGHT, _whole_body_com, _position_tracking_error


def reference_complete(env) -> torch.Tensor:
    """End a finite skill at its final reference sample, including RSI offsets."""
    return REFERENCE.phase_for_env(env) >= 1.0


def end_effector_deviation(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg,
    upright_com_height: float = UPRIGHT_COM_HEIGHT,
) -> torch.Tensor:
    """Tracking failure; ground contact is allowed for the get-up skill."""
    asset = env.scene[asset_cfg.name]
    frame_ids = REFERENCE.frame_ids_for_env(env).to(REFERENCE.device)
    current = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins[:, None, :]
    target = REFERENCE.get_end_effector_pos(frame_ids).to(asset.device)
    error = _position_tracking_error(env, asset, frame_ids, current, target, upright_com_height)
    return torch.any(torch.linalg.norm(error, dim=-1) > threshold, dim=1)


def com_deviation(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg,
    upright_com_height: float = UPRIGHT_COM_HEIGHT,
) -> torch.Tensor:
    """Terminate when whole-body COM tracking error exceeds the threshold in meters."""
    asset = env.scene[asset_cfg.name]
    frame_ids = REFERENCE.frame_ids_for_env(env).to(REFERENCE.device)
    current = _whole_body_com(asset) - env.scene.env_origins
    target = REFERENCE.get_com_pos(frame_ids).to(asset.device)
    error = _position_tracking_error(
        env, asset, frame_ids, current[:, None], target[:, None], upright_com_height
    )
    return torch.linalg.norm(error[:, 0], dim=-1) > threshold
