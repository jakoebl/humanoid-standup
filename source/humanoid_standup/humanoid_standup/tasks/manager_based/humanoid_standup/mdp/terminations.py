from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from .reference import REFERENCE


def reference_complete(env) -> torch.Tensor:
    """End a finite skill at its final reference sample, including RSI offsets."""
    return REFERENCE.phase_for_env(env) >= 1.0


def end_effector_deviation(env, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Tracking failure; ground contact is allowed for the get-up skill."""
    asset = env.scene[asset_cfg.name]
    frame_ids = REFERENCE.frame_ids_for_env(env).to(REFERENCE.device)
    current = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins[:, None, :]
    target = REFERENCE.get_end_effector_pos(frame_ids).to(asset.device)
    return torch.any(torch.linalg.norm(current - target, dim=-1) > threshold, dim=1)
