from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from .reference import Reference


def proprioceptive_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions/velocities, body angular velocity and gravity (64 values)."""
    asset = env.scene[asset_cfg.name]
    joint_ids, names = asset.find_joints(list(Reference.JOINT_NAMES), preserve_order=True)
    if tuple(names) != tuple(Reference.JOINT_NAMES):
        raise ValueError("Proprioceptive observations require all 29 configured G1 joints in order.")
    return torch.cat(
        (
            asset.data.joint_pos[:, joint_ids],
            asset.data.joint_vel[:, joint_ids],
            asset.data.root_ang_vel_b,
            asset.data.projected_gravity_b,
        ),
        dim=-1,
    )
