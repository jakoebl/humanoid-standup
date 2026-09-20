from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from .reference import REFERENCE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_to_reference(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    random_start: bool = True,
):
    """Reset robot pose and velocity to a random frame in the reference clip."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "reference_start_frames"):
        env.reference_start_frames = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # sample a random frame for each env.
    frames = torch.randint(REFERENCE.num_frames - 1, (len(env_ids),), device=env.device) if random_start else torch.zeros(len(env_ids), dtype=torch.long, device=env.device)

    # store the sampled frame for each env so that we can use it to compute rewards and terminations
    env.reference_start_frames[env_ids] = frames

    # get the reference frame and joint indices??
    frame_ids = frames.to(REFERENCE.device)
    frame = REFERENCE.get_frame(frame_ids)
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    robot_joint_indices = {name: index for index, name in enumerate(asset.joint_names)}
    missing = set(REFERENCE.JOINT_NAMES) - robot_joint_indices.keys()
    if missing:
        raise ValueError(f"Reference joints missing from robot: {sorted(missing)}")
    indices = [robot_joint_indices[name] for name in REFERENCE.JOINT_NAMES]

    # set robot state to reference values
    joint_pos[:, indices] = frame["joint_pos"].to(asset.device)
    joint_vel[:, indices] = frame["joint_vel"].to(asset.device)
    root_state = REFERENCE.get_root_state(frame_ids).to(asset.device)
    root_state[:, :3] += env.scene.env_origins[env_ids]

    # write to sim
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
