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
    clip_id: int | None = None,
    start_fraction: float | None = None,
    joint_pos_noise: float = 0.03,
    joint_vel_noise: float = 0.1,
):
    """Reset to a reference frame with uniform joint noise (radians and radians/second)."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "reference_start_frames"):
        env.reference_start_frames = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    if not hasattr(env, "reference_clip_ids"):
        env.reference_clip_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # Choose clips uniformly, then sample a local frame excluding the final sample.
    clip_ids = torch.randint(REFERENCE.num_clips, (len(env_ids),), device=env.device)
    if clip_id is not None:
        if not 0 <= clip_id < REFERENCE.num_clips:
            raise ValueError(f"Invalid reference clip: {clip_id}")
        clip_ids.fill_(clip_id)
    env.reference_clip_ids[env_ids] = clip_ids
    lengths = REFERENCE.clip_lengths.to(env.device)[clip_ids]
    frames = (torch.rand(len(env_ids), device=env.device) * (lengths - 1)).long() if random_start else torch.zeros(len(env_ids), dtype=torch.long, device=env.device)

    if start_fraction is not None:
        if not 0 <= start_fraction < 1:
            raise ValueError("start_fraction must be in [0, 1).")
        frames = ((lengths - 1) * start_fraction).long()

    # store the sampled frame for each env so that we can use it to compute rewards and terminations
    env.reference_start_frames[env_ids] = frames

    frame_ids = frames.to(REFERENCE.device) + REFERENCE.clip_offsets[clip_ids.to(REFERENCE.device)]
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

    # Independently perturb each environment and joint around the reference state.
    joint_pos += torch.empty_like(joint_pos).uniform_(-joint_pos_noise, joint_pos_noise)
    joint_vel += torch.empty_like(joint_vel).uniform_(-joint_vel_noise, joint_vel_noise)
    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, min=limits[..., 0], max=limits[..., 1])

    # write to sim
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
