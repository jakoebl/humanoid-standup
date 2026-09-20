from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat
from .reference import REFERENCE


# Explicit G1 body order: pelvis plus the child body of each of the 29 DOFs.
# No palm or finger bodies can enter the policy through an unrestricted slice.
OBSERVATION_BODY_NAMES = ("pelvis",) + tuple(
    "torso_link" if name == "waist_pitch_joint" else name.replace("_joint", "_link")
    for name in REFERENCE.JOINT_NAMES
)


def deepmimic_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Released DeepMimic CtController raw state layout, adapted to G1/z-up.

    [phase, root height, (position, normal, tangent) per body,
    (linear velocity, angular velocity) per body]. Positions/velocities use
    rigid-body COMs, with positions relative to the root joint. Normal is the
    local up axis (z here, y in DeepMimic); tangent is local x. All vectors are
    expressed in the root heading frame. There are no reference-target,
    joint-coordinate, palm, or finger observations. Size: 2 + 30 * 15 = 452.
    """
    asset = env.scene[asset_cfg.name]
    body_ids, names = asset.find_bodies(list(OBSERVATION_BODY_NAMES), preserve_order=True)
    if tuple(names) != OBSERVATION_BODY_NAMES:
        raise ValueError("DeepMimic observations require all 30 configured G1 bodies in order.")
    root_pos = asset.data.root_link_pos_w - env.scene.env_origins
    root_quat = asset.data.root_link_quat_w
    heading = yaw_quat(root_quat)
    body_heading = heading[:, None, :].expand(-1, len(body_ids), -1)
    body_position = quat_apply_inverse(
        body_heading, asset.data.body_com_pos_w[:, body_ids] - asset.data.root_link_pos_w[:, None, :]
    )

    body_quat = quat_mul(quat_conjugate(body_heading), asset.data.body_link_quat_w[:, body_ids])
    normal = torch.zeros_like(body_position)
    tangent = torch.zeros_like(body_position)
    normal[..., 2] = 1.0
    tangent[..., 0] = 1.0
    pose = torch.cat((body_position, quat_apply(body_quat, normal), quat_apply(body_quat, tangent)), dim=-1)
    velocity = torch.cat(
        (
            quat_apply_inverse(body_heading, asset.data.body_com_lin_vel_w[:, body_ids]),
            quat_apply_inverse(body_heading, asset.data.body_link_ang_vel_w[:, body_ids]),
        ),
        dim=-1,
    )
    return torch.cat(
        (
            REFERENCE.phase_for_env(env).unsqueeze(-1),
            root_pos[:, 2:3],
            pose.flatten(start_dim=1),
            velocity.flatten(start_dim=1),
        ),
        dim=-1,
    )
