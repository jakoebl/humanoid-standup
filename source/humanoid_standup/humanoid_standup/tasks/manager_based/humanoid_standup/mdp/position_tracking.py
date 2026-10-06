"""Shared standing position measurement for rewards, terminations and markers."""
import torch

from .reference import REFERENCE

UPRIGHT_COM_HEIGHT = 0.5


def _whole_body_com(asset) -> torch.Tensor:
    positions = asset.data.body_com_pos_w
    masses = asset.data.default_mass.to(positions.device)
    return (positions * masses[..., None]).sum(dim=1) / masses.sum(dim=1, keepdim=True)


def _quat_product(a, b):
    """Hamilton product for scalar-first link quaternions."""
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), dim=-1)


def _axis_quat(angle, axis):
    result = torch.zeros((*angle.shape, 4), device=angle.device, dtype=angle.dtype)
    result[..., 0] = torch.cos(angle / 2)
    result[..., axis + 1] = torch.sin(angle / 2)
    return result


def _reference_left_foot_quat(frame_ids, device):
    """G1 leg rotation FK, including the fixed hip/knee pitch offsets.

    Both offsets and joint axes follow the G1 29-DOF rev 1.0 asset.
    Existing motion clips already contain the root and leg joint poses.
    """
    joints = REFERENCE.get_joint_pos(frame_ids).to(device)
    quat = REFERENCE.get_root_state(frame_ids).to(device)[:, 3:7]
    for name, axis, fixed_pitch in (
        ("hip_pitch", 1, 0.), ("hip_roll", 0, -.1749),
        ("hip_yaw", 2, 0.), ("knee", 1, .1749),
        ("ankle_pitch", 1, 0.), ("ankle_roll", 0, 0.),
    ):
        angle = joints[:, REFERENCE.JOINT_NAMES.index(f"left_{name}_joint")]
        if fixed_pitch:
            quat = _quat_product(quat, _axis_quat(torch.full_like(angle, fixed_pitch), 1))
        quat = _quat_product(quat, _axis_quat(angle, axis))
    return quat


def _foot_heading(quat):
    """Heading of the foot's forward axis projected onto the ground."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def _foot_yaw_delta(asset, frame_ids, body_ids):
    left_id = body_ids[REFERENCE.END_EFFECTOR_NAMES.index("left_ankle_roll_link")]
    actual = _foot_heading(asset.data.body_link_quat_w[:, left_id])
    reference = _foot_heading(_reference_left_foot_quat(frame_ids, asset.device))
    return actual - reference


def orientation_tracking_targets(env, asset, frame_ids, target, upright_com_height=UPRIGHT_COM_HEIGHT):
    """Apply the same foot heading alignment as upright position targets."""
    upright = (_whole_body_com(asset) - env.scene.env_origins)[:, 2] >= upright_com_height
    if not torch.any(upright):
        return target
    body_ids, _ = asset.find_bodies(list(REFERENCE.END_EFFECTOR_NAMES), preserve_order=True)
    delta = _foot_yaw_delta(asset, frame_ids, body_ids)
    aligned = _quat_product(_axis_quat(delta, 2), target)
    return torch.where(upright[:, None], aligned, target)


def position_tracking_targets(env, asset, frame_ids, target, upright_com_height=UPRIGHT_COM_HEIGHT):
    """Align upright targets to the actual left foot position and heading, in env coordinates."""
    com = _whole_body_com(asset) - env.scene.env_origins
    upright = com[:, 2] >= upright_com_height
    # Avoid resolving feet when all environments use absolute tracking.
    if not torch.any(upright):
        return target
    body_ids, _ = asset.find_bodies(list(REFERENCE.END_EFFECTOR_NAMES), preserve_order=True)
    feet = asset.data.body_link_pos_w[:, body_ids] - env.scene.env_origins[:, None, :]
    reference_feet = REFERENCE.get_end_effector_pos(frame_ids).to(asset.device)
    left_index = REFERENCE.END_EFFECTOR_NAMES.index("left_ankle_roll_link")
    relative = target - reference_feet[:, left_index, None, :]
    left = feet[:, left_index]
    yaw = _foot_yaw_delta(asset, frame_ids, body_ids)
    cosine, sine = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    aligned = torch.stack((
        cosine * relative[..., 0] - sine * relative[..., 1],
        sine * relative[..., 0] + cosine * relative[..., 1],
        relative[..., 2],
    ), dim=-1) + left[:, None, :]
    return torch.where(upright[:, None, None], aligned, target)


def _position_tracking_error(env, asset, frame_ids, current, target, upright_com_height):
    return current - position_tracking_targets(env, asset, frame_ids, target, upright_com_height)
