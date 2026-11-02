"""Validate and remap named Isaac Sim motion targets; independent of Isaac Sim."""

from pathlib import Path

import numpy as np


def load_motion(path, joint_names, body_names, fps):
    with np.load(path, allow_pickle=False) as data:
        actual_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not np.isclose(actual_fps, fps):
            raise ValueError(f"Motion FPS {actual_fps} does not match control FPS {fps}.")

        stored_joints = data["joint_names"].tolist()
        stored_bodies = data["body_names"].tolist()


        joints = [stored_joints.index(name) for name in joint_names]
        bodies = [stored_bodies.index(name) for name in body_names]

        frames = data["joint_pos"].shape[0]

        result = {}
        for key, width in [("joint_pos", None), ("joint_vel", None), ("body_pos_w", 3),
                           ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)]:
            expected = (frames, len(stored_joints)) if width is None else (frames, len(stored_bodies), width)
            array = np.asarray(data[key], dtype=np.float32)
            result[key] = array[:, joints if width is None else bodies].copy()

        norms = np.linalg.norm(result["body_quat_w"], axis=-1)
        if not np.allclose(norms, 1, atol=1e-4):
            raise ValueError("Motion body quaternions must have unit norm.")

        result["fps"] = actual_fps
        return result


def resolve_motion_files(directory, files, motion_file=None):
    """Resolve an explicit list, a directory, or the default single file, in that order."""
    if files:
        paths = [Path(path).expanduser().resolve() for path in files]
    elif directory:
        paths = sorted(Path(directory).expanduser().resolve().glob("*.npz"))
    else:
        paths = [Path(motion_file).expanduser().resolve()] if motion_file else []
    if not paths:
        raise ValueError(f"No prepared NPZ clips found in {directory}; use precompute_beyond_mimic.py first.")
    if len(set(paths)) != len(paths):
        raise ValueError("Motion files must be unique.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths
