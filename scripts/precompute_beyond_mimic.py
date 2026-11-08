#!/usr/bin/env python3
"""Generate named Isaac Lab body targets for BeyondMimic from a G1 CSV.

Accepts legacy centimeter/degree CSVs or mjlab-format meter/radian CSVs.
Sampling defaults to 120 FPS input and 50 FPS output; specify the actual input
rate for other clips. No dynamics are stepped during forward kinematics.
"""

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback


def main():
    from isaaclab.app import AppLauncher

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=root / "data/unprocessed/stand_up_lying_V2.csv")
    parser.add_argument("output", type=Path, nargs="?", default=root / "data/beyondmimic/stand_up_lying_V2.npz")
    parser.add_argument("--input-fps", type=float, default=120)
    parser.add_argument("--output-fps", type=float, default=50)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must differ.")
    if args.input_fps <= 0 or args.output_fps <= 0:
        parser.error("Frame rates must be positive.")
    app = AppLauncher(args).app
    try:
        from humanoid_standup.tasks.manager_based.beyond_mimic.constants import JOINT_NAMES

        print(f"Converting {args.input} to {args.output}", flush=True)
        rows = convert_rows(args.input, JOINT_NAMES)
        print(f"Loaded {len(rows)} source frames; starting forward kinematics", flush=True)
        process(args, rows)
    except BaseException:
        # SimulationApp.close() terminates the process, so report failures first.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    finally:
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
    app.close()


def convert_rows(source, joint_names):
    """Read a G1 clip as rows of XYZ meters, XYZW quaternion and 29 joint radians.

    Legacy named CSVs use centimeters and XYZ Euler/joint angles in degrees.
    Headerless 36-column CSVs already use meters, radians and XYZW quaternions.
    Quaternions are normalized and sign-aligned to keep interpolation continuous.
    """
    columns = (
        [f"root_translate{axis}" for axis in "XYZ"]
        + [f"root_rotate{axis}" for axis in "XYZ"]
        + [f"{name}_dof" for name in joint_names]
    )
    with source.open(newline="") as stream:
        rows = [row for row in csv.reader(stream) if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("Empty clip.")
    legacy = "root_translateX" in rows[0]
    if legacy:
        header = rows.pop(0)
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names.")
        missing = set(columns) - set(header)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        indices = [header.index(name) for name in columns]
    expected = len(header) if legacy else 7 + len(joint_names)
    if len(rows) < 3:
        raise ValueError("At least three frames are required.")

    output = []
    previous_quat = None
    for index, row in enumerate(rows):
        if len(row) != expected:
            raise ValueError(f"Frame {index}: expected {expected} columns, got {len(row)}.")
        values = [float(row[i]) for i in indices] if legacy else [float(cell) for cell in row]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Frame {index}: nonfinite motion value.")
        if legacy:
            position = [value / 100 for value in values[:3]]
            roll, pitch, yaw = [math.radians(value) for value in values[3:6]]
            cr, cp, cy = [math.cos(angle / 2) for angle in (roll, pitch, yaw)]
            sr, sp, sy = [math.sin(angle / 2) for angle in (roll, pitch, yaw)]
            quat = [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                    cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
            joints = [math.radians(value) for value in values[6:]]
        else:
            position, quat, joints = values[:3], values[3:7], values[7:]
        norm = math.hypot(*quat)
        if norm < 1e-12:
            raise ValueError(f"Frame {index}: zero-length quaternion.")
        quat = [value / norm for value in quat]
        if previous_quat is not None and sum(a*b for a, b in zip(previous_quat, quat)) < 0:
            quat = [-value for value in quat]
        previous_quat = quat
        output.append(position + quat + joints)
    return output


def process(args, rows):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    import torch
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from humanoid_standup.tasks.manager_based.beyond_mimic.constants import JOINT_NAMES
    from humanoid_standup.tasks.manager_based.beyond_mimic.robot import robot_cfg

    original = np.asarray(rows)
    source_times = np.arange(len(rows)) / args.input_fps
    times = np.arange(0, source_times[-1], 1 / args.output_fps)
    if len(times) < 3:
        raise ValueError("Resampled motion requires at least three frames.")
    linear = np.stack([np.interp(times, source_times, original[:, i]) for i in [0, 1, 2] + list(range(7, 36))], axis=-1)
    quat = Slerp(source_times, Rotation.from_quat(original[:, 3:7]))(times).as_quat()[:, [3, 0, 1, 2]]
    sim = SimulationContext(SimulationCfg(dt=1 / args.output_fps, device=args.device))
    robot = Articulation(robot_cfg().replace(prim_path="/World/Robot"))
    sim.reset()
    ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if list(names) != JOINT_NAMES:
        raise ValueError("Robot does not contain the expected 29 joints.")
    ankle_roll_ids = [JOINT_NAMES.index(name) for name in ("left_ankle_roll_joint", "right_ankle_roll_joint")]
    positions, rotations, joints = [], [], []
    for index in range(len(times)):
        root = robot.data.default_root_state.clone()
        root[:, :3] = torch.as_tensor(linear[index, :3], device=robot.device, dtype=torch.float32)
        root[:, 3:7] = torch.as_tensor(quat[index], device=robot.device, dtype=torch.float32)
        root[:, 7:] = 0
        joint = robot.data.default_joint_pos.clone()
        joint[:, ids] = torch.as_tensor(linear[index, 3:], device=robot.device, dtype=torch.float32)
        joint[:, [ids[i] for i in ankle_roll_ids]] = 0.0
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(joint, torch.zeros_like(joint))
        sim.physics_sim_view.update_articulations_kinematic()
        robot.update(sim.get_physics_dt())
        positions.append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
        rotations.append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
        joints.append(joint[0].cpu().numpy().copy())
        if not args.headless:
            sim.render()
    pos, rot, joint = map(np.stack, (positions, rotations, joints))
    # Angular velocity is a world-frame SO(3) derivative, including shortest-arc handling.
    angular = np.zeros((*rot.shape[:2], 3))
    for body in range(rot.shape[1]):
        rotation = Rotation.from_quat(rot[:, body, [1, 2, 3, 0]])
        angular[1:-1, body] = (rotation[2:] * rotation[:-2].inv()).as_rotvec() * (args.output_fps / 2)
        angular[0, body] = (rotation[1] * rotation[0].inv()).as_rotvec() * args.output_fps
        angular[-1, body] = (rotation[-1] * rotation[-2].inv()).as_rotvec() * args.output_fps
    data = dict(
        fps=np.array([args.output_fps]), backend=np.array("isaaclab"),
        joint_names=np.array(robot.joint_names), body_names=np.array(robot.body_names),
        joint_pos=joint, joint_vel=np.gradient(joint, 1 / args.output_fps, axis=0),
        body_pos_w=pos, body_quat_w=rot,
        body_lin_vel_w=np.gradient(pos, 1 / args.output_fps, axis=0), body_ang_vel_w=angular,
    )
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite target array {key}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **data)
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Wrote {len(times)} frames at {args.output_fps} FPS to {args.output}", flush=True)


if __name__ == "__main__":
    main()
