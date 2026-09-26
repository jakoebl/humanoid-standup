# Copyright (c) 2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Export G1 link origins and mass-weighted COM directly from Isaac Sim poses.

Uses the training robot asset without importing the task or an existing processed
reference. Pose writes and PhysX FK updates do not integrate gravity/contacts/PD.
"""

import argparse
import csv
import math
import tempfile
import traceback
from pathlib import Path


# CSV order must match Reference.JOINT_NAMES and its positional loader.
JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")] + [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
]
END_EFFECTORS = [
    "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link",
]
MOTION_COLUMNS = (
    ["Frame"] + [f"root_translate{axis}" for axis in "XYZ"]
    + [f"root_rotate{axis}" for axis in "XYZ"] + [f"{name}_dof" for name in JOINT_NAMES]
)
TARGET_COLUMNS = [f"{name}_pos_{axis}" for name in END_EFFECTORS for axis in "xyz"] + [
    f"com_pos_{axis}" for axis in "xyz"
]


def load_rows(path):
    """Validate source poses before launching simulation; retain original cell text."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Input CSV has duplicate column names.")
        missing = set(MOTION_COLUMNS) - set(fields)
        if missing:
            raise ValueError(f"Missing motion columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("Reference motion requires at least two frames.")
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row at frame index {index}.")
        for name in MOTION_COLUMNS:
            if not math.isfinite(float(row[name])):
                raise ValueError(f"Nonfinite {name} at frame index {index}.")
    # Canonical motion + EE + COM order is required by the training loader.
    extra = [name for name in fields if name not in MOTION_COLUMNS + TARGET_COLUMNS]
    return rows, MOTION_COLUMNS + TARGET_COLUMNS + extra


def load_quaternion_rows(path, translation_unit="m", joint_unit="rad"):
    """Convert headerless 30 FPS XYZ/XYZW/29-joint data to legacy 120 FPS rows.

    Accept comma or whitespace delimiters. Hold each sample for four output
    frames, including the final sample. Output units are centimeters/degrees.
    """
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            cells = line.strip().split(",") if "," in line else line.split()
            if len(cells) != 36:
                raise ValueError(f"Line {line_number}: expected 36 values (XYZ, QX QY QZ QW, 29 joints).")
            try:
                values = [float(cell) for cell in cells]
            except ValueError as error:
                raise ValueError(f"Line {line_number}: expected headerless numeric motion data.") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Line {line_number}: nonfinite motion value.")
            x, y, z, w = values[3:7]
            norm = math.hypot(x, y, z, w)
            if norm < 1e-12:
                raise ValueError(f"Line {line_number}: zero-length quaternion.")
            x, y, z, w = (value / norm for value in (x, y, z, w))
            # Inverse of Reference._euler_xyz_to_quat. At gimbal lock,
            # choose yaw=0 and the equivalent coupled roll.
            sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
            if abs(sin_pitch) >= 1.0 - 1e-12:
                roll = 2.0 * math.atan2(x, w)
                pitch = math.copysign(math.pi / 2.0, sin_pitch)
                yaw = 0.0
            else:
                roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                pitch = math.asin(sin_pitch)
                yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            positions = [value * (100.0 if translation_unit == "m" else 1.0) for value in values[:3]]
            angles = [math.degrees(value) for value in (roll, pitch, yaw)]
            joints = [math.degrees(value) if joint_unit == "rad" else value for value in values[7:]]
            motion = dict(zip(MOTION_COLUMNS[1:], positions + angles + joints))
            for _ in range(4):
                rows.append({"Frame": len(rows), **motion})
    if len(rows) < 8:
        raise ValueError("Reference motion requires at least two source frames.")
    return rows, MOTION_COLUMNS + TARGET_COLUMNS


def process(args, rows, fields, simulation_app):
    import torch
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils.math import quat_from_euler_xyz
    from isaaclab_assets.robots.unitree import G1_29DOF_CFG

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device))
    robot = Articulation(G1_29DOF_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    joint_ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    body_ids, body_names = robot.find_bodies(END_EFFECTORS, preserve_order=True)
    if list(names) != JOINT_NAMES or list(body_names) != END_EFFECTORS:
        raise ValueError("Simulation robot does not match the reference joint/end-effector names.")
    masses = robot.root_physx_view.get_masses()[0].to(robot.device)
    if not torch.isfinite(masses).all() or (masses < 0).any() or masses.sum() <= 0:
        raise ValueError("Invalid simulation body masses.")

    # Publish only a complete clip. Existing output survives interrupted/failed runs.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows):
                if not simulation_app.is_running():
                    raise RuntimeError(f"Simulation closed at frame {index}/{len(rows)}.")
                values = torch.tensor(
                    [[float(row[name]) for name in MOTION_COLUMNS[1:]]], device=robot.device,
                    dtype=torch.float32,
                )
                root = robot.data.default_root_state.clone()
                root[:, :3] = values[:, :3] / 100.0
                angles = torch.deg2rad(values[:, 3:6])
                root[:, 3:7] = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
                root[:, 7:] = 0.0
                joints = robot.data.default_joint_pos.clone()
                joints[:, joint_ids] = torch.deg2rad(values[:, 6:])
                # Fingers use the same defaults as training resets and reference playback.
                robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
                robot.write_root_state_to_sim(root)
                sim.physics_sim_view.update_articulations_kinematic()
                robot.update(sim.get_physics_dt())
                ee = robot.data.body_link_pos_w[0, body_ids]
                com = (robot.data.body_com_pos_w[0] * masses[:, None]).sum(dim=0) / masses.sum()
                positions = torch.cat((ee.flatten(), com))
                if not torch.isfinite(positions).all():
                    raise RuntimeError(f"Nonfinite simulation positions at frame {index}.")
                # Robot origin is world zero; output uses the reference coordinate frame.
                for name, value in zip(TARGET_COLUMNS, positions.tolist()):
                    row[name] = f"{value:.9g}"
                writer.writerow(row)
                if not args.headless:
                    sim.render()
        temporary.replace(args.output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Wrote {len(rows)} frames to {args.output}; simulation mass={masses.sum().item():.6f} kg", flush=True)


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Legacy motion CSV or headerless quaternion clip.")
    parser.add_argument("output", type=Path, help="Processed CSV to create or replace; must differ from input.")
    parser.add_argument("--input-format", choices=("legacy", "quaternion"), default="legacy",
                        help="legacy: named Euler CSV at 120 FPS; quaternion: XYZ/XYZW/29 joints at 30 FPS.")
    parser.add_argument("--translation-unit", choices=("m", "cm"), default="m",
                        help="Quaternion input translation unit (default: m); legacy always uses cm.")
    parser.add_argument("--joint-unit", choices=("rad", "deg"), default="rad",
                        help="Quaternion input joint unit (default: rad); legacy always uses degrees.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must differ.")
    if args.input_format == "quaternion":
        rows, fields = load_quaternion_rows(args.input, args.translation_unit, args.joint_unit)
    else:
        rows, fields = load_rows(args.input)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        process(args, rows, fields, simulation_app)
    except Exception:
        traceback.print_exc()
        simulation_app.app.post_quit(1)
        raise
    finally:
        from isaaclab.sim import SimulationContext
        sim = SimulationContext.instance()
        if sim is not None:
            if sim.cfg.create_stage_in_memory:
                import omni.physx
                omni.physx.get_physx_simulation_interface().detach_stage()
                sim.stop()
                sim.clear()
            sim.clear_all_callbacks()
            sim.clear_instance()
        simulation_app.close()


if __name__ == "__main__":
    main()
