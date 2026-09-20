# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Precompute G1 end-effector and center-of-mass trajectories for a motion CSV."""

import argparse
import csv
import math
import urllib.request
from pathlib import Path

import pinocchio as pin

parser = argparse.ArgumentParser(description="Append G1 kinematics targets to a reference motion CSV.")
parser.add_argument("input", type=Path, help="Input motion CSV containing root pose and G1 joint positions.")
parser.add_argument("output", type=Path, help="Output CSV to create with appended kinematics columns.")
parser.add_argument(
    "--urdf",
    type=str,
    default=None,
    help="Local G1 URDF matching the playback robot. Defaults to Unitree's G1 29-DOF with-hand rev 1.0 URDF.",
)
parser.add_argument(
    "--end-effectors",
    nargs="+",
    default=[
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    help="Pinocchio frame names to export as end-effector positions.",
)
parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
args_cli = parser.parse_args()

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def resolve_urdf() -> str:
    if args_cli.urdf is not None:
        return str(Path(args_cli.urdf).expanduser().resolve())

    cache_path = Path("/tmp/g1_29dof_with_hand_rev_1_0.urdf")
    if not cache_path.is_file():
        urdf_url = (
            "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/"
            "master/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
        )
        try:
            urllib.request.urlretrieve(urdf_url, cache_path)
        except Exception as error:
            raise RuntimeError(
                "Could not download the mass-bearing G1 URDF. Pass a local file with --urdf."
            ) from error
    return str(cache_path)


def root_rotation_matrix(row: dict[str, str]):
    roll, pitch, yaw = (math.radians(float(row[name])) for name in ("root_rotateX", "root_rotateY", "root_rotateZ"))
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    quaternion = pin.Quaternion(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    return quaternion.toRotationMatrix()


def build_joint_indices(model: pin.Model) -> dict[str, int]:
    indices = {}
    for name in JOINT_NAMES:
        if not model.existJointName(name):
            raise ValueError(f"Reference joint '{name}' is missing from the URDF.")
        joint_id = model.getJointId(name)
        if model.nqs[joint_id] != 1:
            raise ValueError(f"Reference joint '{name}' must have one position coordinate.")
        indices[name] = model.idx_qs[joint_id]
    if model.nq < len(JOINT_NAMES):
        raise ValueError(f"Expected at least 29 URDF position coordinates, got nq={model.nq}.")
    return indices


def resolve_frame_id(model: pin.Model, frame_name: str) -> int:
    frame_names = [frame.name for frame in model.frames]
    if frame_name in frame_names:
        return frame_names.index(frame_name)

    matches = [index for index, name in enumerate(frame_names) if name.endswith(f"_{frame_name}")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"End-effector frame '{frame_name}' is missing from the URDF.")
    raise ValueError(f"End-effector frame '{frame_name}' is ambiguous in the URDF: {[frame_names[i] for i in matches]}")


def main() -> None:
    if not args_cli.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args_cli.input}")
    if args_cli.input.resolve() == args_cli.output.resolve():
        raise ValueError("Input and output paths must differ to avoid overwriting the source clip.")

    model = pin.buildModelFromUrdf(resolve_urdf())
    data = model.createData()
    joint_indices = build_joint_indices(model)
    frame_ids = {}
    for frame_name in args_cli.end_effectors:
        frame_ids[frame_name] = resolve_frame_id(model, frame_name)

    with args_cli.input.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")
        required_fields = {"root_translateX", "root_translateY", "root_translateZ", "root_rotateX", "root_rotateY", "root_rotateZ"}
        required_fields.update(f"{name}_dof" for name in JOINT_NAMES)
        missing_fields = sorted(required_fields.difference(reader.fieldnames))
        if missing_fields:
            raise ValueError(f"Input CSV is missing required fields: {missing_fields}")

        output_fields = list(reader.fieldnames)
        for frame_name in args_cli.end_effectors:
            for axis in "xyz":
                column = f"{frame_name}_pos_{axis}"
                if column not in output_fields:
                    output_fields.append(column)
        for column in ("com_pos_x", "com_pos_y", "com_pos_z"):
            if column not in output_fields:
                output_fields.append(column)

        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        with args_cli.output.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_fields)
            writer.writeheader()
            row_count = 0
            for row in reader:
                q = pin.neutral(model)
                for name, index in joint_indices.items():
                    q[index] = math.radians(float(row[f"{name}_dof"]))
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                com_position = pin.centerOfMass(model, data, q).copy()
                root_position = [
                    float(row["root_translateX"]) / 100.0,
                    float(row["root_translateY"]) / 100.0,
                    float(row["root_translateZ"]) / 100.0,
                ]
                rotation = root_rotation_matrix(row)

                for frame_name, frame_id in frame_ids.items():
                    position = rotation @ data.oMf[frame_id].translation + root_position
                    for axis, value in zip("xyz", position):
                        row[f"{frame_name}_pos_{axis}"] = f"{value:.9g}"
                world_com = rotation @ com_position + root_position
                for axis, value in zip("xyz", world_com):
                    row[f"com_pos_{axis}"] = f"{value:.9g}"
                writer.writerow(row)
                row_count += 1

    print(f"Wrote {row_count} frames to {args_cli.output}")


if __name__ == "__main__":
    main()
