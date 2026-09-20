# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a reference motion clip directly in Isaac Sim."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trajectory reference motion in Isaac Sim.")
parser.add_argument("--task", type=str, default="Template-Trajectory-Tracker-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--csv",
    type=Path,
    default=None,
    help="CSV created by precompute_reference_kinematics.py.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--check-kinematics", action="store_true",
    help="Compare every frame's exported EE positions against Isaac Sim link origins, then exit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import isaaclab_tasks  # noqa: F401
import trajectory_tracker.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from trajectory_tracker.tasks.manager_based.trajectory_tracker.mdp.reference import REFERENCE


def load_reference_positions(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(
            f"Kinematics CSV not found: {path}. Run scripts/precompute_reference_kinematics.py first."
        )

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32, encoding="utf-8")
    data = np.atleast_1d(data)
    columns = [
        "com_pos_x",
        "com_pos_y",
        "com_pos_z",
        "left_wrist_yaw_link_pos_x",
        "left_wrist_yaw_link_pos_y",
        "left_wrist_yaw_link_pos_z",
        "right_wrist_yaw_link_pos_x",
        "right_wrist_yaw_link_pos_y",
        "right_wrist_yaw_link_pos_z",
        "left_ankle_roll_link_pos_x",
        "left_ankle_roll_link_pos_y",
        "left_ankle_roll_link_pos_z",
        "right_ankle_roll_link_pos_x",
        "right_ankle_roll_link_pos_y",
        "right_ankle_roll_link_pos_z",
    ]
    missing_columns = [column for column in columns if column not in data.dtype.names]
    if missing_columns:
        raise ValueError(f"Kinematics CSV is missing required columns: {missing_columns}")
    return torch.from_numpy(np.stack([data[column] for column in columns], axis=-1))


def create_reference_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ReferenceKinematics",
        markers={
            "com": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)),
            ),
            "left_wrist": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            ),
            "right_wrist": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
            ),
            "left_foot": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.6)),
            ),
            "right_foot": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.5)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def show_reference_markers(
    markers: VisualizationMarkers,
    reference_positions: torch.Tensor,
    frame_id: int,
    env_origin: torch.Tensor,
) -> None:
    positions = reference_positions[frame_id].to(env_origin.device).reshape(5, 3)
    markers.visualize(
        marker_indices=[0, 1, 2, 3, 4],
        translations=positions + env_origin,
    )


def apply_reference_frame(env, frame_id: int) -> None:
    robot = env.unwrapped.scene["robot"]
    frame_ids = torch.tensor([frame_id], device=REFERENCE.device)
    frame = REFERENCE.get_frame(frame_ids)

    joint_indices = {name: index for index, name in enumerate(robot.joint_names)}
    reference_indices = {name: index for index, name in enumerate(REFERENCE.JOINT_NAMES)}
    missing_joints = [name for name in REFERENCE.JOINT_NAMES if name not in joint_indices]
    if missing_joints:
        raise ValueError(f"Reference joints missing from robot: {missing_joints}")

    env_ids = torch.zeros(1, dtype=torch.long, device=robot.device)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    for joint_name in REFERENCE.JOINT_NAMES:
        robot_index = joint_indices[joint_name]
        reference_index = reference_indices[joint_name]
        joint_pos[:, robot_index] = frame["joint_pos"][:, reference_index].to(robot.device)
        joint_vel[:, robot_index] = frame["joint_vel"][:, reference_index].to(robot.device)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = REFERENCE.get_root_state(frame_ids).to(robot.device)
    root_state[:, :3] += env.unwrapped.scene.env_origins[env_ids]
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    kinematics_csv = args_cli.csv
    if kinematics_csv is None:
        kinematics_csv = (
            REFERENCE.path if REFERENCE.end_effector_pos is not None
            else REFERENCE.path.with_name(f"{REFERENCE.path.stem}_kinematics.csv")
        )
    reference_positions = load_reference_positions(kinematics_csv)
    if reference_positions.shape[0] != REFERENCE.num_frames:
        raise ValueError(
            f"Kinematics CSV has {reference_positions.shape[0]} frames, "
            f"but the reference has {REFERENCE.num_frames}."
        )
    reference_markers = create_reference_markers()

    frame_id = 0
    frame_step = round((env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation) / REFERENCE.dt)
    if frame_step < 1:
        raise ValueError("The simulation step must be at least as large as the reference timestep.")
    try:
        if args_cli.check_kinematics:
            robot = env.unwrapped.scene["robot"]
            body_ids, body_names = robot.find_bodies(REFERENCE.END_EFFECTOR_NAMES, preserve_order=True)
            if body_names != REFERENCE.END_EFFECTOR_NAMES:
                raise ValueError(f"Could not resolve reference end effectors in order: {body_names}")
            max_errors = torch.zeros(len(body_ids), device=robot.device)
            for check_frame in range(REFERENCE.num_frames):
                apply_reference_frame(env, check_frame)
                # Explicitly refresh PhysX FK even with rendering/fabric disabled.
                env.unwrapped.sim.physics_sim_view.update_articulations_kinematic()
                actual = robot.data.body_link_pos_w[0, body_ids] - env.unwrapped.scene.env_origins[0]
                expected = reference_positions[check_frame, 3:].to(robot.device).reshape(-1, 3)
                max_errors = torch.maximum(max_errors, torch.linalg.vector_norm(actual - expected, dim=-1))
            for name, error in zip(body_names, max_errors.tolist()):
                print(f"{name}: maximum position error = {error:.6f} m", flush=True)
            return

        while simulation_app.is_running():
            frame_start = time.perf_counter()
            apply_reference_frame(env, frame_id)
            show_reference_markers(
                reference_markers,
                reference_positions,
                frame_id,
                env.unwrapped.scene.env_origins[0],
            )
            # Render the exact reference pose. A physics step applies gravity,
            # contacts and PD actions, moving the robot away from the markers.
            env.unwrapped.sim.physics_sim_view.update_articulations_kinematic()
            env.unwrapped.sim.render()
            time.sleep(max(0.0, frame_step * REFERENCE.dt - (time.perf_counter() - frame_start)))
            frame_id = (frame_id + frame_step) % REFERENCE.num_frames
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
