#!/usr/bin/env python3
"""Display a BeyondMimic NPZ reference directly as a kinematic robot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, nargs="?", default=ROOT / "data/beyondmimic/back.npz")
    parser.add_argument("--once", action="store_true", help="Stop after one pass instead of looping.")
    parser.add_argument("--start-frame", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative.")

    app = AppLauncher(args).app
    try:
        play(args, app)
    finally:
        app.close()


def play(args, app) -> None:
    import torch
    from isaaclab.assets import Articulation
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext

    from humanoid_standup.tasks.manager_based.beyond_mimic.constants import JOINT_NAMES
    from humanoid_standup.tasks.manager_based.beyond_mimic.robot import robot_cfg

    with np.load(args.clip, allow_pickle=False) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        stored_joints = data["joint_names"].tolist()
        stored_bodies = data["body_names"].tolist()
        joint_ids = [stored_joints.index(name) for name in JOINT_NAMES]
        pelvis_id = stored_bodies.index("pelvis")
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)[:, joint_ids]
        pelvis_pos = np.asarray(data["body_pos_w"], dtype=np.float32)[:, pelvis_id]
        pelvis_quat = np.asarray(data["body_quat_w"], dtype=np.float32)[:, pelvis_id]
        pelvis_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float32)[:, pelvis_id]
        pelvis_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float32)[:, pelvis_id]

    if not 0 <= args.start_frame < len(joint_pos):
        raise ValueError(f"--start-frame must be in [0, {len(joint_pos) - 1}].")

    sim = SimulationContext(SimulationCfg(dt=1.0 / fps, device=args.device))
    sim_utils.spawn_ground_plane(
        "/World/ground", sim_utils.GroundPlaneCfg(size=(20.0, 20.0))
    )
    sim_utils.spawn_light(
        "/World/keyLight",
        sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
        translation=(2.0, -2.0, 4.0),
    )
    sim_utils.spawn_light(
        "/World/skyLight",
        sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=800.0),
    )
    robot = Articulation(robot_cfg().replace(prim_path="/World/ReferenceRobot"))
    sim.reset()
    sim.set_camera_view(eye=(2.8, -2.8, 1.8), target=(0.0, 0.0, 0.8))
    robot_joint_ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if list(names) != list(JOINT_NAMES):
        raise ValueError(f"Robot joints do not match the reference: {names}")

    root = robot.data.default_root_state.clone()
    joints = robot.data.default_joint_pos.clone()
    sim_dt = sim.get_physics_dt()
    frame_period = 1.0 / fps
    frame = args.start_frame
    print(f"Playing {args.clip} ({len(joint_pos)} frames at {fps:g} Hz)", flush=True)
    while app.is_running():
        root[0, :3] = torch.from_numpy(pelvis_pos[frame]).to(robot.device)
        root[0, 3:7] = torch.from_numpy(pelvis_quat[frame]).to(robot.device)
        root[0, 7:10] = torch.from_numpy(pelvis_lin_vel[frame]).to(robot.device)
        root[0, 10:13] = torch.from_numpy(pelvis_ang_vel[frame]).to(robot.device)
        joints[0, robot_joint_ids] = torch.from_numpy(joint_pos[frame]).to(robot.device)
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        sim.physics_sim_view.update_articulations_kinematic()
        robot.update(sim_dt)
        sim.render()

        frame += 1
        if frame == len(joint_pos):
            if args.once:
                break
            frame = 0
        time.sleep(frame_period)



if __name__ == "__main__":
    main()
