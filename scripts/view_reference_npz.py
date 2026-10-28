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
    parser.add_argument("--render-every", type=int, default=2,
                        help="Render every N pose updates; 1 renders every update.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative.")
    if args.render_every < 1:
        parser.error("--render-every must be at least 1.")

    app = AppLauncher(args).app
    import carb
    settings = carb.settings.get_settings()
    settings.set("/rtx/post/aa/op", 0)
    settings.set("/rtx/post/aa/quality", 0)
    settings.set("/app/window/width", 960)
    settings.set("/app/window/height", 540)
    for key in (
        "/rtx/shadows/enabled",
        "/rtx/reflections/enabled",
        "/rtx/indirectDiffuse/enabled",
        "/rtx/translucency/enabled",
        "/rtx/ambientOcclusion/enabled",
    ):
        settings.set(key, False)
    try:
        play(args, app)
    finally:
        app.close()


def play(args, app) -> None:
    import omni.ui as ui
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
        sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=1800.0),
        translation=(2.0, -2.0, 4.0),
    )
    robot_cfg_instance = robot_cfg().replace(prim_path="/World/ReferenceRobot")
    robot_cfg_instance.spawn.activate_contact_sensors = False
    robot_cfg_instance.spawn.articulation_props.enabled_self_collisions = False
    robot = Articulation(robot_cfg_instance)
    sim.reset()
    sim.set_camera_view(eye=(2.8, -2.8, 1.8), target=(0.0, 0.0, 0.8))
    robot_joint_ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if list(names) != list(JOINT_NAMES):
        raise ValueError(f"Robot joints do not match the reference: {names}")

    root = robot.data.default_root_state.clone()
    joints = robot.data.default_joint_pos.clone()
    frame_period = 1.0 / fps
    frame = args.start_frame
    wall_start = time.perf_counter()
    stats = ui.Window("Reference timing", width=280, height=100)
    with stats.frame:
        with ui.VStack(spacing=4):
            ui.Label("Reference playback")
            timing_label = ui.Label("")
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
        sim_time = (frame - args.start_frame) / fps
        real_time = time.perf_counter() - wall_start
        timing_label.text = (
            f"Sim time: {sim_time:7.2f} s\n"
            f"Real time: {real_time:7.2f} s\n"
            f"RTF: {sim_time / real_time if real_time > 0 else 0.0:5.2f}x"
        )
        if frame % args.render_every == 0:
            sim.render()

        frame += 1
        if frame == len(joint_pos):
            if args.once:
                break
            frame = 0
        time.sleep(frame_period)



if __name__ == "__main__":
    main()
