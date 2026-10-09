#!/usr/bin/env python3
"""Generate named Isaac Lab body targets for BeyondMimic from a G1 CSV.

Accepts legacy centimeter/degree CSVs or mjlab-format meter/radian CSVs.
Sampling defaults to 120 FPS input and 50 FPS output; specify the actual input
rate for other clips. No dynamics are stepped during forward kinematics.
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile


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
    sys.path.insert(0, str(root))
    from convert_mj import convert_rows

    rows, _ = convert_rows(args.input)
    app = AppLauncher(args).app
    try:
        process(args, rows)
    finally:
        app.close()


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
    positions, rotations, joints = [], [], []
    for index in range(len(times)):
        root = robot.data.default_root_state.clone()
        root[:, :3] = torch.as_tensor(linear[index, :3], device=robot.device, dtype=torch.float32)
        root[:, 3:7] = torch.as_tensor(quat[index], device=robot.device, dtype=torch.float32)
        root[:, 7:] = 0
        joint = robot.data.default_joint_pos.clone()
        joint[:, ids] = torch.as_tensor(linear[index, 3:], device=robot.device, dtype=torch.float32)
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
