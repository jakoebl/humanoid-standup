# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a reference motion clip directly in Isaac Sim."""

import argparse
import csv
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trajectory reference motion in Isaac Sim.")
parser.add_argument("--task", type=str, default="Humanoid-Standup")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--csv",
    type=Path,
    default=None,
    help="Motion CSV to play, including precomputed COM and end-effector positions.",
)
parser.add_argument(
    "--foot-relative-markers", action="store_true",
    help="Align markers to the actual left foot above the standing CoM threshold, as in training.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--check-kinematics", action="store_true",
    help="Compare every frame's COM and EE positions against PhysX, export measurements, then exit.",
)
parser.add_argument("--ee-tolerance", type=float, default=0.001, help="Maximum EE error in meters (default: 1 mm).")
parser.add_argument("--com-tolerance", type=float, default=0.001, help="Maximum COM error in meters (default: 1 mm).")
parser.add_argument(
    "--check-output", type=Path, default=Path("outputs/reference_kinematics_check.csv"),
    help="Per-frame measured/expected coordinates and errors; summary saved alongside as JSON.",
)
parser.add_argument("--record", type=Path, help="Record one pass to an MP4 file, then exit.")
parser.add_argument("--video-fps", type=int, choices=(30, 60, 120), default=120)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.record is not None:
    if args_cli.check_kinematics or args_cli.num_envs != 1:
        parser.error("Recording requires one environment and cannot be combined with --check-kinematics.")
    if args_cli.record.suffix.lower() != ".mp4":
        parser.error("Recording output must have an .mp4 extension.")
    if args_cli.record.exists():
        parser.error(f"Output already exists: {args_cli.record}")
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import isaaclab_tasks  # noqa: F401
import humanoid_standup.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from humanoid_standup.tasks.manager_based.humanoid_standup.mdp.reference import REFERENCE
from humanoid_standup.tasks.manager_based.humanoid_standup.mdp.position_tracking import position_tracking_targets


class PlaybackControls:
    """Select exact CSV frames while keeping the viewport interactive."""

    def __init__(self, num_frames, dt, clip_name):
        import omni.ui as ui

        self.num_frames, self.dt = num_frames, dt
        self.frame_id = 0
        self.playing = True
        self._syncing = False
        self._anchor_time = time.perf_counter()
        self._anchor_frame = 0
        self.window = ui.Window("Reference Playback", width=520, height=180)
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label(clip_name)
                with ui.HStack(height=25, spacing=6):
                    ui.Button("Previous frame", clicked_fn=lambda: self.seek(self.frame_id - 1))
                    self.play_button = ui.Button("Pause", clicked_fn=self.toggle)
                    ui.Button("Next frame", clicked_fn=lambda: self.seek(self.frame_id + 1))
                self.model = ui.SimpleIntModel(0)
                ui.IntSlider(model=self.model, min=0, max=num_frames - 1, height=22)
                with ui.HStack(height=22, spacing=6):
                    ui.Label("Frame (from 0)", width=110)
                    ui.IntField(model=self.model, width=90)
                    self.status = ui.Label("")
                ui.Label("Scrubbing or stepping pauses playback. Play loops the clip.", height=20)
        self.model.add_value_changed_fn(self._on_frame_changed)
        self._sync()

    def _sync(self):
        self._syncing = True
        try:
            self.model.set_value(self.frame_id)
            self.status.text = f"/ {self.num_frames - 1}    {self.frame_id * self.dt:.3f} s"
            self.play_button.text = "Pause" if self.playing else "Play"
        finally:
            self._syncing = False

    def _on_frame_changed(self, model):
        if not self._syncing:
            self.seek(model.as_int)

    def seek(self, frame_id):
        self.frame_id = max(0, min(self.num_frames - 1, frame_id))
        self.playing = False
        self._sync()

    def toggle(self):
        self.playing = not self.playing
        self._anchor_frame = self.frame_id
        self._anchor_time = time.perf_counter()
        self._sync()

    def advance(self):
        if self.playing:
            elapsed_frames = int((time.perf_counter() - self._anchor_time) / self.dt)
            self.frame_id = (self._anchor_frame + elapsed_frames) % self.num_frames
            self._sync()
        return self.frame_id


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
    env=None,
    foot_relative: bool = False,
) -> None:
    positions = reference_positions[frame_id].to(env_origin.device).reshape(5, 3)
    if foot_relative:
        raw = env.unwrapped
        robot = raw.scene["robot"]
        # Refresh FK and cached poses after directly writing the reference state.
        raw.sim.physics_sim_view.update_articulations_kinematic()
        robot.update(REFERENCE.dt)
        frame_ids = torch.full((raw.num_envs,), frame_id, device=REFERENCE.device, dtype=torch.long)
        targets = positions[None].expand(raw.num_envs, -1, -1)
        height = raw.cfg.rewards.com_tracking.params["upright_com_height"]
        positions = position_tracking_targets(raw, robot, frame_ids, targets, height)[0]
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



def check_kinematics(env, reference_positions: torch.Tensor, kinematics_csv: Path) -> None:
    """Read PhysX FK at every exact reference pose, without integrating dynamics."""
    if not all(np.isfinite(t) and t >= 0 for t in (args_cli.ee_tolerance, args_cli.com_tolerance)):
        raise ValueError("Kinematics tolerances must be finite and nonnegative.")
    # Ensure marker targets describe the motion being played.
    motion = np.atleast_2d(np.loadtxt(kinematics_csv, delimiter=",", skiprows=1))[:, :36]
    configured = np.atleast_2d(np.loadtxt(REFERENCE.path, delimiter=",", skiprows=1))[:, :36]
    if motion.shape != configured.shape or not np.allclose(motion, configured, rtol=0, atol=1e-5):
        raise ValueError("Kinematics CSV motion columns do not match the reference being played.")
    if not torch.isfinite(reference_positions).all():
        raise ValueError("Kinematics CSV contains nonfinite positions.")

    robot = env.unwrapped.scene["robot"]
    body_ids, body_names = robot.find_bodies(REFERENCE.END_EFFECTOR_NAMES, preserve_order=True)
    if list(body_names) != list(REFERENCE.END_EFFECTOR_NAMES):
        raise ValueError(f"Could not resolve reference end effectors in order: {body_names}")
    # Read the actual simulation masses, including every articulated body/finger.
    masses = robot.root_physx_view.get_masses()[0].to(robot.device)
    if not torch.isfinite(masses).all() or (masses < 0).any() or masses.sum() <= 0:
        raise ValueError("Simulation body masses must be finite, nonnegative and have positive total mass.")
    names = ["com", *body_names]
    measured = []
    for frame_id in range(REFERENCE.num_frames):
        if not simulation_app.is_running():
            raise RuntimeError(f"Simulation closed after {frame_id}/{REFERENCE.num_frames} frames.")
        apply_reference_frame(env, frame_id)
        env.unwrapped.sim.physics_sim_view.update_articulations_kinematic()
        # Advance buffer timestamps, not physics, to guarantee fresh body pose reads.
        robot.update(REFERENCE.dt)
        origin = env.unwrapped.scene.env_origins[0]
        com = (robot.data.body_com_pos_w[0] * masses[:, None]).sum(dim=0) / masses.sum()
        ee = robot.data.body_link_pos_w[0, body_ids]
        measured.append((torch.cat((com[None], ee), dim=0) - origin).cpu().numpy())
        if not args_cli.headless:
            env.unwrapped.sim.render()

    actual = np.stack(measured)
    expected = reference_positions.numpy().reshape(-1, len(names), 3)
    if not np.isfinite(actual).all():
        raise RuntimeError("Simulation returned nonfinite positions.")
    errors = np.linalg.norm(actual - expected, axis=-1)
    output = args_cli.check_output
    if output.resolve() in (REFERENCE.path.resolve(), kinematics_csv.resolve()):
        raise ValueError("Check output must not overwrite a reference CSV.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "reference_frame", "time_s", "body",
                         "sim_x", "sim_y", "sim_z", "csv_x", "csv_y", "csv_z", "error_m"])
        for frame_id in range(REFERENCE.num_frames):
            for index, name in enumerate(names):
                writer.writerow([frame_id, REFERENCE.frames[frame_id].item(), frame_id * REFERENCE.dt, name,
                                 *actual[frame_id, index], *expected[frame_id, index], errors[frame_id, index]])
    summary = {"reference": str(REFERENCE.path), "targets": str(kinematics_csv),
               "frames": REFERENCE.num_frames, "total_mass_kg": masses.sum().item(), "bodies": {}}
    passed = True
    for index, name in enumerate(names):
        values = errors[:, index]
        tolerance = args_cli.com_tolerance if index == 0 else args_cli.ee_tolerance
        result = {"mean_error_m": float(values.mean()), "rms_error_m": float(np.sqrt(np.mean(values ** 2))),
                  "max_error_m": float(values.max()), "worst_frame_index": int(values.argmax()),
                  "tolerance_m": tolerance, "passed": bool(np.all(values <= tolerance))}
        summary["bodies"][name] = result
        passed = passed and result["passed"]
        print(f"{name}: mean={result['mean_error_m']:.6f} m, RMS={result['rms_error_m']:.6f} m, "
              f"max={result['max_error_m']:.6f} m at frame {result['worst_frame_index']} "
              f"({'PASS' if result['passed'] else 'FAIL'})", flush=True)
    summary["passed"] = passed
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Measurements: {output}; summary: {output.with_suffix('.json')}", flush=True)
    if not passed:
        raise RuntimeError("Reference kinematics failed tolerance checks; see exported measurements.")



def frame_recording_camera(env):
    """Fit a padded sphere around all robot links into the video camera."""
    from pxr import UsdGeom

    sim = env.unwrapped.sim
    robot = env.unwrapped.scene["robot"]
    # Refresh pose buffers after writing the reference and updating PhysX FK.
    robot.update(REFERENCE.dt)
    positions = robot.data.body_link_pos_w[0].detach().cpu().numpy()
    if not np.isfinite(positions).all():
        raise RuntimeError("Cannot frame robot with nonfinite link positions.")
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    # Include link geometry beyond joint origins, fingers and reference markers.
    radius = max(1.2, float(np.linalg.norm(positions - center, axis=1).max()) + 0.35)
    width, height = env.unwrapped.cfg.viewer.resolution
    focal_length, horizontal_aperture = 24.0, 36.0
    vertical_aperture = horizontal_aperture * height / width
    half_fov = np.arctan(min(horizontal_aperture, vertical_aperture) / (2 * focal_length))
    distance = 1.15 * radius / np.sin(half_fov)
    direction = np.array([1.0, -1.0, 0.65])
    direction /= np.linalg.norm(direction)
    camera_path = env.unwrapped.cfg.viewer.cam_prim_path
    camera = UsdGeom.Camera(sim.stage.GetPrimAtPath(camera_path))
    camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
    camera.GetVerticalApertureAttr().Set(vertical_aperture)
    camera.GetHorizontalApertureOffsetAttr().Set(0.0)
    camera.GetVerticalApertureOffsetAttr().Set(0.0)
    sim.set_camera_view(eye=(center + direction * distance).tolist(),
                        target=center.tolist(), camera_prim_path=camera_path)


def record_reference(env, reference_positions, reference_markers, output, fps):
    """Render one pass at fixed sample times without integrating physics."""
    import imageio.v2 as imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".mp4", prefix=".recording-", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary)
    stride = round(1 / (fps * REFERENCE.dt))
    frame_ids = range(0, REFERENCE.num_frames, stride)
    try:
        with imageio.get_writer(str(temporary), format="FFMPEG", mode="I", fps=fps,
                                codec="libx264", pixelformat="yuv420p", macro_block_size=2) as writer:
            for index, frame_id in enumerate(frame_ids):
                if not simulation_app.is_running():
                    raise RuntimeError("Simulation closed before recording finished.")
                apply_reference_frame(env, frame_id)
                show_reference_markers(reference_markers, reference_positions, frame_id,
                                       env.unwrapped.scene.env_origins[0], env=env,
                                       foot_relative=args_cli.foot_relative_markers)
                env.unwrapped.sim.physics_sim_view.update_articulations_kinematic()
                frame_recording_camera(env)
                if index == 0:
                    # Create the RGB annotator, then warm up with the first pose held fixed.
                    env.unwrapped.render()
                    for _ in range(10):
                        env.unwrapped.sim.render()
                env.unwrapped.sim.render()
                rgb = env.unwrapped.render(recompute=True)
                if rgb is None or not rgb.size:
                    raise RuntimeError("Isaac Sim returned an empty RGB frame.")
                writer.append_data(rgb)
                if index % fps == 0:
                    print(f"Recording: {index + 1}/{len(frame_ids)} frames", flush=True)
        # Publish only a completed file, and never overwrite an existing recording.
        os.link(temporary, output)
        print(f"Saved {output} ({len(frame_ids)} frames at {fps} FPS)", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    if args_cli.csv is not None:
        # Managers import this shared object directly. Reload it in place before
        # constructing the config so poses, targets and episode duration agree.
        REFERENCE.__init__(path=args_cli.csv, dt=REFERENCE.dt, device=REFERENCE.device)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # Evaluate and validate reference kinematics at the true robot mass.
    env_cfg.curriculum.robot_mass = None
    if args_cli.record is not None:
        env_cfg.viewer.resolution = (1920, 1080)
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.record is not None else None)
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
    controls = None
    try:
        if args_cli.check_kinematics:
            check_kinematics(env, reference_positions, kinematics_csv)
            return

        if args_cli.record is not None:
            record_reference(env, reference_positions, reference_markers,
                             args_cli.record, args_cli.video_fps)
            return

        if not args_cli.headless:
            controls = PlaybackControls(REFERENCE.num_frames, REFERENCE.dt, kinematics_csv.name)

        while simulation_app.is_running():
            frame_start = time.perf_counter()
            if controls is not None:
                frame_id = controls.advance()
            apply_reference_frame(env, frame_id)
            show_reference_markers(
                reference_markers,
                reference_positions,
                frame_id,
                env.unwrapped.scene.env_origins[0],
                env=env,
                foot_relative=args_cli.foot_relative_markers,
            )
            # Render the exact reference pose. A physics step applies gravity,
            # contacts and PD actions, moving the robot away from the markers.
            env.unwrapped.sim.physics_sim_view.update_articulations_kinematic()
            env.unwrapped.sim.render()
            time.sleep(max(0.0, frame_step * REFERENCE.dt - (time.perf_counter() - frame_start)))
            if controls is None:
                frame_id = (frame_id + frame_step) % REFERENCE.num_frames
    finally:
        if controls is not None:
            controls.window.destroy()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Kit shutdown can terminate Python before a pending exception propagates.
        simulation_app.app.post_quit(1)
        raise
    finally:
        simulation_app.close()
