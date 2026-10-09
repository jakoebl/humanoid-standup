# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--record-policy-dir", type=str, help="Record each clip from frame 0 at 1080p and 60 fps.")
parser.add_argument("--record-policy-repetitions", type=int, default=1,
                    help="Record this many rollouts per clip, each starting at frame 0.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video or args_cli.record_policy_dir:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import humanoid_standup.tasks  # noqa: F401
from humanoid_standup.tasks.manager_based.humanoid_standup.mdp.reference import REFERENCE
from humanoid_standup.tasks.manager_based.humanoid_standup.mdp.position_tracking import position_tracking_targets

# Load all training clips before Hydra constructs the environment config.
IS_TRACKING_TASK = args_cli.task is not None and args_cli.task.split(":")[-1].startswith(("BeyondMimic", "Mimic-Multiclip"))
if not IS_TRACKING_TASK:
    REFERENCE.__init__(Path(__file__).resolve().parents[2] / "data/training", REFERENCE.dt, REFERENCE.device)
elif args_cli.record_policy_dir:
    raise ValueError("BeyondMimic playback supports --video; --record-policy-dir is for Humanoid-Standup clips.")


class KinematicsVisualization(gym.Wrapper):
    """Display clip targets and measured kinematics, including in recorded video."""

    def __init__(self, env):
        super().__init__(env)
        self.robot = env.unwrapped.scene["robot"]
        self.body_ids, names = self.robot.find_bodies(REFERENCE.END_EFFECTOR_NAMES, preserve_order=True)
        if list(names) != list(REFERENCE.END_EFFECTOR_NAMES):
            raise ValueError(f"Could not resolve reference end effectors in order: {names}")
        self.masses = self.robot.root_physx_view.get_masses().to(self.robot.device)
        colors = {
            "com": (1.0, 0.8, 0.0),
            "left_wrist": (1.0, 0.1, 0.1),
            "right_wrist": (0.1, 0.3, 1.0),
            "left_foot": (1.0, 0.1, 0.6),
            "right_foot": (0.1, 1.0, 0.5),
        }
        self.reference_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/ReferenceKinematics",
            markers={name: sim_utils.SphereCfg(
                radius=0.045 if name == "com" else 0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ) for name, color in colors.items()},
        ))
        self.actual_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/ActualKinematics",
            markers={name: sim_utils.CuboidCfg(
                size=(0.04, 0.04, 0.04),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ) for name, color in colors.items()},
        ))
        self.marker_indices = torch.arange(5, device=self.robot.device).repeat(env.unwrapped.num_envs)
        print("[INFO] Kinematics: spheres = clip targets; cubes = actual robot positions. "
              "COM yellow, left/right wrist red/blue, left/right foot pink/green. "
              "Above the standing CoM threshold, targets follow the actual left foot and stance yaw.")

    def update_markers(self):
        env = self.env.unwrapped
        frame_ids = REFERENCE.frame_ids_for_env(env).to(REFERENCE.device)
        targets = torch.cat((REFERENCE.get_com_pos(frame_ids)[:, None, :],
                             REFERENCE.get_end_effector_pos(frame_ids)), dim=1).to(self.robot.device)
        height = env.cfg.rewards.com_tracking.params["upright_com_height"]
        targets = position_tracking_targets(env, self.robot, frame_ids, targets, height)
        targets = targets + env.scene.env_origins[:, None, :]
        com = (self.robot.data.body_com_pos_w * self.masses[..., None]).sum(dim=1)
        com = com / self.masses.sum(dim=1, keepdim=True)
        actual = torch.cat((com[:, None, :], self.robot.data.body_link_pos_w[:, self.body_ids]), dim=1)
        self.reference_markers.visualize(translations=targets.reshape(-1, 3), marker_indices=self.marker_indices)
        self.actual_markers.visualize(translations=actual.reshape(-1, 3), marker_indices=self.marker_indices)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.update_markers()
        return result

    def step(self, action):
        result = self.env.step(action)
        self.update_markers()
        if self.env.unwrapped.sim.has_gui():
            self.env.unwrapped.sim.render()
        return result


def record_policy_rollouts(env, runner, policy, output_dir, action_repeat=1, repetitions=1):
    """Record each clip from frame zero, holding actions between capture steps."""
    import math
    import tempfile
    import imageio.v2 as imageio

    if repetitions < 1:
        raise ValueError("Recording repetitions must be positive.")
    raw = env.unwrapped
    reset_cfg = raw.event_manager.get_term_cfg("reset_robot")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        policy_nn = policy
    else:
        policy_nn = getattr(runner.alg, "policy", None)
        if policy_nn is None:
            policy_nn = runner.alg.actor_critic
    for clip_id, clip in enumerate(REFERENCE.paths):
        folder = output_dir / clip.stem
        folder.mkdir(parents=True, exist_ok=True)
        for repetition in range(repetitions):
            reset_cfg.params.update(clip_id=clip_id, start_fraction=0.0)
            raw.event_manager.set_term_cfg("reset_robot", reset_cfg)
            with torch.inference_mode():
                obs, _ = env.reset()
                policy_nn.reset(torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device))
                start = int(raw.reference_start_frames[0].item())
                remaining = int(REFERENCE.clip_lengths[clip_id].item()) - 1 - start
                steps = math.ceil(remaining * REFERENCE.dt / raw.step_dt)
                raw.render()
                for _ in range(10):
                    raw.sim.render()
                output = folder / f"{repetition}.mp4"
                fd, temporary = tempfile.mkstemp(suffix=".mp4", prefix=".record-", dir=folder)
                os.close(fd)
                try:
                    with imageio.get_writer(temporary, format="FFMPEG", mode="I",
                                            fps=1 / raw.step_dt, codec="libx264",
                                            pixelformat="yuv420p", macro_block_size=2) as writer:
                        for frame in range(steps + 1):
                            if frame:
                                if (frame - 1) % action_repeat == 0:
                                    action = policy(obs)
                                obs, _, _, _ = env.step(action)
                            # Headless steps may skip rendering without RTX sensors.
                            # Flush physics/marker updates before reading the RGB annotator;
                            # render(recompute=True) alone can return its cached frame.
                            raw.sim.render()
                            rgb = raw.render(recompute=True)
                            if rgb is None or not rgb.size:
                                raise RuntimeError("Isaac Sim returned an empty RGB frame.")
                            writer.append_data(rgb)
                    os.replace(temporary, output)
                    print(f"Saved {output}", flush=True)
                finally:
                    Path(temporary).unlink(missing_ok=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    # Evaluate and validate reference kinematics at the true robot mass.
    if not IS_TRACKING_TASK:
        env_cfg.curriculum.robot_mass = None

    # Let playback continue through tracking errors until the reference finishes.
    if IS_TRACKING_TASK:
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.events.push_robot = None
        env_cfg.commands.motion.sampling_mode = "start"
        env_cfg.commands.motion.pose_range = {}
        env_cfg.commands.motion.velocity_range = {}
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    else:
        env_cfg.terminations.ee_deviation = None
        env_cfg.terminations.com_deviation = None
        print(f"[INFO] Playback samples uniformly from {REFERENCE.num_clips} training clips on each reset.")
    if args_cli.record_policy_dir:
        env_cfg.scene.num_envs = 1
        env_cfg.viewer.resolution = (1920, 1080)
        # Capture at 60 Hz while retaining the original policy action period.
        original_decimation = env_cfg.decimation
        recording_decimation = round(1 / (60 * env_cfg.sim.dt))
        if (recording_decimation < 1
                or abs(recording_decimation * env_cfg.sim.dt - 1 / 60) > 1e-9
                or original_decimation % recording_decimation):
            raise ValueError("60 fps recording requires compatible physics and policy timing.")
        recording_action_repeat = original_decimation // recording_decimation
        env_cfg.decimation = recording_decimation
        env_cfg.sim.render_interval = recording_decimation
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.events.reset_robot.params.update(joint_pos_noise=0.0, joint_vel_noise=0.0)
        # Record to the end without automatic resets contaminating the final frame.
        for name in vars(env_cfg.terminations):
            if not name.startswith("_"):
                setattr(env_cfg.terminations, name, None)
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video or args_cli.record_policy_dir else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Update markers before RecordVideo captures each frame.
    if not IS_TRACKING_TASK:
        env = KinematicsVisualization(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.record_policy_dir:
        try:
            record_policy_rollouts(env, runner, policy, Path(args_cli.record_policy_dir),
                                   action_repeat=recording_action_repeat,
                                   repetitions=args_cli.record_policy_repetitions)
        finally:
            env.close()
        return

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
