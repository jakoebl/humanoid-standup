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
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_folder", type=str, default=None, help="Output directory for recorded videos.")
parser.add_argument("--standup-success-file", type=str, default=None,
                    help="Evaluate one full clip from frame 0 in every environment and write standing success rates.")
parser.add_argument("--initial-joint-noise-sweep", type=float, nargs="+",
                    help="Uniform +/- joint position noise in radians; evaluate all clips and save CSV beside success file.")
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
if args_cli.initial_joint_noise_sweep is not None:
    import math
    if not args_cli.standup_success_file:
        parser.error("--initial-joint-noise-sweep requires --standup-success-file")
    if any(not math.isfinite(value) or value < 0 for value in args_cli.initial_joint_noise_sweep):
        parser.error("Initial joint noise amplitudes must be finite and nonnegative.")
# always enable cameras to record video
if args_cli.video:
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

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
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

# Whole-body CoM height counted as standing by the standing evaluation.
UPRIGHT_COM_HEIGHT = 0.5


def evaluate_standup_success(env, runner, policy, output_file):
    """Count trials that reach standing, without early resets or clip looping."""
    raw = env.unwrapped
    motion = raw.command_manager.get_term("motion")
    clip_id = motion.cfg.play_clip_index
    robot = raw.scene["robot"]
    # Use actual randomized masses, rather than the nominal asset masses.
    masses = robot.root_physx_view.get_masses().to(raw.device)

    def standing():
        com = (robot.data.body_com_pos_w * masses[..., None]).sum(dim=1) / masses.sum(dim=1, keepdim=True)
        return com[:, 2] - raw.scene.env_origins[:, 2] >= UPRIGHT_COM_HEIGHT

    with torch.inference_mode():
        obs, _ = env.reset()
        if not torch.all(motion.local_frames == 0) or not torch.all(motion.clip_ids == clip_id):
            raise RuntimeError("Standing evaluation must start every environment at frame 0 of the selected clip.")
        policy_nn = policy if version.parse(installed_version) >= version.parse("4.0.0") else getattr(runner.alg, "policy", None)
        if policy_nn is None:
            policy_nn = runner.alg.actor_critic
        policy_nn.reset(torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device))
        succeeded = standing()
        # The initial state is frame 0; stop at the last frame before the command loops.
        steps = int(motion.motion.lengths[clip_id].item()) - 1
        for _ in range(steps):
            obs, _, dones, _ = env.step(policy(obs))
            if torch.any(dones):
                raise RuntimeError("Unexpected reset during full-clip standing evaluation.")
            succeeded |= standing()
        successes = int(succeeded.sum().item())
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        f"Clip: {motion.motion.paths[clip_id]}\n"
        f"Trials: {raw.num_envs}\nStart frame: 0\nFinal frame: {steps}\n"
        f"Success criterion: whole-body CoM height >= {UPRIGHT_COM_HEIGHT} m at any clip frame\n"
        "Early terminations: disabled\n"
        f"Successes: {successes}\nSuccess rate: {successes / raw.num_envs:.6%}\n"
    )
    print(f"Standing success: {successes}/{raw.num_envs}; saved to {output_file}", flush=True)
    return successes


def evaluate_initial_joint_noise(env, runner, policy, output_file):
    """Sweep physical initial joint perturbations with paired random seeds."""
    import csv
    import json

    raw = env.unwrapped
    motion = raw.command_manager.get_term("motion")
    destination = Path(output_file).parent
    destination.mkdir(parents=True, exist_ok=True)
    seed = raw.cfg.seed
    rows = []
    for amplitude in args_cli.initial_joint_noise_sweep:
        if amplitude < 0:
            raise ValueError("Noise amplitudes must be nonnegative.")
        motion.cfg.joint_position_range = (-amplitude, amplitude)
        for clip_id, clip in enumerate(motion.motion.paths):
            motion.cfg.play_clip_index = clip_id
            raw.seed(seed)
            successes = evaluate_standup_success(
                env, runner, policy, destination / f"noise_{amplitude:g}_clip_{clip_id}.txt")
            rows.append(dict(noise_rad=amplitude, clip=Path(clip).name, trials=raw.num_envs,
                             successes=successes, success_rate=successes / raw.num_envs))
            with (destination / "success_rates.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (destination / "metadata.json").write_text(json.dumps(dict(
        checkpoint=args_cli.checkpoint, seed=seed, noise="independent uniform joint offsets in radians, clamped to soft joint limits",
        amplitudes=args_cli.initial_joint_noise_sweep, trials_per_clip=raw.num_envs,
        success="whole-body CoM >= 0.5 m at any frame; early terminations disabled",
        root_pose_noise=False, velocity_noise=False, observation_noise=False,
    ), indent=2))


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
    # Let playback continue through tracking errors until the reference finishes.
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.events.assist_force = None
    env_cfg.curriculum.assist_force = None
    env_cfg.commands.motion.sampling_mode = "start"
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    # Preserve the task-configured joint position noise on reset.
    if args_cli.video:
        env_cfg.viewer.resolution = (1920, 1080)
    if args_cli.standup_success_file:
        if task_name != "Humanoid-Standup-Play" or env_cfg.commands.motion.play_clip_index is None:
            raise ValueError("Standing evaluation requires Humanoid-Standup-Play with a selected clip.")
        if args_cli.video:
            raise ValueError("Run standing evaluation separately from video recording.")
        for name in vars(env_cfg.terminations):
            if not name.startswith("_"):
                setattr(env_cfg.terminations, name, None)
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_folder or os.path.join(log_dir, "videos", "play"),
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

    if args_cli.standup_success_file:
        try:
            if args_cli.initial_joint_noise_sweep:
                evaluate_initial_joint_noise(env, runner, policy, args_cli.standup_success_file)
            else:
                evaluate_standup_success(env, runner, policy, args_cli.standup_success_file)
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

    # The deployment controller does not provide the lie-down goal observation yet.
    if train_task_name == "Humanoid-Standup":
        from onnx_metadata import attach_tracking_metadata

        attach_tracking_metadata(env.unwrapped, os.path.join(export_model_dir, "policy.onnx"))
        print("[INFO] Attached g1_rl_controller deployment metadata to policy.onnx.")

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
