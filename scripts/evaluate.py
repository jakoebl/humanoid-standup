#!/usr/bin/env python3
"""Run full-clip trials with the selected evaluation.

--checkpoint: model to evaluate; task and clips are inferred from its path.
--criterion: standing_up, foot_motion, lifted_foot, jitteriness, actuator_effort, worst_rewards or standing_push.
Omit to run all unpushed evaluations; standing_push is a separate disturbed rollout.
--task: optional task override for relocated checkpoints.
--num-envs: parallel trials per clip (default: 1000).
--clip-index: evaluate one clip; default is every training clip.
--record: enable video rendering and recording; disabled by default.
--output: fresh result directory; defaults to evaluation/<run>_<model>_<YYYY-MM-DD_HH-MM-SS>.
--seed: evaluation seed (default: 42).
--foot-movement-threshold: post-standing foot displacement tolerance in meters (default: 0.02).
--foot-lift-threshold: upward ankle displacement after standing in meters (default: 0.02).
--device: simulation and policy device (default: cuda:0).
"""

import argparse
from contextlib import ExitStack
import inspect
import json
import math
from pathlib import Path
import sys

from evaluate.checkpoints import infer_task, resolve_clips


def json_safe(value):
    """Preserve config fields, naming Python callables instead of dropping them."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "__dict__"):
        return {key: json_safe(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def effective_event_parameters(events):
    """Include function defaults omitted from EventTermCfg.params (e.g. encoder bias)."""
    result = {}
    for name, term in vars(events).items():
        if name.startswith("_") or term is None or not hasattr(term, "func"):
            continue
        parameters = {}
        if callable(term.func):
            for parameter in inspect.signature(term.func).parameters.values():
                if parameter.default is not inspect.Parameter.empty:
                    parameters[parameter.name] = parameter.default
        parameters.update(term.params)
        result[name] = {"mode": term.mode, "parameters": json_safe(parameters)}
    return result


def configure_evaluation_randomization(cfg, motion_defaults):
    """Keep training observation/pose/joint noise, without periodic velocity pushes."""
    from copy import deepcopy

    cfg.events.push_robot = None
    cfg.observations.policy.enable_corruption = True
    cfg.commands.motion.pose_range = deepcopy(motion_defaults.pose_range)
    cfg.commands.motion.joint_position_range = motion_defaults.joint_position_range


def statistics(value):
    """Summarize sampled tensors without emitting thousands of trials into JSON."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    import numpy as np

    values = np.asarray(value)
    return {"shape": list(values.shape), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std())}


def environment_summary(raw, task, agent_cfg):
    """Capture the actual config after evaluation overrides and manager initialization."""
    cfg = raw.cfg
    motion = raw.command_manager.get_term("motion")
    robot = raw.scene["robot"]
    policy = cfg.observations.policy
    noise = {}
    for name, term in vars(policy).items():
        if name.startswith("_") or not hasattr(term, "noise"):
            continue
        noise[name] = {"enabled": bool(policy.enable_corruption and term.noise is not None),
                       "noise": json_safe(term.noise), "scale": json_safe(term.scale),
                       "clip": json_safe(term.clip), "parameters": json_safe(term.params)}
    sampled = {"body_masses_kg": statistics(robot.root_physx_view.get_masses())}
    if hasattr(motion, "encoder_bias"):
        sampled["encoder_bias_rad"] = statistics(motion.encoder_bias)
    for name in ("get_material_properties", "get_coms"):
        getter = getattr(robot.root_physx_view, name, None)
        if getter is not None:
            sampled[name.removeprefix("get_")] = statistics(getter())
    return {
        "task": task,
        "clip_duration_s": (int(motion.motion.lengths[motion.cfg.play_clip_index].item()) - 1) * raw.step_dt,
        "policy_observation_noise": noise,
        "reset_joint_position_noise_rad": json_safe(motion.cfg.joint_position_range),
        "effective_event_parameters": effective_event_parameters(cfg.events),
        "sampled_randomization_statistics": sampled,
        "resolved_environment_config": json_safe(cfg),
        "resolved_agent_config": json_safe(agent_cfg),
    }


def run_clip(env, policy, clip_index, evaluations, num_frames=None):
    """Run one shared full-clip rollout and send frames to each evaluation."""
    import torch

    raw = env.unwrapped
    motion = raw.command_manager.get_term("motion")
    motion.cfg.play_clip_index = clip_index
    length = int(motion.motion.lengths[clip_index].item())
    if length < 2:
        raise ValueError("Evaluation requires at least two clip frames.")
    if num_frames is not None:
        if num_frames < length:
            raise ValueError("Evaluation cannot truncate the reference clip.")
        if num_frames > length and not motion.cfg.hold_final_frame:
            raise ValueError("Extended rollouts require holding the final reference frame.")
        length = num_frames
    for evaluation in evaluations:
        evaluation.prepare(raw)
    with torch.inference_mode():
        obs, _ = env.reset()
        if not torch.all(motion.local_frames == 0) or not torch.all(motion.clip_ids == clip_index):
            raise RuntimeError("All trials must start at frame zero of the selected clip.")
        policy.reset(torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device))
        for frame in range(length):
            rewards = None
            if frame:
                obs, rewards, dones, _ = env.step(policy(obs))
                if torch.any(dones):
                    raise RuntimeError("Unexpected reset during full-clip evaluation.")
            for evaluation in evaluations:
                # each evaluation stores the data it needs
                evaluation.observe(raw, frame, rewards)
    return length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--criterion", choices=("standing_up", "foot_motion", "lifted_foot", "jitteriness", "actuator_effort", "worst_rewards", "standing_push"))
    parser.add_argument("--task")
    parser.add_argument("--num-envs", type=int, default=1000)
    parser.add_argument("--clip-index", type=int)
    parser.add_argument("--record", action="store_true", help="Enable video rendering and recording.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--foot-movement-threshold", type=float, default=0.1)
    parser.add_argument("--foot-lift-threshold", type=float, default=0.02)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive.")
    if not math.isfinite(args.foot_movement_threshold) or args.foot_movement_threshold < 0:
        parser.error("--foot-movement-threshold must be finite and nonnegative.")
    if not math.isfinite(args.foot_lift_threshold) or args.foot_lift_threshold < 0:
        parser.error("--foot-lift-threshold must be finite and nonnegative.")
    try:
        task = args.task or infer_task(checkpoint)
        clips = resolve_clips(checkpoint, Path(__file__).resolve().parents[1], task)
    except ValueError as error:
        parser.error(str(error))

    if task.split(":")[-1] not in ("BeyondMimic", "BeyondMimic-Play"):
        parser.error("Full-clip evaluation currently requires a BeyondMimic checkpoint.")
    if args.clip_index is not None and not 0 <= args.clip_index < len(clips):
        parser.error(f"--clip-index must be between 0 and {len(clips) - 1}.")
    from datetime import datetime, timezone

    project = Path(__file__).resolve().parents[1]
    output = (args.output.expanduser().resolve() if args.output else
              project / "evaluation" /
              f"{checkpoint.parent.name}_{checkpoint.stem}_{datetime.now(timezone.utc):%Y-%m-%d_%H-%M-%S}")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory to avoid mixing evaluations.")

    # Launch before importing simulator-dependent modules.
    from isaaclab.app import AppLauncher

    print("[evaluate] Starting simulator...", flush=True)
    record_video = args.record and args.criterion in (None, "worst_rewards", "jitteriness", "foot_motion")
    app = AppLauncher(headless=True, device=args.device, enable_cameras=record_video).app
    print("[evaluate] Simulator started; loading evaluation modules...", flush=True)
    env = None
    try:
        from importlib.metadata import version as installed_version

        import gymnasium as gym
        from rsl_rl.algorithms import PPO
        from rsl_rl.models import MLPModel
        from rsl_rl.modules import GaussianDistribution
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        import humanoid_standup.tasks  # noqa: F401
        from evaluate.actuator_effort import ActuatorEffort
        from evaluate.jitteriness import Jitteriness
        from evaluate.lifted_foot import LiftedFoot
        from evaluate.foot_motion import StandingFootMotion
        from evaluate.rewards import RewardEvaluation, TrajectoryCache, render_attempts
        from evaluate.standing_up import StandingUp
        from evaluate.standing_push import StandingPush

        from humanoid_standup.tasks.manager_based.beyond_mimic.env_cfg import CommandsCfg

        # copy the task environment and restore training noise for evaluation
        env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        env_cfg.actions.joint_pos.clip = None
        configure_evaluation_randomization(env_cfg, CommandsCfg().motion)
        # load reference clips
        env_cfg.commands.motion.motion_files = [str(clip) for clip in clips]
        # start from frame 0
        env_cfg.commands.motion.sampling_mode = "start"
        env_cfg.commands.motion.hold_final_frame = args.criterion == "standing_push"
        env_cfg.commands.motion.debug_vis = False

        # Retain configured randomization, but never terminate or reset early.
        env_cfg.commands.motion.play_clip_index = args.clip_index or 0
        for name in vars(env_cfg.terminations):
            if not name.startswith("_"):
                setattr(env_cfg.terminations, name, None)
        env_cfg.viewer.resolution = (1280, 720)
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        rsl_version = installed_version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_version)
        agent_cfg.device = args.device
        agent_cfg.seed = args.seed
        env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if record_video else None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cfg = agent_cfg.to_dict()
        # Pass classes directly so the runner does not scan and import scripts.
        runner_cfg["algorithm"]["class_name"] = PPO
        for name in ("actor", "critic"):
            runner_cfg[name]["class_name"] = MLPModel
            distribution = runner_cfg[name].get("distribution_cfg")
            if distribution:
                distribution["class_name"] = GaussianDistribution
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=args.device)
        print(f"[evaluate] Loading checkpoint: {checkpoint}", flush=True)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=args.device)

        indexes = range(len(clips)) if args.clip_index is None else [args.clip_index]
        for index in indexes:
            destination = output / f"{index:02d}_{clips[index].stem}"
            destination.mkdir(parents=True, exist_ok=True)
            raw = env.unwrapped
            length = int(raw.command_manager.get_term("motion").motion.lengths[index].item())
            with ExitStack() as stack:
                evaluations = []

                if args.criterion in (None, "worst_rewards"):
                    evaluations.append(stack.enter_context(RewardEvaluation(raw, length, destination, record_video)))
                if args.criterion in (None, "foot_motion"):
                    evaluations.append(StandingFootMotion(raw.num_envs, args.foot_movement_threshold))                
                if args.criterion in (None, "lifted_foot"):
                    evaluations.append(LiftedFoot(raw.num_envs, args.foot_lift_threshold))
                if args.criterion in (None, "jitteriness"):
                    evaluations.append(Jitteriness(raw.num_envs, raw.step_dt))
                if args.criterion in (None, "actuator_effort"):
                    evaluations.append(ActuatorEffort(raw.num_envs, raw.step_dt))
                if args.criterion in (None, "standing_up"):
                    evaluations.append(StandingUp(raw.num_envs, raw.step_dt))
                if args.criterion == "standing_push":
                    push_evaluation = StandingPush(raw.num_envs, raw.step_dt, args.foot_movement_threshold)
                    evaluations.append(push_evaluation)
                    length = push_evaluation.rollout_frames(length)


                print(f"Evaluating clip {index}: {raw.num_envs} trials, {length - 1} steps...", flush=True)
                # Share one trajectory cache across all video criteria.
                cache = next((item for item in evaluations if isinstance(item, RewardEvaluation)), None)
                observers = list(evaluations)
                if record_video and cache is None:
                    cache = stack.enter_context(TrajectoryCache(raw, length, destination))
                    observers.append(cache)
                run_clip(env, policy, index, observers, num_frames=length)

                # general information on the evaluation process
                summary = {"checkpoint": str(checkpoint), "clip": str(clips[index]),
                           "trials": raw.num_envs, "frames": length, "fps": 1 / raw.step_dt,
                           "early_termination": False,
                           "criterion": args.criterion or "all",
                           "evaluations": {evaluation.name: evaluation.summary() for evaluation in evaluations},
                           "environment": environment_summary(raw, task, agent_cfg)}                           
                (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

                # save data for each evaluation
                for evaluation in evaluations:
                    evaluation.save(destination)
                for evaluation in evaluations:
                    criterion = {"jitteriness": "jitteriness", "standing_foot_motion": "foot_motion",
                                 "standing_push": "standing_push"}.get(evaluation.name)
                    if record_video and criterion is not None:
                        render_attempts(raw, cache.states, evaluation.video_attempts(), destination, criterion=criterion)
        print(f"Evaluation results: {output}", flush=True)
    
    
    
    except Exception as error:
        import traceback

        print(f"Error during evaluation: {error}", flush=True)
        # Print before simulator shutdown, which can obscure the outer traceback.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    
    
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            app.close()


if __name__ == "__main__":
    main()
