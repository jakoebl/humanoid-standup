#!/usr/bin/env python3
"""Record reference motion or checkpoint policy rollouts in Isaac Sim."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, nargs="?", help="Processed motion CSV (.csv may be omitted).")
    parser.add_argument("--output", type=Path, help="Output MP4; defaults to vids/<clip-name>.mp4 in the project.")
    parser.add_argument("--fps", type=int, choices=(30, 60, 120), default=120,
                        help="Video FPS (default: 120, preserving every reference frame).")
    parser.add_argument("--gui", action="store_true", help="Show Isaac Sim while recording.")
    parser.add_argument("--device", default="cuda:0", help="Simulation device (default: cuda:0).")
    parser.add_argument("--checkpoint", type=Path, help="Record the policy on every training clip.")
    parser.add_argument("--iteration", type=int, help="Iteration subfolder; defaults to model_<iteration>.pt number.")
    parser.add_argument("--training-name", help="Output directory name; defaults to the checkpoint run name.")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Number of policy recordings per clip, each from frame 0 (default: 1).")
    parser.add_argument("--task", default="Humanoid-Standup")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("Repetitions must be positive.")
    if args.repetitions != 1 and not args.checkpoint:
        parser.error("--repetitions requires --checkpoint.")
    project = Path(__file__).resolve().parents[1]
    python = "/isaac-sim/python.sh" if Path("/isaac-sim/python.sh").exists() else sys.executable
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")
        name = args.training_name or checkpoint.parent.name
        if Path(name).name != name or name in (".", ".."):
            parser.error("Training name must be a single directory name.")
        iteration = args.iteration
        if iteration is None:
            prefix, separator, number = checkpoint.stem.rpartition("_")
            if prefix != "model" or not separator or not number.isdecimal():
                parser.error("Provide --iteration for a checkpoint not named model_<iteration>.pt.")
            iteration = int(number)
        if iteration < 0:
            parser.error("Iteration must be nonnegative.")
        command = [python, str(project / "scripts/rsl_rl/play.py"),
                   "--checkpoint", str(checkpoint), "--task", args.task, "--agent", args.agent,
                   "--num_envs", "1", "--device", args.device,
                   "--record-policy-dir", str(project / "vids" / name / str(iteration)),
                   "--record-policy-repetitions", str(args.repetitions)]
        if not args.gui:
            command.append("--headless")
        return subprocess.call(command)
    if args.clip is None:
        parser.error("Provide a clip or --checkpoint.")
    clip = args.clip.expanduser().resolve()
    if not clip.exists() and clip.suffix != ".csv":
        clip = Path(f"{clip}.csv")
    if not clip.is_file():
        parser.error(f"CSV not found: {clip}")
    output = (args.output.expanduser().resolve() if args.output else
              (project / "vids") / f"{clip.stem}.mp4")
    if output.suffix.lower() != ".mp4":
        parser.error("Output must have an .mp4 extension.")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    command = [python, str(project / "scripts/play_reference.py"),
               "--csv", str(clip.resolve()), "--record", str(output.resolve()),
               "--video-fps", str(args.fps), "--device", args.device]
    if not args.gui:
        command.append("--headless")
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
