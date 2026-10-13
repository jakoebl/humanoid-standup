#!/usr/bin/env python3
"""Record reference motion or checkpoint policy rollouts in Isaac Sim.

CLI options:
Important:
clip: reference CSV path; .csv suffix is optional.
--checkpoint: policy checkpoint to record on all training clips.

Overrides:
--output: reference MP4 path (default: vids/<clip>.mp4).
--fps: reference video FPS: 30, 60, or 120 (default: 120).
--device: simulation device (default: cuda:0).
--agent: agent config entry point (default: rsl_rl_cfg_entry_point).
--task: override the task inferred from the checkpoint path.
--clips-dir: override the automatically resolved training clip directory.
--repetitions: recordings per policy clip (default: 1)


-h, --help: show usage and exit.
"""

import argparse
from pathlib import Path
import subprocess
import sys

from evaluate.checkpoints import infer_task, resolve_clips


def record_multiclip(args, checkpoint, project, python, output):
    """Record prepared motion clips headlessly."""
    import numpy as np
    clips = resolve_clips(checkpoint, project, args.task, args.clips_dir)
    # Validate references before launching expensive simulator processes.
    lengths = []
    for clip in clips:
        with np.load(clip, allow_pickle=False) as data:
            lengths.append(len(data["joint_pos"]))
    motion_files = "env.commands.motion.motion_files=[" + ",".join('"' + str(path) + '"' for path in clips) + "]"
    for index, (clip, length) in enumerate(zip(clips, lengths)):
        folder = output / f"{index:02d}_{clip.stem}"
        base = [python, str(project / "scripts/rsl_rl/play.py"),
                "--task", "BeyondMimic-Play", "--agent", args.agent,
                "--device", args.device, "--checkpoint", str(checkpoint),
                f"env.commands.motion.play_clip_index={index}", motion_files]
        for repetition in range(args.repetitions):
            destination = folder if args.repetitions == 1 else folder / str(repetition)
            print(f"Recording {clip.name} ({length} frames) to {destination}", flush=True)
            command = base + ["--num_envs", "1", "--video", "--reference-ghost",
                              "--video_length", str(length), "--video_folder", str(destination)]
            command.append("--headless")
            subprocess.run(command, cwd=project, check=True)
    print(f"Videos saved to {output}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, nargs="?", help="Processed motion CSV (.csv may be omitted).")
    parser.add_argument("--output", type=Path, help="Output MP4; defaults to vids/<clip-name>.mp4 in the project.")
    parser.add_argument("--fps", type=int, choices=(30, 60, 120), default=120,
                        help="Video FPS (default: 120, preserving every reference frame).")
    parser.add_argument("--device", default="cuda:0", help="Simulation device (default: cuda:0).")
    parser.add_argument("--checkpoint", type=Path, help="Record the policy on every training clip.")
    parser.add_argument("--clips-dir", type=Path, help="Override the training clips inferred from the checkpoint and task.")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Number of policy recordings per clip, each from frame 0 (default: 1).")
    parser.add_argument("--task", help="Override task inferred from the checkpoint experiment directory.")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("Repetitions must be positive.")
    if args.repetitions != 1 and not args.checkpoint:
        parser.error("--repetitions requires --checkpoint.")
    if args.clips_dir and not args.checkpoint:
        parser.error("--clips-dir requires --checkpoint.")
    project = Path(__file__).resolve().parents[1]
    python = "/isaac-sim/python.sh" if Path("/isaac-sim/python.sh").exists() else sys.executable
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")
        try:
            args.task = args.task or infer_task(checkpoint)
        except ValueError as error:
            parser.error(str(error))
        task = args.task.split(":")[-1]
        multiclip = task in ("BeyondMimic", "BeyondMimic-Play")
        if not multiclip and task not in ("Humanoid-Standup", "Humanoid-Standup-Play"):
            parser.error(f"Unsupported recording task: {args.task}")
        prefix, separator, number = checkpoint.stem.rpartition("_")
        if prefix != "model" or not separator or not number.isdecimal():
            parser.error("Checkpoint must be named model_<iteration>.pt.")
        output = project / "vids" / checkpoint.parent.name / str(int(number))
        if multiclip:
            try:
                return record_multiclip(args, checkpoint, project, python, output)
            except ValueError as error:
                parser.error(str(error))
        try:
            clips = resolve_clips(checkpoint, project, args.task, args.clips_dir)
        except ValueError as error:
            parser.error(str(error))
        directories = {clip.parent for clip in clips}
        if len(directories) != 1:
            parser.error("Humanoid-Standup recording requires clips in one directory.")
        command = [python, str(project / "scripts/rsl_rl/play.py"),
                   "--checkpoint", str(checkpoint), "--task", args.task, "--agent", args.agent,
                   "--num_envs", "1", "--device", args.device,
                   "--reference-clips-dir", str(next(iter(directories))),
                   "--record-policy-dir", str(output),
                   "--record-policy-repetitions", str(args.repetitions)]
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
    command.append("--headless")
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
