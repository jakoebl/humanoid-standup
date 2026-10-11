#!/usr/bin/env python3
"""Record each training reference using a Mimic-Multiclip checkpoint."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--clips-dir", type=Path, help="Override the saved training clip list.")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    project = Path(__file__).resolve().parents[1]
    if args.clips_dir:
        clips = sorted(args.clips_dir.expanduser().resolve().glob("*.npz"))
    else:
        manifest = checkpoint.parent / "params/training_clips.yaml"
        if not manifest.is_file():
            parser.error("Missing training_clips.yaml; specify --clips-dir.")
        with manifest.open() as stream:
            saved = yaml.safe_load(stream)
        clips = [Path(path) for path in saved["clips"]]
    if not clips:
        parser.error("No clips found.")
    # Validate all references before launching expensive simulator processes.
    lengths = []
    for clip in clips:
        with np.load(clip, allow_pickle=False) as data:
            lengths.append(len(data["joint_pos"]))
    output = project / "vids" / checkpoint.parent.name
    for index, (clip, length) in enumerate(zip(clips, lengths)):
        folder = output / f"{index:02d}_{clip.stem}"
        print(f"Recording {clip.name} ({length} frames) to {folder}", flush=True)
        subprocess.run([
            sys.executable, str(project / "scripts/rsl_rl/play.py"),
            "--task", "Mimic-Multiclip-Play", "--num_envs", "1",
            "--checkpoint", str(checkpoint), "--headless", "--video",
            "--video_length", str(length), "--video_folder", str(folder),
            f"env.commands.motion.play_clip_index={index}",
            "env.commands.motion.motion_files=" + "[" + ",".join('"' + str(path) + '"' for path in clips) + "]",
        ], cwd=project, check=True, env=os.environ.copy())
    print(f"Videos saved to {output}")


if __name__ == "__main__":
    main()
