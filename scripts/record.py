#!/usr/bin/env python3
"""Record one full CSV reference clip in Isaac Sim as an MP4."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="Processed motion CSV (.csv may be omitted).")
    parser.add_argument("--output", type=Path, help="Output MP4; defaults to vids/<clip-name>.mp4 in the project.")
    parser.add_argument("--fps", type=int, choices=(30, 60, 120), default=120,
                        help="Video FPS (default: 120, preserving every reference frame).")
    parser.add_argument("--gui", action="store_true", help="Show Isaac Sim while recording.")
    parser.add_argument("--device", default="cuda:0", help="Simulation device (default: cuda:0).")
    args = parser.parse_args()
    clip = args.clip.expanduser().resolve()
    if not clip.exists() and clip.suffix != ".csv":
        clip = Path(f"{clip}.csv")
    if not clip.is_file():
        parser.error(f"CSV not found: {clip}")
    output = (args.output.expanduser().resolve() if args.output else
              Path("/workspace/humanoid_standup/vids") / f"{clip.stem}.mp4")
    if output.suffix.lower() != ".mp4":
        parser.error("Output must have an .mp4 extension.")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    command = ["/isaac-sim/python.sh", "/workspace/humanoid_standup/scripts/play_reference.py",
               "--csv", str(clip.resolve()), "--record", str(output.resolve()),
               "--video-fps", str(args.fps), "--device", args.device]
    if not args.gui:
        command.append("--headless")
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
