#!/usr/bin/env python3
"""Record reference motion in Isaac Sim.

CLI options:
clip: reference CSV path; .csv suffix is optional.

Overrides:
--output: reference MP4 path (default: vids/<clip>.mp4).
--fps: reference video FPS: 30, 60, or 120 (default: 120).
--device: simulation device (default: cuda:0).

-h, --help: show usage and exit.
"""

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
    parser.add_argument("--device", default="cuda:0", help="Simulation device (default: cuda:0).")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    python = "/isaac-sim/python.sh" if Path("/isaac-sim/python.sh").exists() else sys.executable
    if args.clip is None:
        parser.error("Provide a clip.")
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
