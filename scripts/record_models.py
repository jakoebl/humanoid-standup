#!/usr/bin/env python3
"""Record saved policies at iterations 0, 1000, 2000, ... using record.py."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]


def find_checkpoints(logs):
    """Find eligible checkpoints, sorted by run and numeric iteration."""
    checkpoints = []
    for path in logs.rglob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if path.is_file() and match:
            iteration = int(match[1])
            if iteration % 1000 == 0:
                checkpoints.append((path, iteration))
    return sorted(checkpoints, key=lambda item: (str(item[0].parent), item[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="?", type=Path, default=PROJECT / "logs",
                        help="Run directory or logs tree to scan recursively (default: project logs).")
    parser.add_argument("--task", default="Humanoid-Standup")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting Isaac Sim.")
    args = parser.parse_args()
    logs = args.logs.expanduser().resolve()
    if not logs.is_dir():
        parser.error(f"Logs directory not found: {logs}")
    checkpoints = find_checkpoints(logs)
    if not checkpoints:
        parser.error(f"No checkpoints at multiples of 1000 found in {logs}")

    # Output paths use the run basename: reject ambiguous names across experiments.
    runs = {}
    for checkpoint, _ in checkpoints:
        run = checkpoint.parent
        if run.name in runs and runs[run.name] != run:
            parser.error(f"Duplicate run name {run.name}; select a single experiment or run directory.")
        runs[run.name] = run

    child_env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        child_env.pop(key, None)
    child_env.update(PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")
    failures = 0
    for checkpoint, iteration in checkpoints:
        command = [sys.executable, str(PROJECT / "scripts/record.py"),
                   "--checkpoint", str(checkpoint), "--iteration", str(iteration),
                   "--training-name", checkpoint.parent.name, "--task", args.task,
                   "--agent", args.agent, "--device", args.device]
        if args.gui:
            command.append("--gui")
        if args.dry_run:
            print(shlex.join(command))
            continue
        recording_log = checkpoint.parent / "recordings" / f"{iteration}.log"
        print(f"[INFO] Recording {checkpoint}; log: {recording_log}", flush=True)
        try:
            recording_log.parent.mkdir(parents=True, exist_ok=True)
            with recording_log.open("w") as output:
                result = subprocess.run(command, env=child_env, stdout=output,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                failures += 1
                print(f"[ERROR] Recording failed (exit {result.returncode}): {checkpoint}\n"
                      f"Full output: {recording_log}\n"
                      f"{recording_log.read_text(errors='replace')[-8000:]}", file=sys.stderr)
        except OSError as error:
            failures += 1
            print(f"[ERROR] Could not record {checkpoint}: {error}", file=sys.stderr)
    if not args.dry_run:
        print(f"Finished: {len(checkpoints) - failures} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
