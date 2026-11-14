"""Build a static dashboard of git history with training runs and evaluations attached.

Usage:
    python scripts/dashboard/build.py

Open logs/dashboard/index.html in a browser. Training curves are parsed from the
TensorBoard event files once and cached in logs/dashboard/runs/.
"""

import json
import math
import re
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs" / "rsl_rl"
EVALUATIONS = ROOT / "evaluation"
OUT = ROOT / "logs" / "dashboard"
POINTS_PER_CURVE = 400

EVAL_METRICS = [
    ("standing_up.success_rate", "Stand-up success", "high"),
    ("rewards.mean_total_reward", "Mean reward", "high"),
    ("standing_foot_motion.fraction_of_standing_trials", "Feet moved", "low"),
    ("lifted_foot.fraction_of_standing_trials", "Foot lifted", "low"),
    ("jitteriness.policy_trial_rms.mean", "Jitter RMS", "low"),
    ("actuator_effort.metrics.peak_torque_nm.mean", "Peak torque", "low"),
    ("actuator_effort.metrics.rms_torque_nm.mean", "RMS torque", "low"),
    ("actuator_effort.metrics.time_at_torque_limit_s.mean", "Time at limit", "low"),
    ("actuator_effort.metrics.absolute_mechanical_work_j.mean", "Mech. work", "low"),
    ("joints.mean_position_tracking_rms", "Joint pos err", "low"),
    ("standing_push.success_rate", "Push survival", "high"),
    ("standing_push.foot_movement.fraction_of_standing_trials", "Feet moved (push)", "low"),
]


def git_graph():
    fields = ["hash", "short", "date", "refs", "subject"]
    output = subprocess.run(
        ["git", "log", "--branches", "--remotes", "--tags", "--graph", "--date-order", "--format=%x1f%H%x1f%h%x1f%aI%x1f%D%x1f%s"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        graph, *values = line.split("\x1f")
        rows.append({"graph": graph.rstrip(), **dict(zip(fields, values))})
    return rows


def git_info(run_dir):
    path = run_dir / "git" / "humanoid_standup.diff"
    if not path.exists():
        return {"commit": "", "branch": "", "diff": ""}
    text = path.read_text()
    branch = re.search(r"On branch (\S+)", text)
    return {
        "commit": re.search(r"--- git commit ---\n(\w*)", text)[1],
        "branch": branch[1] if branch else "",
        "diff": text.split("--- git diff ---\n", 1)[1].strip(),
    }


def diff_stat(diff):
    stat = {}
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            changes = stat.setdefault(line.split(" b/", 1)[1], [0, 0])
        elif line.startswith("+") and not line.startswith("+++"):
            changes[0] += 1
        elif line.startswith("-") and not line.startswith("---"):
            changes[1] += 1
    return stat


def rounded(value):
    return float(f"{value:.5g}") if math.isfinite(value) else None


def scalars(run_dir):
    run_id = f"{run_dir.parent.name}__{run_dir.name}"
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    key = [[event.name, event.stat().st_size] for event in events]
    cache = OUT / "runs" / f"{run_id}.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    if cached.get("key") == key:
        return cached["scalars"]
    print(f"  parsing {run_dir.name}", flush=True)
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    curves = {}
    for tag in accumulator.Tags()["scalars"]:
        events = accumulator.Scalars(tag)
        stride = max(1, math.ceil(len(events) / POINTS_PER_CURVE))
        sampled = events[::stride] + ([events[-1]] if (len(events) - 1) % stride else [])
        curves[tag] = [[event.step for event in sampled], [rounded(event.value) for event in sampled]]
    cache.write_text(json.dumps({"key": key, "scalars": curves}))
    return curves


def runs():
    result = []
    run_dirs = sorted(LOGS.glob("*/*/"))
    with Pool() as pool:
        all_curves = pool.map(scalars, run_dirs)
    for run_dir, curves in zip(run_dirs, all_curves):
        run_id = f"{run_dir.parent.name}__{run_dir.name}"
        git = git_info(run_dir)
        details = {"scalars": curves, "diff": git["diff"]}
        (OUT / "runs" / f"{run_id}.js").write_text(f"RUN_DETAILS[{json.dumps(run_id)}] = {json.dumps(details)};\n")
        checkpoints = [int(path.stem.split("_")[1]) for path in run_dir.glob("model_*.pt")]
        clips = run_dir / "params" / "training_clips.yaml"
        reward = curves.get("Train/mean_reward", [[], []])[1]
        result.append({
            "id": run_id,
            "experiment": run_dir.parent.name,
            "name": run_dir.name,
            "started": datetime.strptime(run_dir.name[:19], "%Y-%m-%d_%H-%M-%S").isoformat(),
            "label": run_dir.name[20:] or run_dir.name[:19],
            "iterations": max(checkpoints, default=0),
            "final_reward": reward[-1] if reward else None,
            "clips": re.findall(r"([^/\s]+)\.npz", clips.read_text()) if clips.exists() else [],
            "commit": git["commit"],
            "branch": git["branch"],
            "diff_stat": diff_stat(git["diff"]),
        })
    return result


def lookup(summary, path):
    for key in path.split("."):
        summary = summary.get(key) if isinstance(summary, dict) else None
    return summary


def evaluations():
    result = []
    for eval_dir in sorted(EVALUATIONS.glob("*_model_*")):
        match = re.fullmatch(r"(.+)_model_(\d+)_(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)", eval_dir.name)
        clips = []
        for clip_dir in sorted(eval_dir.iterdir()):
            summary_path = clip_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            tracking = lookup(summary["evaluations"], "joints.position_tracking_error_rad")
            if tracking:
                summary["evaluations"]["joints"]["mean_position_tracking_rms"] = (
                    sum(joint["rms"] for joint in tracking.values()) / len(tracking)
                )
            clips.append({
                "clip": clip_dir.name,
                "trials": summary["trials"],
                "early_termination": summary["early_termination"],
                "metrics": {path: lookup(summary["evaluations"], path) for path, _, _ in EVAL_METRICS},
                "media": sorted(
                    str(path.relative_to(clip_dir))
                    for pattern in ["*.svg", "*.mp4", "joints/index.html"]
                    for path in clip_dir.glob(pattern)
                ),
                "path": f"../../evaluation/{eval_dir.name}/{clip_dir.name}",
            })
        result.append({
            "run": match[1],
            "checkpoint": int(match[2]),
            "created": datetime.strptime(match[3], "%Y-%m-%d_%H-%M-%S").isoformat(),
            "name": eval_dir.name,
            "clips": clips,
        })
    return result


def main():
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    data = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "commits": git_graph(),
        "runs": runs(),
        "evaluations": evaluations(),
        "metrics": [{"path": path, "label": label, "better": better} for path, label, better in EVAL_METRICS],
    }
    template = (Path(__file__).parent / "template.html").read_text()
    (OUT / "index.html").write_text(template.replace("__DATA__", json.dumps(data)))
    print(f"wrote {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
