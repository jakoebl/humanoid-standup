"""Joint jerk sampled at the control rate, with one RMS score per full-clip trial."""

import csv
from collections import deque
from pathlib import Path

import numpy as np


def describe(values):
    values = np.asarray(values)
    return {"count": int(values.size), "mean": float(values.mean()),
            "stddev": float(values.std(ddof=0))}


def save_distribution(values, output, stem, label, reference=None):
    """Write counts and a standalone SVG; keep the reference marker in view."""
    counts, edges = np.histogram(values, bins="auto")
    (Path(output) / "raw").mkdir(parents=True, exist_ok=True)
    with (Path(output) / "raw" / f"{stem}_bins.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bin_left_rad_s3", "bin_right_rad_s3", "count"])
        writer.writerows(zip(edges[:-1], edges[1:], counts))
    lo, hi = edges[0], edges[-1]
    if reference is not None:
        lo, hi = min(lo, reference), max(hi, reference)
    padding = (hi - lo) * .03
    lo, hi = lo - padding, hi + padding
    x = lambda value: 80 + 740 * (value - lo) / (hi - lo)
    peak = max(int(counts.max()), 1)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 510" role="img">',
           f'<title>{label}</title>', '<rect width="900" height="510" fill="white"/>',
           '<g font-family="sans-serif" font-size="14">',
           f'<text x="450" y="30" text-anchor="middle">{label}</text>']
    for left, right, count in zip(edges[:-1], edges[1:], counts):
        height = 330 * int(count) / peak
        svg.append(f'<rect x="{x(left)}" y="{395 - height}" width="{x(right) - x(left)}" '
                   f'height="{height}" fill="#4477aa" stroke="white">'
                   f'<title>{left:.6g} to {right:.6g}: {count} samples</title></rect>')
    if reference is not None:
        svg.extend([f'<path d="M {x(reference)} 60 V 395" stroke="#cc3311" stroke-dasharray="6 4"/>',
                    f'<text x="450" y="455" text-anchor="middle">Reference RMS: {reference:.6g} rad/s³</text>'])
    for value in np.linspace(lo, hi, 6):
        svg.append(f'<text x="{x(value)}" y="420" text-anchor="middle">{value:.4g}</text>')
    svg.extend(['<path d="M 80 60 V 395 H 820" fill="none" stroke="black"/>',
                '<text x="450" y="485" text-anchor="middle">RMS joint jerk (rad/s³)</text>',
                f'<text x="80" y="50">Count (maximum {peak})</text>', '</g></svg>'])
    (output / f"{stem}_histogram.svg").write_text("\n".join(svg), encoding="utf-8")


class Jitteriness:
    name = "jitteriness"

    def __init__(self, num_envs, step_dt):
        if num_envs < 1 or not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Jitteriness requires positive trial count and finite positive step duration.")
        self.num_envs = num_envs
        self.step_dt = step_dt
        self.history = deque(maxlen=4)
        self.sum_squared = np.zeros(num_envs + 1, dtype=np.float64)
        self.sample_count = 0
        self.last_frame = -1

    def prepare(self, raw):
        self.motion = raw.command_manager.get_term("motion")
        index = self.motion.cfg.play_clip_index
        self.reference_start = int(self.motion.motion.offsets[index].item())
        if int(self.motion.motion.lengths[index].item()) < 4:
            raise ValueError("Joint jerk evaluation requires at least four clip frames.")
        self.joint_names = [raw.scene["robot"].joint_names[i] for i in self.motion.joint_indexes]

    def observe(self, raw, frame, rewards=None):
        # Read the unperturbed clip directly, excluding randomized reset/encoder offsets.
        reference = self.motion.motion.joint_pos[self.reference_start + frame][self.motion.joint_indexes]
        policy = self.motion.robot_joint_pos
        self.update(frame, policy.cpu().numpy(), reference.cpu().numpy())

    def update(self, frame, policy, reference):
        if frame != self.last_frame + 1:
            raise ValueError("Jitteriness requires consecutive frames starting at zero.")
        policy, reference = np.asarray(policy, dtype=np.float64), np.asarray(reference, dtype=np.float64)
        if policy.ndim != 2 or policy.shape[0] != self.num_envs or policy.shape[1] == 0:
            raise ValueError("Policy positions must have shape (trials, joints).")
        if reference.shape != policy.shape[1:]:
            raise ValueError("Reference and policy must use the same joints.")
        positions = np.concatenate((policy, reference[None]), axis=0)
        if not np.isfinite(positions).all():
            raise ValueError("Joint positions must be finite for jitteriness evaluation.")
        self.history.append(positions.copy())
        self.last_frame = frame
        if len(self.history) == 4:
            a, b, c, d = self.history
            jerk = (d - 3 * c + 3 * b - a) / self.step_dt ** 3
            squared = np.mean(jerk ** 2, axis=1)
            self.sum_squared += squared
            self.sample_count += 1

    @property
    def scores(self):
        if not self.sample_count:
            raise ValueError("Joint jerk evaluation requires at least four observed frames.")
        return np.sqrt(self.sum_squared / self.sample_count)

    def video_attempts(self):
        scores = self.scores[:-1]
        selected = np.argsort(-scores, kind="stable")[:10]
        return [{"rank": rank, "env_id": int(env_id), "rms_joint_jerk_rad_s3": float(scores[env_id]),
                 "video": f"vids/jitteriness_{rank:02d}_env_{env_id:04d}.mp4"}
                for rank, env_id in enumerate(selected, start=1)]

    def summary(self):
        scores = self.scores
        return {"measurement": "RMS third finite difference of joint positions / step_dt^3, over time and joints",
                "units": "rad/s^3", "lower_is_smoother": True, "stddev_ddof": 0,
                "step_dt_s": self.step_dt, "jerk_samples_per_trial": self.sample_count,
                "joints": self.joint_names,
                "policy_trial_rms": describe(scores[:-1]),
                "reference_clip_rms": float(scores[-1]),
                "video_attempts": self.video_attempts(), "combined_video": "jitteriness.mp4"}

    def save(self, output):
        output = Path(output)
        scores = self.scores
        (Path(output) / "raw").mkdir(parents=True, exist_ok=True)
        with (Path(output) / "raw" / "jitteriness.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "rms_joint_jerk_rad_s3"])
            writer.writerows(enumerate(scores[:-1]))
        save_distribution(scores[:-1], output, "jitteriness", "Policy trial jitteriness", scores[-1])
