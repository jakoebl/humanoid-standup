"""Evaluate sustained COM height at the end of a clip."""

import csv
import math
from pathlib import Path

import numpy as np


class StandingUp:
    name = "standing_up"

    def __init__(self, num_envs, step_dt, height_threshold=0.5, duration=1.0):
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Evaluation step duration must be finite and positive.")
        self.step_dt = step_dt
        self.height_threshold = height_threshold
        self.duration = duration
        # N consecutive samples span (N - 1) simulation intervals.
        self.required_intervals = math.ceil(duration / step_dt - 1e-9)
        self.trailing_samples = np.zeros(num_envs, dtype=np.int64)
        self.last_frame = -1

    def prepare(self, raw):
        self.masses = raw.scene["robot"].root_physx_view.get_masses().to(raw.device)

    def observe(self, raw, frame, rewards=None):
        robot = raw.scene["robot"]
        com = (robot.data.body_com_pos_w * self.masses[..., None]).sum(dim=1) / self.masses.sum(dim=1, keepdim=True)
        self.update(frame, (com[:, 2] - raw.scene.env_origins[:, 2]).cpu().numpy())

    def update(self, frame, com_height):
        if frame != self.last_frame + 1:
            raise ValueError("Standing evaluation requires consecutive frames starting at zero.")
        above = np.isfinite(com_height) & (np.asarray(com_height) > self.height_threshold)
        self.trailing_samples = np.where(above, self.trailing_samples + 1, 0)
        self.last_frame = frame

    @property
    def succeeded(self):
        return self.trailing_samples >= self.required_intervals + 1

    @property
    def trailing_duration(self):
        return np.maximum(self.trailing_samples - 1, 0) * self.step_dt

    def summary(self):
        count = int(self.succeeded.sum())
        return {
            "com_height_threshold_m": self.height_threshold,
            "required_duration_s": self.duration,
            "measurement": "mass-weighted COM height relative to environment origin; strictly above threshold",
            "timing": "continuous sampled hold ending at the final clip frame",
            "succeeded": count,
            "failed": len(self.trailing_samples) - count,
            "success_rate": count / len(self.trailing_samples),
        }

    def save(self, output):
        (Path(output) / "raw").mkdir(parents=True, exist_ok=True)
        with (Path(output) / "raw" / "standing_up.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "succeeded", "final_standing_duration_s"])
            for env_id, (success, duration) in enumerate(zip(self.succeeded, self.trailing_duration)):
                writer.writerow([env_id, bool(success), float(duration)])
        print(f"Standing up: {int(self.succeeded.sum())}/{len(self.trailing_samples)} succeeded", flush=True)
