"""Control-rate actuator-model effort statistics, one score per full-clip trial."""

import csv
from html import escape
from pathlib import Path

import numpy as np


METRICS = {
    "peak_torque_nm": ("Peak absolute joint torque", "N m"),
    "rms_torque_nm": ("RMS joint torque", "N m"),
    "time_at_torque_limit_s": ("Time with any joint near its torque limit", "s"),
    "absolute_mechanical_work_j": ("Absolute mechanical work", "J"),
}


def save_histogram(values, output, metric):
    label, unit = METRICS[metric]
    counts, edges = np.histogram(values, bins="auto")
    with (output / "raw" / f"actuator_effort_{metric}_bins.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bin_left", "bin_right", "count"])
        writer.writerows(zip(edges[:-1], edges[1:], counts))
    peak = max(int(counts.max()), 1)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 500" role="img">',
           f'<title>{escape(label)}</title>', '<rect width="900" height="500" fill="white"/>',
           '<g font-family="sans-serif" font-size="14">',
           f'<text x="450" y="30" text-anchor="middle">{escape(label)}</text>']
    for index, count in enumerate(counts):
        height = 330 * int(count) / peak
        width = 740 / len(counts)
        svg.append(f'<rect x="{80 + index * width}" y="{395 - height}" width="{width}" '
                   f'height="{height}" fill="#4477aa" stroke="white">'
                   f'<title>{edges[index]:.6g} to {edges[index+1]:.6g}: {int(count)} runs</title></rect>')
    for tick in sorted({round(peak * i / 5) for i in range(6)}):
        svg.append(f'<text x="65" y="{400 - 330 * tick / peak}" text-anchor="end">{tick}</text>')
    for fraction in np.linspace(0, 1, 6):
        value = edges[0] + fraction * (edges[-1] - edges[0])
        svg.append(f'<text x="{80 + 740*fraction}" y="420" text-anchor="middle">{value:.4g}</text>')
    svg.extend(['<path d="M 80 65 V 395 H 820" fill="none" stroke="black"/>',
                f'<text x="450" y="465" text-anchor="middle">{escape(label)} ({unit})</text>',
                '<text transform="translate(25 230) rotate(-90)" text-anchor="middle">Number of runs</text>',
                '</g></svg>'])
    (output / f"actuator_effort_{metric}_histogram.svg").write_text("\n".join(svg), encoding="utf-8")


class ActuatorEffort:
    name = "actuator_effort"

    def __init__(self, num_envs, step_dt):
        if num_envs < 1 or not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Actuator effort requires positive trial count and finite positive timestep.")
        self.num_envs = num_envs
        self.step_dt = step_dt
        self.limit_ratio = .99
        self.peak = np.zeros(num_envs)
        self.squared_sum = np.zeros(num_envs)
        self.limit_steps = np.zeros(num_envs, dtype=np.int64)
        self.work = np.zeros(num_envs)
        self.last_frame = -1
        self.samples = 0
        self.joint_names = []

    def prepare(self, raw):
        robot = raw.scene["robot"]
        indexes = set()
        all_indexes = list(range(robot.num_joints))
        for actuator in robot.actuators.values():
            ids = actuator.joint_indices
            indexes.update(all_indexes[ids] if isinstance(ids, slice) else [int(i) for i in ids])
        if not indexes:
            raise ValueError("No actuated joints found for effort evaluation.")
        self.joint_indexes = sorted(indexes)
        self.joint_names = [robot.joint_names[i] for i in self.joint_indexes]

    def observe(self, raw, frame, rewards=None):
        data = raw.scene["robot"].data
        ids = self.joint_indexes
        self.update(frame, data.applied_torque[:, ids].cpu().numpy(),
                    data.joint_vel[:, ids].cpu().numpy(), data.joint_effort_limits[:, ids].cpu().numpy())

    def update(self, frame, torque, velocity, limits):
        if frame != self.last_frame + 1:
            raise ValueError("Actuator effort requires consecutive frames starting at zero.")
        # Reset buffers can contain torque from the preceding rollout; frame zero has no elapsed interval.
        if frame == 0:
            self.last_frame = frame
            return
        torque, velocity, limits = (np.asarray(value, dtype=np.float64) for value in (torque, velocity, limits))
        if torque.ndim != 2 or torque.shape[0] != self.num_envs or torque.shape[1] == 0:
            raise ValueError("Torque must have shape (trials, actuated joints).")
        if velocity.shape != torque.shape or limits.shape != torque.shape:
            raise ValueError("Torque, velocity and limits must have matching shapes.")
        if not all(np.isfinite(value).all() for value in (torque, velocity, limits)) or (limits <= 0).any():
            raise ValueError("Effort signals must be finite and torque limits strictly positive.")
        if hasattr(self, "num_joints") and self.num_joints != torque.shape[1]:
            raise ValueError("Actuated joint count changed during evaluation.")
        self.num_joints = torque.shape[1]
        self.peak = np.maximum(self.peak, np.abs(torque).max(axis=1))
        self.squared_sum += np.square(torque).sum(axis=1)
        self.limit_steps += (np.abs(torque) >= self.limit_ratio * limits).any(axis=1)
        self.work += np.abs(torque * velocity).sum(axis=1) * self.step_dt
        self.samples += 1
        self.last_frame = frame

    @property
    def values(self):
        if not self.samples:
            raise ValueError("Actuator effort requires at least one post-reset sample.")
        return {"peak_torque_nm": self.peak,
                "rms_torque_nm": np.sqrt(self.squared_sum / (self.samples * self.num_joints)),
                "time_at_torque_limit_s": self.limit_steps * self.step_dt,
                "absolute_mechanical_work_j": self.work}

    def summary(self):
        return {"torque_source": "robot.data.applied_torque; clipped actuator-model estimate for implicit actuators",
                "sampling": "control-rate samples after each step; excludes reset frame; may miss substep peaks",
                "step_dt_s": self.step_dt, "samples_per_run": self.samples,
                "sampled_duration_s": self.samples * self.step_dt, "joints": self.joint_names,
                "torque_limit_ratio": self.limit_ratio,
                "definitions": {"peak_torque_nm": "max absolute torque over time and actuated joints",
                                "rms_torque_nm": "sqrt(mean squared torque over time and actuated joints)",
                                "time_at_torque_limit_s": "dt * samples with any joint at >= 99% of its limit; simultaneous joints count once",
                                "absolute_mechanical_work_j": "dt * sum over time and joints of abs(torque * joint_velocity); mechanical, not electrical energy"},
                "stddev_ddof": 0,
                "metrics": {name: {"count": len(values), "units": METRICS[name][1],
                                   "mean": float(values.mean()), "stddev": float(values.std()),
                                   "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
                            for name, values in self.values.items()}}

    def save(self, output):
        output = Path(output)
        (output / "raw").mkdir(parents=True, exist_ok=True)
        values = self.values
        with (output / "raw" / "actuator_effort.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", *values])
            writer.writerows((env_id, *(float(v[env_id]) for v in values.values()))
                             for env_id in range(self.num_envs))
        for metric, samples in values.items():
            save_histogram(samples, output, metric)
