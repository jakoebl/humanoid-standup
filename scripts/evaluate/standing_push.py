"""Evaluate three seconds of recovery after one 0.5 m/s horizontal velocity push."""

import csv
import math
from pathlib import Path

import numpy as np

from .standing_up import StandingUp
from .foot_motion import StandingFootMotion
PUSH_VELOCITY = 2.0

def push_frame(length, step_dt):
    """Choose the first sample within the final second, allowing response steps."""
    if not math.isfinite(step_dt) or not 0 < step_dt <= 1:
        raise ValueError("Standing pushes require a step duration in (0, 1] seconds.")
    if (length - 1) * step_dt < 1.0 - 1e-9:
        raise ValueError("Standing pushes require a clip at least one second long.")
    return length - 1 - math.floor(1.0 / step_dt + 1e-9)


class StandingPush(StandingUp):
    name = "standing_push"

    def __init__(self, num_envs, step_dt, movement_threshold=0.02):
        super().__init__(num_envs, step_dt)
        self.foot_motion = StandingFootMotion(num_envs, movement_threshold)

    def rollout_frames(self, reference_length):
        return push_frame(reference_length, self.step_dt) + math.ceil(3.0 / self.step_dt - 1e-9) + 1

    def prepare(self, raw):
        super().prepare(raw)
        self.foot_motion.prepare(raw)

        motion = raw.command_manager.get_term("motion")
        length = int(motion.motion.lengths[motion.cfg.play_clip_index].item())
        self.push_frame = push_frame(length, self.step_dt)
        self.required_intervals = self.rollout_frames(length) - 1 - self.push_frame
        self.response_duration = self.required_intervals * self.step_dt
        self.pushed = False
        # Sample without replacement, independently of simulation RNG consumption.
        rng = np.random.default_rng(np.random.SeedSequence([raw.cfg.seed, motion.cfg.play_clip_index]))
        # Independent uniform headings: fixed magnitude, rather than uniform x/y components.
        angles = rng.uniform(0.0, 2.0 * math.pi, raw.num_envs)
        self.velocity_offsets = PUSH_VELOCITY * np.column_stack((np.cos(angles), np.sin(angles)))
        selected = rng.choice(raw.num_envs, size=min(10, raw.num_envs), replace=False)
        self.attempts = [
            {"rank": rank, "env_id": int(env_id),
             "video": f"vids/standing_push_{rank:02d}_env_{env_id:04d}.mp4"}
            for rank, env_id in enumerate(selected, start=1)
        ]

    def observe(self, raw, frame, rewards=None):
        super().observe(raw, frame, rewards)
        if frame >= self.push_frame:
            self.foot_motion.observe(raw, frame, rewards)
        if frame == self.push_frame and not self.pushed:
            self.apply_push(raw)
            self.pushed = True

    def apply_push(self, raw):
        import torch

        robot = raw.scene["robot"]
        velocities = robot.data.root_vel_w.clone()
        velocities[:, :2] += torch.tensor(self.velocity_offsets, device=raw.device, dtype=velocities.dtype)
        robot.write_root_velocity_to_sim(velocities)

    def summary(self):
        result = super().summary()
        result["required_duration_s"] = self.response_duration
        result["timing"] = "standing continuously from push sample through end of three-second response window"
        feet = self.foot_motion.summary()
        # Random pushed trials are the videos for this criterion.
        feet.pop("video_attempts")
        feet.pop("combined_video")
        feet["tracking"] = "from first standing sample at or after push through end of three-second response window, including falls"
        result.update({
            "standing_success": dict(result),
            "foot_movement": feet,
            "video_attempts": self.video_attempts(),
            "combined_video": "standing_push.mp4",
            "video_sampling": "up to 10 random trials without replacement per reference clip, seeded by evaluation seed and clip index",
            "required_duration_s": self.response_duration,
            "timing": "standing continuously from push sample through end of three-second response window",
            "pushes_per_trial": int(self.pushed),
            "push_frame": self.push_frame,
            "push_time_s": self.push_frame * self.step_dt,
            "standing_window": "push at start of final reference second; observe for three seconds; no measured-height gate",
            "horizontal_velocity_change_m_s": PUSH_VELOCITY,
            "direction": "independent uniform world-frame horizontal heading per trial",
            "vertical_velocity_change_m_s": 0.0,
            "angular_velocity_change_rad_s": 0.0,
        })
        return result

    def video_attempts(self):
        return self.attempts

    def save(self, output):
        self.foot_motion.save(output)
        folder = Path(output) / "raw"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "standing_push.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "succeeded", "final_standing_duration_s", "push_frame", "push_delta_vx_m_s", "push_delta_vy_m_s"])
            for env_id, (success, duration) in enumerate(zip(self.succeeded, self.trailing_duration)):
                writer.writerow([env_id, bool(success), float(duration), self.push_frame,
                                 *self.velocity_offsets[env_id].tolist()])
        print(f"Standing after push: {int(self.succeeded.sum())}/{len(self.trailing_samples)} succeeded", flush=True)
