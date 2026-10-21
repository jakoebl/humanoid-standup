"""Independent per-environment clip selection with per-clip adaptive sampling."""

import math

import torch
from isaaclab.utils import configclass

from .commands import MotionCommand, MotionCommandCfg, MotionLoader
from ..motion_data import resolve_motion_files


class MotionLibrary(MotionLoader):
    """Pack validated clips into contiguous storage without joining their timelines."""

    def __init__(self, paths, body_indexes, joint_names, body_names, fps, device):
        clips = [MotionLoader(path, body_indexes, joint_names, body_names, fps, device) for path in paths]
        self.paths = tuple(str(path) for path in paths)
        self.fps = fps
        self._body_indexes = body_indexes
        self.lengths = torch.tensor([clip.time_step_total for clip in clips], device=device)
        self.offsets = self.lengths.cumsum(0) - self.lengths
        self.time_step_total = int(self.lengths.sum())
        for field in ("joint_pos", "joint_vel", "_body_pos_w", "_body_quat_w", "_body_lin_vel_w", "_body_ang_vel_w"):
            setattr(self, field, torch.cat([getattr(clip, field) for clip in clips]))




class MulticlipMotionCommand(MotionCommand):
    """Reuse tracking/reset mechanics while selecting clips uniformly at each reset."""

    def _load_motion(self, env):
        return MotionLibrary(
            resolve_motion_files(self.cfg.motion_directory, self.cfg.motion_files, self.cfg.motion_file),
            self.body_indexes, self.robot.joint_names, self.robot.body_names, 1 / env.step_dt, self.device,
        )

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if cfg.play_clip_index is not None and not 0 <= cfg.play_clip_index < len(self.motion.paths):
            raise ValueError("play_clip_index is outside the motion library.")
        # Histograms remain independent: difficult frames never change clip selection odds.
        self.clip_bin_counts = [int(length // self.motion.fps) + 1 for length in self.motion.lengths.tolist()]
        self.failed_bins = [torch.zeros(count, device=self.device) for count in self.clip_bin_counts]
        self.pending_failures = [torch.zeros_like(counts) for counts in self.failed_bins]
        self.metrics["clip_id"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def local_frames(self):
        return self.time_steps - self.motion.offsets[self.clip_ids]

    def _sample_frames(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self.cfg.sampling_mode == "adaptive":
            self._record_failures(env_ids)
        if self.cfg.play_clip_index is None:
            self.clip_ids[env_ids] = torch.randint(len(self.motion.paths), (len(env_ids),), device=self.device)
        else:
            self.clip_ids[env_ids] = self.cfg.play_clip_index
        for clip_id, count in enumerate(self.clip_bin_counts):
            selected = env_ids[self.clip_ids[env_ids] == clip_id]
            if not len(selected):
                continue
            length = int(self.motion.lengths[clip_id])
            if self.cfg.sampling_mode == "start":
                frames = torch.zeros(len(selected), dtype=torch.long, device=self.device)
            elif self.cfg.sampling_mode == "uniform":
                frames = torch.randint(length, (len(selected),), device=self.device)
            elif self.cfg.sampling_mode == "adaptive":
                probabilities = self.failed_bins[clip_id] + self.cfg.adaptive_uniform_ratio / count
                padded = torch.nn.functional.pad(probabilities[None, None], (0, len(self.kernel) - 1), mode="replicate")
                probabilities = torch.nn.functional.conv1d(padded, self.kernel[None, None]).flatten()
                probabilities /= probabilities.sum()
                bins = torch.multinomial(probabilities, len(selected), replacement=True)
                frames = ((bins + torch.rand(len(selected), device=self.device)) / count * (length - 1)).long()
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                normalized_entropy = entropy / math.log(count) if count > 1 else 0
                self.metrics["sampling_entropy"][selected] = normalized_entropy
                maximum, index = probabilities.max(0)
                self.metrics["sampling_top1_prob"][selected] = maximum
                self.metrics["sampling_top1_bin"][selected] = index.float() / count
            else:
                raise ValueError(f"Unknown sampling mode: {self.cfg.sampling_mode}")
            self.time_steps[selected] = self.motion.offsets[clip_id] + frames
        self.metrics["clip_id"][env_ids] = self.clip_ids[env_ids].float()

    def _record_failures(self, env_ids):
        failed = env_ids[self._env.termination_manager.terminated[env_ids]]
        for clip_id, count in enumerate(self.clip_bin_counts):
            selected = failed[self.clip_ids[failed] == clip_id]
            bins = (self.local_frames[selected] * count // self.motion.lengths[clip_id]).clamp(0, count - 1)
            self.pending_failures[clip_id] += torch.bincount(bins, minlength=count)

    def _update_command(self):
        self.time_steps += 1
        ends = self.motion.offsets[self.clip_ids] + self.motion.lengths[self.clip_ids]
        if self.cfg.hold_final_frame:
            # Evaluation can keep simulating after the reference ends without resetting the robot.
            self.time_steps = torch.minimum(self.time_steps, ends - 1)
        else:
            self._resample_command(torch.where(self.time_steps >= ends)[0])
        self.update_relative_body_poses()
        for counts, pending in zip(self.failed_bins, self.pending_failures):
            counts.lerp_(pending, self.cfg.adaptive_alpha)
            pending.zero_()


@configclass
class MulticlipMotionCommandCfg(MotionCommandCfg):
    class_type: type = MulticlipMotionCommand
    motion_directory: str = ""
    motion_files: list[str] = []
    play_clip_index: int | None = None
    hold_final_frame: bool = False
