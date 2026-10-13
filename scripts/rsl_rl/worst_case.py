"""Full-clip tracking evaluation and exact state replay of threshold breaches."""
import json
from functools import wraps
import tempfile
from pathlib import Path

import numpy as np
import torch


class BreachMonitor:
    """Evaluate termination functions at their normal step timing, suppressing resets."""

    def __init__(self):
        self.first = {}
        self.thresholds = {}

    def configure(self, cfg):
        for name, term in vars(cfg.terminations).items():
            if name.startswith("_") or term is None:
                continue
            if term.time_out:
                setattr(cfg.terminations, name, None)
                continue
            self.thresholds[name] = term.params.get("threshold")
            term.func = self.wrap(name, term.func)

    def wrap(self, name, function):
        @wraps(function)
        def monitor(env, **params):
            breached = function(env, **params)
            if name not in self.first:
                self.first[name] = torch.full_like(env.episode_length_buf, -1)
            first = self.first[name]
            mask = breached & (first < 0)
            first[mask] = env.episode_length_buf[mask]
            return torch.zeros_like(breached)
        return monitor


def evaluate(env, policy, policy_nn, monitor, output_dir, checkpoint):
    """Cache physical states once, then render only the failed trials without resimulation."""
    import imageio.v2 as imageio

    raw = env.unwrapped
    motion = raw.command_manager.get_term("motion")
    clip_id = getattr(motion.cfg, "play_clip_index", None)
    if hasattr(motion.motion, "lengths"):
        if clip_id is None:
            raise ValueError("Worst-case evaluation requires a selected play_clip_index.")
        length = int(motion.motion.lengths[clip_id].item())
        clip = motion.motion.paths[clip_id]
    else:
        length = motion.motion.time_step_total
        clip = motion.cfg.motion_file
    robot = raw.scene["robot"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists() or any(output_dir.glob("env_*.mp4")):
        raise ValueError("Use a fresh output directory to avoid mixing evaluations.")
    width = 13 + 2 * robot.num_joints
    with tempfile.TemporaryDirectory(prefix="worst-case-", dir=output_dir) as cache:
        states = np.memmap(Path(cache) / "states.dat", dtype=np.float32, mode="w+",
                           shape=(length, raw.num_envs, width))
        with torch.inference_mode():
            monitor.first.clear()
            obs, _ = env.reset()
            frames = getattr(motion, "local_frames", motion.time_steps)
            if not torch.all(frames == 0):
                raise RuntimeError("Evaluation must start at frame zero.")
            raw.termination_manager.compute()  # Include threshold breaches in the initial state.
            policy_nn.reset(torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device))
            for frame in range(length):
                if frame:
                    obs, _, dones, _ = env.step(policy(obs))
                    if torch.any(dones):
                        raise RuntimeError("Unexpected reset during worst-case evaluation.")
                states[frame] = torch.cat((robot.data.root_state_w, robot.data.joint_pos,
                                          robot.data.joint_vel), dim=-1).cpu().numpy()
            failed = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
            for first in monitor.first.values():
                failed |= first >= 0
            ids = failed.nonzero().flatten().tolist()
        trials = []
        for env_id in ids:
            breaches = {name: int(first[env_id].item()) for name, first in monitor.first.items()
                        if first[env_id] >= 0}
            trials.append({"env_id": env_id, "first_breach_frame": min(breaches.values()),
                           "breaches": breaches, "video": f"env_{env_id:04d}.mp4"})
        trials.sort(key=lambda trial: (trial["first_breach_frame"], trial["env_id"]))
        summary = {"checkpoint": str(checkpoint), "clip": str(clip), "seed": raw.cfg.seed,
                   "trials": raw.num_envs, "frames": length, "fps": 1 / raw.step_dt,
                   "failed": len(ids), "failure_rate": len(ids) / raw.num_envs,
                   "thresholds": monitor.thresholds, "early_termination": False, "failures": trials}
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Threshold breaches: {len(ids)}/{raw.num_envs}; report: {output_dir / 'summary.json'}", flush=True)
        # Replay each captured trial on the robot at env 0, translating its world origin.
        target_ids = torch.tensor([0], device=raw.device)
        origin = raw.scene.env_origins[0]
        raw.sim.set_camera_view(eye=(origin + torch.tensor([2.5, 2.5, 1.8], device=raw.device)).tolist(),
                                target=(origin + torch.tensor([0., 0., 0.7], device=raw.device)).tolist())
        for trial in trials:
            env_id = trial["env_id"]
            output = output_dir / trial["video"]
            temporary = output.with_suffix(".partial.mp4")
            try:
                with imageio.get_writer(temporary, fps=1 / raw.step_dt, codec="libx264",
                                        pixelformat="yuv420p", macro_block_size=2) as writer:
                    for frame in range(length):
                        state = torch.tensor(states[frame, env_id:env_id + 1], device=raw.device)
                        state[:, :3] += origin - raw.scene.env_origins[env_id]
                        robot.write_root_state_to_sim(state[:, :13], env_ids=target_ids)
                        robot.write_joint_state_to_sim(state[:, 13:13 + robot.num_joints],
                                                      state[:, 13 + robot.num_joints:], env_ids=target_ids)
                        raw.scene.write_data_to_sim()
                        raw.sim.forward()
                        raw.scene.update(raw.physics_dt)
                        if frame == 0:
                            raw.render()
                            for _ in range(10):
                                raw.sim.render()
                        raw.sim.render()
                        rgb = raw.render(recompute=True)
                        if rgb is None or not rgb.size:
                            raise RuntimeError("Isaac Sim returned an empty RGB frame.")
                        writer.append_data(rgb)
                temporary.replace(output)
                print(f"Saved {output}", flush=True)
            finally:
                temporary.unlink(missing_ok=True)
