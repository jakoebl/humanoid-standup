"""Cumulative reward ranking and video replay of the ten worst attempts."""

import csv
from pathlib import Path
import tempfile

import numpy as np
import torch


def worst_attempts(totals, count=10):
    """Rank total rewards ascending, breaking ties by environment index."""
    return [
        {"rank": rank, "env_id": int(env_id), "total_reward": float(totals[env_id]),
         "video": f"vids/worst_{rank:02d}_env_{env_id:04d}.mp4"}
        for rank, env_id in enumerate(np.argsort(totals, kind="stable")[:count], start=1)
    ]


def save_reward_histogram(totals, output):
    """Save automatic cumulative-reward bins and a standalone SVG count plot."""
    output = Path(output)
    counts, edges = np.histogram(totals, bins="auto")
    (Path(output) / "raw").mkdir(parents=True, exist_ok=True)
    with (Path(output) / "raw" / "reward_bins.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bin_left", "bin_right", "count"])
        writer.writerows(zip(edges[:-1], edges[1:], counts.tolist()))

    # Fixed viewBox keeps the plot scalable and usable without plotting dependencies.
    left, top, width, height = 85, 65, 760, 350
    peak = max(int(counts.max()), 1)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 510" role="img">',
        '<title>Cumulative reward distribution</title>',
        '<desc>Trial counts in automatic reward bins. Bins include the left edge; '
        'only the final bin includes its right edge.</desc>',
        '<rect width="900" height="510" fill="white"/>',
        '<g font-family="sans-serif" font-size="14" fill="#222">',
        '<text x="450" y="30" text-anchor="middle" font-size="20">Cumulative reward distribution</text>',
    ]
    for tick in sorted({round(peak * index / 5) for index in range(6)}):
        y = top + height * (1 - tick / peak)
        svg.append(f'<path d="M {left} {y} h {width}" stroke="#ddd"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 5}" text-anchor="end">{tick}</text>')
    bar_width = width / len(counts)
    for index, count in enumerate(counts):
        bar_height = height * int(count) / peak
        x = left + index * bar_width
        y = top + height - bar_height
        svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
                   f'fill="#4477aa" stroke="white" stroke-width="0.5">'
                   f'<title>[{edges[index]:.6g}, {edges[index + 1]:.6g}'
                   f'{"]" if index == len(counts) - 1 else ")"}: {int(count)} trials</title></rect>')
    for fraction in np.linspace(0, 1, 6):
        x = left + width * fraction
        value = edges[0] + (edges[-1] - edges[0]) * fraction
        svg.append(f'<text x="{x}" y="{top + height + 25}" text-anchor="middle">{value:.4g}</text>')
    svg.extend([
        f'<path d="M {left} {top} v {height} h {width}" fill="none" stroke="#222"/>',
        '<text x="465" y="475" text-anchor="middle">Cumulative reward per trial</text>',
        '<text transform="translate(25 240) rotate(-90)" text-anchor="middle">Number of trials</text>',
        '</g></svg>',
    ])
    (output / "reward_histogram.svg").write_text("\n".join(svg), encoding="utf-8")


@torch.inference_mode()
def render_attempts(raw, states, attempts, output, criterion="worst_rewards"):
    """Render cached physical states on environment zero without stepping the policy."""
    if not attempts:
        return
    import imageio.v2 as imageio

    robot = raw.scene["robot"]
    target_ids = torch.tensor([0], device=raw.device)
    origin = raw.scene.env_origins[0]
    raw.sim.set_camera_view(
        eye=(origin + torch.tensor([2.5, 2.5, 1.8], device=raw.device)).tolist(),
        target=(origin + torch.tensor([0., 0., 0.7], device=raw.device)).tolist(),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    combined_destination = output / f"{criterion}.mp4"
    combined_temporary = output / f"{criterion}.partial.mp4"
    try:
        with imageio.get_writer(combined_temporary, fps=1 / raw.step_dt, codec="libx264",
                                pixelformat="yuv420p", macro_block_size=2) as combined:
            for attempt in attempts:
                env_id = attempt["env_id"]
                destination = output / attempt["video"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".partial.mp4")
                try:
                    with imageio.get_writer(temporary, fps=1 / raw.step_dt, codec="libx264",
                                            pixelformat="yuv420p", macro_block_size=2) as writer:
                        for frame in range(len(states)):
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
                            combined.append_data(rgb)
                    temporary.replace(destination)
                    print(f"Saved {destination}", flush=True)
                finally:
                    temporary.unlink(missing_ok=True)
        combined_temporary.replace(combined_destination)
        print(f"Saved {combined_destination}", flush=True)
    finally:
        combined_temporary.unlink(missing_ok=True)



class TrajectoryCache:
    """One disk-backed full-rollout cache shared by video criteria."""

    def __init__(self, raw, length, output):
        self.raw = raw
        self.length = length
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self.cache = tempfile.TemporaryDirectory(prefix="trajectory-cache-", dir=self.output)
        robot = self.raw.scene["robot"]
        self.states = np.memmap(Path(self.cache.name) / "states.dat", dtype=np.float32, mode="w+",
                                shape=(self.length, self.raw.num_envs, 13 + 2 * robot.num_joints))
        return self

    def __exit__(self, *args):
        del self.states
        self.cache.cleanup()

    def prepare(self, raw):
        pass

    def observe(self, raw, frame, rewards):
        robot = raw.scene["robot"]
        self.states[frame] = torch.cat(
            (robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel), dim=-1
        ).cpu().numpy()
        if not np.isfinite(self.states[frame]).all():
            raise RuntimeError(f"Nonfinite state or reward at frame {frame}.")


class RewardEvaluation(TrajectoryCache):
    name = "rewards"

    def __init__(self, raw, length, output, record_video=False):
        super().__init__(raw, length, output)
        self.record_video = record_video
        self.totals = np.zeros(raw.num_envs, dtype=np.float64)

    def observe(self, raw, frame, rewards):
        super().observe(raw, frame, rewards)
        if rewards is not None:
            self.totals += rewards.cpu().numpy()
        if not np.isfinite(self.totals).all():
            raise RuntimeError(f"Nonfinite reward at frame {frame}.")

    def summary(self):
        return {"ranking": "ascending cumulative reward", "reward_steps": self.length - 1,
                "mean_total_reward": float(self.totals.mean()), "combined_video": "worst_rewards.mp4",
                "worst_attempts": worst_attempts(self.totals)}

    def save(self, output):
        (Path(output) / "raw").mkdir(parents=True, exist_ok=True)
        with (Path(output) / "raw" / "rewards.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "total_reward"])
            writer.writerows(enumerate(self.totals.tolist()))
        save_reward_histogram(self.totals, output)
        attempts = worst_attempts(self.totals)
        if self.record_video:
            print(f"Worst total reward: {attempts[0]['total_reward']:.6f}; rendering {len(attempts)} videos...",
                  flush=True)
            render_attempts(self.raw, self.states, attempts, Path(output))
