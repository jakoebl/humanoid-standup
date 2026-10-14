"""Track each foot's displacement after the first standing-height crossing."""

import csv
from pathlib import Path

import numpy as np


class StandingFootMotion:
    name = "standing_foot_motion"

    def __init__(self, num_envs, movement_threshold=0.02):
        self.movement_threshold = movement_threshold
        self.standing_height = 0.5
        self.first_standing_frame = np.full(num_envs, -1, dtype=np.int64)
        self.anchors = np.zeros((num_envs, 2, 3))
        self.max_displacement = np.zeros((num_envs, 2))

    def prepare(self, raw):
        robot = raw.scene["robot"]
        self.masses = robot.root_physx_view.get_masses().to(raw.device)
        foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.foot_ids, names = robot.find_bodies(foot_names, preserve_order=True)
        if list(names) != foot_names:
            raise ValueError("Could not resolve both feet for standing displacement measurements.")

    def observe(self, raw, frame, rewards=None):
        robot = raw.scene["robot"]
        com = (robot.data.body_com_pos_w * self.masses[..., None]).sum(dim=1) / self.masses.sum(dim=1, keepdim=True)
        heights = (com[:, 2] - raw.scene.env_origins[:, 2]).cpu().numpy()
        feet = robot.data.body_link_pos_w[:, self.foot_ids].cpu().numpy()
        self.update(frame, heights, feet)

    def save(self, output):
        moved = self.moved
        (Path(output) / "raw").mkdir(parents=True, exist_ok=True)
        with (Path(output) / "raw" / "foot_motion.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "first_standing_frame", "moved_feet_after_standing",
                             "max_left_foot_displacement_m", "max_right_foot_displacement_m"])
            for env_id in range(len(moved)):
                writer.writerow([env_id, int(self.first_standing_frame[env_id]), bool(moved[env_id]),
                                 *self.max_displacement[env_id].tolist()])
        self.save_graph(output)
        summary = self.summary()
        print(f"Foot movement after standing: {summary['moved_feet_after_standing']}/{len(moved)}; "
              f"stood: {summary['stood']}, never stood: {summary['never_stood']}", flush=True)

    def save_graph(self, output):
        """Plot each foot's maximum displacement for trials that reached standing height."""
        stood = self.first_standing_frame >= 0
        values = self.max_displacement[stood]
        crossed_count = int(self.moved[stood].sum())
        crossing_description = (f"{crossed_count}/{int(stood.sum())} standing trials exceeded "
                                f"{self.movement_threshold:g} m on either foot (each trial counted once).")
        # Shared bins and axes allow comparison of the two feet.
        upper = max(float(values.max()) if values.size else 0., self.movement_threshold, .01) * 1.05
        edges = np.histogram_bin_edges(values.ravel(), bins="auto", range=(0., upper))
        counts = [np.histogram(values[:, foot], bins=edges)[0] for foot in range(2)]
        peak = max(max(int(count.max()) for count in counts), 1)
        (Path(output) / "raw").mkdir(parents=True, exist_ok=True)
        with (Path(output) / "raw" / "foot_motion_bins.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["bin_left_m", "bin_right_m", "left_count", "right_count"])
            writer.writerows(zip(edges[:-1], edges[1:], counts[0].tolist(), counts[1].tolist()))
        svg = [
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 540" role="img">',
            '<title>Foot movement after standing</title>',
            '<desc>Maximum 3D ankle displacement from the first standing-height crossing. '
            'Only trials reaching standing height are included. Tracking includes later falls. '
            f'{crossing_description}</desc>',
            '<rect width="1000" height="540" fill="white"/>',
            '<g font-family="sans-serif" font-size="13" fill="#222">',
            '<text x="500" y="30" text-anchor="middle" font-size="20">Foot movement after standing</text>',
            f'<text x="500" y="55" text-anchor="middle">{int(stood.sum())} standing trials; '
            f'{int((~stood).sum())} never-standing trials excluded</text>',
        ]
        top, width, height = 110, 380, 300
        for foot, label in enumerate(("Left foot", "Right foot")):
            left = 90 + foot * 480
            svg.append(f'<text x="{left + width / 2}" y="90" text-anchor="middle">{label}</text>')
            for tick in sorted({round(peak * index / 5) for index in range(6)}):
                y = top + height * (1 - tick / peak)
                svg.append(f'<path d="M {left} {y} h {width}" stroke="#ddd"/>')
                svg.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end">{tick}</text>')
            for index, count in enumerate(counts[foot]):
                x = left + width * edges[index] / upper
                bar_width = width * (edges[index + 1] - edges[index]) / upper
                bar_height = height * int(count) / peak
                svg.append(f'<rect x="{x}" y="{top + height - bar_height}" width="{bar_width}" '
                           f'height="{bar_height}" fill="#4477aa" stroke="white">'
                           f'<title>{label}: {edges[index]:.6g} to {edges[index + 1]:.6g} m; '
                           f'{int(count)} trials</title></rect>')
            threshold_x = left + width * self.movement_threshold / upper
            svg.append(f'<path d="M {threshold_x} {top} v {height}" '
                       'stroke="#cc4422" stroke-width="2" stroke-dasharray="6 4"/>')
            svg.append(f'<path d="M {left} {top} v {height} h {width}" fill="none" stroke="#222"/>')
            for fraction in np.linspace(0, 1, 5):
                svg.append(f'<text x="{left + width * fraction}" y="435" text-anchor="middle">'
                           f'{upper * fraction:.3g}</text>')
            if not values.size:
                svg.append(f'<text x="{left + width / 2}" y="260" text-anchor="middle">No standing trials</text>')
        svg.extend([
            '<text x="500" y="470" text-anchor="middle">Maximum 3D displacement from first standing frame (m)</text>',
            '<text transform="translate(25 260) rotate(-90)" text-anchor="middle">Number of standing trials</text>',
            f'<text x="500" y="495" text-anchor="middle" fill="#cc4422">'
            f'Dashed line: movement threshold = {self.movement_threshold:g} m; '
            f'{crossed_count}/{int(stood.sum())} standing trials exceeded it on either foot</text>',
            '<text x="500" y="520" text-anchor="middle">Tracking continues after standing, including later falls.</text>',
            '</g></svg>',
        ])
        (Path(output) / "foot_motion_histogram.svg").write_text("\n".join(svg), encoding="utf-8")

    def update(self, frame, com_height, feet):
        if not np.isfinite(com_height).all() or not np.isfinite(feet).all():
            raise RuntimeError(f"Nonfinite standing measurements at frame {frame}.")
        newly_standing = (self.first_standing_frame < 0) & (com_height >= self.standing_height)
        self.first_standing_frame[newly_standing] = frame
        self.anchors[newly_standing] = feet[newly_standing]
        stood = self.first_standing_frame >= 0
        displacement = np.linalg.norm(feet[stood] - self.anchors[stood], axis=-1)
        self.max_displacement[stood] = np.maximum(self.max_displacement[stood], displacement)

    @property
    def moved(self):
        return (self.max_displacement > self.movement_threshold).any(axis=1)

    def video_attempts(self):
        displacement = self.max_displacement.max(axis=1)
        eligible = (self.first_standing_frame >= 0) & self.moved
        selected = np.argsort(-displacement, kind="stable")
        selected = selected[eligible[selected]][:10]
        return [{"rank": rank, "env_id": int(env_id), "max_foot_displacement_m": float(displacement[env_id]),
                 "video": f"vids/foot_motion_{rank:02d}_env_{env_id:04d}.mp4"}
                for rank, env_id in enumerate(selected, start=1)]

    def summary(self):
        stood = self.first_standing_frame >= 0
        stood_count = int(stood.sum())
        moved_count = int(self.moved.sum())
        return {
            "video_attempts": self.video_attempts(),
            "combined_video": "foot_motion.mp4" if self.video_attempts() else None,
            "standing_com_height_m": self.standing_height,
            "movement_threshold_m": self.movement_threshold,
            "measurement": "3D ankle-roll link displacement from first standing frame; either foot",
            "tracking": "continues through the rest of the clip, including later falls",
            "stood": stood_count,
            "never_stood": int((~stood).sum()),
            "moved_feet_after_standing": moved_count,
            "stood_without_foot_movement": stood_count - moved_count,
            "fraction_of_all_trials": moved_count / len(stood),
            "fraction_of_standing_trials": moved_count / stood_count if stood_count else None,
        }
