"""Approximate reset floor checks using feet, hands, knees, and elbows."""

import math

import torch


class CollisionChecker:
    """Check eight link positions against the floor and keep the feet separated."""

    def __init__(self, robot, env_origins, tolerance=1e-4, min_foot_distance=0.1):
        # Store floor and foot-separation thresholds.
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Floor penetration tolerance must be finite and nonnegative.")
        if not math.isfinite(min_foot_distance) or min_foot_distance < 0:
            raise ValueError("Minimum foot distance must be finite and nonnegative.")
        self.robot = robot
        self.floor_heights = env_origins[:, 2]
        self.tolerance = tolerance
        self.min_foot_distance = min_foot_distance

        # Feet come first; wrists represent the hands for this inexpensive point check.
        names = ["left_ankle_roll_link", "right_ankle_roll_link",
                 "left_wrist_yaw_link", "right_wrist_yaw_link",
                 "left_knee_link", "right_knee_link", "left_elbow_link", "right_elbow_link"]
        self.point_ids, resolved = robot.find_bodies(names, preserve_order=True)
        if list(resolved) != names:
            raise ValueError("Could not resolve all eight reset-check points.")

    def invalid_envs(self, env_ids):
        # Gather only the eight points for environments still being retried.
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.robot.device)
        positions = self.robot.data.body_link_pos_w[ids[:, None], self.point_ids]
        if not torch.isfinite(positions).all():
            raise RuntimeError("Nonfinite body positions during reset checking.")

        # Reject any checked point below the floor, allowing numerical tolerance.
        floor = self.floor_heights[ids] - self.tolerance
        below_floor = (positions[:, :, 2] < floor[:, None]).any(dim=1)

        # Reject close feet using the same 3D distance threshold as before.
        distance = torch.linalg.vector_norm(positions[:, 0] - positions[:, 1], dim=-1)
        invalid = below_floor | (distance < self.min_foot_distance)
        return ids[invalid].cpu().tolist()
