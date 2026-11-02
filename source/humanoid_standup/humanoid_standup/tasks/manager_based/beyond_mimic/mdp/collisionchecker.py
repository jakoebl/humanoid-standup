"""Approximate reset floor checks using feet, hands, knees, and elbows."""
import math
import torch


class CollisionChecker:
    """Check eight link positions against the floor and keep the feet separated."""

    def __init__(self, robot, env_origins, tolerance=1e-4, min_foot_distance=0.1):
        # Store floor and foot-separation thresholds.
        self.robot = robot
        self.floor_heights = env_origins[:, 2]
        self.tolerance = tolerance
        self.min_foot_distance = min_foot_distance

        # Feet come first; wrists represent the hands for this inexpensive point check.
        names = ["left_ankle_roll_link", "right_ankle_roll_link",
                 "left_wrist_yaw_link", "right_wrist_yaw_link",
                 "left_knee_link", "right_knee_link", "left_elbow_link", "right_elbow_link"]
        self.point_ids, resolved = robot.find_bodies(names, preserve_order=True)


    def invalid_envs(self, env_ids):
        # Gather only the eight points for environments still being retried.
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.robot.device)
        positions = self.robot.data.body_link_pos_w[ids[:, None], self.point_ids]

        # Reject any checked point below the floor
        floor = self.floor_heights[ids] - self.tolerance
        below_floor = (positions[:, :, 2] < floor[:, None]).any(dim=1)

        # Reject feet collisions
        foot_distance = torch.linalg.vector_norm(positions[:, 0] - positions[:, 1], dim=-1)
        invalid = below_floor | (foot_distance < self.min_foot_distance)
        return ids[invalid].cpu().tolist()
