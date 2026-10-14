"""Measure upward foot displacement after the first standing-height crossing."""

import csv
from pathlib import Path

import numpy as np

from .foot_motion import StandingFootMotion


class LiftedFoot(StandingFootMotion):
    name = "lifted_foot"

    def __init__(self, num_envs, lift_threshold=0.02):
        super().__init__(num_envs)
        self.lift_threshold = lift_threshold
        self.max_lift = np.zeros((num_envs, 2))

    def update(self, frame, com_height, feet):
        super().update(frame, com_height, feet)
        stood = self.first_standing_frame >= 0
        lift = feet[stood, :, 2] - self.anchors[stood, :, 2]
        self.max_lift[stood] = np.maximum(self.max_lift[stood], lift)

    @property
    def lifted(self):
        return (self.max_lift > self.lift_threshold).any(axis=1)

    def summary(self):
        stood = self.first_standing_frame >= 0
        stood_count = int(stood.sum())
        lifted_count = int(self.lifted.sum())
        return {
            "standing_com_height_m": self.standing_height,
            "lift_threshold_m": self.lift_threshold,
            "measurement": "upward ankle-roll link displacement from first standing frame; either foot; strictly above lift threshold",
            "tracking": "starts at first COM height >= standing threshold and continues through the rest of the clip, including later falls",
            "stood": stood_count,
            "never_stood": int((~stood).sum()),
            "lifted_foot_after_standing": lifted_count,
            "stood_without_foot_lift": stood_count - lifted_count,
            "fraction_of_all_trials": lifted_count / len(stood),
            "fraction_of_standing_trials": lifted_count / stood_count if stood_count else None,
        }

    def save(self, output):
        lifted = self.lifted
        (Path(output) / "raw").mkdir(parents=True, exist_ok=True)
        with (Path(output) / "raw" / "lifted_foot.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "first_standing_frame", "lifted_foot_after_standing",
                             "max_left_foot_lift_m", "max_right_foot_lift_m"])
            for env_id in range(len(lifted)):
                writer.writerow([env_id, int(self.first_standing_frame[env_id]), bool(lifted[env_id]),
                                 *self.max_lift[env_id].tolist()])
        summary = self.summary()
        print(f"Foot lift after standing: {summary['lifted_foot_after_standing']}/{len(lifted)}; "
              f"stood: {summary['stood']}, never stood: {summary['never_stood']}", flush=True)
