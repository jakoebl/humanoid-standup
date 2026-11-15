"""Video replay of ten randomly chosen attempts."""

import numpy as np


class RandomAttempts:
    name = "random_attempts"

    def __init__(self, num_envs, seed):
        self.env_ids = np.sort(np.random.default_rng(seed).permutation(num_envs)[:10])

    def prepare(self, raw):
        pass

    def observe(self, raw, frame, rewards):
        pass

    def video_attempts(self):
        return [{"rank": rank, "env_id": int(env_id), "video": f"vids/random_{rank:02d}_env_{env_id:04d}.mp4"}
                for rank, env_id in enumerate(self.env_ids, start=1)]

    def summary(self):
        return {"selection": "uniform without replacement, seeded by --seed",
                "video_attempts": self.video_attempts(), "combined_video": "random_attempts.mp4"}

    def save(self, output):
        pass
