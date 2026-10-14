"""Verify analytic joint jerk and exported distributions without Isaac Sim."""

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts/evaluate/jitteriness.py"
spec = importlib.util.spec_from_file_location("jitteriness_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class JitterinessTest(unittest.TestCase):
    def test_cubic_motion_known_jerk_and_constant_acceleration(self):
        tracker = module.Jitteriness(3, .1)
        tracker.joint_names = ["a", "b"]
        for frame in range(10):
            t = frame * .1
            # Third derivative of t^3 is 6; constant acceleration has zero jerk.
            tracker.update(frame, [[t**3, t**3], [2*t**3, 2*t**3], [t**2, t**2]], [t**3, 0])
        np.testing.assert_allclose(tracker.scores, [6, 12, 0, 6 / np.sqrt(2)], atol=1e-10)
        summary = tracker.summary()
        self.assertAlmostEqual(summary["policy_trial_rms"]["mean"], 6)
        self.assertAlmostEqual(summary["policy_trial_rms"]["stddev"], np.std([6, 12, 0]))
        self.assertAlmostEqual(summary["reference_clip_rms"], 6 / np.sqrt(2))
        self.assertNotIn("reference_frame_rms", summary)
        self.assertEqual(summary["jerk_samples_per_trial"], 7)

    def test_observe_uses_selected_clip_and_matching_joint_order(self):
        class Tensor(np.ndarray):
            def cpu(self):
                return self

            def numpy(self):
                return np.asarray(self)

        clip = np.array([[999., 999.]] * 4 + [[frame**3, 0.] for frame in range(4)]).view(Tensor)
        motion = SimpleNamespace(cfg=SimpleNamespace(play_clip_index=1), joint_indexes=[1, 0],
                                 motion=SimpleNamespace(offsets=np.array([0, 4]), lengths=np.array([4, 4]),
                                                        joint_pos=clip))
        raw = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: motion),
                              scene={"robot": SimpleNamespace(joint_names=["a", "b"])})
        tracker = module.Jitteriness(1, 1)
        tracker.prepare(raw)
        for frame in range(4):
            motion.robot_joint_pos = np.array([[0., frame**3]]).view(Tensor)
            tracker.observe(raw, frame)
        np.testing.assert_allclose(tracker.scores, [6 / np.sqrt(2)] * 2)
        self.assertEqual(tracker.summary()["joints"], ["b", "a"])

    def test_exports_counts_and_reference_for_constant_and_varied_trials(self):
        for amplitudes in ([0], [1, 1, 1], [0, 1, 2]):
            with self.subTest(amplitudes=amplitudes), tempfile.TemporaryDirectory() as directory:
                tracker = module.Jitteriness(len(amplitudes), 1)
                for frame in range(5):
                    tracker.update(frame, np.asarray(amplitudes)[:, None] * frame**3, [3 * frame**3])
                tracker.save(directory)
                output = Path(directory)
                for stem, count in (("jitteriness", len(amplitudes)),):
                    with (output / "raw" / f"{stem}_bins.csv").open() as stream:
                        bins = list(csv.DictReader(stream))
                    self.assertEqual(sum(int(row["count"]) for row in bins), count)
                    ET.parse(output / f"{stem}_histogram.svg")
                with (output / "raw" / "jitteriness.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), len(amplitudes))
                self.assertFalse((output / "raw" / "jitteriness_reference.csv").exists())

    def test_video_selection_uses_top_ten_policy_scores_with_stable_ties(self):
        tracker = module.Jitteriness(12, 1)
        amplitudes = np.array([0, 11, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        for frame in range(4):
            tracker.update(frame, amplitudes[:, None] * frame**3, [999 * frame**3])
        attempts = tracker.video_attempts()
        self.assertEqual([item["env_id"] for item in attempts], [1, 2, 11, 10, 9, 8, 7, 6, 5, 4])
        self.assertTrue(all(item["video"].startswith("vids/jitteriness_") for item in attempts))

    def test_rejects_missing_frames_short_clips_and_nonfinite_data(self):
        tracker = module.Jitteriness(1, .02)
        with self.assertRaisesRegex(ValueError, "four observed"):
            tracker.summary()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            tracker.update(1, [[0]], [0])
        with self.assertRaisesRegex(ValueError, "finite"):
            tracker.update(0, [[np.nan]], [0])
        with self.assertRaises(ValueError):
            module.Jitteriness(1, 0)


if __name__ == "__main__":
    unittest.main()
