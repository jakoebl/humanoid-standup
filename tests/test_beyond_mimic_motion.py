"""Named motion ordering and validation checks; no Isaac Sim startup required."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / (
    "source/humanoid_standup/humanoid_standup/tasks/manager_based/beyond_mimic/motion_data.py"
)
spec = importlib.util.spec_from_file_location("motion_data", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NamedMotionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "clip.npz"
        rotations = np.zeros((3, 2, 4))
        rotations[..., 0] = 1
        self.data = dict(
            backend=np.array("isaaclab"), fps=np.array([50]),
            joint_names=np.array(["b", "a"]), body_names=np.array(["right", "left"]),
            joint_pos=np.tile([20, 10], (3, 1)), joint_vel=np.tile([2, 1], (3, 1)),
            body_pos_w=np.tile([[2, 0, 0], [1, 0, 0]], (3, 1, 1)),
            body_quat_w=rotations, body_lin_vel_w=np.zeros((3, 2, 3)),
            body_ang_vel_w=np.zeros((3, 2, 3)),
        )

    def load(self, fps=50):
        np.savez(self.path, **self.data)
        return module.load_motion(self.path, ["a", "b"], ["left", "right"], fps)

    def test_reorders_joints_and_bodies_by_name(self):
        actual = self.load()
        np.testing.assert_equal(actual["joint_pos"][0], [10, 20])
        np.testing.assert_equal(actual["joint_vel"][0], [1, 2])
        np.testing.assert_equal(actual["body_pos_w"][0, :, 0], [1, 2])

    def test_rejects_unnamed_mjlab_motion(self):
        self.data.pop("body_names")
        with self.assertRaisesRegex(ValueError, "unnamed mjlab"):
            self.load()

    def test_rejects_wrong_fps(self):
        with self.assertRaisesRegex(ValueError, "control FPS"):
            self.load(fps=30)

    def test_rejects_nonfinite_and_nonunit_data(self):
        self.data["body_ang_vel_w"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "body_ang_vel_w"):
            self.load()
        self.data["body_ang_vel_w"][0, 0, 0] = 0
        self.data["body_quat_w"] *= 2
        with self.assertRaisesRegex(ValueError, "unit norm"):
            self.load()


if __name__ == "__main__":
    unittest.main()
