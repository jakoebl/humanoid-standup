"""Check the actual command advancement method without simulator imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class FinalReferenceHoldTest(unittest.TestCase):
    def test_clamps_each_clip_without_resampling_and_preserves_default_behavior(self):
        path = Path(__file__).resolve().parents[1] / (
            "source/humanoid_standup/humanoid_standup/tasks/manager_based/"
            "beyond_mimic/mdp/library_commands.py"
        )
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MulticlipMotionCommand")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_update_command")
        namespace = {"torch": SimpleNamespace(minimum=np.minimum, where=np.where)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        for hold in (True, False):
            resampled, updated = [], []
            command = SimpleNamespace(
                cfg=SimpleNamespace(hold_final_frame=hold),
                time_steps=np.array([1, 6]), clip_ids=np.array([0, 1]),
                motion=SimpleNamespace(offsets=np.array([0, 3]), lengths=np.array([3, 4])),
                failed_bins=[], pending_failures=[],
                _resample_command=lambda ids: resampled.append(ids.copy()),
                update_relative_body_poses=lambda: updated.append(True),
            )
            for _ in range(5):
                namespace["_update_command"](command)
            if hold:
                np.testing.assert_array_equal(command.time_steps, [2, 6])
                self.assertEqual(resampled, [])
            else:
                self.assertEqual(len(resampled), 5)
                np.testing.assert_array_equal(resampled[0], [1])
            self.assertEqual(len(updated), 5)


if __name__ == "__main__":
    unittest.main()
