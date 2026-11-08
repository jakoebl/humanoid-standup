"""Check reading the training action clip from a run's saved config."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


path = Path(__file__).resolve().parents[1] / "scripts/evaluate/checkpoints.py"
spec = importlib.util.spec_from_file_location("evaluation_checkpoints", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

CLIPPED = """scene:
  robot:
    joint_ids: !!python/object/apply:builtins.slice
    - null
    - null
    - null
actions:
  joint_pos:
    asset_name: robot
    clip:
      left_hip_pitch_joint: !!python/tuple
      - -2.5307
      - 2.8798
      left_knee_joint: !!python/tuple
      - -0.087267
      - 2.8798
    scale: 0.25
"""


class TrainingActionClipTest(unittest.TestCase):
    def checkpoint(self, directory, env_yaml):
        run = Path(directory)
        (run / "params").mkdir()
        (run / "params/env.yaml").write_text(env_yaml)
        return run / "model_100.pt"

    def test_returns_training_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = module.training_action_clip(self.checkpoint(directory, CLIPPED))
        self.assertEqual(clip, {"left_hip_pitch_joint": (-2.5307, 2.8798),
                                "left_knee_joint": (-0.087267, 2.8798)})

    def test_returns_none_for_unclipped_training(self):
        with tempfile.TemporaryDirectory() as directory:
            unclipped = CLIPPED.split("    clip:")[0] + "    clip: null\n"
            self.assertIsNone(module.training_action_clip(self.checkpoint(directory, unclipped)))

    def test_rejects_missing_config_and_bad_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                module.training_action_clip(Path(directory) / "model_100.pt")
            reversed_bounds = CLIPPED.replace("- -2.5307\n      - 2.8798", "- 2.8798\n      - -2.5307")
            with self.assertRaises(ValueError):
                module.training_action_clip(self.checkpoint(directory, reversed_bounds))


if __name__ == "__main__":
    unittest.main()
