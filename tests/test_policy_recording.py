"""Check headless frame freshness without importing Isaac Sim."""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class PolicyRecordingTest(unittest.TestCase):
    def test_headless_frames_follow_physics(self):
        self.check_recordings(repetitions=1)

    def test_ten_rollouts_restart_at_frame_zero(self):
        self.check_recordings(repetitions=10)

    def check_recordings(self, repetitions):
        source = Path(__file__).resolve().parents[1] / 'scripts/rsl_rl/play.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'record_policy_rollouts')
        frames = []
        recordings = []
        policy_observations = []
        resets = []

        class Writer:
            def __enter__(self):
                frames.clear()
                return self

            def append_data(self, rgb):
                frames.append(rgb.pose)

            def __exit__(self, *args):
                recordings.append(list(frames))

        class HeadlessEnv:
            num_envs = 1
            device = 'cpu'
            step_dt = 1.0
            reference_start_frames = [SimpleNamespace(item=lambda: 0)]

            def __init__(self):
                self.unwrapped = self
                self.pose = self.visible_pose = 0
                self.sim = SimpleNamespace(render=self.refresh)
                cfg = SimpleNamespace(params={})
                self.event_manager = SimpleNamespace(get_term_cfg=lambda name: cfg,
                                                     set_term_cfg=lambda *args: None)

            def refresh(self):
                self.visible_pose = self.pose

            def reset(self):
                resets.append(dict(self.event_manager.get_term_cfg('reset_robot').params))
                self.pose = 0
                return 0, {}

            def step(self, action):
                # Physics advances but no GUI or RTX sensor triggers a render.
                self.pose += 1
                return self.pose, 0, False, {}

            def render(self, recompute=False):
                # Match the installed Isaac Lab implementation: recompute=True reads cache.
                if not recompute:
                    self.refresh()
                return SimpleNamespace(size=3, pose=self.visible_pose)

        imageio = ModuleType('imageio')
        imageio.__path__ = []
        imageio_v2 = ModuleType('imageio.v2')
        imageio_v2.get_writer = lambda *args, **kwargs: Writer()
        imageio.v2 = imageio_v2
        policy = SimpleNamespace(reset=lambda dones: None)
        namespace = dict(
            os=os, Path=Path,
            torch=SimpleNamespace(inference_mode=nullcontext, bool=bool,
                                  ones=lambda *args, **kwargs: [True]),
            version=SimpleNamespace(parse=lambda value: tuple(map(int, value.split('.')))),
            installed_version='3.0.0',
            REFERENCE=SimpleNamespace(paths=[Path('clip.csv')], dt=1.0,
                                      clip_lengths=[SimpleNamespace(item=lambda: 4)]),
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, {'imageio': imageio, 'imageio.v2': imageio_v2}):
            namespace['record_policy_rollouts'](
                HeadlessEnv(), SimpleNamespace(alg=SimpleNamespace(policy=policy)),
                lambda obs: policy_observations.append(obs) or obs, Path(directory),
                action_repeat=2, repetitions=repetitions)
            self.assertEqual({p.name for p in Path(directory).rglob('*.mp4')},
                             {f'{i}.mp4' for i in range(repetitions)})
        self.assertEqual(recordings, [[0, 1, 2, 3]] * repetitions)
        self.assertEqual(policy_observations, [0, 2] * repetitions)
        self.assertEqual(resets, [{'clip_id': 0, 'start_fraction': 0.0}] * repetitions)


if __name__ == '__main__':
    unittest.main()
