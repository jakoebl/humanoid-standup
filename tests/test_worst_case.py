"""Verify threshold monitoring without Isaac Sim or PyTorch."""
import ast
from functools import wraps
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class BreachMonitorTest(unittest.TestCase):
    def test_checks_continue_without_resets_and_keep_first_breach(self):
        path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/worst_case.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        namespace = {"wraps": wraps, "torch": SimpleNamespace(full_like=np.full_like, zeros_like=np.zeros_like)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        monitor = namespace["BreachMonitor"]()
        calls = []
        def threshold(env, threshold):
            calls.append(threshold)
            return env.errors > threshold
        cfg = SimpleNamespace(terminations=SimpleNamespace(
            time_out=SimpleNamespace(time_out=True),
            anchor=SimpleNamespace(time_out=False, func=threshold, params={"threshold": 0.25}),
            disabled=None,
        ))
        monitor.configure(cfg)
        self.assertEqual(inspect.signature(cfg.terminations.anchor.func), inspect.signature(threshold))
        self.assertIsNone(cfg.terminations.time_out)
        env = SimpleNamespace(episode_length_buf=np.array([0, 0, 0]), errors=np.array([0., .3, 0.]))
        for frame, errors in enumerate(([0., .3, 0.], [.5, 0., 0.], [.6, .5, 0.])):
            env.episode_length_buf[:] = frame
            env.errors[:] = errors
            done = cfg.terminations.anchor.func(env, **cfg.terminations.anchor.params)
            self.assertFalse(done.any())
        np.testing.assert_array_equal(monitor.first["anchor"], [1, 0, -1])
        self.assertEqual(calls, [.25, .25, .25])
        self.assertEqual(monitor.thresholds, {"anchor": .25})


if __name__ == "__main__":
    unittest.main()
