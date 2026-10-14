"""Check analytic effort metrics, joint selection and distribution exports without Isaac Sim."""

import csv
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

path = Path(__file__).resolve().parents[1] / 'scripts/evaluate/actuator_effort.py'
spec = importlib.util.spec_from_file_location('effort_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EffortTest(unittest.TestCase):
    def test_analytic_metrics_exclude_reset_and_sum_absolute_work(self):
        tracker = module.ActuatorEffort(2, .1)
        tracker.update(0, None, None, None)
        tracker.update(1, [[10, -2], [0, 0]], [[2, 3], [0, 0]], [[10, 10], [10, 10]])
        tracker.update(2, [[0, 4], [3, -3]], [[5, -1], [2, 2]], [[10, 4], [10, 10]])
        values = tracker.values
        np.testing.assert_allclose(values['peak_torque_nm'], [10, 3])
        np.testing.assert_allclose(values['rms_torque_nm'], [np.sqrt(30), np.sqrt(4.5)])
        np.testing.assert_allclose(values['time_at_torque_limit_s'], [.2, 0])
        np.testing.assert_allclose(values['absolute_mechanical_work_j'], [3, 1.2])
        self.assertAlmostEqual(tracker.summary()['metrics']['peak_torque_nm']['mean'], 6.5)
        self.assertAlmostEqual(tracker.summary()['metrics']['peak_torque_nm']['stddev'], 3.5)

    def test_saturation_threshold_and_simultaneous_joints_count_once(self):
        tracker = module.ActuatorEffort(1, .02)
        tracker.update(0, None, None, None)
        tracker.update(1, [[9.9, -9.9]], [[0, 0]], [[10, 10]])
        tracker.update(2, [[9.89, -9.89]], [[0, 0]], [[10, 10]])
        self.assertAlmostEqual(tracker.values['time_at_torque_limit_s'][0], .02)

    def test_prepare_and_observe_select_actuated_joints(self):
        class Tensor(np.ndarray):
            def cpu(self):
                return self
            def numpy(self):
                return np.asarray(self)
        def tensor(value):
            return np.array(value).view(Tensor)
        robot = SimpleNamespace(num_joints=4, joint_names=['a', 'b', 'c', 'passive'],
                                actuators={'a': SimpleNamespace(joint_indices=slice(0, 2)),
                                           'b': SimpleNamespace(joint_indices=[2])},
                                data=SimpleNamespace(applied_torque=tensor([[1, 2, 3, 999]]),
                                                     joint_vel=tensor([[1, 1, 1, 999]]),
                                                     joint_effort_limits=tensor([[10, 10, 10, 0]])))
        raw = SimpleNamespace(scene={'robot': robot})
        tracker = module.ActuatorEffort(1, .1)
        tracker.prepare(raw)
        tracker.observe(raw, 0)
        tracker.observe(raw, 1)
        self.assertEqual(tracker.joint_names, ['a', 'b', 'c'])
        self.assertEqual(tracker.values['peak_torque_nm'][0], 3)
        self.assertAlmostEqual(tracker.values['absolute_mechanical_work_j'][0], .6)

    def test_exports_constant_single_and_varied_distributions(self):
        for torques in ([[0]], [[2], [2]], [[1, 2], [3, 4], [5, 6]]):
            with self.subTest(torques=torques), tempfile.TemporaryDirectory() as directory:
                tracker = module.ActuatorEffort(len(torques), .1)
                tracker.update(0, None, None, None)
                tracker.update(1, torques, np.ones_like(torques), np.full_like(torques, 10))
                tracker.save(directory)
                output = Path(directory)
                self.assertEqual(len(list(output.glob('*.svg'))), 4)
                self.assertEqual(list(output.glob('*.csv')), [])
                self.assertEqual(list(output.rglob('*.mp4')), [])
                with (output / 'raw/actuator_effort.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), len(torques))
                for metric in module.METRICS:
                    with (output / 'raw' / f'actuator_effort_{metric}_bins.csv').open() as stream:
                        self.assertEqual(sum(int(row['count']) for row in csv.DictReader(stream)), len(torques))
                    ET.parse(output / f'actuator_effort_{metric}_histogram.svg')

    def test_rejects_invalid_inputs_and_incomplete_rollouts(self):
        tracker = module.ActuatorEffort(1, .1)
        with self.assertRaises(ValueError):
            tracker.summary()
        with self.assertRaises(ValueError):
            tracker.update(1, [[1]], [[1]], [[10]])
        tracker.update(0, None, None, None)
        for torque, velocity, limits in (([[np.nan]], [[1]], [[10]]), ([[1]], [[1]], [[0]]),
                                         ([[1]], [[1, 2]], [[10]])):
            with self.assertRaises(ValueError):
                tracker.update(1, torque, velocity, limits)


if __name__ == '__main__':
    unittest.main()
