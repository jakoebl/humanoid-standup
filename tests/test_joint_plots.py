"""Check joint trajectory plots without Isaac Sim."""

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


path = Path(__file__).resolve().parents[1] / "scripts/evaluate/joints.py"
spec = importlib.util.spec_from_file_location("joint_plots_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class JointPlotsTest(unittest.TestCase):
    def test_records_one_run_and_exports_limits(self):
        tracker = module.JointPlots(.1)
        tracker.joint_names = ["hip", "knee"]
        tracker.position_limits = np.array([[-1., 2.], [-3., 3.]])
        tracker.torque_limits = np.array([10., 20.])
        for frame in range(3):
            tracker.update(frame, [frame, -frame], [frame * 2, -frame])

        with tempfile.TemporaryDirectory() as directory:
            tracker.save(directory)
            output = Path(directory) / "joints"
            self.assertEqual(len(list(output.glob("*.svg"))), 4)
            with (output / "joints.csv").open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0], ["time_s", "hip_position_rad", "knee_position_rad",
                                       "hip_torque_nm", "knee_torque_nm", "hip_unclipped_action_rad",
                                       "knee_unclipped_action_rad", "hip_clipped_action_rad",
                                       "knee_clipped_action_rad"])
            for plot in output.glob("*.svg"):
                tree = ET.parse(plot)
                if plot.name.endswith("_position.svg"):
                    text = "".join(tree.getroot().itertext())
                    self.assertIn("actual joint position", text)
                    self.assertIn("unclipped action", text)
                    self.assertIn("clipped action", text)
                    polylines = tree.findall(".//{http://www.w3.org/2000/svg}polyline")
                    self.assertEqual(len(polylines), 3)
                    self.assertNotIn("nan", text.lower())

    def test_excludes_hand_joints_and_keeps_all_position_traces_visible(self):
        tracker = module.JointPlots(.1)
        tracker.joint_names = ["hip", "knee"]
        tracker.position_limits = np.array([[-1., 1.], [-1., 1.]])
        tracker.torque_limits = np.array([10., 10.])
        tracker.update(0, [0., 0.], [0., 0.], [5., 5.], [1., 1.])
        with tempfile.TemporaryDirectory() as directory:
            tracker.save(directory)
            text = (Path(directory) / "joints/hip_position.svg").read_text()
            self.assertIn("actual joint position", text)
            self.assertIn("unclipped action", text)
            self.assertIn("clipped action", text)

    def test_observe_reads_environment_zero_and_validates_frames(self):
        class Tensor(np.ndarray):
            def cpu(self):
                return self

            def numpy(self):
                return np.asarray(self)

        def tensor(value):
            return np.asarray(value).view(Tensor)

        robot = SimpleNamespace(
            joint_names=["hip", "left_wrist_roll_joint", "knee"],
            data=SimpleNamespace(
                soft_joint_pos_limits=tensor([[[-1., 2.], [-2., 2.], [-3., 3.]]]),
                joint_effort_limits=tensor([[10., 20., 30.]]),
                joint_pos=tensor([[0., 99., 1.], [99., 99., 99.]]),
                applied_torque=tensor([[2., 99., 3.], [99., 99., 99.]]),
            ),
        )
        action_term = SimpleNamespace(raw_actions=tensor([[4., 99., 5.]]),
                                      processed_actions=tensor([[.4, 99., .5]]),
                                      _joint_ids=[0, 1, 2],
                                      _scale=tensor([[0.1, 0.1, 0.1]]), _offset=tensor([[1.0, 1.0, 1.0]]),
                                      _clip=tensor([[[-1.0, 1.0], [-2.0, 2.0], [-1.4, 1.4]]]))
        raw = SimpleNamespace(scene={"robot": robot},
                              action_manager=SimpleNamespace(get_term=lambda name: action_term))
        tracker = module.JointPlots(.02)
        tracker.prepare(raw)
        tracker.observe_action(raw)
        tracker.observe(raw, 0)
        self.assertEqual(tracker.joint_names, ["hip", "knee"])
        np.testing.assert_allclose(tracker.positions, [[0., 1.]])
        np.testing.assert_allclose(tracker.torques, [[2., 3.]])
        np.testing.assert_allclose(tracker.unclipped_actions, [[1.4, 1.5]])
        np.testing.assert_allclose(tracker.clipped_actions, [[1.0, 1.4]])
        with self.assertRaises(ValueError):
            tracker.observe(raw, 2)

    def test_clips_recorded_action_to_action_term_bounds(self):
        tracker = module.JointPlots(.02)
        tracker.joint_names = ["left_ankle_roll_joint"]
        tracker.position_limits = np.array([[-1., 1.]])
        tracker.torque_limits = np.array([10.])
        tracker.joint_indices = np.array([0])
        tracker._pending_unclipped_action = np.full(1, np.nan)
        tracker._pending_clipped_action = np.full(1, np.nan)
        action_term = SimpleNamespace(
            raw_actions=np.array([[-10.]]),
            processed_actions=np.array([[-10.]]),
            _joint_ids=[0],
            _scale=1.0,
            _offset=0.0,
            _clip=np.array([[[[-0.2618, 0.2618]]]]),
            cfg=SimpleNamespace(clip={"left_ankle_roll_joint": (-0.2618, 0.2618)}),
        )
        raw = SimpleNamespace(
            scene={"robot": SimpleNamespace(joint_names=["left_ankle_roll_joint"])},
            action_manager=SimpleNamespace(get_term=lambda name: action_term),
        )
        tracker.observe_action(raw)
        np.testing.assert_allclose(tracker._pending_unclipped_action, [-10.])
        np.testing.assert_allclose(tracker._pending_clipped_action, [-0.2618])

    def test_rejects_nonfinite_samples(self):
        tracker = module.JointPlots(.1)
        tracker.joint_names = ["hip"]
        tracker.position_limits = np.array([[-1., 1.]])
        tracker.torque_limits = np.array([10.])
        with self.assertRaises(ValueError):
            tracker.update(0, [np.nan], [0.])

    def test_accepts_transposed_batched_clip_bounds(self):
        tracker = module.JointPlots(.02)
        tracker.joint_names = ["left_ankle_roll_joint", "knee"]
        tracker.position_limits = np.array([[-1., 1.], [-1., 1.]])
        tracker.torque_limits = np.array([10., 10.])
        tracker.joint_indices = np.array([0, 1])
        action_term = SimpleNamespace(
            raw_actions=np.array([[-10., 10.]]),
            processed_actions=np.array([[-10., 10.]]),
            _joint_ids=[0, 1],
            _scale=1.0,
            _offset=0.0,
            _clip=np.array([[[-0.2618, 0.2618], [-0.5, 0.5]]]),
        )
        raw = SimpleNamespace(
            scene={"robot": SimpleNamespace(joint_names=["left_ankle_roll_joint", "knee"])},
            action_manager=SimpleNamespace(get_term=lambda name: action_term),
        )
        tracker.observe_action(raw)
        np.testing.assert_allclose(tracker._pending_clipped_action, [-0.2618, 0.5])

    def test_accepts_batched_clip_bounds_without_joint_axis_detection(self):
        tracker = module.JointPlots(.02)
        tracker.joint_names = ["left_ankle_roll_joint", "knee"]
        tracker.position_limits = np.array([[-1., 1.], [-1., 1.]])
        tracker.torque_limits = np.array([10., 10.])
        tracker.joint_indices = np.array([0, 1])
        action_term = SimpleNamespace(
            raw_actions=np.array([[-10., 10.]]),
            processed_actions=np.array([[-10., 10.]]),
            _joint_ids=[0, 1],
            _scale=1.0,
            _offset=0.0,
            _clip=np.array([[[[-0.2618, 0.2618], [-0.5, 0.5]]]]),
        )
        raw = SimpleNamespace(
            scene={"robot": SimpleNamespace(joint_names=["left_ankle_roll_joint", "knee"])},
            action_manager=SimpleNamespace(get_term=lambda name: action_term),
        )
        tracker.observe_action(raw)
        np.testing.assert_allclose(tracker._pending_clipped_action, [-0.2618, 0.5])

    def test_resolves_collapsed_clip_from_action_config(self):
        tracker = module.JointPlots(.02)
        tracker.joint_names = ["left_ankle_roll_joint", "knee"]
        tracker.position_limits = np.array([[-1., 1.], [-1., 1.]])
        tracker.torque_limits = np.array([10., 10.])
        tracker.joint_indices = np.array([0, 1])
        action_term = SimpleNamespace(
            raw_actions=np.array([[-10., 10.]]),
            processed_actions=np.array([[-10., 10.]]),
            _joint_ids=[0, 1],
            _joint_names=["left_ankle_roll_joint", "knee"],
            _scale=1.0,
            _offset=0.0,
            _clip=np.array([[-1., 1.]]),
            cfg=SimpleNamespace(clip={
                "left_ankle_roll_joint": (-0.2618, 0.2618),
                "knee": (-0.5, 0.5),
            }),
        )
        raw = SimpleNamespace(
            scene={"robot": SimpleNamespace(joint_names=["left_ankle_roll_joint", "knee"])},
            action_manager=SimpleNamespace(get_term=lambda name: action_term),
        )
        tracker.observe_action(raw)
        np.testing.assert_allclose(tracker._pending_clipped_action, [-0.2618, 0.5])

    def test_resolves_collapsed_clip_from_task_constants(self):
        tracker = module.JointPlots(.02)
        tracker.joint_names = ["left_ankle_roll_joint"]
        tracker.position_limits = np.array([[-1., 1.]])
        tracker.torque_limits = np.array([10.])
        tracker.joint_indices = np.array([0])
        action_term = SimpleNamespace(
            raw_actions=np.array([[-10.]]),
            processed_actions=np.array([[-10.]]),
            _joint_ids=[0],
            _joint_names=["left_ankle_roll_joint"],
            _scale=1.0,
            _offset=0.0,
            _clip=np.array([[-1., 1.]]),
            cfg=SimpleNamespace(clip={"left_ankle_roll_joint": (-0.2618, 0.2618)}),
        )
        raw = SimpleNamespace(
            scene={"robot": SimpleNamespace(joint_names=["left_ankle_roll_joint"])},
            action_manager=SimpleNamespace(get_term=lambda name: action_term),
        )
        tracker.observe_action(raw)
        np.testing.assert_allclose(tracker._pending_clipped_action, [-0.2618])
