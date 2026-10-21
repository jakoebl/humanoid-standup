"""Check full-clip reward ranking and cached playback without Isaac Sim."""

from contextlib import ContextDecorator
import csv
import importlib.util
from pathlib import Path
import tempfile
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class Tensor(np.ndarray):
    def clone(self):
        return self.copy()

    def to(self, device):
        return self

    def sum(self, dim=None, keepdim=False):
        return tensor(np.asarray(self).sum(axis=dim, keepdims=keepdim))

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(value, **kwargs):
    return np.array(value).view(Tensor)


class PolicyEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.inference_depth = 0
        test = self

        class InferenceMode(ContextDecorator):
            def __enter__(self):
                test.inference_depth += 1
                return self

            def __exit__(self, *args):
                test.inference_depth -= 1

        torch = SimpleNamespace(
            inference_mode=InferenceMode, bool=bool, all=np.all, any=np.any, tensor=tensor,
            ones=lambda count, **kwargs: tensor(np.ones(count, dtype=bool)),
            cat=lambda values, dim: tensor(np.concatenate(values, axis=dim)),
        )
        path = Path(__file__).resolve().parents[1] / "scripts/evaluate/rewards.py"
        spec = importlib.util.spec_from_file_location("policy_evaluation_test", path)
        self.module = importlib.util.module_from_spec(spec)
        torch_patch = patch.dict("sys.modules", {"torch": torch})
        torch_patch.start()
        self.addCleanup(torch_patch.stop)
        with patch.object(sys, "path", [str(path.parents[1])] + sys.path):
            spec.loader.exec_module(self.module)
            from evaluate.foot_motion import StandingFootMotion
            self.foot_motion = StandingFootMotion
            from evaluate.lifted_foot import LiftedFoot
            self.lifted_foot = LiftedFoot
            from evaluate.standing_push import StandingPush, push_frame
            self.standing_push = StandingPush
            self.push_frame = push_frame
            from evaluate.standing_up import StandingUp
            self.standing_up = StandingUp
            main_spec = importlib.util.spec_from_file_location("evaluate_main_test", path.parents[1] / "evaluate.py")
            self.main = importlib.util.module_from_spec(main_spec)
            main_spec.loader.exec_module(self.main)

        self.frame = 0
        self.reset_count = 0
        self.motion = SimpleNamespace(
            cfg=SimpleNamespace(play_clip_index=0), local_frames=np.zeros(3), clip_ids=np.zeros(3),
            motion=SimpleNamespace(lengths=np.array([3]), paths=["clip.npz"]),
        )
        self.robot = SimpleNamespace(
            num_joints=1, data=SimpleNamespace(root_state_w=np.zeros((3, 13)),
                                             joint_pos=np.zeros((3, 1)), joint_vel=np.zeros((3, 1)),
                                             body_com_pos_w=tensor([[[0., 0., .6]], [[0., 0., .4]], [[0., 0., .6]]]),
                                             body_link_pos_w=tensor(np.zeros((3, 2, 3)))),
            root_physx_view=SimpleNamespace(get_masses=lambda: tensor(np.ones((3, 1)))),
            find_bodies=lambda names, **kwargs: ([0, 1], names),
        )
        class Scene(dict):
            env_origins = tensor(np.zeros((3, 3)))

        raw = SimpleNamespace(num_envs=3, device="cpu", step_dt=.02, cfg=SimpleNamespace(seed=42),
                              command_manager=SimpleNamespace(get_term=lambda name: self.motion),
                              scene=Scene(robot=self.robot))
        self.env = SimpleNamespace(unwrapped=raw, reset=self.reset, step=self.step)
        # A callable object with a state-reset method.
        self.policy = lambda obs: obs
        self.policy.reset = lambda dones: None

    def reset(self):
        self.frame = 0
        self.reset_count += 1
        self.robot.data.root_state_w[:] = 0
        return None, {}

    def step(self, actions):
        self.frame += 1
        self.robot.data.root_state_w[:] = self.frame
        self.robot.data.body_link_pos_w[0, 0, 0] = .03 * self.frame
        return None, tensor([2., -1., 3.]), tensor([False] * 3), {}

    def test_standing_push_criterion_pushes_once_and_measures_recovery(self):
        self.motion.motion.lengths = np.array([101])
        calls = []
        criterion = self.standing_push(3, .02)
        self.motion.cfg.hold_final_frame = True
        length = criterion.rollout_frames(101)
        with tempfile.TemporaryDirectory() as folder:
            with self.module.TrajectoryCache(self.env.unwrapped, length, folder) as cache:
                with patch.object(criterion, "apply_push", lambda raw: calls.append(self.frame)):
                    self.main.run_clip(self.env, self.policy, 0, [criterion, cache], num_frames=length)
                self.assertEqual(len(cache.states), 201)
                self.assertEqual(cache.states[-1, 0, 0], 200)
        self.assertEqual(self.frame, 200)
        self.assertEqual(self.reset_count, 1)
        self.assertEqual(calls, [50])
        summary = criterion.summary()
        self.assertEqual(summary["pushes_per_trial"], 1)
        self.assertEqual(summary["standing_success"]["succeeded"], 2)
        self.assertEqual(summary["foot_movement"]["moved_feet_after_standing"], 1)
        self.assertEqual(summary["combined_video"], "standing_push.mp4")
        ids = [attempt["env_id"] for attempt in criterion.video_attempts()]
        self.assertEqual(sorted(ids), [0, 1, 2])
        # Motion before the push does not contribute to push-response displacement.
        self.assertAlmostEqual(criterion.foot_motion.max_displacement[0, 0], 4.5)
        self.assertEqual(summary["standing_success"]["required_duration_s"], 3.0)
        np.testing.assert_array_equal(criterion.succeeded, [True, False, True])
        with tempfile.TemporaryDirectory() as folder:
            criterion.save(folder)
            with (Path(folder) / "raw/standing_push.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["push_frame"] for row in rows], ["50"] * 3)
        # Losing standing within the response window must fail even if recovered.
        criterion = self.standing_push(1, .02)
        criterion.required_intervals = 150
        for frame in range(201):
            criterion.update(frame, np.array([.4 if frame == 175 else .6]))
        self.assertFalse(criterion.succeeded[0])

    def test_push_video_selection_is_seeded_unique_and_limited_to_ten(self):
        raw = self.env.unwrapped
        raw.num_envs = 100
        self.motion.motion.lengths = np.array([101])
        first = self.standing_push(100, .02)
        second = self.standing_push(100, .02)
        first.prepare(raw)
        second.prepare(raw)
        np.testing.assert_array_equal(first.velocity_offsets, second.velocity_offsets)
        np.testing.assert_allclose(np.linalg.norm(first.velocity_offsets, axis=1), .5)
        self.assertEqual(first.video_attempts(), second.video_attempts())
        ids = [attempt["env_id"] for attempt in first.video_attempts()]
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids)), 10)
        self.assertTrue(all(0 <= index < 100 for index in ids))

    def test_horizontal_push_adds_fixed_speed_and_preserves_other_velocities(self):
        criterion = self.standing_push(3, .02)
        self.motion.motion.lengths = np.array([101])
        raw = self.env.unwrapped
        criterion.prepare(raw)
        before = tensor([[1., 2., 3., 4., 5., 6.], [-2., .3, -.4, .5, -.6, .7], [0., 0., 0., 0., 0., 0.]])
        self.robot.data.root_vel_w = before.copy()
        written = []
        self.robot.write_root_velocity_to_sim = lambda value: written.append(value.copy())
        criterion.apply_push(raw)
        delta = written[0] - before
        np.testing.assert_allclose(np.linalg.norm(delta[:, :2], axis=1), .5)
        np.testing.assert_array_equal(delta[:, 2:], 0.)
        np.testing.assert_array_equal(self.robot.data.root_vel_w, before)
        self.assertEqual(criterion.summary()["horizontal_velocity_change_m_s"], .5)

    def test_extended_rollout_requires_final_reference_hold(self):
        self.motion.cfg.hold_final_frame = False
        with self.assertRaisesRegex(ValueError, "holding the final reference frame"):
            self.main.run_clip(self.env, self.policy, 0, [], num_frames=103)

    def test_standing_push_timing_rejects_short_clips(self):
        self.assertEqual(self.push_frame(51, .02), 0)
        self.assertEqual(self.push_frame(101, .03), 67)
        for length, dt in ((3, .02), (101, 0), (101, float("nan"))):
            with self.subTest(length=length, dt=dt), self.assertRaises(ValueError):
                self.push_frame(length, dt)

    def test_full_clip_reports_all_rewards_and_renders_original_states(self):
        rendered = []

        def render(raw, states, attempts, output):
            rendered.append(states.copy())
            self.assertEqual([attempt["env_id"] for attempt in attempts], [1, 0, 2])

        with tempfile.TemporaryDirectory() as directory, patch.object(self.module, "render_attempts", render):
            with self.module.RewardEvaluation(self.env.unwrapped, 3, directory) as reward:
                feet = self.foot_motion(3)
                self.main.run_clip(self.env, self.policy, 0, [reward, feet])
                summary = reward.summary()
                self.assertEqual(summary["reward_steps"], 2)
                self.assertEqual([entry["total_reward"] for entry in summary["worst_attempts"]], [-2., 4., 6.])
                feet.save(directory)
                reward.save(directory)
                self.assertEqual(feet.summary()["stood"], 2)
                self.assertEqual(feet.summary()["never_stood"], 1)
                self.assertEqual(feet.summary()["moved_feet_after_standing"], 1)
            with (Path(directory) / "raw" / "rewards.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["total_reward"] for row in rows], ["4.0", "-2.0", "6.0"])
            with (Path(directory) / "raw" / "foot_motion.csv").open() as stream:
                foot_rows = list(csv.DictReader(stream))
            self.assertEqual([row["moved_feet_after_standing"] for row in foot_rows], ["True", "False", "False"])
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()),
                             ["foot_motion_histogram.svg", "raw", "reward_histogram.svg"])
        self.assertEqual(self.frame, 2)
        self.assertEqual(self.reset_count, 1)
        np.testing.assert_array_equal(rendered[0][:, 1, 0], [0, 1, 2])

    def test_reward_histogram_bins_preserve_all_trials_and_match_plot(self):
        import xml.etree.ElementTree as ET

        for totals in (np.array([-4., -2., 0., 0., 6.]), np.full(3, 2.), np.array([0.])):
            with self.subTest(totals=totals.tolist()), tempfile.TemporaryDirectory() as directory:
                self.module.save_reward_histogram(totals, directory)
                expected_counts, expected_edges = np.histogram(totals, bins="auto")
                with (Path(directory) / "raw" / "reward_bins.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(sum(int(row["count"]) for row in rows), len(totals))
                np.testing.assert_array_equal([int(row["count"]) for row in rows], expected_counts)
                np.testing.assert_allclose([float(row["bin_left"]) for row in rows], expected_edges[:-1])
                np.testing.assert_allclose([float(row["bin_right"]) for row in rows], expected_edges[1:])
                svg = ET.parse(Path(directory) / "reward_histogram.svg")
                bars = [rect for rect in svg.findall(".//{http://www.w3.org/2000/svg}rect")
                        if rect.find("{http://www.w3.org/2000/svg}title") is not None]
                self.assertEqual(len(bars), len(rows))
                for bar, row in zip(bars, rows):
                    self.assertIn(f": {row['count']} trials", bar[0].text)
                    self.assertGreaterEqual(float(bar.attrib["height"]), 0.)

    def test_selects_exactly_ten_lowest_rewards_with_stable_ties(self):
        totals = np.array([4, 1, -2, 1, 8, 9, 0, 7, 3, 6, 5, 2])
        attempts = self.module.worst_attempts(totals)
        self.assertEqual([entry["env_id"] for entry in attempts], [2, 6, 1, 3, 11, 8, 0, 10, 9, 7])

    def test_standing_requires_full_second_and_strict_height_at_clip_end(self):
        tracker = self.standing_up(4, .02)
        for frame in range(50):
            tracker.update(frame, np.array([.6, .6, .5, .6]))
        self.assertFalse(tracker.succeeded.any())  # 50 samples span only 0.98 seconds.
        tracker.update(50, np.array([.6, .4, .5, np.nan]))
        np.testing.assert_array_equal(tracker.succeeded, [True, False, False, False])
        self.assertEqual(tracker.summary()["success_rate"], .25)
        tracker.update(51, np.array([.4, .6, .6, .6]))
        self.assertFalse(tracker.succeeded.any())  # An earlier hold does not count.

    def test_standing_observes_mass_weighted_com_relative_to_ground_and_saves(self):
        raw = self.env.unwrapped
        self.robot.root_physx_view.get_masses = lambda: tensor([[1., 3.]] * 3)
        self.robot.data.body_com_pos_w = tensor([[[0., 0., .1], [0., 0., .7]]] * 3)
        raw.scene.env_origins[1, 2] = .1
        tracker = self.standing_up(3, .3)
        tracker.prepare(raw)
        for frame in range(5):
            tracker.observe(raw, frame)
        np.testing.assert_array_equal(tracker.succeeded, [True, False, True])
        with tempfile.TemporaryDirectory() as directory:
            tracker.save(directory)
            with (Path(directory) / "raw" / "standing_up.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["succeeded"] for row in rows], ["True", "False", "True"])

    def test_rejects_automatic_resets(self):
        self.env.step = lambda action: (None, tensor([1.] * 3), tensor([True, False, False]), {})
        with self.assertRaisesRegex(RuntimeError, "Unexpected reset"):
            self.main.run_clip(self.env, self.policy, 0, [])

    def test_foot_anchors_capture_first_crossing_and_persist_through_falls(self):
        tracker = self.foot_motion(3)
        feet = np.zeros((3, 2, 3))
        feet[1] = 10.  # Foot positions can have arbitrary world origins.
        tracker.update(0, np.array([.5, .4, .4]), feet)
        feet[0, 0, 0] = .019
        feet[1, 1, 2] += 1.  # Before standing: must not count.
        tracker.update(1, np.array([.6, .5, .4]), feet)
        np.testing.assert_array_equal(tracker.moved, [False, False, False])
        feet[0, 0, 0] = .021
        feet[1, 1, 2] += .03
        tracker.update(2, np.array([.2, .6, .4]), feet)
        np.testing.assert_array_equal(tracker.moved, [True, True, False])
        feet[0, 0, 0] = 0.
        tracker.update(3, np.array([.6, .6, .4]), feet)
        np.testing.assert_array_equal(tracker.first_standing_frame, [0, 1, -1])
        np.testing.assert_array_equal(tracker.moved, [True, True, False])

    def test_foot_lift_counts_only_upward_motion_after_standing(self):
        tracker = self.lifted_foot(4)
        feet = np.zeros((4, 2, 3))
        tracker.update(0, np.array([.5, .4, .5, .4]), feet)
        feet[0, 0, 0] = 1.  # Horizontal movement is not a lift.
        feet[1, 1, 2] = 1.  # Pre-standing movement becomes the anchor.
        feet[2, 0, 2] = -.1  # Downward movement is not a lift.
        feet[3, 0, 2] = 1.  # Never-standing trial must not count.
        tracker.update(1, np.array([.6, .5, .6, .4]), feet)
        np.testing.assert_array_equal(tracker.lifted, [False] * 4)
        feet[0, 0, 2] = .02  # Exactly the tolerance must not count.
        tracker.update(2, np.array([.6, .6, .6, .4]), feet)
        self.assertFalse(tracker.lifted.any())
        feet[0, 0, 2] = .021
        feet[1, 1, 2] += .03
        tracker.update(3, np.array([.2, .6, .6, .4]), feet)
        feet[:, :, 2] = 0.
        tracker.update(4, np.array([.2, .6, .6, .4]), feet)
        np.testing.assert_array_equal(tracker.lifted, [True, True, False, False])
        self.assertEqual(tracker.summary()["lifted_foot_after_standing"], 2)
        with tempfile.TemporaryDirectory() as directory:
            tracker.save(directory)
            with (Path(directory) / "raw" / "lifted_foot.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["lifted_foot_after_standing"] for row in rows],
                             ["True", "True", "False", "False"])
        never_stood = self.lifted_foot(1)
        self.assertIsNone(never_stood.summary()["fraction_of_standing_trials"])

    def test_foot_graphs_exclude_never_stood_and_handle_empty_or_constant_values(self):
        import xml.etree.ElementTree as ET

        for stood_frames in ([0, 1, -1], [-1, -1, -1], [0, 0, 0]):
            tracker = self.foot_motion(3)
            tracker.first_standing_frame[:] = stood_frames
            if stood_frames == [0, 1, -1]:
                tracker.max_displacement[:] = [[.01, .03], [.04, .02], [100., 100.]]
            with self.subTest(frames=stood_frames), tempfile.TemporaryDirectory() as directory:
                tracker.save_graph(directory)
                with (Path(directory) / "raw" / "foot_motion_bins.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                expected = int((tracker.first_standing_frame >= 0).sum())
                for foot in ("left", "right"):
                    self.assertEqual(sum(int(row[f"{foot}_count"]) for row in rows), expected)
                if stood_frames == [0, 1, -1]:
                    self.assertLess(float(rows[-1]["bin_right_m"]), 1.)
                svg = ET.parse(Path(directory) / "foot_motion_histogram.svg")
                bars = [rect for rect in svg.findall(".//{http://www.w3.org/2000/svg}rect")
                        if rect.find("{http://www.w3.org/2000/svg}title") is not None]
                self.assertEqual(len(bars), 2 * len(rows))
                for foot, label in enumerate(("left", "right")):
                    for bar, row in zip(bars[foot * len(rows):(foot + 1) * len(rows)], rows):
                        self.assertIn(f"{row[f'{label}_count']} trials", bar[0].text)
                markers = [path for path in svg.findall(".//{http://www.w3.org/2000/svg}path")
                           if path.attrib.get("stroke-dasharray")]
                self.assertEqual(len(markers), 2)

    def test_no_standing_trials_has_no_standing_fraction(self):
        tracker = self.foot_motion(3)
        tracker.update(0, np.zeros(3), np.zeros((3, 2, 3)))
        self.assertIsNone(tracker.summary()["fraction_of_standing_trials"])
        self.assertEqual(tracker.summary()["moved_feet_after_standing"], 0)

    def test_foot_video_selection_caps_at_ten_and_filters_strict_threshold(self):
        tracker = self.foot_motion(15, .1)
        tracker.first_standing_frame[:] = 0
        tracker.max_displacement[:, 0] = np.arange(15) / 10
        tracker.first_standing_frame[14] = -1
        tracker.max_displacement[12] = tracker.max_displacement[13]  # Stable tie.
        attempts = tracker.video_attempts()
        self.assertEqual([item["env_id"] for item in attempts], [12, 13, 11, 10, 9, 8, 7, 6, 5, 4])
        self.assertTrue(all(item["video"].startswith("vids/foot_motion_") for item in attempts))
        tracker.max_displacement[:] = .1
        self.assertEqual(tracker.video_attempts(), [])
        self.assertIsNone(tracker.summary()["combined_video"])

    def test_empty_video_selection_does_not_import_encoder_or_render(self):
        self.module.render_attempts(None, None, [], None, criterion="foot_motion")

    def test_shared_cache_records_states_without_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.module.TrajectoryCache(self.env.unwrapped, 3, directory) as cache:
                self.main.run_clip(self.env, self.policy, 0, [cache])
                np.testing.assert_array_equal(cache.states[:, 0, 0], [0, 1, 2])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_video_replays_cached_states_without_stepping_or_mutating_cache(self):
        raw = self.env.unwrapped
        robot = self.robot

        class Scene(dict):
            env_origins = tensor([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]])

            def write_data_to_sim(self):
                pass

            def update(self, dt):
                pass

        def write_root(state, env_ids):
            self.assertGreater(self.inference_depth, 0)
            robot.data.root_state_w[env_ids] = state

        def write_joints(positions, velocities, env_ids):
            self.assertGreater(self.inference_depth, 0)
            robot.data.joint_pos[env_ids] = positions
            robot.data.joint_vel[env_ids] = velocities

        robot.write_root_state_to_sim = write_root
        robot.write_joint_state_to_sim = write_joints
        raw.scene = Scene(robot=robot)
        raw.physics_dt = .005
        raw.sim = SimpleNamespace(set_camera_view=lambda **kwargs: None, forward=lambda: None, render=lambda: None)
        raw.render = lambda **kwargs: robot.data.root_state_w[0, :3].copy()
        states = np.zeros((3, 3, 15), dtype=np.float32)
        states[:, 1, 0] = [10, 11, 12]
        original = states.copy()

        written = {}

        class Writer:
            def __init__(self, path, **kwargs):
                Path(path).touch()
                self.frames = written.setdefault(Path(path).name, [])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def append_data(self, rgb):
                self.frames.append(rgb[0])

        imageio = ModuleType("imageio")
        imageio.__path__ = []
        imageio_v2 = ModuleType("imageio.v2")
        imageio_v2.get_writer = Writer
        imageio.v2 = imageio_v2
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "sys.modules", {"imageio": imageio, "imageio.v2": imageio_v2}):
            output = Path(directory)
            self.module.render_attempts(raw, states, [
                {"env_id": 1, "video": "vids/worst_01.mp4"},
                {"env_id": 1, "video": "vids/worst_02.mp4"}], output, criterion="jitteriness")
            self.assertTrue((output / "vids/worst_01.mp4").exists())
            self.assertTrue((output / "vids/worst_02.mp4").exists())
            self.assertTrue((output / "jitteriness.mp4").exists())
            self.assertFalse(list(output.rglob("*.partial.mp4")))
        self.assertEqual(written["worst_01.partial.mp4"], [0., 1., 2.])
        self.assertEqual(written["worst_02.partial.mp4"], [0., 1., 2.])
        self.assertEqual(written["jitteriness.partial.mp4"], [0., 1., 2.] * 2)
        self.assertEqual(self.frame, 0)
        np.testing.assert_array_equal(states, original)


if __name__ == "__main__":
    unittest.main()
