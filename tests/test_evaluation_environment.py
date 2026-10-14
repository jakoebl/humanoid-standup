"""Verify evaluation metadata without Isaac Sim."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


path = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
spec = importlib.util.spec_from_file_location("evaluation_environment", path)
metadata = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(path.parent)] + sys.path):
    spec.loader.exec_module(metadata)


def encoder_bias(env, env_ids, bias_range=(-.01, .01)):
    pass


class EvaluationEnvironmentTest(unittest.TestCase):
    def test_evaluation_disables_pushes_and_restores_requested_training_noise(self):
        defaults = SimpleNamespace(pose_range={"x": (-.05, .05), "yaw": (-.2, .2)},
                                   joint_position_range=(-.3, .3))
        for play in (False, True):
            velocity_range = {} if play else {"x": (-.5, .5)}
            cfg = SimpleNamespace(
                events=SimpleNamespace(push_robot=None if play else object(), material=object()),
                observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=not play)),
                commands=SimpleNamespace(motion=SimpleNamespace(
                    pose_range={} if play else defaults.pose_range.copy(),
                    joint_position_range=(0., 0.) if play else defaults.joint_position_range,
                    velocity_range=velocity_range)),
            )
            material = cfg.events.material
            metadata.configure_evaluation_randomization(cfg, defaults)
            self.assertIsNone(cfg.events.push_robot)
            self.assertTrue(cfg.observations.policy.enable_corruption)
            self.assertEqual(cfg.commands.motion.pose_range, defaults.pose_range)
            self.assertEqual(cfg.commands.motion.joint_position_range, (-.3, .3))
            self.assertIs(cfg.events.material, material)
            self.assertIs(cfg.commands.motion.velocity_range, velocity_range)
            cfg.commands.motion.pose_range["x"] = (0., 0.)
            self.assertEqual(defaults.pose_range["x"], (-.05, .05))

    def test_resolved_noise_defaults_and_config_are_json_safe(self):
        term = SimpleNamespace(noise=SimpleNamespace(n_min=-.01, n_max=.01),
                               scale=1., clip=None, params={"biased": True})
        event = SimpleNamespace(func=encoder_bias, mode="startup", params={})
        cfg = SimpleNamespace(
            observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=False, joint_pos=term)),
            events=SimpleNamespace(encoder_bias=event, push_robot=None),
            sim=SimpleNamespace(dt=.005, gravity=(0., 0., -9.81)),
            actions=SimpleNamespace(joint_pos=SimpleNamespace(scale=.25)),
        )
        motion = SimpleNamespace(cfg=SimpleNamespace(play_clip_index=0, joint_position_range=(-.3, .3)),
                                 motion=SimpleNamespace(lengths=np.array([51])),
                                 encoder_bias=np.array([[-.01, .01]]))
        robot = SimpleNamespace(root_physx_view=SimpleNamespace(
            get_masses=lambda: np.array([[1., 3.]]),
            get_material_properties=lambda: np.array([[[.3, .4, 0.]]]),
            get_coms=lambda: np.array([[[0., 0., .02, 0., 0., 0., 1.]]]),
        ))
        raw = SimpleNamespace(cfg=cfg, step_dt=.02, physics_dt=.005, scene={"robot": robot},
                              command_manager=SimpleNamespace(get_term=lambda name: motion))
        summary = metadata.environment_summary(raw, "BeyondMimic", {"clip_actions": 1.})
        encoded = json.loads(json.dumps(summary, allow_nan=False))
        self.assertFalse(encoded["policy_observation_noise"]["joint_pos"]["enabled"])
        self.assertEqual(encoded["reset_joint_position_noise_rad"], [-.3, .3])
        self.assertEqual(encoded["effective_event_parameters"]["encoder_bias"]["parameters"]["bias_range"],
                         [-.01, .01])
        self.assertEqual(encoded["clip_duration_s"], 1.)
        self.assertEqual(encoded["sampled_randomization_statistics"]["body_masses_kg"]["mean"], 2.)
        self.assertEqual(encoded["resolved_environment_config"]["sim"]["gravity"], [0., 0., -9.81])
        self.assertTrue(encoded["resolved_environment_config"]["events"]["encoder_bias"]["func"].endswith(
            ".encoder_bias"))
        event.params["bias_range"] = (-.02, .02)
        self.assertEqual(metadata.effective_event_parameters(cfg.events)["encoder_bias"]["parameters"]["bias_range"],
                         [-.02, .02])


if __name__ == "__main__":
    unittest.main()
