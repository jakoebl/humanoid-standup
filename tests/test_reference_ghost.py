"""Check reference synchronization and render timing without Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class ReferenceGhostTest(unittest.TestCase):
    def test_nested_link_paths_preserve_body_order(self):
        path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/reference_ghost.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        def prim(name, path, rigid=True):
            return SimpleNamespace(GetName=lambda: name, GetPath=lambda: path,
                                   GetAppliedSchemas=lambda: ["PhysicsRigidBodyAPI"] if rigid else [])
        prims = [prim("pelvis", "/Robot/pelvis"),
                 prim("left_hand_palm_link", "/Robot/links/left_hand_palm_link"),
                 prim("pelvis", "/Robot/pelvis/visuals/pelvis", rigid=False),
                 prim("pelvis", "/Robot/pelvis/collisions/pelvis", rigid=False),
                 prim("left_hand_palm_link", "/Robot/links/left_hand_palm_link/visuals/left_hand_palm_link",
                      rigid=False)]
        resolve = namespace["resolve_link_paths"]
        self.assertEqual(resolve(prims, ["left_hand_palm_link", "pelvis"]),
                         ["/Robot/links/left_hand_palm_link", "/Robot/pelvis"])
        with self.assertRaisesRegex(ValueError, "missing"):
            resolve(prims, ["missing"])
        with self.assertRaisesRegex(ValueError, "Expected one"):
            resolve(prims + [prim("pelvis", "/Robot/other/pelvis")], ["pelvis"])

    def test_world_reference_and_fresh_frame(self):
        path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/reference_ghost.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        cls.bases = []
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name != "__init__"]
        namespace = {}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        ghost = namespace["ReferenceGhost"]()
        class Tensor(np.ndarray):
            def cpu(self):
                return self
        def tensor(values):
            return np.array(values).view(Tensor)
        values = {}
        calls = []
        ghost.Gf = SimpleNamespace(Vec3d=lambda *values: tuple(values), Quatd=lambda real, imag: (real, imag))
        ghost.links = [(SimpleNamespace(Set=lambda value: values.update(position=value)),
                        SimpleNamespace(Set=lambda value: values.update(quaternion=value)))]
        frames = tensor([0])
        ghost.motion = SimpleNamespace(time_steps=frames, motion=SimpleNamespace(
            _body_pos_w=tensor([[[1., 2., 3.]], [[4., 5., 6.]]]),
            _body_quat_w=tensor([[[1., 0., 0., 0.]], [[0., 1., 0., 0.]]])))
        def step(action):
            frames[:] = 1
            calls.append("physics")
            return "result"
        raw = SimpleNamespace(scene=SimpleNamespace(env_origins=tensor([[10., 20., 0.]])),
                              sim=SimpleNamespace(render=lambda: calls.append(dict(values))))
        ghost.env = SimpleNamespace(unwrapped=raw, step=step, reset=lambda **kwargs: "reset")
        self.assertEqual(ghost.reset(), "reset")
        self.assertEqual(values["position"], (11., 22., 3.))
        self.assertEqual(ghost.step(None), "result")
        self.assertEqual(calls[0], "physics")
        self.assertEqual(calls[1]["position"], (14., 25., 6.))
        self.assertEqual(calls[1]["quaternion"], (0., (1., 0., 0.)))


if __name__ == "__main__":
    unittest.main()
