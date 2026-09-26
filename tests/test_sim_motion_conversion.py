"""CPU-only input conversion checks; no Isaac Sim startup required."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/precompute_reference_kinematics_sim.py'
spec = importlib.util.spec_from_file_location('sim_precompute', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def quaternion(roll, pitch, yaw):
    cr, cp, cy = (math.cos(v / 2) for v in (roll, pitch, yaw))
    sr, sp, sy = (math.sin(v / 2) for v in (roll, pitch, yaw))
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


class ConversionTests(unittest.TestCase):
    def convert(self, poses, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.csv'
            path.write_text('\n'.join(','.join(map(str, pose)) for pose in poses))
            return module.load_quaternion_rows(path, **kwargs)

    def test_hold_units_and_order(self):
        pose = [1, 2, 3, 0, 0, 0, 1] + [i / 10 for i in range(29)]
        rows, fields = self.convert([pose, [4] + pose[1:]])
        self.assertEqual(fields, module.MOTION_COLUMNS + module.TARGET_COLUMNS)
        self.assertEqual([row['Frame'] for row in rows], list(range(8)))
        self.assertEqual([row['root_translateX'] for row in rows], [100]*4 + [400]*4)
        for i, name in enumerate(module.JOINT_NAMES):
            self.assertAlmostEqual(rows[0][name + '_dof'], math.degrees(i / 10))
        raw, _ = self.convert([pose, pose], translation_unit='cm', joint_unit='deg')
        self.assertEqual(raw[0]['root_translateX'], 1)
        self.assertEqual(raw[0][module.JOINT_NAMES[-1] + '_dof'], 2.8)

    def test_orientation_round_trip(self):
        for pitch in (0.4, -0.8, math.pi/2, -math.pi/2):
            q = quaternion(0.7, pitch, -1.1)
            pose = [0, 0, 0] + [v * 3 for v in q] + [0]*29
            rows, _ = self.convert([pose, pose])
            recovered = quaternion(*(math.radians(rows[0]['root_rotate' + axis]) for axis in 'XYZ'))
            self.assertAlmostEqual(abs(sum(a*b for a, b in zip(q, recovered))), 1.0, places=10)

    def test_invalid_input(self):
        valid = [0, 0, 0, 0, 0, 0, 1] + [0]*29
        for bad in (valid[:-1], [0]*36, [float('nan')] + valid[1:]):
            with self.assertRaises(ValueError):
                self.convert([valid, bad])

    def test_legacy_still_loads(self):
        path = SCRIPT.parents[1] / 'data/unprocessed/stand_up_lying_R_002__A472_M.csv'
        rows, fields = module.load_rows(path)
        self.assertGreater(len(rows), 1)
        self.assertEqual(fields[:36], module.MOTION_COLUMNS)
        self.assertIsInstance(rows[0]['root_rotateX'], str)


if __name__ == '__main__':
    unittest.main()
