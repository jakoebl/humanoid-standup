"""CPU checks for proportional scaling, reset isolation, and curriculum scheduling."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

path = (Path(__file__).resolve().parents[1] / 'source/humanoid_standup/humanoid_standup/tasks'
        '/manager_based/humanoid_standup/mdp/curriculums.py')
spec = importlib.util.spec_from_file_location('mass_curriculums', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class View:
    def __init__(self, masses, inertias):
        self.masses, self.inertias = masses.clone(), inertias.clone()

    def get_masses(self):
        return self.masses

    def get_inertias(self):
        return self.inertias

    def set_masses(self, values, ids):
        self.masses[ids] = values[ids]

    def set_inertias(self, values, ids):
        self.inertias[ids] = values[ids]


masses = torch.tensor([[1., 2., 0.05], [1., 2., 0.05]])
inertias = torch.arange(54).reshape(2, 3, 9).float() + 1
view = View(masses, inertias)
robot = SimpleNamespace(root_physx_view=view, data=SimpleNamespace(default_mass=masses, default_inertia=inertias))
env = SimpleNamespace(scene={'robot': robot}, num_envs=2, common_step_counter=0)
apply = module.robot_mass_curriculum
assert apply(env, [0], ramp_steps=100) == 0.5
for _ in range(3):
    apply(env, [0], ramp_steps=100)
torch.testing.assert_close(view.masses[0], masses[0] * 0.5)
torch.testing.assert_close(view.inertias[0], inertias[0] * 0.5)
torch.testing.assert_close(view.masses[1], masses[1])
torch.testing.assert_close(view.inertias[1], inertias[1])
positions = torch.tensor([[1., 2., 3.], [-3., 5., 2.], [10., -4., 2.]])
def com(m):
    return (m[:, None] * positions).sum(0) / m.sum()
torch.testing.assert_close(com(view.masses[0]), com(masses[0]))
env.common_step_counter = 50
assert apply(env, [1], ramp_steps=100) == 0.75
torch.testing.assert_close(view.masses[1], masses[1] * 0.75)
env.mass_curriculum_step_offset = 50
assert apply(env, None, ramp_steps=100) == 1.0
torch.testing.assert_close(view.masses, masses)
torch.testing.assert_close(view.inertias, inertias)
env.common_step_counter = 200
assert apply(env, None, ramp_steps=100) == 1.0
assert apply(env, [], ramp_steps=100) == 1.0
for kwargs in ({'initial_scale': 0}, {'initial_scale': 1.1}, {'initial_scale': float('nan')},
               {'ramp_steps': 0}, {'ramp_steps': 1.5}):
    try:
        apply(env, [0], **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Invalid parameters accepted: {kwargs}')
print('PASS: uniform mass/inertia scaling, COM invariance, partial resets, no compounding, ramp/resume/clamp, validation')
