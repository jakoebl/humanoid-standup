"""CPU reward regressions; no Isaac Sim startup required."""
import ast
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'source/humanoid_standup/humanoid_standup/tasks/manager_based/humanoid_standup/mdp/rewards.py'
tree = ast.parse(path.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
names = ['left_wrist_yaw_link', 'right_wrist_yaw_link', 'left_ankle_roll_link', 'right_ankle_roll_link']
target_ee = torch.tensor([[[.2, .3, 1.], [.2, -.3, 1.], [0., .1, 0.], [0., -.1, 0.]]]).expand(4, -1, -1)
target_com = torch.tensor([[.05, 0., .9]]).expand(4, -1)
reference = types.SimpleNamespace(
    END_EFFECTOR_NAMES=names, device='cpu',
    JOINT_NAMES=[f'left_{name}_joint' for name in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')],
    get_joint_pos=lambda ids: torch.zeros(4, 6),
    get_root_state=lambda ids: torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]).expand(4, -1),
    frame_ids_for_env=lambda env: torch.zeros(4, dtype=torch.long),
    get_end_effector_pos=lambda ids: target_ee,
    get_com_pos=lambda ids: target_com,
)
namespace = dict(UPRIGHT_COM_HEIGHT=.65, torch=torch, REFERENCE=reference, Articulation=object, ManagerBasedRLEnv=object,
                 SceneEntityCfg=lambda name: types.SimpleNamespace(name=name, body_ids=list(range(4))))
tracking = ast.parse((path.parent / "position_tracking.py").read_text())
functions = [node for node in tracking.body if isinstance(node, ast.FunctionDef)] + functions
exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)

class Scene(dict):
    env_origins = torch.tensor([[0., 0., 0.], [10., -5., 2.], [-2., 7., -1.], [4., 3., 0.]])

angles = torch.tensor([0., .5, -1.2, 2.8])
rotation = torch.zeros(4, 3, 3)
rotation[:, 0, 0] = rotation[:, 1, 1] = angles.cos()
rotation[:, 0, 1] = -angles.sin()
rotation[:, 1, 0] = angles.sin()
rotation[:, 2, 2] = 1.
translation = torch.tensor([[0., 0., 0.], [2., -3., 0.], [-4., 1., 0.], [1., 6., 0.]])
scene = Scene()
# Nonzero errors ensure invariance is tested beyond perfect tracking.
ee = target_ee.clone(); ee[:, 0, 0] += .04
com = target_com + torch.tensor([.02, .03, 0.])
transform = lambda points: points @ rotation.transpose(1, 2) + translation[:, None] + scene.env_origins[:, None]
asset = types.SimpleNamespace(device='cpu', find_bodies=lambda names, preserve_order: (list(range(4)), names),
    data=types.SimpleNamespace(body_link_pos_w=transform(ee), body_com_pos_w=transform(com[:, None]),
                               body_link_quat_w=torch.stack((torch.cos(angles / 2), torch.zeros(4),
                                   torch.zeros(4), torch.sin(angles / 2)), -1)[:, None].expand(-1, 4, -1).clone(),
                               default_mass=torch.ones(4, 1)))
scene['robot'] = asset
env = types.SimpleNamespace(scene=scene)
for name in ('end_effector_pos_target_exp', 'com_pos_target_exp'):
    reward = namespace[name](env)
    torch.testing.assert_close(reward, reward[:1].expand(4), atol=2e-5, rtol=2e-5)
    # Below threshold the original world-position tracking is retained.
    reward = namespace[name](env, upright_com_height=1.)
    positions, targets = (asset.data.body_link_pos_w, target_ee) if name.startswith('end') else (asset.data.body_com_pos_w, target_com[:, None])
    scale = 40. if name.startswith('end') else 10.
    expected = torch.exp(-scale * ((positions - scene.env_origins[:, None] - targets)**2).sum((1, 2)))
    torch.testing.assert_close(reward, expected)
print('PASS: upright CoM/EE translation and yaw invariance with posture errors; below-threshold world tracking')

# Separate arm and leg terms select matching targets in both tracking modes.
for height in (.65, 1.):
    for indices in ((0, 1), (2, 3)):
        cfg = types.SimpleNamespace(name='robot', body_ids=list(indices))
        reward = namespace['end_effector_pos_target_exp'](
            env, asset_cfg=cfg, reference_body_indices=indices, upright_com_height=height)
        targets = namespace['position_tracking_targets'](
            env, asset, torch.zeros(4, dtype=torch.long), target_ee, height)
        error = (asset.data.body_link_pos_w - scene.env_origins[:, None] - targets)[:, list(indices)]
        torch.testing.assert_close(reward, torch.exp(-40. * error.square().sum((1, 2))))
leg_cfg = types.SimpleNamespace(name='robot', body_ids=[2, 3])
leg_reward = namespace['end_effector_pos_target_exp'](env, asset_cfg=leg_cfg, reference_body_indices=(2, 3))
torch.testing.assert_close(leg_reward, torch.ones(4), atol=2e-5, rtol=2e-5)
print('PASS: independent arm/leg tracking in upright and absolute coordinates')

# Optional arm reward is guaranteed per episode after first standing, with inclusive cutoff.
class ManagerTermBase:
    def __init__(self, cfg, env):
        pass

namespace['ManagerTermBase'] = ManagerTermBase
arm_class = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == 'end_effector_pos_tracking')
exec(compile(ast.Module(body=[arm_class], type_ignores=[]), str(path), 'exec'), namespace)
env.num_envs = 4
arm_cfg = types.SimpleNamespace(name='robot', body_ids=[0, 1])
arm_term = namespace['end_effector_pos_tracking'](
    types.SimpleNamespace(params={'asset_cfg': arm_cfg}), env)
arm_params = dict(asset_cfg=arm_cfg, reference_body_indices=(0, 1), upright_com_height=.65)
original_com = asset.data.body_com_pos_w.clone()
asset.data.body_com_pos_w[:, 0, 2] = scene.env_origins[:, 2] + torch.tensor([.5, .65, .9, .4])
baseline = namespace['end_effector_pos_target_exp'](env, **arm_params)
torch.testing.assert_close(arm_term(env, **arm_params), baseline)
torch.testing.assert_close(arm_term(env, guaranteed_when_standing=True, **arm_params),
                           torch.where(torch.tensor([False, True, True, False]), 1., baseline))
# Falling preserves the guaranteed reward, and disabling the option preserves the reward.
asset.data.body_com_pos_w[:, 0, 2] = scene.env_origins[:, 2] + .4
baseline = namespace['end_effector_pos_target_exp'](env, **arm_params)
torch.testing.assert_close(arm_term(env, guaranteed_when_standing=True, **arm_params),
                           torch.where(torch.tensor([False, True, True, False]), 1., baseline))
torch.testing.assert_close(arm_term(env, **arm_params), baseline)
# Partial resets resume tracking only for the selected environment; full resets clear all gates.
arm_term.reset(torch.tensor([1]))
torch.testing.assert_close(arm_term(env, guaranteed_when_standing=True, **arm_params),
                           torch.where(torch.tensor([False, False, True, False]), 1., baseline))
arm_term.reset()
torch.testing.assert_close(arm_term(env, guaranteed_when_standing=True, **arm_params), baseline)
# The gate uses the configured height rather than a hardcoded standing cutoff.
asset.data.body_com_pos_w[:, 0, 2] = scene.env_origins[:, 2] + .75
arm_params['upright_com_height'] = .8
baseline = namespace['end_effector_pos_target_exp'](env, **arm_params)
torch.testing.assert_close(arm_term(env, guaranteed_when_standing=True, **arm_params), baseline)
asset.data.body_com_pos_w[:, 0, 2] = scene.env_origins[:, 2] + .85
torch.testing.assert_close(arm_term(env, guaranteed_when_standing=True, **arm_params), torch.ones(4))
asset.data.body_com_pos_w.copy_(original_com)
print('PASS: optional guaranteed arm reward, inclusive/custom threshold, falling, and partial/full resets')

# Exercise real termination functions with the same transforms and strict cutoffs.
term_tree = ast.parse((path.parent / 'terminations.py').read_text())
exec(compile(ast.Module(body=[n for n in term_tree.body if isinstance(n, ast.FunctionDef)],
                       type_ignores=[]), 'terminations.py', 'exec'), namespace)
cfg = types.SimpleNamespace(name='robot', body_ids=list(range(4)))
for name in ('end_effector_deviation', 'com_deviation'):
    for threshold, expected in ((.1, False), (.01, True)):
        result = namespace[name](env, threshold, cfg)
        assert result.tolist() == [expected] * 4
    result = namespace[name](env, .1, cfg, upright_com_height=1.)
    assert result.tolist() == [False, True, True, True]

# A mixed batch uses absolute targets below threshold and relative targets above.
asset.data.body_com_pos_w[0, 0, 2] = .5
aligned = namespace['position_tracking_targets'](env, asset, torch.zeros(4, dtype=torch.long), target_ee)
torch.testing.assert_close(aligned[0], target_ee[0])
torch.testing.assert_close(aligned[1:] + scene.env_origins[1:, None], transform(target_ee)[1:])

# Execute the actual playback marker update without importing Isaac Sim.
play_tree = ast.parse((ROOT / 'scripts/rsl_rl/play.py').read_text())
visualization = next(n for n in play_tree.body if isinstance(n, ast.ClassDef) and n.name == 'KinematicsVisualization')
update = next(n for n in visualization.body if isinstance(n, ast.FunctionDef) and n.name == 'update_markers')
exec(compile(ast.Module(body=[update], type_ignores=[]), 'play.py', 'exec'), namespace)
class Markers:
    def visualize(self, translations, marker_indices):
        self.positions = translations.reshape(4, 5, 3)
env.cfg = types.SimpleNamespace(rewards=types.SimpleNamespace(
    com_tracking=types.SimpleNamespace(params={'upright_com_height': .65})))
wrapper = types.SimpleNamespace(env=types.SimpleNamespace(unwrapped=env), robot=asset,
    masses=asset.data.default_mass, body_ids=list(range(4)),
    reference_markers=Markers(), actual_markers=Markers(), marker_indices=torch.arange(5).repeat(4))
namespace['update_markers'](wrapper)
all_targets = torch.cat((target_com[:, None], target_ee), dim=1)
expected = namespace['position_tracking_targets'](env, asset, torch.zeros(4, dtype=torch.long), all_targets)
torch.testing.assert_close(wrapper.reference_markers.positions, expected + scene.env_origins[:, None])
torch.testing.assert_close(wrapper.actual_markers.positions[:, 1:], asset.data.body_link_pos_w)
print('PASS: relative terminations, posture failures, mixed height gating, and playback marker alignment')

# Feet can turn in place without changing the line between the ankles.
asset.data.body_com_pos_w[0, 0, 2] = .9
old_targets = namespace['position_tracking_targets'](env, asset, torch.zeros(4, dtype=torch.long), all_targets)
turn = torch.full((4,), .6)
turn_quat = namespace['_axis_quat'](turn, 2)
asset.data.body_link_quat_w[:, 2] = namespace['_quat_product'](turn_quat, asset.data.body_link_quat_w[:, 2])
new_targets = namespace['position_tracking_targets'](env, asset, torch.zeros(4, dtype=torch.long), all_targets)
left = asset.data.body_link_pos_w[:, 2] - scene.env_origins
relative = old_targets - left[:, None]
c, s = torch.cos(turn)[:, None], torch.sin(turn)[:, None]
expected = torch.stack((c * relative[..., 0] - s * relative[..., 1],
                        s * relative[..., 0] + c * relative[..., 1], relative[..., 2]), -1) + left[:, None]
torch.testing.assert_close(new_targets, expected, atol=2e-5, rtol=2e-5)
root_target = reference.get_root_state(None)[:, 3:7]
aligned_root = namespace['orientation_tracking_targets'](env, asset, torch.zeros(4, dtype=torch.long), root_target)
expected_root = namespace['_axis_quat'](angles + turn, 2)
torch.testing.assert_close((aligned_root * expected_root).sum(-1).abs(), torch.ones(4), atol=2e-5, rtol=2e-5)
asset.data.body_com_pos_w[0, 0, 2] = .5
aligned_root = namespace['orientation_tracking_targets'](env, asset, torch.zeros(4, dtype=torch.long), root_target)
torch.testing.assert_close(aligned_root[0], root_target[0])
print('PASS: foot rotation in place aligns position and root targets; height gating retained')

# Independently compose rotation matrices for nonzero leg poses and fixed offsets.
from scipy.spatial.transform import Rotation
joint_samples = torch.tensor([[.4, -.3, .7, 1.1, -.4, .2]]).expand(4, -1)
reference.get_joint_pos = lambda ids: joint_samples
actual_quat = namespace['_reference_left_foot_quat'](torch.zeros(4, dtype=torch.long), 'cpu')
matrix = Rotation.identity()
for axis, angle in (('y', .4), ('y', -.1749), ('x', -.3), ('z', .7),
                    ('y', .1749), ('y', 1.1), ('y', -.4), ('x', .2)):
    matrix = matrix * Rotation.from_euler(axis, angle)
expected_quat = torch.tensor(matrix.as_quat()[[3, 0, 1, 2]], dtype=torch.float32).expand(4, -1)
torch.testing.assert_close(actual_quat, expected_quat, atol=2e-6, rtol=2e-6)
print('PASS: reference foot rotation FK matches independent matrix composition')

# Reference playback explicitly opts in and refreshes poses before alignment.
reference_tree = ast.parse((ROOT / 'scripts/play_reference.py').read_text())
show = next(n for n in reference_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'show_reference_markers')
namespace['VisualizationMarkers'] = object
exec(compile(ast.Module(body=[show], type_ignores=[]), 'play_reference.py', 'exec'), namespace)
reference.dt = 1 / 120
refresh = []
env.num_envs = 4
env.unwrapped = env
env.sim = types.SimpleNamespace(physics_sim_view=types.SimpleNamespace(
    update_articulations_kinematic=lambda: refresh.append('fk')))
asset.update = lambda dt: refresh.append('buffers')
class ReferenceMarkers:
    def visualize(self, marker_indices, translations):
        self.positions = translations
markers = ReferenceMarkers()
show_markers = namespace['show_reference_markers']
show_markers(markers, all_targets, 0, scene.env_origins[0])
torch.testing.assert_close(markers.positions, all_targets[0] + scene.env_origins[0])
assert not refresh
for height in (.5, .9):
    asset.data.body_com_pos_w[0, 0, 2] = height
    show_markers(markers, all_targets, 0, scene.env_origins[0], env=env, foot_relative=True)
    expected = namespace['position_tracking_targets'](env, asset, torch.zeros(4, dtype=torch.long),
                                                     all_targets[:1].expand(4, -1, -1), .65)[0]
    torch.testing.assert_close(markers.positions, expected + scene.env_origins[0])
assert refresh == ['fk', 'buffers', 'fk', 'buffers']
print('PASS: optional reference markers share alignment, height gating and refreshed FK')

# Action-rate scaling uses current local CoM height, including the threshold.
original_com = asset.data.body_com_pos_w.clone()
asset.data.body_com_pos_w[:, 0, 2] = scene.env_origins[:, 2] + torch.tensor([.5, .75, .9, .4])
env.action_manager = types.SimpleNamespace(
    action=torch.tensor([[1., 2.], [2., 3.], [3., 4.], [4., 5.]]),
    prev_action=torch.tensor([[0., 1.], [0., 1.], [0., 1.], [0., 1.]]),
)
base_penalty = torch.tensor([2., 8., 18., 32.])
action_rate = namespace['standing_action_rate_l2']
torch.testing.assert_close(action_rate(env), base_penalty)
for multiplier in (0., .5, 3.):
    torch.testing.assert_close(action_rate(env, threshold=.75, multiplier=multiplier),
                               base_penalty * torch.tensor([1., multiplier, multiplier, 1.]))
torch.testing.assert_close(action_rate(env, threshold=1., multiplier=3.), base_penalty)
asset.data.body_com_pos_w[:, 0, 2] = scene.env_origins[:, 2] + .4
torch.testing.assert_close(action_rate(env, threshold=.75, multiplier=3.), base_penalty)
env.action_manager.prev_action.copy_(env.action_manager.action)
torch.testing.assert_close(action_rate(env, multiplier=3.), torch.zeros(4))
asset.data.body_com_pos_w.copy_(original_com)
print('PASS: standing action-rate scaling, inclusive/custom threshold, falling, and unchanged actions')
