"""CPU regression checks without starting Isaac Sim."""
import ast
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / 'source/humanoid_standup/humanoid_standup/tasks/manager_based/humanoid_standup/mdp'
for path in ROOT.rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

package = types.ModuleType('tracker_test')
package.__path__ = [str(MDP)]
sys.modules['tracker_test'] = package
reference = load('tracker_test.reference', MDP/'reference.py')
ref = reference.REFERENCE
# Constant world-axis rotation, with an initially tilted body and sign flips.
dt = 0.01
axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
t = np.arange(101)*dt
rot = np.column_stack((np.cos(t/2), np.sin(t/2)[:,None]*axis))
base = np.array([np.cos(.4), np.sin(.4), 0., 0.])
q = np.column_stack((rot[:,0]*base[0]-rot[:,1:]@base[1:], rot[:,:1]*base[1:]+base[0]*rot[:,1:]+np.cross(rot[:,1:],base[1:])))
q[::2] *= -1
np.testing.assert_allclose(ref._compute_angular_velocity(q, dt), np.broadcast_to(axis, (101,3)), atol=2e-5)
np.testing.assert_allclose(ref._compute_angular_velocity(np.tile(base,(3,1)),dt), 0, atol=1e-6)
steps = torch.tensor([0,1,999999])
starts = torch.tensor([5,10,ref.num_frames-2])
assert ref.get_frame_ids(steps,4*ref.dt,starts).tolist() == [5,14,ref.num_frames-1]
# Permit manager observation shape inference before the first reset.
env = types.SimpleNamespace(episode_length_buf=torch.zeros(3,dtype=torch.long), step_dt=4*ref.dt)
assert torch.equal(ref.frame_ids_for_env(env),env.episode_length_buf)
assert torch.equal(ref.phase_for_env(env),torch.zeros(3))

# Minimal asset interfaces exercise the real reset and termination functions.
class SceneEntityCfg:
    def __init__(self, name): self.name=name
managers = types.ModuleType('isaaclab.managers'); managers.SceneEntityCfg=SceneEntityCfg
sys.modules['isaaclab']=types.ModuleType('isaaclab')
sys.modules['isaaclab.managers']=managers
events=load('tracker_test.events', MDP/'events.py')
terms=load('tracker_test.terminations', MDP/'terminations.py')
class Robot:
    device='cpu'
    joint_names=list(reversed(ref.JOINT_NAMES))
    data=types.SimpleNamespace(default_joint_pos=torch.zeros(8,29),default_joint_vel=torch.zeros(8,29))
    def write_joint_state_to_sim(self,p,v,env_ids): self.p,self.v,self.ids=p,v,env_ids
    def write_root_state_to_sim(self,r,env_ids): self.r=r
class Scene(dict): pass
robot=Robot(); scene=Scene(robot=robot); scene.env_origins=torch.arange(24).reshape(8,3).float()
robot.data.soft_joint_pos_limits = torch.tensor([-float('inf'), float('inf')]).expand(8, 29, 2)
env=types.SimpleNamespace(scene=scene,num_envs=8,device='cpu',step_dt=4*ref.dt,episode_length_buf=torch.zeros(8,dtype=torch.long))
ids=torch.tensor([1,3,6])
torch.manual_seed(5)
events.reset_to_reference(env,ids,joint_pos_noise=0,joint_vel_noise=0)
f=env.reference_start_frames[ids]
assert f.unique().numel()>1
assert (env.reference_start_frames[torch.tensor([0,2,4,5,7])]==0).all()
torch.testing.assert_close(robot.p,ref.joint_pos[f].flip(1))
torch.testing.assert_close(robot.v,ref.joint_vel[f].flip(1))
torch.testing.assert_close(robot.r[:,:3]-scene.env_origins[ids],ref.root_pos[f])
torch.testing.assert_close(robot.r[:,3:],ref.get_root_state(f)[:,3:])
events.reset_to_reference(env,ids,random_start=False)
assert (env.reference_start_frames[ids]==0).all()
env.reference_start_frames[ids]=ref.num_frames-2
env.episode_length_buf[ids]=1
assert terms.reference_complete(env)[ids].all()
assert (ref.frame_ids_for_env(env)[ids]==ref.num_frames-1).all()
print('PASS: Python syntax, quaternion world velocities/sign invariance, noncyclic sampling, pre-reset phase, synchronized partial RSI, deterministic reset, clip completion')

# Only encoders and local inertial signals are available; world/reference reads fail.
observations = load('tracker_test.observations', MDP/'observations.py')
names = list(ref.JOINT_NAMES)
all_names = ['finger_joint', *reversed(names)]
count = len(all_names)
data = types.SimpleNamespace(
    joint_pos=torch.arange(2*count).reshape(2,count).float(),
    joint_vel=torch.arange(2*count).reshape(2,count).float() + 100,
    root_ang_vel_b=torch.tensor([[1.,2.,3.], [-1.,-2.,-3.]]),
    projected_gravity_b=torch.tensor([[0.,0.,-1.], [1.,0.,0.]]),
)
asset = types.SimpleNamespace(data=data)
asset.find_joints = lambda requested, preserve_order: ([all_names.index(n) for n in requested], requested)
env = types.SimpleNamespace(scene=Scene(robot=asset))
state = observations.proprioceptive_state(env)
assert state.shape == (2,64)
for i, name in enumerate(names):
    j = all_names.index(name)
    torch.testing.assert_close(state[:,i],data.joint_pos[:,j])
    torch.testing.assert_close(state[:,29+i],data.joint_vel[:,j])
torch.testing.assert_close(state[:,58:61],data.root_ang_vel_b)
torch.testing.assert_close(state[:,61:64],data.projected_gravity_b)
data.joint_pos[:,0] = float('nan')
data.joint_vel[:,0] = float('nan')
torch.testing.assert_close(observations.proprioceptive_state(env),state,rtol=0,atol=0)
print('PASS: proprioceptive layout, joint ordering, finger exclusion, no world/reference inputs')


# COM failure uses mass weighting, environment-local coordinates, and RSI frame offsets.
com_scene = Scene()
com_scene.env_origins = torch.tensor([[0., 0., 0.], [10., -4., 2.], [-8., 3., 1.], [2., 5., -3.]])
com_env = types.SimpleNamespace(
    scene=com_scene, step_dt=ref.dt,
    episode_length_buf=torch.tensor([0, 1, 2, 3]),
    reference_start_frames=torch.tensor([5, 10, 15, 20]),
)
original_com = ref.com_pos
try:
    # Exactly representable targets make the strict threshold boundary deterministic.
    ref.com_pos = torch.zeros_like(original_com)
    ref.com_pos[torch.tensor([5, 11, 17, 23], device=ref.device)] = 2.0
    target = torch.full((4, 3), 2.0)
    offsets = torch.tensor([[0., 0., 0.], [.5, 0., 0.], [0., 0., .75], [.375, .375, 0.]])
    center = target + offsets + com_scene.env_origins
    # Unequal masses: weighted offsets cancel; an unweighted mean would fail.
    body_com = torch.stack((center + torch.tensor([3., 0., 0.]), center - torch.tensor([1., 0., 0.])), dim=1)
    com_scene['robot'] = types.SimpleNamespace(
        device='cpu', data=types.SimpleNamespace(
            body_com_pos_w=body_com, default_mass=torch.tensor([[1., 3.]]).expand(4, -1),
        ),
    )
    result = terms.com_deviation(com_env, threshold=.5, asset_cfg=SceneEntityCfg('robot'),
                                 upright_com_height=float('inf'))
    assert result.dtype == torch.bool and result.shape == (4,)
    assert result.tolist() == [False, False, True, True]
finally:
    ref.com_pos = original_com
print('PASS: COM deviation mass weighting, environment origins, RSI offsets, strict threshold and 3D distance')

# Playback must select the requested motion before config construction, preserving
# the object already imported by managers. Extract main to avoid launching Kit.
import tempfile
playback_tree = ast.parse((ROOT / 'scripts/play_reference.py').read_text())
playback_main = next(node for node in playback_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
class ConfigReached(Exception):
    pass
with tempfile.TemporaryDirectory() as directory:
    clip = Path(directory) / 'other_clip.csv'
    motion = np.loadtxt(ref.path, delimiter=',', skiprows=1)[:3].copy()
    motion[:, 1] += 100.0
    np.savetxt(clip, motion, delimiter=',', header=ref.path.open().readline().strip(), comments='')
    selected = reference.Reference(ref.path, ref.dt)
    shared = selected
    def inspect_config(*args, **kwargs):
        assert selected is shared
        assert selected.path == clip and selected.num_frames == 3
        np.testing.assert_allclose(selected.root_pos[:, 0].numpy(), motion[:, 1] / 100, atol=1e-6)
        assert selected.com_pos.shape == (3, 3)
        raise ConfigReached
    namespace = dict(REFERENCE=selected, args_cli=types.SimpleNamespace(
        csv=clip, task='Humanoid-Standup', device='cpu', num_envs=1, disable_fabric=False,
    ), parse_env_cfg=inspect_config)
    exec(compile(ast.Module(body=[playback_main], type_ignores=[]), 'play_reference.py', 'exec'), namespace)
    try:
        namespace['main']()
    except ConfigReached:
        pass
    else:
        raise AssertionError('Playback did not construct its config')
print('PASS: playback CSV selects motion in place before environment configuration')

# Mixed lengths must preserve clip boundaries and independent derivatives.
with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    for name, count, offset in [('a', 3, 0), ('b', 7, 10000)]:
        motion = np.zeros((count, 51), dtype=np.float32)
        motion[:, 0] = np.arange(count)
        motion[:, 1] = offset + np.arange(count)
        motion[:, 7:36] = offset
        motion[:, 36:] = offset
        np.savetxt(directory / f'{name}.csv', motion, delimiter=',', header='test', comments='')
    multi = reference.Reference(directory, .01)
    assert multi.clip_lengths.tolist() == [3, 7]
    assert multi.clip_offsets.tolist() == [0, 3]
    assert multi.max_clip_frames == 7
    torch.testing.assert_close(multi.root_lin_vel[:, 0], torch.ones(10), atol=.001, rtol=0)
    torch.testing.assert_close(multi.joint_vel, torch.zeros(10, 29))
    mixed = types.SimpleNamespace(
        episode_length_buf=torch.tensor([2, 2, 999]), step_dt=.01,
        reference_start_frames=torch.tensor([0, 0, 1]), reference_clip_ids=torch.tensor([0, 1, 1]),
    )
    assert multi.frame_ids_for_env(mixed).tolist() == [2, 5, 9]
    torch.testing.assert_close(multi.phase_for_env(mixed), torch.tensor([1., 1/3, 1.]))
    targets = multi.get_end_effector_pos(multi.frame_ids_for_env(mixed))
    assert targets[:, 0, 0].tolist() == [0., 10000., 10000.]
    events.REFERENCE = terms.REFERENCE = multi
    env = types.SimpleNamespace(scene=scene,num_envs=8,device='cpu',step_dt=.01,
                                episode_length_buf=torch.zeros(8,dtype=torch.long))
    try:
        torch.manual_seed(12)
        seen = set()
        for _ in range(20):
            events.reset_to_reference(env, ids, joint_pos_noise=0, joint_vel_noise=0)
            clips = env.reference_clip_ids[ids]
            seen.update(clips.tolist())
            starts = env.reference_start_frames[ids]
            assert (starts < multi.clip_lengths[clips] - 1).all()
            frames = multi.frame_ids_for_env(env)[ids]
            torch.testing.assert_close(robot.p, multi.joint_pos[frames].flip(1))
            torch.testing.assert_close(robot.r[:, :3] - scene.env_origins[ids], multi.root_pos[frames])
            assert env.reference_clip_ids[[0, 2, 4, 5, 7]].tolist() == [0]*5
        assert seen == {0, 1}
        events.reset_to_reference(env, ids, random_start=False, joint_pos_noise=0, joint_vel_noise=0)
        assert (env.reference_start_frames[ids] == 0).all()
        env.episode_length_buf[ids] = multi.clip_lengths[env.reference_clip_ids[ids]] - 1
        assert terms.reference_complete(env)[ids].all()
        assert not terms.reference_complete(env)[[0, 2, 4, 5, 7]].any()
    finally:
        events.REFERENCE = terms.REFERENCE = ref
    empty = directory / 'empty'
    empty.mkdir()
    try:
        reference.Reference(empty, .01)
    except ValueError as error:
        assert 'No training CSV' in str(error)
    else:
        raise AssertionError('Empty training directory accepted')
    np.savetxt(directory / 'missing_targets.csv', np.zeros((3, 36)), delimiter=',', header='test')
    try:
        reference.Reference(directory, .01)
    except ValueError as error:
        assert 'missing_targets.csv' in str(error)
    else:
        raise AssertionError('Training clip without kinematics accepted')

training = reference.Reference(ROOT / 'data/training', ref.dt)
assert training.num_clips >= 1
print(f'PASS: multi-clip boundaries, velocities, targets, partial resets, completion, validation; {training.num_clips} training clips loaded')
