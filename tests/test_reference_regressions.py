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
env=types.SimpleNamespace(scene=scene,num_envs=8,device='cpu',step_dt=4*ref.dt,episode_length_buf=torch.zeros(8,dtype=torch.long))
ids=torch.tensor([1,3,6])
torch.manual_seed(5)
events.reset_to_reference(env,ids)
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

# Exercise the actual Isaac Lab quaternion math without launching the simulator.
math_path = ROOT.parent / 'isaaclab/source/isaaclab/isaaclab/utils/math.py'
sys.modules['isaaclab.utils'] = types.ModuleType('isaaclab.utils')
math_utils = load('isaaclab.utils.math', math_path)
observations = load('tracker_test.observations', MDP/'observations.py')
names = observations.OBSERVATION_BODY_NAMES
assert len(names) == len(set(names)) == 30
assert not any('hand' in name or 'finger' in name for name in names)
# Interleave excluded bodies to detect accidental slicing or ordering mistakes.
all_names = ['left_hand_index_0_link', *reversed(names), 'right_hand_thumb_2_link']
count = len(all_names)
data = types.SimpleNamespace(
    root_link_pos_w=torch.tensor([[3., 4., 2.]]),
    root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
    body_com_pos_w=torch.arange(count * 3).reshape(1, count, 3).float(),
    body_link_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(1, count, 1),
    body_com_lin_vel_w=torch.arange(count * 3).reshape(1, count, 3).float() + 100,
    body_link_ang_vel_w=torch.arange(count * 3).reshape(1, count, 3).float() + 200,
)
asset = types.SimpleNamespace(data=data)
asset.find_bodies = lambda requested, preserve_order: ([all_names.index(n) for n in requested], requested)
scene = Scene(robot=asset)
scene.env_origins = torch.tensor([[3., 4., 0.]])
env = types.SimpleNamespace(scene=scene, step_dt=ref.dt, episode_length_buf=torch.tensor([0]))
state = observations.deepmimic_state(env)
assert state.shape == (1, 452)
assert state[0, 0] == 0 and state[0, 1] == 2
# Independent expected per-body packing at identity heading.
for i, name in enumerate(names):
    j = all_names.index(name)
    pose_start, vel_start = 2 + 9*i, 2 + 9*len(names) + 6*i
    expected_pose = torch.cat((data.body_com_pos_w[0,j]-data.root_link_pos_w[0], torch.tensor([0.,0.,1.,1.,0.,0.])))
    torch.testing.assert_close(state[0,pose_start:pose_start+9], expected_pose)
    torch.testing.assert_close(state[0,vel_start:vel_start+6], torch.cat((data.body_com_lin_vel_w[0,j],data.body_link_ang_vel_w[0,j])))
# Finger pose and velocity changes must have absolutely no direct observation effect.
for field in ('body_com_pos_w','body_link_quat_w','body_com_lin_vel_w','body_link_ang_vel_w'):
    getattr(data,field)[:,[0,count-1]] = float('nan')
torch.testing.assert_close(observations.deepmimic_state(env), state, rtol=0, atol=0)
# Reference targets are not read: phase is the only reference-dependent input.
old_get_root_state = ref.get_root_state
ref.get_root_state = lambda *_: (_ for _ in ()).throw(AssertionError('Reference target leaked into observations'))
torch.testing.assert_close(observations.deepmimic_state(env), state)
ref.get_root_state = old_get_root_state
# A common yaw and horizontal translation must leave the local state unchanged.
yaw = torch.tensor([[2**-0.5, 0., 0., 2**-0.5]])
shift = torch.tensor([[11., -7., 0.]])
data.root_link_pos_w = math_utils.quat_apply(yaw, data.root_link_pos_w) + shift
data.root_link_quat_w = yaw
body_yaw = yaw[:,None,:].expand(1,count,4)
data.body_com_pos_w = math_utils.quat_apply(body_yaw,data.body_com_pos_w) + shift[:,None,:]
data.body_link_quat_w = math_utils.quat_mul(body_yaw,data.body_link_quat_w)
data.body_com_lin_vel_w = math_utils.quat_apply(body_yaw,data.body_com_lin_vel_w)
data.body_link_ang_vel_w = math_utils.quat_apply(body_yaw,data.body_link_ang_vel_w)
torch.testing.assert_close(observations.deepmimic_state(env),state,atol=1e-4,rtol=1e-5)
print('PASS: 452-value DeepMimic layout, body ordering, finger exclusion, no target leakage, heading/translation invariance')
