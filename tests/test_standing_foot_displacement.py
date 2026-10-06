"""CPU regressions for standing foot anchors; no Isaac Sim startup required."""
import ast
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / 'source/humanoid_standup/humanoid_standup/tasks/manager_based/humanoid_standup/mdp'


class ManagerTermBase:
    def __init__(self, cfg, env):
        pass


def test_standing_foot_displacement():
    reward_tree = ast.parse((MDP / 'rewards.py').read_text())
    tracking_tree = ast.parse((MDP / 'position_tracking.py').read_text())
    nodes = [node for node in tracking_tree.body if isinstance(node, ast.FunctionDef)
             and node.name == '_whole_body_com']
    nodes += [node for node in reward_tree.body if isinstance(node, ast.ClassDef)
              and node.name == 'standing_foot_displacement']
    namespace = dict(torch=torch, ManagerTermBase=ManagerTermBase, ManagerBasedRLEnv=object,
                     SceneEntityCfg=object, Articulation=object, UPRIGHT_COM_HEIGHT=.65)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MDP / 'rewards.py'), 'exec'), namespace)

    class Scene(dict):
        env_origins = torch.tensor([[0., 0., 0.], [10., -5., 2.], [-2., 7., -1.]])

    scene = Scene()
    asset = types.SimpleNamespace(device='cpu', data=types.SimpleNamespace(
        body_link_pos_w=torch.zeros(3, 4, 3) + scene.env_origins[:, None],
        body_com_pos_w=scene.env_origins[:, None].clone(), default_mass=torch.ones(3, 1)))
    scene['robot'] = asset
    env = types.SimpleNamespace(scene=scene, num_envs=3)
    cfg = types.SimpleNamespace(name='robot', body_ids=[2, 3])
    term = namespace['standing_foot_displacement'](types.SimpleNamespace(params={'asset_cfg': cfg}), env)
    feet = asset.data.body_link_pos_w
    com = asset.data.body_com_pos_w

    # Capture asynchronously, with equality at the threshold and nonzero origins.
    com[:, 0, 2] += torch.tensor([.65, .5, .8])
    torch.testing.assert_close(term(env, cfg), torch.zeros(3))
    feet[:, 2, 0] += .3
    feet[:, 3, 2] += .4
    torch.testing.assert_close(term(env, cfg), torch.tensor([.7, 0., .7]))
    com[1, 0, 2] = scene.env_origins[1, 2] + .65
    torch.testing.assert_close(term(env, cfg), torch.tensor([.7, 0., .7]))

    # Anchors remain fixed through falling and standing again; costs aren't cumulative.
    com[0, 0, 2] = .2
    torch.testing.assert_close(term(env, cfg), torch.tensor([.7, 0., .7]))
    com[0, 0, 2] = .9
    torch.testing.assert_close(term(env, cfg), torch.tensor([.7, 0., .7]))
    feet[0, 2, 0] -= .3
    feet[0, 3, 2] -= .4
    torch.testing.assert_close(term(env, cfg), torch.tensor([0., 0., .7]))

    # Reset one episode without disturbing another episode's anchors.
    term.reset(torch.tensor([2]))
    com[2, 0, 2] = scene.env_origins[2, 2] + .2
    feet[2, 2, 0] += 1.
    torch.testing.assert_close(term(env, cfg), torch.zeros(3))
    com[2, 0, 2] = scene.env_origins[2, 2] + .8
    torch.testing.assert_close(term(env, cfg), torch.zeros(3))
    feet[1, 3, 1] += .2
    feet[2, 2, 0] += .1
    torch.testing.assert_close(term(env, cfg), torch.tensor([0., .2, .1]), atol=1e-6, rtol=1e-5)
    term.reset()
    torch.testing.assert_close(term(env, cfg), torch.zeros(3))


if __name__ == '__main__':
    test_standing_foot_displacement()
    print('PASS: standing foot distances, first-crossing anchors, falling, and episode resets')
