from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from isaaclab.utils import configclass
import torch

class MultiCritic(MLPModel):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int = 1,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        num_critics = obs["clip"].shape[-1]
        self.mlps = torch.nn.ModuleList([MLP(self._get_latent_dim(), output_dim, hidden_dims, activation) for _ in range(num_critics)])
        del self.mlp


    def forward(self, obs: TensorDict, masks = None, hidden_state = None) -> torch.Tensor:
        # find index from one-hot encoding
        clip = obs["clip"].argmax(-1)
        # normalizing observations
        latent = self.get_latent(obs)

        values = torch.empty(latent.shape[0], 1, dtype=latent.dtype, device=latent.device)

        # mask for each clip index
        for i, mlp in enumerate(self.mlps):
            mask = clip == i
            values[mask] = mlp(latent[mask])
        return values


@configclass
class MultiCriticCfg(RslRlMLPModelCfg):
    class_name: str = "humanoid_standup.tasks.manager_based.humanoid_standup.critic:MultiCritic"
