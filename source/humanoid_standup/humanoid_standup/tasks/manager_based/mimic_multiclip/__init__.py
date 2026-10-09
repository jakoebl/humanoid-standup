"""Registration for multi-clip training and deterministic selected-clip playback."""
import gymnasium as gym

for task, config in (
    ("Mimic-Multiclip", "MimicMulticlipEnvCfg"),
    ("Mimic-Multiclip-Play", "MimicMulticlipPlayEnvCfg"),
):
    gym.register(
        id=task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:MimicMulticlipPPORunnerCfg",
        },
    )
