import gymnasium as gym

for task, cfg in [("BeyondMimic", "BeyondMimicEnvCfg"), ("BeyondMimic-Play", "BeyondMimicPlayEnvCfg")]:
    gym.register(
        id=task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:BeyondMimicPPORunnerCfg",
        },
    )
