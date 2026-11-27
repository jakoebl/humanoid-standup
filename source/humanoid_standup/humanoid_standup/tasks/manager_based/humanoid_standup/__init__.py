import gymnasium as gym

for task, cfg, agent in [
    ("Humanoid-Standup", "HumanoidStandupEnvCfg", "HumanoidStandupPPORunnerCfg"),
    ("Humanoid-Standup-Play", "HumanoidStandupPlayEnvCfg", "HumanoidStandupPPORunnerCfg"),
    ("Humanoid-Standup-Liedown", "HumanoidStandupLiedownEnvCfg", "HumanoidStandupLiedownPPORunnerCfg"),
    ("Humanoid-Standup-Liedown-Play", "HumanoidStandupLiedownPlayEnvCfg", "HumanoidStandupLiedownPPORunnerCfg"),
]:
    gym.register(
        id=task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:{agent}",
        },
    )
