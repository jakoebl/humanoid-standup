"""Multi-clip variants share all robot and learning settings with BeyondMimic."""
from pathlib import Path

from isaaclab.utils import configclass
from ..beyond_mimic.env_cfg import BeyondMimicEnvCfg, BeyondMimicPlayEnvCfg
from ..beyond_mimic.agents import BeyondMimicPPORunnerCfg
from .commands import MulticlipMotionCommandCfg


def multiclip_command(single_clip_command):
    values = vars(single_clip_command).copy()
    values.pop("class_type", None)
    return MulticlipMotionCommandCfg(
        **values,
        motion_directory=str(Path(__file__).resolve().parents[6] / "data/beyondmimic"),
    )


@configclass
class MimicMulticlipEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = multiclip_command(self.commands.motion)


@configclass
class MimicMulticlipPlayEnvCfg(BeyondMimicPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = multiclip_command(self.commands.motion)
        self.commands.motion.play_clip_index = 0


@configclass
class MimicMulticlipPPORunnerCfg(BeyondMimicPPORunnerCfg):
    experiment_name = "mimic_multiclip"
