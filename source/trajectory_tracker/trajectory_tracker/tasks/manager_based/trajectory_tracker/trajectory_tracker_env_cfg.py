# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.reference import REFERENCE, Reference

##
# Pre-defined configs
##

from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # isort:skip


##
# Scene definition
##


@configclass
class TrajectoryTrackerSceneCfg(InteractiveSceneCfg):
    """Ground-plane scene for G1 motion imitation."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # IMU attached to the robot pelvis
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        gravity_bias=(0.0, 0.0, 9.81),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(Reference.JOINT_NAMES),
        scale=0.25,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        deepmimic_state = ObsTerm(
            func=mdp.deepmimic_state,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_robot = EventTerm(
        func=mdp.reset_to_reference,
        mode="reset",
        params={"random_start": False},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    joint_position_tracking = RewTerm(
        func=mdp.joint_pos_target_exp,
        weight=0.65,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(Reference.JOINT_NAMES)), "scale": 4.0},
    )

    joint_velocity_tracking = RewTerm(
        func=mdp.joint_vel_target_exp,
        weight=0.10,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(Reference.JOINT_NAMES)), "scale": 0.1},
    )

    end_effector_tracking = RewTerm(
        func=mdp.end_effector_pos_target_exp,
        weight=0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=list(Reference.END_EFFECTOR_NAMES),
                preserve_order=True,
            ),
            "scale": 40.0,
        },
    )

    com_tracking = RewTerm(
        func=mdp.com_pos_target_exp,
        weight=0.10,
        params={"asset_cfg": SceneEntityCfg("robot"), "scale": 10.0},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reference_complete = DoneTerm(func=mdp.reference_complete)
    
    # The stand-up reference includes intentional torso/limb ground contact.
    ee_deviation = DoneTerm(
        func=mdp.end_effector_deviation,
        params={
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=list(Reference.END_EFFECTOR_NAMES), preserve_order=True,
            ),
        },
    )


##
# Environment configuration
##


@configclass
class TrajectoryTrackerEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: TrajectoryTrackerSceneCfg = TrajectoryTrackerSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = (REFERENCE.num_frames - 1) * REFERENCE.dt + self.decimation / 120.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # Isaac Lab integrates rewards over dt; expose DeepMimic per-step rewards.
        for term in vars(self.rewards).values():
            if isinstance(term, RewTerm):
                term.weight /= self.sim.dt * self.decimation
