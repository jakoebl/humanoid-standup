"""G1 asset converted from humanoid-mrl's MJCF, with mjlab's motor gains and nominal pose."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from pxr import Usd, UsdPhysics

from .constants import MOTOR_GROUPS, NATURAL_FREQ

# Thigh and shin capsules collide with the other links but pass through the ground.
GROUND_FILTERED_BODY_NAMES = ["left_hip_roll_link", "right_hip_roll_link", "left_knee_link", "right_knee_link"]


def spawn_without_leg_ground_contact(prim_path, cfg, *args, **kwargs):
    """Spawn the robot and filter the thigh and shin colliders against the colliders under ``/World/ground``."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, *args, **kwargs)
    stage = prim.GetStage()
    ground = [p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath("/World/ground")) if p.HasAPI(UsdPhysics.CollisionAPI)]
    for robot in sim_utils.find_matching_prims(prim_path):
        for name in GROUND_FILTERED_BODY_NAMES:
            UsdPhysics.FilteredPairsAPI.Apply(robot.GetChild(name)).CreateFilteredPairsRel().SetTargets(ground)
    return prim


def robot_cfg():
    cfg = G1_29DOF_CFG.copy()
    # Converted by scripts/convert_g1_mjcf.py; has collision capsules on the forearms and fixed hands.
    cfg.spawn.usd_path = str(Path(__file__).parent / "assets" / "g1_29dof.usd")
    cfg.spawn.activate_contact_sensors = True
    cfg.spawn.articulation_props.enabled_self_collisions = True
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312, ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363, ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    )
    cfg.actuators = {
        name: ImplicitActuatorCfg(
            joint_names_expr=expressions, effort_limit_sim=effort, velocity_limit_sim=velocity,
            stiffness=armature * NATURAL_FREQ**2,
            damping=4 * armature * NATURAL_FREQ, armature=armature,
        )
        for name, (expressions, armature, effort, velocity) in MOTOR_GROUPS.items()
    }
    return cfg
