"""Isaac Lab G1 asset with mjlab's motor gains and nominal pose."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from .constants import MOTOR_GROUPS, NATURAL_FREQ


def robot_cfg():
    cfg = G1_29DOF_CFG.copy()
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
    hands = cfg.actuators["hands"]
    cfg.actuators = {
        name: ImplicitActuatorCfg(
            joint_names_expr=expressions, effort_limit_sim=effort, velocity_limit_sim=velocity,
            stiffness=armature * NATURAL_FREQ**2,
            damping=4 * armature * NATURAL_FREQ, armature=armature,
        )
        for name, (expressions, armature, effort, velocity) in MOTOR_GROUPS.items()
    }
    cfg.actuators["hands"] = hands
    return cfg
