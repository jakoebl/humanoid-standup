"""Isaac Lab G1 asset with mjlab's motor gains and nominal pose."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from pxr import Usd, UsdGeom, UsdPhysics

from .constants import MOTOR_GROUPS, NATURAL_FREQ

# The robot spawns before the terrain, so the ground collider is referenced by path.
GROUND_COLLIDER = "/World/ground/terrain/GroundPlane/CollisionPlane"


def is_hand(name):
    return "_hand_" in name or name.endswith("_wrist_yaw_link")


def spawn_with_thigh_colliders(prim_path, cfg, *args, **kwargs):
    """Spawn the robot with humanoid-mrl's MJCF thigh capsules, colliding only with the hands."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, *args, **kwargs)
    for robot in sim_utils.find_matching_prims(prim_path):
        links = [robot.GetChild(f"{side}_hip_roll_link") for side in ("left", "right")]
        for link in links:
            capsule = UsdGeom.Capsule.Define(robot.GetStage(), link.GetPath().AppendChild("thigh_collision"))
            capsule.CreateRadiusAttr(0.05)
            capsule.CreateHeightAttr(0.2)
            capsule.CreateAxisAttr(UsdGeom.Tokens.z)
            capsule.CreatePurposeAttr(UsdGeom.Tokens.guide)
            capsule.AddTranslateOp().Set((0.02, 0.0, -0.1))
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        # Target colliders, not bodies: a filter on the pelvis, the articulation root, would filter every link.
        colliders = [
            collider.GetPath() for body in robot.GetChildren()
            if body.HasAPI(UsdPhysics.RigidBodyAPI) and not is_hand(body.GetName())
            for collider in Usd.PrimRange(body, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
            if collider.HasAPI(UsdPhysics.CollisionAPI)
        ]
        for link in links:
            UsdPhysics.FilteredPairsAPI.Apply(link).CreateFilteredPairsRel().SetTargets(
                [GROUND_COLLIDER] + [path for path in colliders if not path.HasPrefix(link.GetPath())]
            )
    return prim


def robot_cfg():
    cfg = G1_29DOF_CFG.copy()
    cfg.spawn.func = spawn_with_thigh_colliders
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
