"""Robot ordering and control constants shared by the task and motion exporter."""

JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")] + [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
]
BODY_NAMES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
CONTACT_BODY_NAMES = ["pelvis"] + [
    "torso_link" if joint == "waist_pitch_joint" else joint.replace("_joint", "_link")
    for joint in JOINT_NAMES
]
VELOCITY_RANGE = {
    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
}
# Groups: joint expressions, armature, effort limit, velocity limit.
MOTOR_GROUPS = {
    "small": ([".*_elbow_joint", ".*_shoulder_.*_joint", ".*_wrist_roll_joint"], 0.003609725, 25.0, 37.0),
    "hip_yaw_pitch": ([".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"], 0.01017752004132231, 88.0, 32.0),
    "hip_roll_knee": ([".*_hip_roll_joint", ".*_knee_joint"], 0.025101925, 139.0, 20.0),
    "wrist": ([".*_wrist_pitch_joint", ".*_wrist_yaw_joint"], 0.00425, 5.0, 22.0),
    "waist": (["waist_pitch_joint", "waist_roll_joint"], 0.00721945, 50.0, 37.0),
    "ankle": ([".*_ankle_pitch_joint", ".*_ankle_roll_joint"], 0.00721945, 50.0, 37.0),
}
NATURAL_FREQ = 10 * 2 * 3.1415926535
ACTION_SCALE = {
    expression: 0.25 * effort / (armature * NATURAL_FREQ**2)
    for expressions, armature, effort, _ in MOTOR_GROUPS.values()
    for expression in expressions
}
