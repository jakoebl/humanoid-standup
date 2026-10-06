from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Reference:
    """Loads a Unitree G1 MuJoCo-compatible CSV motion.

    Expected CSV format:

        Frame,
        root_translateX,
        root_translateY,
        root_translateZ,
        root_rotateX,
        root_rotateY,
        root_rotateZ,
        <29 G1 joint DOFs>

    Assumptions:
        - root translation is in centimeters
        - root rotation is in degrees
        - joint positions are in degrees
        - root rotation is XYZ Euler angles
        - joint ordering matches the CSV column ordering
    """

    JOINT_NAMES = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",

        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",

        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",

        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",

        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    NUM_JOINTS = 29
    END_EFFECTOR_NAMES = [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    KINEMATICS_COLUMNS = 3 * len(END_EFFECTOR_NAMES) + 3

    def __init__(
        self,
        path: str | Path,
        dt: float,
        device: str | torch.device = "cpu",
    ):
        self.path = Path(path)
        self.dt = dt
        if dt <= 0:
            raise ValueError("Reference dt must be positive.")
        self.device = torch.device(device)

        if not self.path.exists():
            raise FileNotFoundError(f"Motion file not found: {self.path}")

        if self.path.is_dir():
            self.paths = sorted(self.path.glob("*.csv"))
            if not self.paths:
                raise ValueError(f"No training CSV clips found in {self.path}")
            clips = [Reference(path, dt, device) for path in self.paths]
            for clip in clips:
                if clip.end_effector_pos is None or clip.com_pos is None:
                    raise ValueError(f"Training clip requires precomputed EE and COM targets: {clip.path}")
            # Differentiate each clip separately before joining storage.
            for name in ("frames", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
                         "joint_pos", "joint_vel", "end_effector_pos", "com_pos"):
                setattr(self, name, torch.cat([getattr(clip, name) for clip in clips]))
            lengths = [clip.num_frames for clip in clips]
            self.num_frames = sum(lengths)
        else:
            self.paths = [self.path]
            self._load()
            lengths = [self.num_frames]
        self.num_clips = len(lengths)
        self.max_clip_frames = max(lengths)
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.clip_offsets = self.clip_lengths.cumsum(0) - self.clip_lengths

    def _load(self) -> None:
        """Load and preprocess the CSV."""

        data = np.loadtxt(
            self.path,
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )

        if data.ndim == 1:
            data = data[None, :]
        if len(data) < 2:
            raise ValueError("Reference motion requires at least two frames.")
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite motion values in {self.path}")

        expected_columns = 1 + 3 + 3 + self.NUM_JOINTS

        if data.shape[1] < expected_columns:
            raise ValueError(
                f"Expected at least {expected_columns} columns, "
                f"got {data.shape[1]}"
            )
        kinematics_data = None
        if data.shape[1] >= expected_columns + self.KINEMATICS_COLUMNS:
            kinematics_data = data[:, expected_columns : expected_columns + self.KINEMATICS_COLUMNS]
        data = data[:, :expected_columns]

        # ---------------------------------------------------------
        # Raw CSV
        # ---------------------------------------------------------

        self.frames = torch.from_numpy(data[:, 0]).to(self.device)

        root_pos = data[:, 1:4]
        root_euler = data[:, 4:7]
        joint_pos = data[:, 7:36]

        # ---------------------------------------------------------
        # Unit conversion
        # ---------------------------------------------------------

        # cm -> m
        root_pos = root_pos / 100.0

        # degrees -> radians
        root_euler = np.deg2rad(root_euler)
        joint_pos = np.deg2rad(joint_pos)

        # ---------------------------------------------------------
        # Convert root Euler angles to quaternion
        # ---------------------------------------------------------

        root_quat = self._euler_xyz_to_quat(root_euler)

        # ---------------------------------------------------------
        # Compute velocities
        # ---------------------------------------------------------

        joint_vel = np.gradient(
            joint_pos,
            self.dt,
            axis=0,
        )

        root_lin_vel = np.gradient(
            root_pos,
            self.dt,
            axis=0,
        )

        root_ang_vel = self._compute_angular_velocity(
            root_quat,
            self.dt,
        )

        # ---------------------------------------------------------
        # Convert to torch
        # ---------------------------------------------------------

        self.root_pos = torch.from_numpy(root_pos).to(
            self.device,
            dtype=torch.float32,
        )

        # Isaac Lab convention: w, x, y, z
        self.root_quat = torch.from_numpy(root_quat).to(
            self.device,
            dtype=torch.float32,
        )

        self.root_lin_vel = torch.from_numpy(root_lin_vel).to(
            self.device,
            dtype=torch.float32,
        )

        self.root_ang_vel = torch.from_numpy(root_ang_vel).to(
            self.device,
            dtype=torch.float32,
        )

        self.joint_pos = torch.from_numpy(joint_pos).to(
            self.device,
            dtype=torch.float32,
        )

        self.joint_vel = torch.from_numpy(joint_vel).to(
            self.device,
            dtype=torch.float32,
        )

        if kinematics_data is not None:
            self.end_effector_pos = torch.from_numpy(
                kinematics_data[:, : 3 * len(self.END_EFFECTOR_NAMES)].reshape(-1, len(self.END_EFFECTOR_NAMES), 3)
            ).to(self.device, dtype=torch.float32)
            self.com_pos = torch.from_numpy(kinematics_data[:, -3:]).to(self.device, dtype=torch.float32)
        else:
            self.end_effector_pos = None
            self.com_pos = None

        self.num_frames = self.joint_pos.shape[0]

    # ------------------------------------------------------------------
    # Quaternion utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _euler_xyz_to_quat(euler: np.ndarray) -> np.ndarray:
        """Convert XYZ Euler angles to quaternions.

        Input:
            [..., 3] containing [roll, pitch, yaw]

        Output:
            [..., 4] containing [w, x, y, z]
        """

        roll = euler[:, 0]
        pitch = euler[:, 1]
        yaw = euler[:, 2]

        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.stack([w, x, y, z], axis=-1).astype(np.float32)

    @staticmethod
    def _compute_angular_velocity(
        quat: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """World angular velocity from shortest-arc quaternion logarithms.

        q_next * inverse(q_previous) expresses the rotation in world axes.
        Average adjacent interval velocities at interior samples.
        """
        q = quat.astype(np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        a, b = q[1:], q[:-1]
        scalar = np.sum(a * b, axis=1, keepdims=True)
        vector = -a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] - np.cross(a[:, 1:], b[:, 1:])
        vector *= np.where(scalar < 0, -1.0, 1.0)
        scalar = np.abs(scalar)
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        angle = 2.0 * np.arctan2(norm, scalar)
        scale = np.divide(angle, norm, out=np.full_like(norm, 2.0), where=norm > 1e-12)
        intervals = vector * scale / dt
        velocity = np.empty((len(q), 3), dtype=np.float32)
        velocity[0], velocity[-1] = intervals[0], intervals[-1]
        velocity[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
        return velocity

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_frame(self, frame: int | torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a reference state at one or more frame indices."""

        return {
            "root_pos": self.root_pos[frame],
            "root_quat": self.root_quat[frame],
            "root_lin_vel": self.root_lin_vel[frame],
            "root_ang_vel": self.root_ang_vel[frame],
            "joint_pos": self.joint_pos[frame],
            "joint_vel": self.joint_vel[frame],
        }

    def get_frame_ids(self, step_ids: torch.Tensor, step_dt: float, start_frames=0) -> torch.Tensor:
        """Sample this noncyclic motion without wrapping back to its beginning."""
        frames = start_frames + torch.floor(step_ids.to(torch.float64) * (step_dt / self.dt) + 1e-8).long()
        return frames.clamp(0, self.num_frames - 1)

    def frame_ids_for_env(self, env) -> torch.Tensor:
        offsets, lengths = self.clip_bounds_for_env(env)
        frames = getattr(env, "reference_start_frames", 0) + torch.floor(
            env.episode_length_buf.to(torch.float64) * (env.step_dt / self.dt) + 1e-8
        ).long()
        return offsets + torch.minimum(frames.clamp(min=0), lengths - 1)

    def clip_bounds_for_env(self, env):
        clip_ids = getattr(env, "reference_clip_ids", torch.zeros_like(env.episode_length_buf))
        return (self.clip_offsets.to(clip_ids.device)[clip_ids],
                self.clip_lengths.to(clip_ids.device)[clip_ids])

    def phase_for_env(self, env) -> torch.Tensor:
        frames = getattr(env, "reference_start_frames", 0) + env.episode_length_buf * (env.step_dt / self.dt)
        _, lengths = self.clip_bounds_for_env(env)
        return (frames / (lengths - 1)).clamp(0.0, 1.0)

    def get_joint_pos(self, frame: torch.Tensor) -> torch.Tensor:
        return self.joint_pos[frame]

    def get_joint_vel(self, frame: torch.Tensor) -> torch.Tensor:
        return self.joint_vel[frame]

    def get_end_effector_pos(self, frame: torch.Tensor) -> torch.Tensor:
        if self.end_effector_pos is None:
            raise RuntimeError("Reference CSV does not contain precomputed end-effector positions.")
        return self.end_effector_pos[frame]

    def get_com_pos(self, frame: torch.Tensor) -> torch.Tensor:
        if self.com_pos is None:
            raise RuntimeError("Reference CSV does not contain a precomputed center-of-mass trajectory.")
        return self.com_pos[frame]

    def get_root_state(self, frame: torch.Tensor) -> torch.Tensor:
        """Return [pos, quat, lin_vel, ang_vel]."""

        return torch.cat(
            [
                self.root_pos[frame],
                self.root_quat[frame],
                self.root_lin_vel[frame],
                self.root_ang_vel[frame],
            ],
            dim=-1,
        )




REFERENCE = Reference(path=Path(__file__).resolve().parents[7] / "data/processed/stand_up_lying_V2.csv", dt=1.0 / 120.0)
