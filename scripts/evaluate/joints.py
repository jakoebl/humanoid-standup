"""Plot one rollout of every joint with its position and torque limits."""

import csv
from html import escape
from pathlib import Path
import re
from collections.abc import Mapping

import numpy as np


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _clip_bounds(action_term, joint_names):
    clip = _as_numpy(getattr(action_term, "_clip", np.array([[-np.inf, np.inf]])))
    bound_axes = [axis for axis, size in enumerate(clip.shape) if size == 2]
    if bound_axes:
        clip = np.moveaxis(clip, bound_axes[-1], -1)
        while clip.ndim > 2:
            clip = clip[0]
    configured = getattr(getattr(action_term, "cfg", None), "clip", None)
    if clip.shape == (len(joint_names), 2) and len(joint_names) != 1:
        return clip
    if not isinstance(configured, Mapping):
        from humanoid_standup.tasks.manager_based.beyond_mimic.constants import JOINT_POSITION_LIMITS
        configured = JOINT_POSITION_LIMITS
    bounds = np.full((len(joint_names), 2), [-np.inf, np.inf], dtype=np.float64)
    for expression, values in configured.items():
        pattern = re.compile(expression)
        for index, name in enumerate(joint_names):
            if pattern.fullmatch(str(name)):
                bounds[index] = values
    if not np.isfinite(bounds).all():
        raise ValueError(f"No per-joint clip bounds found for action joints: {joint_names}.")
    return bounds


def _plot_svg(path, title, ylabel, times, values, lower, upper, lower_label, upper_label, series=None):
    series = series or []
    values = np.asarray(values, dtype=np.float64)
    lower, upper = float(lower), float(upper)
    finite_series = [np.asarray(item[1], dtype=np.float64) for item in series]
    all_values = np.concatenate([values, *(item[np.isfinite(item)] for item in finite_series)])
    plot_min = min(lower, float(all_values.min()))
    plot_max = max(upper, float(all_values.max()))
    if not np.isfinite([plot_min, plot_max]).all():
        raise ValueError(f"Cannot plot nonfinite limits or samples for {title}.")
    if plot_min == plot_max:
        padding = max(abs(plot_min) * .05, 1.)
    else:
        padding = (plot_max - plot_min) * .05
    plot_min -= padding
    plot_max += padding
    left, top, width, height = 90., 55., 820., 365.
    time_max = max(float(times[-1]), 1e-12)
    x = lambda time: left + width * float(time) / time_max
    y = lambda value: top + height * (plot_max - float(value)) / (plot_max - plot_min)
    def points_for(series_values):
        return " ".join(
            f"{x(time):.2f},{y(value):.2f}"
            for time, value in zip(times, series_values)
            if np.isfinite(value)
        )

    points = points_for(values)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 500" role="img">',
        f"<title>{escape(title)}</title>",
        '<rect width="1000" height="500" fill="white"/>',
        '<g font-family="sans-serif" font-size="14" fill="#222">',
        f'<text x="500" y="28" text-anchor="middle" font-size="20">{escape(title)}</text>',
    ]
    for fraction in np.linspace(0, 1, 6):
        value = plot_min + fraction * (plot_max - plot_min)
        svg.append(f'<path d="M {left} {y(value):.2f} h {width}" stroke="#e5e5e5"/>')
        svg.append(f'<text x="{left - 10}" y="{y(value) + 5:.2f}" text-anchor="end">{value:.4g}</text>')
    for value, label, color in ((lower, lower_label, "#cc4422"), (upper, upper_label, "#cc4422")):
        svg.extend([
            f'<path d="M {left} {y(value):.2f} h {width}" stroke="{color}" stroke-dasharray="7 5"/>',
            f'<text x="920" y="{y(value) + 5:.2f}" fill="{color}">{escape(label)}</text>',
        ])
    svg.extend([
        f'<polyline points="{points}" fill="none" stroke="#4477aa" stroke-width="2"/>',
        f'<text x="{left:.2f}" y="48" fill="#4477aa">actual joint position</text>',
        f'<path d="M {left} {top} V {top + height} H {left + width}" fill="none" stroke="#222"/>',
    ])
    for offset, (label, series_values, color) in enumerate(series):
        series_points = points_for(series_values)
        svg.append(f'<polyline points="{series_points}" fill="none" stroke="{color}" '
                   f'stroke-width="2" stroke-dasharray="5 3"/>')
        svg.append(f'<text x="{left + offset * 145:.2f}" y="48" fill="{color}">{escape(label)}</text>')
    for fraction in np.linspace(0, 1, 6):
        svg.append(f'<text x="{x(time_max * fraction):.2f}" y="445" text-anchor="middle">'
                   f'{time_max * fraction:.4g}</text>')
    svg.extend([
        '<text x="500" y="480" text-anchor="middle">Time (s)</text>',
        f'<text transform="translate(22 240) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>',
        '</g></svg>',
    ])
    Path(path).write_text("\n".join(svg), encoding="utf-8")


class JointPlots:
    name = "joints"

    def __init__(self, step_dt):
        if not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Joint plots require a finite positive timestep.")
        self.step_dt = float(step_dt)
        self.last_frame = -1
        self.times = []
        self.positions = []
        self.torques = []
        self.unclipped_actions = []
        self.clipped_actions = []

    def prepare(self, raw):
        robot = raw.scene["robot"]
        self.joint_indices = np.asarray(
            [index for index, name in enumerate(robot.joint_names)
             if not any(term in name.lower() for term in ("wrist", "hand", "finger"))],
            dtype=np.int64,
        )
        if self.joint_indices.size == 0:
            raise ValueError("Joint evaluation found no non-hand joints.")
        self.joint_names = [robot.joint_names[index] for index in self.joint_indices]
        data = robot.data
        self.position_limits = _as_numpy(data.soft_joint_pos_limits[0])[self.joint_indices]
        effort_limits = _as_numpy(data.joint_effort_limits[0])
        if effort_limits.ndim == 1:
            self.torque_limits = np.abs(effort_limits)[self.joint_indices]
        else:
            self.torque_limits = np.max(np.abs(effort_limits), axis=-1)[self.joint_indices]
        if self.position_limits.shape != (len(self.joint_names), 2):
            raise ValueError("Joint position limits do not match the robot joints.")
        if self.torque_limits.shape != (len(self.joint_names),):
            raise ValueError("Joint effort limits do not match the robot joints.")
        if not np.isfinite(self.position_limits).all() or not np.isfinite(self.torque_limits).all():
            raise ValueError("Joint limits must be finite for joint plots.")
        if (self.position_limits[:, 0] > self.position_limits[:, 1]).any() or (self.torque_limits <= 0).any():
            raise ValueError("Joint limits must be ordered and torque limits strictly positive.")
        self._pending_unclipped_action = np.full(len(self.joint_names), np.nan)
        self._pending_clipped_action = np.full(len(self.joint_names), np.nan)

    def observe(self, raw, frame, rewards=None):
        if frame != self.last_frame + 1:
            raise ValueError("Joint plots require consecutive frames starting at zero.")
        data = raw.scene["robot"].data
        self.update(frame, _as_numpy(data.joint_pos[0])[self.joint_indices],
                    _as_numpy(data.applied_torque[0])[self.joint_indices],
                    self._pending_unclipped_action, self._pending_clipped_action)

    def observe_action(self, raw):
        action_term = raw.action_manager.get_term("joint_pos")
        joint_ids = getattr(action_term, "_joint_ids", slice(None))
        if isinstance(joint_ids, slice):
            joint_ids = range(len(raw.scene["robot"].joint_names))[joint_ids]
        joint_ids = np.asarray(list(joint_ids), dtype=np.int64)
        controlled_joint_names = getattr(action_term, "_joint_names", None)
        if controlled_joint_names is None:
            robot_joint_names = raw.scene["robot"].joint_names
            controlled_joint_names = [robot_joint_names[index] for index in joint_ids]
        plot_ids = {int(index): position for position, index in enumerate(self.joint_indices)}
        action_positions = [position for position, index in enumerate(joint_ids) if int(index) in plot_ids]
        plot_action_ids = np.asarray([plot_ids[int(joint_ids[position])] for position in action_positions], dtype=np.int64)
        raw_action = _as_numpy(action_term.raw_actions[0])
        clipped_action = _as_numpy(action_term.processed_actions[0])
        if raw_action.shape != (len(joint_ids),) or clipped_action.shape != (len(joint_ids),):
            raise ValueError("Action-term values must match its controlled joint count.")
        raw_action = raw_action[action_positions]
        clipped_action = clipped_action[action_positions]
        scale = _as_numpy(getattr(action_term, "_scale", 1.0))
        offset = _as_numpy(getattr(action_term, "_offset", 0.0))
        if scale.ndim > 1:
            scale = scale[0]
        if offset.ndim > 1:
            offset = offset[0]
        for name, parameter in (("scale", scale), ("offset", offset)):
            if parameter.ndim > 0 and parameter.shape == (len(joint_ids),):
                parameter = parameter[action_positions]
            if parameter.ndim > 0 and parameter.shape != raw_action.shape:
                raise ValueError(f"Action-term {name} must be scalar or match its controlled joints.")
            if name == "scale":
                scale = parameter
            else:
                offset = parameter
        unclipped_action = raw_action * scale + offset
        clip = _clip_bounds(action_term, controlled_joint_names)
        clip = clip[action_positions]
        clipped_action = np.clip(unclipped_action, clip[:, 0], clip[:, 1])
        self._pending_unclipped_action = np.full(len(self.joint_names), np.nan)
        self._pending_clipped_action = np.full(len(self.joint_names), np.nan)
        self._pending_unclipped_action[plot_action_ids] = unclipped_action
        self._pending_clipped_action[plot_action_ids] = clipped_action

    def update(self, frame, positions, torques, action=None, clipped_action=None):
        if frame != self.last_frame + 1:
            raise ValueError("Joint plots require consecutive frames starting at zero.")
        positions, torques = _as_numpy(positions), _as_numpy(torques)
        expected = (len(self.joint_names),)
        if positions.shape != expected or torques.shape != expected:
            raise ValueError("Joint positions and torques must match the robot joint count.")
        if not np.isfinite(positions).all() or not np.isfinite(torques).all():
            raise ValueError("Joint positions and torques must be finite.")
        if action is None:
            action = np.full(expected, np.nan)
        if clipped_action is None:
            clipped_action = np.full(expected, np.nan)
        action, clipped_action = _as_numpy(action), _as_numpy(clipped_action)
        if action.shape != expected or clipped_action.shape != expected:
            raise ValueError("Actions must match the robot joint count.")
        if not (np.isfinite(action) | np.isnan(action)).all() or not (np.isfinite(clipped_action) | np.isnan(clipped_action)).all():
            raise ValueError("Actions must be finite or unavailable for passive joints.")
        self.times.append(frame * self.step_dt)
        self.positions.append(positions.copy())
        self.torques.append(torques.copy())
        self.unclipped_actions.append(action.copy())
        self.clipped_actions.append(clipped_action.copy())
        self.last_frame = frame

    def summary(self):
        if not self.times:
            raise ValueError("Joint plots require at least one observed frame.")
        return {"run_env_id": 0, "samples": len(self.times), "step_dt_s": self.step_dt,
                "joints": self.joint_names, "position_units": "rad", "torque_units": "N m"}

    def save(self, output):
        if not self.times:
            raise ValueError("Joint plots require at least one observed frame.")
        output = Path(output) / "joints"
        output.mkdir(parents=True, exist_ok=True)
        times = np.asarray(self.times)
        positions, torques = np.asarray(self.positions), np.asarray(self.torques)
        unclipped_actions, clipped_actions = np.asarray(self.unclipped_actions), np.asarray(self.clipped_actions)
        with (output / "joints.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s"] + [f"{name}_position_rad" for name in self.joint_names] +
                            [f"{name}_torque_nm" for name in self.joint_names] +
                            [f"{name}_unclipped_action_rad" for name in self.joint_names] +
                            [f"{name}_clipped_action_rad" for name in self.joint_names])
            writer.writerows([time, *position, *torque, *unclipped_action, *clipped_action]
                             for time, position, torque, unclipped_action, clipped_action
                             in zip(times, positions, torques, unclipped_actions, clipped_actions))
        for index, name in enumerate(self.joint_names):
            stem = name.replace("/", "_")
            position_limits = self.position_limits[index]
            torque_limit = self.torque_limits[index]
            _plot_svg(output / f"{stem}_position.svg", f"{name} joint position", "Position (rad)",
                      times, positions[:, index], position_limits[0], position_limits[1],
                      "lower limit", "upper limit",
                      series=[("unclipped action", unclipped_actions[:, index], "#ddaa33"),
                              ("clipped action", clipped_actions[:, index], "#33aa55")])
            _plot_svg(output / f"{stem}_torque.svg", f"{name} joint torque", "Torque (N m)",
                      times, torques[:, index], -torque_limit, torque_limit,
                      "negative limit", "positive limit")
