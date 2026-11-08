"""Plot one rollout of every joint against its reference motion and position and torque limits."""

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


# Fixed categorical order; dash patterns keep series distinct without color.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACTUAL = ("#2a78d6", None)
REFERENCE = ("#eb6834", "8 4")
CLIPPED_ACTION = ("#1baf7a", "3 3")
LIMIT_DASH = "6 4"
BODY_GROUPS = (("left_leg", "Left leg"), ("right_leg", "Right leg"), ("waist", "Waist"),
               ("left_arm", "Left arm"), ("right_arm", "Right arm"), ("other", "Other"))


def _body_group(name):
    name = str(name).lower()
    if "waist" in name:
        return "waist"
    side = "left" if name.startswith("left") else "right" if name.startswith("right") else None
    if side and any(part in name for part in ("hip", "knee", "ankle")):
        return f"{side}_leg"
    if side and any(part in name for part in ("shoulder", "elbow")):
        return f"{side}_arm"
    return "other"


def _plot_svg(path, title, ylabel, times, series, limits):
    """Write a line chart; series are (label, values, (color, dash)), limits are (label, value)."""
    series = [(label, np.asarray(values, dtype=np.float64), style) for label, values, style in series]
    limit_values = [float(value) for _, value in limits]
    all_values = np.concatenate([values[np.isfinite(values)] for _, values, _ in series] + [limit_values])
    if all_values.size == 0 or not np.isfinite(limit_values).all():
        raise ValueError(f"Cannot plot nonfinite limits or samples for {title}.")
    plot_min, plot_max = float(all_values.min()), float(all_values.max())
    padding = max(abs(plot_min) * .05, 1.) if plot_min == plot_max else (plot_max - plot_min) * .05
    plot_min, plot_max = plot_min - padding, plot_max + padding
    left, top, width, height = 80., 76., 760., 350.
    time_max = max(float(times[-1]), 1e-12)
    x = lambda time: left + width * float(time) / time_max
    y = lambda value: top + height * (plot_max - float(value)) / (plot_max - plot_min)

    def points_for(values):
        return " ".join(f"{x(time):.2f},{y(value):.2f}"
                        for time, value in zip(times, values) if np.isfinite(value))

    def line(points, color, dash, width_px=2):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width_px}" '
                f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>')

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 500" role="img">',
        f"<title>{escape(title)}</title>",
        f'<rect width="1000" height="500" fill="{SURFACE}"/>',
        f'<g font-family="sans-serif" font-size="13" fill="{INK}">',
        f'<text x="{left}" y="28" font-size="18" font-weight="600">{escape(title)}</text>',
    ]
    # Legend row: swatch in series color, label in text ink.
    legend_x = left
    for label, _, (color, dash) in [*series, ("limits", None, (MUTED, LIMIT_DASH))]:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<path d="M {legend_x:.1f} 52 h 28" stroke="{color}" stroke-width="2"{dash_attr}/>')
        svg.append(f'<text x="{legend_x + 34:.1f}" y="56">{escape(label)}</text>')
        legend_x += 34 + 7.5 * len(label) + 26
    for value in np.linspace(plot_min, plot_max, 6):
        svg.append(f'<path d="M {left} {y(value):.2f} h {width}" stroke="{GRID}"/>')
        svg.append(f'<text x="{left - 8}" y="{y(value) + 4:.2f}" text-anchor="end" fill="{MUTED}">{value:.3g}</text>')
    for fraction in np.linspace(0, 1, 6):
        svg.append(f'<text x="{x(time_max * fraction):.2f}" y="{top + height + 20}" text-anchor="middle" '
                   f'fill="{MUTED}">{time_max * fraction:.3g}</text>')
    for label, value in limits:
        svg.append(f'<path d="M {left} {y(value):.2f} h {width}" stroke="{MUTED}" stroke-dasharray="{LIMIT_DASH}"/>')
        svg.append(f'<text x="{left + width + 8}" y="{y(value) + 4:.2f}" fill="{MUTED}">'
                   f'{escape(label)} {float(value):.3g}</text>')
    # Draw back to front so the actual trajectory stays on top.
    for label, values, (color, dash) in reversed(series):
        svg.append(line(points_for(values), color, dash))
    svg.extend([
        f'<path d="M {left} {top} V {top + height} H {left + width}" fill="none" stroke="{MUTED}"/>',
        f'<text x="{left + width / 2}" y="{top + height + 44}" text-anchor="middle" fill="{MUTED}">Time (s)</text>',
        f'<text transform="translate(22 {top + height / 2}) rotate(-90)" text-anchor="middle" fill="{MUTED}">'
        f'{escape(ylabel)}</text>',
        '</g></svg>',
    ])
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def _index_html(title, groups, errors):
    """Overview page: one section per body part, position and torque side by side."""
    rows = []
    for group, label, joints in groups:
        rows.append(f'<h2>{escape(label)}</h2>')
        for name, position_file, torque_file in joints:
            rms, peak = errors.get(name, (np.nan, np.nan))
            rows.append(
                f'<section><h3>{escape(name)}</h3><p>RMS tracking error {rms:.3f} rad · '
                f'max {peak:.3f} rad</p><div class="pair">'
                f'<img src="{escape(position_file)}" alt="{escape(name)} position">'
                f'<img src="{escape(torque_file)}" alt="{escape(name)} torque"></div></section>')
    return "\n".join([
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>{escape(title)}</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#fcfcfb;color:#0b0b0b}"
        "h2{border-bottom:1px solid #e4e3df;padding-bottom:4px;margin-top:32px}"
        "h3{margin:16px 0 2px;font-size:15px}p{margin:0 0 6px;color:#52514e;font-size:13px}"
        ".pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:8px}"
        "img{width:100%;border:1px solid #e4e3df}</style></head><body>",
        f"<h1>{escape(title)}</h1>", *rows, "</body></html>",
    ])


class JointPlots:
    name = "joints"

    def __init__(self, step_dt):
        if not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Joint plots require a finite positive timestep.")
        self.step_dt = float(step_dt)
        self.last_frame = -1
        self.times = []
        self.positions = []
        self.reference_positions = []
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
        self.position_limits = _as_numpy(data.joint_pos_limits[0])[self.joint_indices]
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
        # The motion command lists reference joints by robot joint index.
        motion_joint_ids = [int(index) for index in raw.command_manager.get_term("motion").joint_indexes]
        plot_ids = {int(index): position for position, index in enumerate(self.joint_indices)}
        self.reference_columns = np.asarray(
            [column for column, index in enumerate(motion_joint_ids) if index in plot_ids], dtype=np.int64)
        self.reference_plot_ids = np.asarray(
            [plot_ids[index] for index in motion_joint_ids if index in plot_ids], dtype=np.int64)

    def observe(self, raw, frame, rewards=None):
        if frame != self.last_frame + 1:
            raise ValueError("Joint plots require consecutive frames starting at zero.")
        data = raw.scene["robot"].data
        reference = np.full(len(self.joint_names), np.nan)
        motion_reference = _as_numpy(raw.command_manager.get_term("motion").joint_pos[0])
        reference[self.reference_plot_ids] = motion_reference[self.reference_columns]
        self.update(frame, _as_numpy(data.joint_pos[0])[self.joint_indices],
                    _as_numpy(data.applied_torque[0])[self.joint_indices],
                    self._pending_unclipped_action, self._pending_clipped_action, reference)

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

    def update(self, frame, positions, torques, action=None, clipped_action=None, reference=None):
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
        if reference is None:
            reference = np.full(expected, np.nan)
        action, clipped_action, reference = _as_numpy(action), _as_numpy(clipped_action), _as_numpy(reference)
        if action.shape != expected or clipped_action.shape != expected:
            raise ValueError("Actions must match the robot joint count.")
        if reference.shape != expected:
            raise ValueError("Reference joint positions must match the robot joint count.")
        if not (np.isfinite(action) | np.isnan(action)).all() or not (np.isfinite(clipped_action) | np.isnan(clipped_action)).all():
            raise ValueError("Actions must be finite or unavailable for passive joints.")
        if not (np.isfinite(reference) | np.isnan(reference)).all():
            raise ValueError("Reference joint positions must be finite or unavailable.")
        self.times.append(frame * self.step_dt)
        self.positions.append(positions.copy())
        self.reference_positions.append(reference.copy())
        self.torques.append(torques.copy())
        self.unclipped_actions.append(action.copy())
        self.clipped_actions.append(clipped_action.copy())
        self.last_frame = frame

    def tracking_errors(self):
        """Per-joint RMS and maximum absolute position error against the reference, in rad."""
        error = np.asarray(self.positions) - np.asarray(self.reference_positions)
        result = {}
        for index, name in enumerate(self.joint_names):
            values = error[:, index][np.isfinite(error[:, index])]
            if values.size:
                result[name] = (float(np.sqrt(np.mean(values ** 2))), float(np.abs(values).max()))
        return result

    def summary(self):
        if not self.times:
            raise ValueError("Joint plots require at least one observed frame.")
        errors = self.tracking_errors()
        return {"run_env_id": 0, "samples": len(self.times), "step_dt_s": self.step_dt,
                "joints": self.joint_names, "position_units": "rad", "torque_units": "N m",
                "position_tracking_error_rad": {name: {"rms": rms, "max": peak}
                                                for name, (rms, peak) in errors.items()}}

    def save(self, output):
        if not self.times:
            raise ValueError("Joint plots require at least one observed frame.")
        output = Path(output) / "joints"
        output.mkdir(parents=True, exist_ok=True)
        times = np.asarray(self.times)
        positions, torques = np.asarray(self.positions), np.asarray(self.torques)
        references = np.asarray(self.reference_positions)
        unclipped_actions, clipped_actions = np.asarray(self.unclipped_actions), np.asarray(self.clipped_actions)
        with (output / "joints.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s"] + [f"{name}_position_rad" for name in self.joint_names] +
                            [f"{name}_reference_position_rad" for name in self.joint_names] +
                            [f"{name}_torque_nm" for name in self.joint_names] +
                            [f"{name}_unclipped_action_rad" for name in self.joint_names] +
                            [f"{name}_clipped_action_rad" for name in self.joint_names])
            writer.writerows([time, *position, *reference, *torque, *unclipped_action, *clipped_action]
                             for time, position, reference, torque, unclipped_action, clipped_action
                             in zip(times, positions, references, torques, unclipped_actions, clipped_actions))
        errors = self.tracking_errors()
        # Group plots by body part; robot joint order is kinematic within each group.
        grouped = {group: [] for group, _ in BODY_GROUPS}
        for index, name in enumerate(self.joint_names):
            grouped[_body_group(name)].append(index)
        groups = []
        for group, label in BODY_GROUPS:
            if not grouped[group]:
                continue
            (output / group).mkdir(exist_ok=True)
            joints = []
            for order, index in enumerate(grouped[group], start=1):
                name = self.joint_names[index]
                stem = f"{group}/{order:02d}_{name.replace('/', '_')}"
                lower, upper = self.position_limits[index]
                torque_limit = self.torque_limits[index]
                _plot_svg(output / f"{stem}_position.svg", f"{name} position", "Position (rad)", times,
                          [("actual", positions[:, index], ACTUAL),
                           ("reference", references[:, index], REFERENCE),
                           ("clipped action", clipped_actions[:, index], CLIPPED_ACTION)],
                          [("upper", upper), ("lower", lower)])
                _plot_svg(output / f"{stem}_torque.svg", f"{name} torque", "Torque (N m)", times,
                          [("applied torque", torques[:, index], ACTUAL)],
                          [("+limit", torque_limit), ("-limit", -torque_limit)])
                joints.append((name, f"{stem}_position.svg", f"{stem}_torque.svg"))
            groups.append((group, label, joints))
        (output / "index.html").write_text(_index_html("Joint evaluation (environment 0)", groups, errors),
                                           encoding="utf-8")
