"""Deployment metadata for the tracking policies supported by g1_rl_controller."""

import json

import torch


def tracking_policy_metadata(env):
    """Read nominal control settings in policy order from an initialized environment.

    Keep this adapter explicit: unsupported observation layouts must not be labelled
    as compatible with the deployment controller. Encoder noise is training-only.
    IMU frame agreement with the deployment robot must still be checked separately.
    """
    action_manager = env.action_manager
    if action_manager.active_terms != ["joint_pos"]:
        raise ValueError("Deployment export requires a single joint_pos action term.")
    action = action_manager.get_term("joint_pos")
    if type(action).__name__ != "JointPositionAction":
        raise ValueError("Deployment export requires absolute joint-position actions.")
    # IsaacLab exposes resolved scales, offsets and joint selection on the action term.
    robot = action._asset
    joint_ids = action._joint_ids
    joint_order = list(action._joint_names)
    count = len(joint_order)
    if action.cfg.clip is not None:
        from humanoid_standup.tasks.manager_based.beyond_mimic.constants import JOINT_POSITION_LIMITS

        expected = torch.tensor(
            [JOINT_POSITION_LIMITS[name] for name in joint_order],
            device=action._clip.device, dtype=action._clip.dtype,
        )
        if not torch.allclose(action._clip, expected.expand_as(action._clip)):
            raise ValueError("Action target clipping does not match the deployment joint limits.")
    command = env.command_manager.get_term("motion")
    if [robot.joint_names[i] for i in command.joint_indexes] != joint_order:
        raise ValueError("Observation and action joint ordering differ.")

    def vector(value):
        value = torch.as_tensor(value).detach().cpu()
        if value.ndim == 2:
            if not torch.allclose(value, value[:1].expand_as(value)):
                raise ValueError("Deployment settings differ between environments.")
            value = value[0]
        if value.ndim == 0:
            value = value.expand(count)
        if value.shape != (count,) or not torch.isfinite(value).all():
            raise ValueError("Expected a finite value for each policy joint.")
        return value.tolist()

    if torch.count_nonzero(robot.data.default_joint_vel[:, joint_ids]):
        raise ValueError("Deployment joint-velocity observations require zero default velocity.")
    names = env.observation_manager.active_terms["policy"]
    expected = ["gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"]
    if names != expected:
        raise ValueError(f"Unsupported deployment observation layout: {names}")
    aliases = ["projected_gravity", "base_ang_vel", "joint_pos_rel", "joint_vel_rel", "last_action"]
    functions = ["imu_projected_gravity", "imu_ang_vel", "selected_joint_pos_rel",
                 "selected_joint_vel_rel", "last_action"]
    group = env.observation_manager.cfg.policy
    obs_terms = {}
    for name, alias, function in zip(names, aliases, functions):
        cfg = getattr(group, name)
        if cfg.func.__name__ != function:
            raise ValueError(f"Unsupported observation function for {name}.")
        history = group.history_length if group.history_length is not None else cfg.history_length
        if history not in (0, 1) or cfg.clip is not None or cfg.modifiers:
            raise ValueError("Deployment adapter supports unmodified observations without history.")
        scale = torch.as_tensor(1.0 if cfg.scale is None else cfg.scale)
        if scale.numel() != 1 or not torch.isfinite(scale).all():
            raise ValueError("Deployment adapter requires finite scalar observation scales.")
        obs_terms[alias] = {"scale": scale.item()}
    return {
        "control_dt": float(env.step_dt),
        "joint_order": joint_order,
        "obs_hist": 1,
        "obs_terms": obs_terms,
        "default_joint_pos": vector(robot.data.default_joint_pos[:, joint_ids]),
        "action_scales": vector(action._scale),
        "action_offsets": vector(action._offset),
        "stiffness": vector(robot.data.default_joint_stiffness[:, joint_ids]),
        "damping": vector(robot.data.default_joint_damping[:, joint_ids]),
    }


def attach_tracking_metadata(env, model_path):
    """Validate the exported interface and save JSON metadata, preserving other keys."""
    import onnx

    metadata = tracking_policy_metadata(env)
    model = onnx.load(model_path)
    count = len(metadata["joint_order"])
    for values, width in ((model.graph.input, 6 + 3 * count), (model.graph.output, count)):
        if len(values) != 1:
            raise ValueError("Deployment requires one model input and one output.")
        tensor = values[0].type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in tensor.shape.dim] != [1, width]:
            raise ValueError(f"Expected a float32 model tensor with shape [1, {width}].")
    properties = {prop.key: prop.value for prop in model.metadata_props}
    properties.update({key: json.dumps(value, allow_nan=False) for key, value in metadata.items()})
    onnx.helper.set_model_props(model, properties)
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
