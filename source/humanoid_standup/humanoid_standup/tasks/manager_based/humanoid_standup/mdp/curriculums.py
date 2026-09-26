"""Reset-time curricula for stand-up training."""

from __future__ import annotations

import math

import torch


def robot_mass_curriculum(
    env,
    env_ids,
    initial_scale: float = 0.5,
    ramp_steps: int = 80_000,
) -> float:
    """Linearly restore true mass, uniformly scaling all link masses/inertias.

    Called by the curriculum manager before resetting selected environments.
    Always scale immutable asset defaults, never the previously scaled values.
    Local COM offsets and all relative link mass ratios remain unchanged.
    """
    if not math.isfinite(initial_scale) or not 0.0 < initial_scale <= 1.0:
        raise ValueError("initial_scale must be finite and in (0, 1].")
    if not isinstance(ramp_steps, int) or isinstance(ramp_steps, bool) or ramp_steps <= 0:
        raise ValueError("ramp_steps must be a positive integer.")
    steps = env.common_step_counter + getattr(env, "mass_curriculum_step_offset", 0)
    progress = min(max(steps / ramp_steps, 0.0), 1.0)
    scale = initial_scale + (1.0 - initial_scale) * progress
    robot = env.scene["robot"]
    ids = (
        torch.arange(env.num_envs, dtype=torch.long, device="cpu")
        if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
    )
    if ids.numel() == 0:
        return scale
    # PhysX mass/inertia setters use CPU buffers with full articulation dimensions.
    masses = robot.root_physx_view.get_masses().clone()
    inertias = robot.root_physx_view.get_inertias().clone()
    masses[ids] = robot.data.default_mass.cpu()[ids] * scale
    inertias[ids] = robot.data.default_inertia.cpu()[ids] * scale
    robot.root_physx_view.set_masses(masses, ids)
    robot.root_physx_view.set_inertias(inertias, ids)
    return scale
