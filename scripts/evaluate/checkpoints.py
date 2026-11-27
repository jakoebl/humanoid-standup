"""Locate a checkpoint's training clips and read its training config."""

import math
from pathlib import Path


TASKS = {"humanoid_standup": "Humanoid-Standup", "humanoid_standup_liedown": "Humanoid-Standup-Liedown"}


def task_for(checkpoint):
    """Name the task from the run's experiment folder."""
    return TASKS[checkpoint.parent.parent.name]


def resolve_clips(checkpoint, project, override=None):
    """Use the run's saved clips, falling back to data/beyondmimic."""
    import yaml

    directory = project / "data" / "beyondmimic"
    manifest = checkpoint.parent / "params/training_clips.yaml"
    if override:
        clips = sorted(override.expanduser().resolve().glob("*.npz"))
    elif manifest.is_file():
        with manifest.open() as stream:
            saved = yaml.safe_load(stream)
        clips = []
        for value in saved["clips"]:
            path = Path(value).expanduser()
            candidates = [path] if path.is_absolute() else [checkpoint.parent / path, project / path]
            # Training may have used another checkout or container mount.
            if "data" in path.parts:
                candidates.append(project.joinpath(*path.parts[path.parts.index("data"):]))
            candidates.append(directory / path.name)
            resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                raise ValueError(f"Training clip not found: {value}; use --clips-dir to override.")
            clips.append(resolved)
    else:
        clips = sorted(directory.glob("*.npz"))
    if not clips:
        raise ValueError("No .npz clips found.")
    return clips


def _load_env_yaml(path):
    """Read an Isaac Lab params/env.yaml, which stores tuples and slices as Python tags."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:python/tuple",
                           lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))
    Loader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice",
                           lambda loader, node: slice(*loader.construct_sequence(node, deep=True)))
    with path.open() as stream:
        return yaml.load(stream, Loader=Loader)


def training_action_clip(checkpoint):
    """Return the joint-position action clip the run trained with, or None if it trained unclipped."""
    path = checkpoint.parent / "params/env.yaml"
    if not path.is_file():
        raise ValueError(f"Training config not found: {path}; cannot tell whether actions were clipped.")
    try:
        clip = _load_env_yaml(path)["actions"]["joint_pos"]["clip"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"No actions.joint_pos.clip entry in {path}.") from error
    if clip is None:
        return None
    if not isinstance(clip, dict) or not clip:
        raise ValueError(f"Unsupported action clip in {path}: {clip!r}.")
    result = {}
    for name, bounds in clip.items():
        if len(bounds) != 2 or not all(math.isfinite(float(value)) for value in bounds) or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid action clip bounds for {name} in {path}: {bounds!r}.")
        result[str(name)] = (float(bounds[0]), float(bounds[1]))
    return result
