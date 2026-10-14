"""Infer checkpoint tasks and locate their training clips."""

from pathlib import Path


def infer_task(checkpoint):
    """Identify the task from the checkpoint's experiment directory."""
    experiments = {
        "humanoid_standup": "Humanoid-Standup",
        "beyond_mimic": "BeyondMimic",
        "mimic_multiclip": "BeyondMimic",  # Historical experiment directory.
    }
    for parent in checkpoint.parents:
        if parent.name in experiments:
            return experiments[parent.name]
    raise ValueError("Cannot infer task from checkpoint path; specify --task Humanoid-Standup or BeyondMimic.")


def resolve_clips(checkpoint, project, task, override=None):
    """Use the run's saved clips, falling back to the task's data directory."""
    import yaml

    tracking = task.split(":")[-1].startswith("BeyondMimic")
    extension = ".npz" if tracking else ".csv"
    directory = project / "data" / ("beyondmimic" if tracking else "training")
    manifest = checkpoint.parent / "params/training_clips.yaml"
    if override:
        clips = sorted(override.expanduser().resolve().glob(f"*{extension}"))
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
        clips = sorted(directory.glob(f"*{extension}"))
    if not clips:
        raise ValueError(f"No {extension} clips found for {task}.")
    return clips
