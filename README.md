# Humanoid Standup

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

Open `humanoid-standup.code-workspace` in VS Code to display the project name.

Install the package in your Isaac Lab Python environment:

```bash
python -m pip install -e source/humanoid_standup
```

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/humanoid_standup
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Humanoid-Standup"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/humanoid_standup/humanoid_standup/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/humanoid_standup"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
## DeepMimic-style stand-up training

The task uses synchronized random reference-state initialization: root pose,
root world velocity, joint state, observation phase, rewards, and termination
share one per-environment starting frame. The RSL-RL runner must keep
`init_at_random_ep_len=False`. For evaluation from the beginning of the clip,
set `env_cfg.events.reset_robot.params["random_start"] = False` before creating
an environment.

The reference is noncyclic. Sampling clamps at the last frame and completing
its remaining duration is a terminal transition, not a time-limit truncation.
The generic episode timeout remains a separate safeguard. Ground contacts are
allowed for the lying-to-standing motion; a wrist/ankle deviation above 0.5 m
terminates failed tracking. This intentionally differs from contact-based fall
termination used for locomotion skills.

Control runs at 30 Hz over 120 Hz physics. The pose reward combines root
quaternion error with wrapped revolute-joint angle errors in one exponential,
weighted 0.65; velocity, end-effector, and COM weights are 0.10/0.15/0.10.
Configuration compensates Isaac Lab's timestep multiplication so a perfect
step receives reward 1. The G1 PD action scaling remains 0.25 about its default
joint pose; it is an action parameterization, not a residual on the reference.

Observations match the released DeepMimic `CtController` raw feature layout:
scalar phase, root height, per-body position/normal/tangent, followed by
per-body linear/angular velocity. Six rotation features match the code rather
than the paper's quaternion description. An explicit list selects pelvis plus
29 G1 joint child bodies (452 values), excluding palms and all fingers.
No reference-target or joint-coordinate features are included. Positions and
linear velocities use body COMs; orientations use link frames. Features use
the root heading frame with z-up instead of DeepMimic's y-up convention.
The feature schema matches; the dimension differs with the robot skeleton,
and DeepMimic's separate input normalization is not reproduced here.
Networks use ReLU. This
changes observation dimensions and training semantics: train a fresh policy
rather than resuming old checkpoints. G1 revolute joints, actuator limits,
and tracking-based failure detection remain robot/skill-specific adaptations.

CPU regression checks (use a Python environment with NumPy and PyTorch):

```sh
python tests/test_reference_regressions.py
```
