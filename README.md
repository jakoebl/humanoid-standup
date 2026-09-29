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
root world velocity, joint state, rewards, and termination
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
step receives tracking reward 1 before penalties. An action magnitude penalty
subtracts `0.001 * sum(action ** 2)` per control step, using raw policy actions
before joint-position scaling. An action-rate penalty also subtracts
`0.005 * sum((action - previous_action) ** 2)` per control step to discourage
rapid target oscillations. This is an initial tuning value: evaluate endpoint
jitter and stand-up tracking after further training. Reward changes do not alter
existing checkpoint behavior during playback. The G1 PD action scaling remains 0.25 about its default
joint pose; it is an action parameterization, not a residual on the reference.

Observations contain only proprioceptive information (64 values): 29 joint
positions and 29 joint velocities in action order, followed by pelvis angular
velocity and projected gravity expressed in the pelvis frame. Finger joints,
reference phase/targets, root height, global pose, and linear velocities are
excluded.
Networks use ReLU. This
changes observation dimensions and training semantics: train a fresh policy
rather than resuming old checkpoints. G1 revolute joints, actuator limits,
and tracking-based failure detection remain robot/skill-specific adaptations.

CPU regression checks (use a Python environment with NumPy and PyTorch):

```sh
python tests/test_reference_regressions.py
```

### Verify processed COM and end-effector targets

Read positions from PhysX at every reference pose (all frames, including the
lying and standing portions):

```sh
python scripts/play_reference.py --check-kinematics --headless --device cpu
```

Use the Isaac Sim Python environment (or `/isaac-sim/python.sh` in this container).
The check compares mass-weighted body COM positions and wrist/ankle link origins
with the processed CSV, in meters relative to the environment origin. It refreshes
kinematics without stepping dynamics, so gravity, contact and PD tracking errors
do not affect the comparison. Non-reference joints retain their playback defaults.
Omit `--headless` to render the checked poses.

`--ee-tolerance` and `--com-tolerance` default to 0.001 m. Measurements are saved to
`outputs/reference_kinematics_check.csv`, with mean/RMS/maximum errors, worst frame
indices and pass/fail results in the adjacent JSON file. Change the destination
with `--check-output`. `--csv path/to/processed_clip.csv` selects the motion and
its marker targets together, for both playback and checking. Omit `--csv` to use
the configured reference. Clips must use the reference timestep (120 FPS).
Failed checks return a nonzero
exit status; source reference data is not modified.

### Process reference targets using Isaac Sim

Use the simulation-based processor to generate EE and COM targets from the same
`G1_29DOF_CFG` asset used by training:

```sh
python scripts/precompute_reference_kinematics_sim.py \
  data/unprocessed/stand_up_lying_R_002__A472_M.csv \
  data/processed/stand_up_lying_R_002__A472_M.csv \
  --headless --device cpu
```

In this container, replace `python` with `/isaac-sim/python.sh`. The explicit output
path is created or replaced only after every frame succeeds; input and output
must differ. The script works without an existing processed clip. It preserves
motion values, accepts columns by name, and writes canonical motion/EE/COM column
order for the training loader. Existing EE/COM columns are recomputed.

Root translations are interpreted as centimeters and XYZ Euler/joint angles as
degrees. Exported positions are meters. Each pose is written directly to PhysX,
with no dynamics step; end effectors use link origins and COM uses every body's
simulation mass and world COM position. Unspecified finger joints retain the
robot's default pose, matching training resets and playback.

For 30 FPS G1 clips containing headerless rows of `X Y Z QX QY QZ QW`
followed by the same 29 joints, use quaternion mode (comma or whitespace separated):

```sh
/isaac-sim/python.sh scripts/precompute_reference_kinematics_sim.py \
  path/to/clip.csv data/processed/clip.csv \
  --input-format quaternion --headless --device cpu
```

This mode assumes root translation in meters and joint angles in radians; use
`--translation-unit cm` and/or `--joint-unit deg` if needed. It normalizes root
quaternions, converts them to XYZ Euler degrees, and converts motion units to the
existing centimeters/degrees schema. Zero-order hold repeats each 30 FPS pose
four times at 120 FPS, including the final pose, with sequential `Frame` indices
starting at zero. EE/COM targets are then computed by the same simulation path.
The default `--input-format legacy` preserves existing CSV behavior.

To play another processed clip, use `python scripts/play_reference.py --csv path/to/processed_clip.csv`.
The graphical viewer's **Reference Playback** panel has play/pause, a frame slider,
previous/next frame buttons, and an editable frame number starting at zero.
Scrubbing or stepping pauses on the exact CSV pose; press **Play** to resume looping.
The camera stays interactive while paused. Steps use the reference's 120 FPS
samples, including repeated poses in clips converted from 30 FPS. Controls are
omitted in headless mode and during kinematics checks.

To verify its targets, add `--check-kinematics` as below:

```sh
python scripts/play_reference.py --check-kinematics --headless --device cpu \
  --csv data/processed/stand_up_lying_R_002__A472_M.csv
```

### Robot mass curriculum

Training starts at 50% of the robot's original mass and ramps linearly to 100%
over 80,000 control steps (5,000 PPO iterations at 16 steps per rollout). Configure
`curriculum.robot_mass.params.initial_scale` and `ramp_steps` in the environment
configuration. Steps are shared across environments, not summed across the batch.

At each environment reset, every link—including fingers—receives the same scalar
multiplying its original mass and inertia tensor. Scaling always uses the original
asset values, so repeated resets do not compound it. Local COM offsets and relative
link masses stay unchanged; consequently the whole-body COM at a given pose and
the processed reference targets remain valid. Updates happen only on reset to
avoid changing dynamics mid-episode; active episodes keep their previous scale.
The curriculum manager logs the scheduled scale under `robot_mass`.

Policy and reference playback disable the curriculum and use true mass. Resumed
RSL-RL training restores schedule progress from the checkpoint's learning iteration
multiplied by the configured rollout length, then resets all environments. This is
an iteration-based approximation (keep the rollout length unchanged when resuming).
Set `initial_scale=1.0` to train at full mass throughout.

Regression check: `python tests/test_mass_curriculum.py`.

### Record a CSV reference clip

Record one full pass of a processed CSV to a 1280×720 MP4:

```sh
/isaac-sim/python.sh record.py data/processed/standup_1_32.csv
```

The video is saved in `/workspace/humanoid_standup/vids/standup_1_32.mp4`. Recording runs headless
by default; add `--gui` to show Isaac Sim. Use `--output path/to/video.mp4` for a
custom destination. Existing recordings are never overwritten.

The default `--fps 120` preserves every reference sample. `--fps 30` or
`--fps 60` samples fewer poses while keeping the playback speed. Recording uses
a camera that follows and frames all robot links with padding, plus the same
COM/end-effector markers as reference playback. It writes exact
CSV poses without stepping physics, and exits after one pass. Rendering can be
slower than real time without changing the video's playback speed. Run with the
Isaac Sim Python environment; RGB rendering still requires a compatible GPU in
headless mode.
