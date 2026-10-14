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

### BeyondMimic single- and multi-clip task

`BeyondMimic` is a separate Isaac Lab task matching the customized mjlab G1
tracking setup. Its actor uses 93 values (IMU projected gravity/angular velocity, 29 relative
joint positions, 29 relative joint velocities, and 29 previous actions). The
critic retains 286 values including reference commands, anchor transforms, body
poses, and IMU linear velocity. Projected gravity is a unit downward vector in the
IMU frame, with actor training noise of ±0.05 per component. Deployment must supply
the same gravity estimate, rather than raw accelerometer measurements (which also
contain motion acceleration). This replaces the previous actor velocity input;
train a fresh policy even though its input dimension is unchanged. Other actor
noise and encoder bias follow mjlab's settings. Actions use
the same per-motor scales, gains, torque limits, and nominal bent-knee pose.
Training defaults to 2048 environments, 200 Hz physics, and 50 Hz control.
PPO uses 24 steps per rollout, ELU networks of size 512/256/128, observation
normalization, and 30000 iterations. The eight reward terms exclude joint-limit
penalties. Ground contacts are permitted; tracking failure thresholds and
adaptive reference-state initialization follow the mjlab setup. Clip completion
resamples the reference state, rather than terminating the episode.

The prepared stand-up reference is `data/beyondmimic/stand_up_lying_V2.npz`.
To generate another reference, use the Isaac Sim environment:

```bash
/isaac-sim/python.sh scripts/precompute_beyond_mimic.py \
  data/unprocessed/stand_up_lying_V2.csv \
  data/beyondmimic/stand_up_lying_V2.npz \
  --input-fps 120 --output-fps 50 --headless
```

The exporter also accepts the headerless clips in `data/mjlab`. Set their actual
input FPS. It recomputes body poses on the Isaac Lab asset and stores joint/body
names; existing mjlab NPZs are rejected because their body order and asset differ.

Start a fresh policy inside the Isaac Lab container:

```bash
cd /workspace/humanoid_standup
/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task BeyondMimic --headless --num_envs 2048 --run_name beyond_mimic
```

Use `env.commands.motion.motion_file=/absolute/path/to/clip.npz` to select another
prepared clip. The same task also supports multi-clip training: set
`env.commands.motion.motion_directory=/absolute/path/to/clips` to load sorted
`*.npz` files, or pass
`'env.commands.motion.motion_files=[/data/first.npz,/data/second.npz]'`.
Selection precedence is explicit file list, directory, then `motion_file`.
The default remains the prepared V2 single clip. Clips are selected uniformly,
with independent adaptive frame sampling per clip and no crossing clip boundaries.
Use `BeyondMimic-Play` with `env.commands.motion.play_clip_index=0` to play a
selected clip; set it to `null` for random clip selection. Supply the same library
configuration during training and playback. The resolved file list is saved in
`params/training_clips.yaml`; keep it unchanged when resuming.
Its checkpoints live under `logs/rsl_rl/beyond_mimic`; existing
mjlab or `Humanoid-Standup` checkpoints cannot be resumed directly.

```bash
/isaac-sim/python.sh scripts/rsl_rl/play.py \
  --task BeyondMimic-Play --num_envs 1 \
  --checkpoint logs/rsl_rl/beyond_mimic/<run>/model_<iteration>.pt
```

Playback starts at frame 0 and uses a native Isaac Sim window. Add `--video
--video_length 259 --headless` to record a full reference-length rollout. The
`record_models.py` batch recorder also supports BeyondMimic with
`--task BeyondMimic`.

PhysX uses the repository's G1 USD asset, including finger links. Motor control,
reference math, and network inputs match the custom task; dynamics and collision
geometry are simulator-specific. Self-contact is counted over the 30 main bodies
using robot-only filtered contact histories, so ground contact is not penalized.
Finger-only contacts are excluded. Friction uses PhysX material buckets.
See the task's `PROVENANCE.md` for the upstream source and adaptations.

Validation:

```bash
/isaac-sim/python.sh tests/test_beyond_mimic_motion.py
/isaac-sim/python.sh tests/check_beyond_mimic.py --headless
```

### Humanoid-Standup task

Training loads every `*.csv` directly in `/workspace/humanoid_standup/data/training`
(`data/training` in the host checkout). Put processed 120 FPS clips with precomputed
end-effector and COM columns there. Each environment independently selects a clip
uniformly on reset, then samples a starting frame within that clip. Clips may have
different lengths; rewards and completion always follow the selected clip.
The training command is unchanged:

```bash
python scripts/rsl_rl/train.py --task Humanoid-Standup --headless
```

The selected filenames are printed at startup and saved in `params/training_clips.yaml`
in the run directory. Keep this set of files when resuming a run. Reference playback
and policy evaluation retain their existing single-clip behavior.

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

To disable the arm end-effector tracking reward after first standing, append
`env.rewards.arm_end_effector_tracking.params.stop_when_standing=true` to the
training command. The option defaults to `false`. Standing means whole-body
CoM height relative to the environment origin reaches the arm term's
`upright_com_height` (currently 0.5 m). Once reached, that environment's arm
reward stays zero until its episode resets, even if the robot falls below the
threshold. Leg tracking, joint tracking, and tracking-failure terminations
continue as configured.

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

Add `--foot-relative-markers` to align the marker targets to the actual left foot position and heading when the robot's CoM is above the configured standing threshold (default 0.65 m), matching training. Below the threshold, markers retain their absolute reference positions. This option also applies to `--record`; kinematics validation still compares the original CSV coordinates.
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
only. Use `--output path/to/video.mp4` for a
custom destination. Existing recordings are never overwritten.

The default `--fps 120` preserves every reference sample. `--fps 30` or
`--fps 60` samples fewer poses while keeping the playback speed. Recording uses
a camera that follows and frames all robot links with padding, plus the same
COM/end-effector markers as reference playback. It writes exact
CSV poses without stepping physics, and exits after one pass. Rendering can be
slower than real time without changing the video's playback speed. Run with the
Isaac Sim Python environment; RGB rendering still requires a compatible GPU in
headless mode.

### Record policy videos after training

Run the standalone batch recorder in your Isaac Sim environment:

```bash
# Scan all runs under the project's logs directory.
python scripts/record_models.py --last-three

# Record only one training run.
python scripts/record_models.py logs/rsl_rl/humanoid_standup/<run> --last-three

# Record the latest checkpoint ten times per clip, each starting from frame 0.
python scripts/record_models.py logs/rsl_rl/humanoid_standup/<run> --last

# Preview selected checkpoints without starting the simulator.
python scripts/record_models.py --dry-run
```

The script selects the latest three `model_<iteration>.pt` checkpoints per training run
by iteration number and calls `scripts/record.py` sequentially for each checkpoint.
Runs with fewer than three checkpoints record all available checkpoints.
`--last-three` is the default when neither selection flag is supplied. `--last`
instead selects the highest-numbered checkpoint per run and records ten separate
rollouts for every training clip. Each rollout resets the environment and policy
state at frame 0; videos are named `0.mp4` through `9.mp4` in the clip folder.
Training saves checkpoints for offline recording and no longer starts these recordings.
The existing optional `--video` training flag still enables its original Gym video capture.

For every clip in `data/training`, each checkpoint records the policy from
frame 0 through that clip to its end at 1920×1080 and 60 fps. The policy
keeps its original action frequency while rendering captures intermediate physics
steps. Videos are saved under
`vids/<training run directory name>/<iteration>/<clip name>/0.mp4`.
Earlier iterations are preserved; rerunning an iteration replaces its videos.
Recordings use noise-free resets and the true robot mass, matching policy playback.
Keep the training clips available on the recording machine.

Simulator output is saved to `<run>/recordings/<iteration>.log`.
Failures are reported and the batch continues with the next checkpoint;
the script exits with a nonzero status if any recording failed.
Use `--device`, `--task`, or `--agent` to override playback defaults.
In the Isaac Sim container, you can use `/isaac-sim/python.sh` instead of `python`.

Recording infers the task from the checkpoint’s `humanoid_standup` or
`beyond_mimic` experiment directory. Use `--task` to override it for relocated
checkpoints.

To record a single checkpoint:

```bash
python scripts/record.py --checkpoint logs/rsl_rl/humanoid_standup/<run>/model_1000.pt
```

### BeyondMimic multi-clip training

BeyondMimic supports multiple prepared motion clips through the same motion command.

`BeyondMimic` shares BeyondMimic's gravity-based 93-value actor, 286-value
critic, robot, actions, rewards, and PPO settings. It loads all prepared `*.npz`
files directly in `data/beyondmimic` (sorted by filename). Each environment selects
clips uniformly at reset and clip completion. Starting frames use independent
per-clip adaptive failure histograms, so long or difficult clips do not receive
more clip-selection probability. Different clip lengths are supported; reference
frames never advance into a neighboring clip. One clip also works.

Prepare additional clips with `scripts/precompute_beyond_mimic.py` as described
above; all references must use the same robot and 50 Hz control rate. Raw CSVs and
unnamed mjlab NPZ files cannot be used directly. The current directory contains
only the prepared V2 clip; add other prepared references for multi-clip training.

```bash
cd /workspace/humanoid_standup
/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task BeyondMimic --headless --num_envs 2048 --run_name multiclip
```

Override `env.commands.motion.motion_directory=/absolute/path/to/clips`, or use
an explicit list (which takes precedence over the directory):
`'env.commands.motion.motion_files=[/data/first.npz,/data/second.npz]'`.
The resolved file list is saved in `params/training_clips.yaml`. Runs are stored
under `logs/rsl_rl/beyond_mimic`. Keep the clip library unchanged when resuming.

```bash
/isaac-sim/python.sh scripts/rsl_rl/play.py \
  --task BeyondMimic-Play --num_envs 1 \
  --checkpoint logs/rsl_rl/beyond_mimic/<run>/model_<iteration>.pt \
  env.commands.motion.play_clip_index=0
```

`scripts/record.py --checkpoint <checkpoint>` overlays a translucent cyan reference
robot in each recorded clip. Recording is headless and reads
`params/training_clips.yaml`, resolving paths from other checkouts locally.
Without a saved list, it uses `data/beyondmimic` (`data/training` for
Humanoid-Standup). Use `--clips-dir` only to override the inferred directory.
Outputs are saved under `vids/<run>/<iteration>/<index>_<clip>/`.
Use `--repetitions` for multiple recordings per clip. Worst-case
tracking evaluation runs only when explicitly requested with `--worst-case-dir`.
For direct playback or `--video`, add
`--reference-ghost` to show the same visual-only overlay in world coordinates.

Playback selects the zero-based index in the resolved file list, starts at frame
0, and hides debug arrows. Use `--video --video_length 259 --headless` for video;
adjust the length to the selected clip. Set `play_clip_index=null` to randomly
select clips at resets.

Evaluate standing-up success with 1,000 parallel full-clip trials:

```bash
/isaac-sim/python.sh scripts/evaluate.py \
  --checkpoint logs/rsl_rl/beyond_mimic/<run>/model_<iteration>.pt \
  --criterion standing_up
```

Success requires mass-weighted COM height strictly above 0.5 m relative to the
environment origin throughout at least the final 1 second of the clip. The check
includes both endpoints at the control sampling rate; clips shorter than 1 second
cannot succeed. Results include success counts and rate in `summary.json` and each
trial's result and final sustained standing duration in `standing_up.csv`.
Each summary also records reset joint noise, policy observation noise and whether
it is enabled, event parameters including encoder-bias defaults, sampled physical
randomization statistics, simulation timing, and the complete resolved environment
and agent configurations used for evaluation.
Omit `--criterion` to run all evaluations, or use `--clip-index` to select one clip.

Evaluate jitteriness with `scripts/evaluate.py --checkpoint <checkpoint> --criterion jitteriness`
(default: 1,000 trials per clip). Lower scores indicate smoother motion. The score is
RMS joint jerk in rad/s³: the third finite difference of joint angles divided by the
control timestep cubed, averaged in squared form over all tracked joints and valid
full-clip samples, then square-rooted. Policy and unperturbed reference use identical
joints, frames and control sampling. At least four frames are required. This measures
rapid changes in acceleration, including deliberate fast movements; it is sensitive
to the sampling rate and does not use smoothing.

`summary.json` includes population mean/stddev across the 1,000 separate full-clip
policy scores and the single reference clip RMS score. `jitteriness.csv` stores every
trial score. `jitteriness_bins.csv` and `jitteriness_histogram.svg` show the distribution
across runs, with the reference clip RMS marked on the plot. Included when running
all evaluations without `--criterion`.

Evaluate worst-case tracking for a checkpoint with 1,000 parallel full-clip trials:

```bash
/isaac-sim/python.sh scripts/rsl_rl/play.py \
  --task BeyondMimic-Play --headless --num_envs 1000 \
  --checkpoint logs/rsl_rl/beyond_mimic/<run>/model_<iteration>.pt \
  --worst-case-dir outputs/worst_case/model_<iteration>/clip_0 \
  env.commands.motion.play_clip_index=0
```

`--worst-case-dir` defaults to 1,000 environments when `--num_envs` is omitted.
All trials start at frame 0 and run through the final frame. The configured
anchor-height, anchor-orientation, and end-effector-height termination checks
still run at their normal timing, but never reset the robot; timeouts are disabled.
`summary.json` reports the failure rate, thresholds, and first breach frame for
each term and failing trial, ordered by earliest breach. Each failed trial gets
its own full-length `env_NNNN.mp4` at 720p and the control frame rate. Videos replay
captured physical states, so they preserve the evaluated trajectory rather than
running the policy again. Temporary state storage is removed after recording.
Use a fresh output directory for each evaluation. Repeat with another clip index
to evaluate other clips; `play_clip_index=null` is not supported in this mode.
This also supports `BeyondMimic-Play` for its single clip.

Evaluation uses the playback settings (no observation noise, reset pose/velocity
noise, or pushes), while retaining startup physical randomization. Use `--seed`
to control trial sampling. These are full-clip tracking-failure rates, not standing
success rates or a search over adversarial disturbances.

The actor receives no clip identity or reference trajectory. Use compatible
stand-up variations; different intended behaviors from indistinguishable states
would need an additional deployable command. Critic reference observations remain
available during training.

Run the unequal-length multi-clip integration check with:

```bash
/isaac-sim/python.sh tests/check_mimic_multiclip.py --headless
```
