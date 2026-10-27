# BeyondMimic training commands

## Training with a single clip

```bash
/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task BeyondMimic \
  --headless \
  --num_envs 2048 \
  env.commands.motion.motion_file=/workspace/humanoid_standup/data/beyondmimic/side.npz \
  --run_name <name>
```


## Training with multiple clips

```bash
/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task BeyondMimic \
  --headless \
  --num_envs 2048 \
  env.commands.motion.motion_directory=/workspace/humanoid_standup/data/beyondmimic \
  --run_name <name>
```


The clip list is saved in the run's `params/training_clips.yaml`.

## Resuming single-clip training

```bash
/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task BeyondMimic \
  --headless \
  --resume \
  --load_run logs/rsl_rl/beyond_mimic/? \
  --checkpoint model_?.pt \
  --max_iterations 30000 \
  env.commands.motion.motion_file=/workspace/humanoid_standup/data/beyondmimic/side.npz
  --run_name <name>
```

## Resuming multi-clip training

```bash
/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task BeyondMimic \
  --headless \
  --resume \
  --load_run logs/rsl_rl/beyond_mimic/? \
  --checkpoint model_?.pt \
  --max_iterations 30000 \
  env.commands.motion.motion_directory=/workspace/humanoid_standup/data/beyondmimic
  --run_name <name>
```
