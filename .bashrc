train() {
    if [ "$#" -eq 0 ]; then
        echo 'Usage: train NAME [additional arguments]'
        return 1
    fi

    local run_name="$1"
    shift

    isaaclab -p /workspace/humanoid_standup/scripts/rsl_rl/train.py \
        --task Humanoid-Standup \
        --headless \
        --run_name "$run_name" \
        "$@"
}

play() {
    if [ "$#" -eq 0 ]; then
        echo 'Usage: play RUN_REGEX [additional arguments]'
        return 1
    fi

    local pattern="$1"
    shift

    local root="/workspace/humanoid_standup/logs/rsl_rl/humanoid_standup"
    local run checkpoint

    run=$(
        find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
            grep -E -- "$pattern" |
            LC_ALL=C sort |
            tail -n 1
    )

    if [ -z "$run" ]; then
        echo "No run matches: $pattern" >&2
        return 1
    fi

    checkpoint=$(
        find "$root/$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' |
            grep -E '^model_[0-9]+\.pt$' |
            sort -V |
            tail -n 1
    )

    if [ -z "$checkpoint" ]; then
        echo "No training checkpoints found in: $root/$run" >&2
        return 1
    fi

    echo "Playing: $root/$run/$checkpoint"
    isaaclab -p /workspace/humanoid_standup/scripts/rsl_rl/play.py \
        --task Humanoid-Standup \
        --num_envs 1 \
        --checkpoint "$root/$run/$checkpoint" \
        "$@"
}