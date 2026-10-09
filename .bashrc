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

    local root="/workspace/humanoid_standup/logs"
    local run checkpoint task

    run=$(
        find "$root" -mindepth 0 -maxdepth 3 -type d -printf '%P\n' |
            grep -E -- "$pattern" |
            LC_ALL=C sort |
            tail -n 1
    )

    if [ -z "$run" ]; then
        echo "No run matches: $pattern" >&2
        return 1
    fi

    case "$run" in
        rsl_rl/humanoid_standup/*) task="Humanoid-Standup" ;;
        rsl_rl/beyond_mimic/*) task="BeyondMimic-Play" ;;
        *)
            echo "Cannot determine task for log directory: $root/$run" >&2
            return 1
            ;;
    esac

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

    echo "Playing ($task): $root/$run/$checkpoint"
    isaaclab -p /workspace/humanoid_standup/scripts/rsl_rl/play.py \
        --task "$task" \
        --num_envs 1 \
        --checkpoint "$root/$run/$checkpoint" \
        "$@"
}
