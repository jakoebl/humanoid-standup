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

    local root="/workspace/humanoid_standup/logs/rsl_rl"
    local run checkpoint task

    # Newest matching run by name across both experiments; the liedown folder exists after its first run.
    run=$(
        find "$root/humanoid_standup" "$root/humanoid_standup_liedown" -mindepth 1 -maxdepth 1 -type d \
            -printf '%f\t%p\n' 2>/dev/null |
            grep -E -- "$pattern" |
            LC_ALL=C sort |
            tail -n 1 |
            cut -f 2
    )

    if [ -z "$run" ]; then
        echo "No run matches: $pattern" >&2
        return 1
    fi

    case "${run%/*}" in
        "$root/humanoid_standup") task="Humanoid-Standup-Play" ;;
        "$root/humanoid_standup_liedown") task="Humanoid-Standup-Liedown-Play" ;;
    esac

    checkpoint=$(
        find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' |
            grep -E '^model_[0-9]+\.pt$' |
            sort -V |
            tail -n 1
    )

    if [ -z "$checkpoint" ]; then
        echo "No training checkpoints found in: $run" >&2
        return 1
    fi

    echo "Playing ($task): $run/$checkpoint"
    isaaclab -p /workspace/humanoid_standup/scripts/rsl_rl/play.py \
        --task "$task" \
        --num_envs 1 \
        --checkpoint "$run/$checkpoint" \
        "$@"
}
