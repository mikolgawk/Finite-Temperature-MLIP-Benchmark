#!/usr/bin/env bash
# Run every rmse_*.py beside this file, using each script's uv dependencies.
set -uo pipefail

usage() {
    echo "Usage: $0 [--dry-run]"
    echo "Run RMSE scripts sequentially; save logs under logs/rmse/<timestamp>/."
    echo "Set CUDA_VISIBLE_DEVICES to choose a GPU. Failures do not stop the batch."
}

dry_run=false
case "${1:-}" in
    --dry-run) dry_run=true ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    usage >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd -- "$script_dir" || exit 1
shopt -s nullglob
scripts=(rmse_*.py)
if (( ${#scripts[@]} == 0 )); then
    echo "No rmse_*.py scripts found in $script_dir" >&2
    exit 1
fi

if "$dry_run"; then
    printf 'Working directory: %s\n' "$script_dir"
    for script in "${scripts[@]}"; do
        printf 'uv run --script %q\n' "$script"
    done
    exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or is not on PATH." >&2
    exit 127
fi

log_dir="$script_dir/logs/rmse/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p -- "$log_dir" || exit 1
export PYTHONUNBUFFERED=1
trap 'echo "RMSE batch interrupted." >&2; exit 130' INT
trap 'echo "RMSE batch terminated." >&2; exit 143' TERM

failed=()
for script in "${scripts[@]}"; do
    printf '\nRunning %s\n' "$script"
    uv run --script "$script" 2>&1 | tee "$log_dir/${script%.py}.log"
    statuses=("${PIPESTATUS[@]}")
    if (( statuses[0] == 130 || statuses[0] == 143 )); then
        exit "${statuses[0]}"
    fi
    if (( statuses[0] != 0 || statuses[1] != 0 )); then
        failed+=("$script (uv: ${statuses[0]}, logging: ${statuses[1]})")
        printf 'FAILED: %s\n' "${failed[-1]}" >&2
    fi
done

printf '\nFinished %d scripts. Logs: %s\n' "${#scripts[@]}" "$log_dir"
if (( ${#failed[@]} > 0 )); then
    printf 'Failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "All RMSE scripts completed successfully."
