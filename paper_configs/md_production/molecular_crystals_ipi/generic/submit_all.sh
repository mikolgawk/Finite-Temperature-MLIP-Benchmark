#!/bin/bash -l
#
# Submits one i-PI job per (system, model) pair -- the full 5 x 17 grid that
# the per-directory tree used to encode by hand.
#
#   ./submit_all.sh                       # everything
#   ./submit_all.sh --model mace-mpa-0    # one model, all systems
#   ./submit_all.sh --system picene_295K_Sharma_S
#   ./submit_all.sh --dry-run

set -euo pipefail

GENERIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ONLY_SYSTEM=""
ONLY_MODEL=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --system) ONLY_SYSTEM="$2"; shift 2 ;;
        --model)  ONLY_MODEL="$2";  shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Systems come from the settings table, models from the shared catalog, so
# the grid has exactly one definition apiece.
SYSTEMS=$(tail -n +2 "$GENERIC_DIR/ipi_settings_ref.csv" | cut -d, -f1)
MODELS=$(python3 -c "
import json
with open('$GENERIC_DIR/../../model_calculators.json') as f:
    print('\n'.join(m['name'] for m in json.load(f)['models']))
")

for system in $SYSTEMS; do
    [ -n "$ONLY_SYSTEM" ] && [ "$system" != "$ONLY_SYSTEM" ] && continue
    for model in $MODELS; do
        [ -n "$ONLY_MODEL" ] && [ "$model" != "$ONLY_MODEL" ] && continue

        job_name="${system%%_*}-${model}-ipi"
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "would submit: $job_name"
            continue
        fi

        sbatch --job-name="$job_name" \
               --export="ALL,SYSTEM=$system,MODEL_NAME=$model" \
               "$GENERIC_DIR/submit-generic.sh"
    done
done
