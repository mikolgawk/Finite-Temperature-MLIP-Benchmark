#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100:1

# Generic i-PI submission script. Replaces the 85 per-(system, model) copies.
#
#   sbatch --job-name=naphthalene-mace-mpa-0-ipi \
#          --export=ALL,SYSTEM=naphthalene_295K_Sharma_S,MODEL_NAME=mace-mpa-0 \
#          submit-generic.sh
#
# Also runnable outside SLURM:
#   SYSTEM=... MODEL_NAME=... ./submit-generic.sh
#
# Python environment is yours to choose. Either activate one beforehand, or
# name it via CONDA_ENV and this script will activate it:
#
#   CONDA_ENV=my-mace-env SYSTEM=... MODEL_NAME=... ./submit-generic.sh
#
# Different models need different packages (mace-torch, fairchem-core,
# tensorpotential, ...); see the "package" field of each entry in
# model_calculators.json for what a given model requires.

set -euo pipefail

: "${SYSTEM:?Set SYSTEM to a system name from ipi_settings_ref.csv}"
: "${MODEL_NAME:?Set MODEL_NAME to a model name from model_calculators.json}"

GENERIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNDIR="${RUNDIR:-$GENERIC_DIR/runs/$SYSTEM/$MODEL_NAME}"

# Unique per job, so concurrent runs cannot collide on /tmp/ipi_driver.
JOB_TAG="${SLURM_JOB_ID:-$$}"
export IPI_ADDRESS="ipi_${SYSTEM}_${MODEL_NAME}_${JOB_TAG}"

echo "System:   $SYSTEM"
echo "Model:    $MODEL_NAME"
echo "Run dir:  $RUNDIR"
echo "Address:  $IPI_ADDRESS"
echo "Env:      ${CONDA_ENV:-(already-active environment)}"

python3 "$GENERIC_DIR/prepare_run.py" \
    --system "$SYSTEM" \
    --model "$MODEL_NAME" \
    --rundir "$RUNDIR" \
    --address "$IPI_ADDRESS"

# Opt-in: activate an environment only if the caller named one, otherwise
# use whatever is already active.
if [ -n "${CONDA_ENV:-}" ]; then
    conda activate "$CONDA_ENV"
fi
echo "python:   $(command -v python3)"

export TMPDIR="${TMPDIR:-/home/mjgawkowski/tmp/}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CRAY_CUDA_MPS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$RUNDIR"

# Only this run's own stale socket -- never the shared /tmp/ipi_* glob, which
# would kill every other i-PI job on the node.
rm -f "/tmp/${IPI_ADDRESS}"

if [ -e RESTART ]; then
    echo "RESTART file found. Running i-PI with RESTART."
    i-pi RESTART &> log.ipi &
else
    echo "RESTART file not found. Running i-PI with input.xml."
    i-pi input.xml &> log.ipi &
fi

sleep 60

python3 "$GENERIC_DIR/run-ase-generic.py" &

wait
