#!/bin/bash

#SBATCH --job-name=daymet_zarr_parallel
#SBATCH --output=./logs/slurm-%x-%A_%a.out
#SBATCH --error=./logs/slurm-%x-%A_%a.err
#SBATCH --time=42:00:00
#SBATCH --partition=serc
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

mkdir -p ./logs

source ~/uv_activations/activate_lfmc_process_py312.sh

MODE="${MODE:-worker}"
ROOT="${ROOT:?ROOT is required}"
OUT="${OUT:?OUT is required}"
COORD_DIR="${COORD_DIR:?COORD_DIR is required}"
OVERWRITE_OUT="${OVERWRITE_OUT:-0}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"
MAX_MONTHS="${MAX_MONTHS:-}"
MAKE_QA_PLOTS="${MAKE_QA_PLOTS:-1}"
QA_PLOT_SEED="${QA_PLOT_SEED:-0}"
QA_PLOT_BASE_DIR="${QA_PLOT_BASE_DIR:-$(dirname "${OUT}")/qc_plots}"

echo "MODE=${MODE}"
echo "ROOT=${ROOT}"
echo "OUT=${OUT}"
echo "COORD_DIR=${COORD_DIR}"
echo "OVERWRITE_OUT=${OVERWRITE_OUT}"
echo "REBUILD_INDEX=${REBUILD_INDEX}"
echo "MAX_MONTHS=${MAX_MONTHS}"
echo "MAKE_QA_PLOTS=${MAKE_QA_PLOTS}"
echo "QA_PLOT_SEED=${QA_PLOT_SEED}"
echo "QA_PLOT_BASE_DIR=${QA_PLOT_BASE_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"

cmd=(
    python3 -u daymet_to_zarr_worker.py
    --mode "${MODE}"
    --coord-dir "${COORD_DIR}"
    --root "${ROOT}"
    --out "${OUT}"
)

if [[ "${MODE}" == "init" && "${OVERWRITE_OUT}" == "1" ]]; then
    cmd+=(--overwrite-out)
fi

if [[ "${REBUILD_INDEX}" == "1" ]]; then
    cmd+=(--rebuild-index)
fi

if [[ -n "${MAX_MONTHS}" ]]; then
    cmd+=(--max-months "${MAX_MONTHS}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

if [[ "${MODE}" == "finalize" && "${MAKE_QA_PLOTS}" == "1" ]]; then
    qa_out_dir="${QA_PLOT_BASE_DIR}/daymet_finalize_qc_${SLURM_JOB_ID:-manual}"
    mkdir -p "${qa_out_dir}"
    qa_cmd=(
        python3 -u daymet_zarr_qc_plots.py
        --zarr "${OUT}"
        --out-dir "${qa_out_dir}"
        --seed "${QA_PLOT_SEED}"
    )
    echo "Running QA plots: ${qa_cmd[*]}"
    "${qa_cmd[@]}"
fi

echo "Processing Complete"
