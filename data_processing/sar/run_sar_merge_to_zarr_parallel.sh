#!/bin/bash

#SBATCH --job-name=sar_merge_parallel
#SBATCH --output=./logs/slurm-%x-%A_%a.out
#SBATCH --error=./logs/slurm-%x-%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=serc
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

mkdir -p ./logs

source ~/uv_activations/activate_lfmc_process_py312.sh

MODE="${MODE:-worker}"
VH_ZARR="${VH_ZARR:?VH_ZARR is required}"
VV_ZARR="${VV_ZARR:?VV_ZARR is required}"
OUTPUT_ZARR="${OUTPUT_ZARR:?OUTPUT_ZARR is required}"
COORD_DIR="${COORD_DIR:?COORD_DIR is required}"
TIME_BLOCK_DAYS="${TIME_BLOCK_DAYS:-16}"
OVERWRITE_OUT="${OVERWRITE_OUT:-0}"
MAX_CLAIMS="${MAX_CLAIMS:-}"
MAKE_QA_PLOTS="${MAKE_QA_PLOTS:-1}"
QA_PLOT_SEED="${QA_PLOT_SEED:-42}"
QA_PLOT_BASE_DIR="${QA_PLOT_BASE_DIR:-$(dirname "${OUTPUT_ZARR}")/qc_plots}"

echo "MODE=${MODE}"
echo "VH_ZARR=${VH_ZARR}"
echo "VV_ZARR=${VV_ZARR}"
echo "OUTPUT_ZARR=${OUTPUT_ZARR}"
echo "COORD_DIR=${COORD_DIR}"
echo "TIME_BLOCK_DAYS=${TIME_BLOCK_DAYS}"
echo "QA_PLOT_BASE_DIR=${QA_PLOT_BASE_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"

cmd=(
    python3 -u sar_merge_to_zarr_parallel.py
    --mode "${MODE}"
    --vh-path "${VH_ZARR}"
    --vv-path "${VV_ZARR}"
    --out "${OUTPUT_ZARR}"
    --coord-dir "${COORD_DIR}"
    --time-block-days "${TIME_BLOCK_DAYS}"
)

if [[ "${MODE}" == "init" && "${OVERWRITE_OUT}" == "1" ]]; then
    cmd+=(--overwrite-out)
fi

if [[ -n "${MAX_CLAIMS}" ]]; then
    cmd+=(--max-claims "${MAX_CLAIMS}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

if [[ "${MODE}" == "finalize" && "${MAKE_QA_PLOTS}" == "1" ]]; then
    qa_dir="${QA_PLOT_BASE_DIR}/sar_finalize_qc_${SLURM_JOB_ID:-local}"
    mkdir -p "${qa_dir}"
    qa_cmd=(
        python3 -u sar_zarr_qc_plots.py
        --zarr-path "${OUTPUT_ZARR}"
        --out-dir "${qa_dir}"
        --seed "${QA_PLOT_SEED}"
    )
    echo "Running QC plots: ${qa_cmd[*]}"
    "${qa_cmd[@]}"
fi

echo "Processing Complete"
