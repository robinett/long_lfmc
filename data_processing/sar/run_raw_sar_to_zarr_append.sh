#!/usr/bin/env bash

#SBATCH --job-name=sar_zarr_append
#SBATCH --output=./logs/slurm-%x-%A_%a.out
#SBATCH --error=./logs/slurm-%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

mkdir -p ./logs

source ~/uv_activations/activate_lfmc_process_py312.sh

RAW_DIR="${RAW_DIR:?RAW_DIR is required}"
OUT_ZARR="${OUT_ZARR:?OUT_ZARR is required}"
VAR_NAME="${VAR_NAME:?VAR_NAME is required}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
APPEND="${APPEND:-1}"
OVERWRITE="${OVERWRITE:-0}"

echo "MODE=raw_to_zarr"
echo "RAW_DIR=${RAW_DIR}"
echo "OUT_ZARR=${OUT_ZARR}"
echo "VAR_NAME=${VAR_NAME}"
echo "START_DATE=${START_DATE}"
echo "END_DATE=${END_DATE}"
echo "APPEND=${APPEND}"
echo "OVERWRITE=${OVERWRITE}"

cmd=(
    python3 -u raw_sar_to_zarr.py
    --raw_dir "${RAW_DIR}"
    --out_zarr "${OUT_ZARR}"
    --var_name "${VAR_NAME}"
)

if [[ -n "${START_DATE}" ]]; then
    cmd+=(--start_date "${START_DATE}")
fi
if [[ -n "${END_DATE}" ]]; then
    cmd+=(--end_date "${END_DATE}")
fi
if [[ "${APPEND}" == "1" ]]; then
    cmd+=(--append)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Processing Complete"
