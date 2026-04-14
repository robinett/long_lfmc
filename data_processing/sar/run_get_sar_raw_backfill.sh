#!/usr/bin/env bash

#SBATCH --job-name=sar_raw_backfill
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

START_DATE="${START_DATE:?START_DATE is required}"
END_DATE="${END_DATE:?END_DATE is required}"
CHUNK_MONTHS="${CHUNK_MONTHS:-1}"
POLARIZATION="${POLARIZATION:?POLARIZATION is required}"
FLIGHT_DIRECTION="${FLIGHT_DIRECTION:-DESCENDING}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
OUT_VAR_NAME="${OUT_VAR_NAME:?OUT_VAR_NAME is required}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_NUM="${SLURM_JOB_ID:-0}${TASK_ID}"

chunk_start="$(date -d "${START_DATE} + $((CHUNK_MONTHS * TASK_ID)) months" +%F)"
chunk_end="$(date -d "${chunk_start} + ${CHUNK_MONTHS} months - 1 day" +%F)"

if [[ "$(date -d "$chunk_end" +%s)" -gt "$(date -d "$END_DATE" +%s)" ]]; then
    chunk_end="$END_DATE"
fi

if [[ "$(date -d "$chunk_start" +%s)" -gt "$(date -d "$END_DATE" +%s)" ]]; then
    echo "chunk_start $chunk_start is after END_DATE $END_DATE; nothing to do."
    exit 0
fi

echo "MODE=download"
echo "POLARIZATION=${POLARIZATION}"
echo "FLIGHT_DIRECTION=${FLIGHT_DIRECTION}"
echo "TASK_ID=${TASK_ID}"
echo "RANGE=${chunk_start} -> ${chunk_end}"
echo "OUT_DIR=${OUT_DIR}"
echo "OUT_VAR_NAME=${OUT_VAR_NAME}"
echo "JOB_NUM=${JOB_NUM}"

cmd=(
    python3 -u get_sar_raw.py
    --start_date "${chunk_start}"
    --end_date "${chunk_end}"
    --job_num "${JOB_NUM}"
    --polarization "${POLARIZATION}"
    --flight_direction "${FLIGHT_DIRECTION}"
    --out_dir "${OUT_DIR}"
    --out_var_name "${OUT_VAR_NAME}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip_existing)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Processing Complete"
