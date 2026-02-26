#!/bin/bash

#SBATCH --job-name=interp_new
#SBATCH --output=./logs/slurm-%x-%A_%a.out
#SBATCH --error=./logs/slurm-%x-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=serc
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

mkdir -p ./logs

#source ~/.bashrc
source ~/uv_activations/activate_lfmc_process_py312.sh

MODE="${MODE:-worker}"
START_DATE="${START_DATE:?START_DATE is required}"
END_DATE="${END_DATE:?END_DATE is required}"
BASE_PATH="${BASE_PATH:?BASE_PATH is required}"
OUTPUT_ZARR="${OUTPUT_ZARR:?OUTPUT_ZARR is required}"
MAX_INTERPOLATION_DAYS="${MAX_INTERPOLATION_DAYS:?MAX_INTERPOLATION_DAYS is required}"
XY_CHUNK_SIZE="${XY_CHUNK_SIZE:-128}"
TIME_CHUNK_SIZE="${TIME_CHUNK_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
BUFFER_DAYS="${BUFFER_DAYS:-}"
OVERWRITE_ZARR="${OVERWRITE_ZARR:-0}"
DRY_RUN_CHUNK_PLAN="${DRY_RUN_CHUNK_PLAN:-0}"

WORKER_ID="${WORKER_ID:-0}"
if [[ "${MODE}" == "worker" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    WORKER_ID="${SLURM_ARRAY_TASK_ID}"
fi

echo "MODE=${MODE}"
echo "START_DATE=${START_DATE}"
echo "END_DATE=${END_DATE}"
echo "BASE_PATH=${BASE_PATH}"
echo "OUTPUT_ZARR=${OUTPUT_ZARR}"
echo "MAX_INTERPOLATION_DAYS=${MAX_INTERPOLATION_DAYS}"
echo "XY_CHUNK_SIZE=${XY_CHUNK_SIZE}"
echo "TIME_CHUNK_SIZE=${TIME_CHUNK_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "WORKER_ID=${WORKER_ID}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"

cmd=(
    python3 -u interpolate_new.py
    --mode "${MODE}"
    --start_date "${START_DATE}"
    --end_date "${END_DATE}"
    --base_path "${BASE_PATH}"
    --output_zarr "${OUTPUT_ZARR}"
    --max_interpolation_days "${MAX_INTERPOLATION_DAYS}"
    --xy_chunk_size "${XY_CHUNK_SIZE}"
    --time_chunk_size "${TIME_CHUNK_SIZE}"
    --num_workers "${NUM_WORKERS}"
)

if [[ -n "${BUFFER_DAYS}" ]]; then
    cmd+=(--buffer_days "${BUFFER_DAYS}")
fi

if [[ "${MODE}" == "worker" ]]; then
    cmd+=(--worker_id "${WORKER_ID}")
fi

if [[ "${MODE}" == "init" && "${OVERWRITE_ZARR}" == "1" ]]; then
    cmd+=(--overwrite_zarr)
fi

if [[ "${DRY_RUN_CHUNK_PLAN}" == "1" ]]; then
    cmd+=(--dry_run_chunk_plan)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Processing Complete"
