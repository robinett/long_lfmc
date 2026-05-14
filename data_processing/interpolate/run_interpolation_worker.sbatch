#!/bin/bash
#SBATCH --job-name=modis_interp
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o /home/users/trobinet/long_lfmc/logs/low_latency_parallel/modis_interp_%A_%a.out
#SBATCH -e /home/users/trobinet/long_lfmc/logs/low_latency_parallel/modis_interp_%A_%a.err

set -euo pipefail

repo_root="/home/users/trobinet/long_lfmc"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
script_path="${repo_root}/data_processing/interpolate/interpolate_new.py"

mkdir -p "${repo_root}/logs/low_latency_parallel"
source "${process_env}"

worker_id="${SLURM_ARRAY_TASK_ID:-0}"
cmd=(
    python3 "${script_path}"
    --base_path "${INTERP_BASE_PATH}"
    --start_date "${INTERP_START_DATE}"
    --end_date "${INTERP_END_DATE}"
    --output_zarr "${INTERP_OUTPUT_ZARR}"
    --max_interpolation_days "${INTERP_MAX_DAYS}"
    --buffer_days "${INTERP_BUFFER_DAYS}"
    --xy_chunk_size "${INTERP_XY_CHUNK_SIZE}"
    --time_chunk_size "${INTERP_TIME_CHUNK_SIZE}"
    --mode worker
    --worker_id "${worker_id}"
    --num_workers "${INTERP_NUM_WORKERS}"
)

echo "Running MODIS interpolation worker"
echo "  worker_id=${worker_id}"
echo "  num_workers=${INTERP_NUM_WORKERS}"
echo "  output_zarr=${INTERP_OUTPUT_ZARR}"
"${cmd[@]}"
