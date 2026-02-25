#!/bin/bash
set -euo pipefail

# Edit these defaults for interactive testing.
START_DATE="2023-01-01"
END_DATE="2023-01-31"
BASE_PATH="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_regrid"
OUTPUT_ZARR="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/testing/debug_dryrun_only.zarr"
MAX_INTERPOLATION_DAYS="15"
BUFFER_DAYS=""
XY_CHUNK_SIZE="128"
NUM_WORKERS="8"

#source ~/.bashrc
source ~/uv_activations/activate_lfmc_process_py312.sh

cmd=(
    python3 interpolate_new.py
    --mode worker
    --dry_run_chunk_plan
    --start_date "${START_DATE}"
    --end_date "${END_DATE}"
    --base_path "${BASE_PATH}"
    --output_zarr "${OUTPUT_ZARR}"
    --max_interpolation_days "${MAX_INTERPOLATION_DAYS}"
    --xy_chunk_size "${XY_CHUNK_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --worker_id 0
)

if [[ -n "${BUFFER_DAYS}" ]]; then
    cmd+=(--buffer_days "${BUFFER_DAYS}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
