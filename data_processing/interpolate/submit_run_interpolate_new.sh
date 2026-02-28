#!/bin/bash
# Submit a parallel-safe MODIS interpolation workflow:
#   1) init zarr store
#   2) worker array writes disjoint spatial chunks
#   3) finalize metadata

set -euo pipefail

# User-configurable settings
START_DATE="2000-01-01"
END_DATE="2024-12-31"
BASE_PATH="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_regrid"
OUTPUT_ZARR="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_regrid_interpolated/modis_interp_5d.zarr"
MAX_INTERPOLATION_DAYS="5"
BUFFER_DAYS=""
XY_CHUNK_SIZE="512"
TIME_CHUNK_SIZE="16"
NUM_WORKERS="32"
OVERWRITE_ZARR="1"
DRY_RUN_CHUNK_PLAN="0"
PLOT_ONLY="0"

if [[ "${NUM_WORKERS}" -lt 1 ]]; then
    echo "NUM_WORKERS must be >= 1"
    exit 1
fi

mkdir -p ./logs

common_export=(
    "START_DATE=${START_DATE}"
    "END_DATE=${END_DATE}"
    "BASE_PATH=${BASE_PATH}"
    "OUTPUT_ZARR=${OUTPUT_ZARR}"
    "MAX_INTERPOLATION_DAYS=${MAX_INTERPOLATION_DAYS}"
    "XY_CHUNK_SIZE=${XY_CHUNK_SIZE}"
    "TIME_CHUNK_SIZE=${TIME_CHUNK_SIZE}"
    "NUM_WORKERS=${NUM_WORKERS}"
    "OVERWRITE_ZARR=${OVERWRITE_ZARR}"
    "DRY_RUN_CHUNK_PLAN=${DRY_RUN_CHUNK_PLAN}"
    "PLOT_ONLY=${PLOT_ONLY}"
)

if [[ -n "${BUFFER_DAYS}" ]]; then
    common_export+=("BUFFER_DAYS=${BUFFER_DAYS}")
fi

join_export() {
    local mode="$1"
    local arr=("ALL" "MODE=${mode}")
    local item
    for item in "${common_export[@]}"; do
        arr+=("${item}")
    done
    local out=""
    local first=1
    for item in "${arr[@]}"; do
        if [[ ${first} -eq 1 ]]; then
            out="${item}"
            first=0
        else
            out="${out},${item}"
        fi
    done
    echo "${out}"
}

echo "Submitting init job"
init_job_id=$(sbatch --parsable \
    --export="$(join_export init)" \
    run_interpolate_new.sh)
echo "init_job_id=${init_job_id}"

if [[ "${DRY_RUN_CHUNK_PLAN}" == "1" ]]; then
    echo "DRY_RUN_CHUNK_PLAN=1, not submitting worker/finalize jobs."
    exit 0
fi

array_max=$((NUM_WORKERS - 1))
echo "Submitting worker array: 0-${array_max}"
worker_job_id=$(sbatch --parsable \
    --dependency="afterok:${init_job_id}" \
    --array="0-${array_max}" \
    --export="$(join_export worker)" \
    run_interpolate_new.sh)
echo "worker_job_id=${worker_job_id}"

echo "Submitting finalize job"
finalize_job_id=$(sbatch --parsable \
    --dependency="afterok:${worker_job_id}" \
    --export="$(join_export finalize)" \
    run_interpolate_new.sh)
echo "finalize_job_id=${finalize_job_id}"

echo "Submitted workflow:"
echo "  init:     ${init_job_id}"
echo "  workers:  ${worker_job_id} (array size ${NUM_WORKERS})"
echo "  finalize: ${finalize_job_id}"
