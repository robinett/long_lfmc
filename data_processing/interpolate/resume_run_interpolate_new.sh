#!/bin/bash
# Resume a partially completed MODIS interpolation workflow after worker failures.
# Default behavior:
#   1) remove leftover Zarr *.partial temp files
#   2) rerun only specified failed worker IDs
#   3) run finalize after successful rerun

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# User-configurable settings (edit these)
START_DATE="2000-01-01"
END_DATE="2024-12-31"
BASE_PATH="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_regrid"
OUTPUT_ZARR="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_regrid_interpolated/modis_interp_7d.zarr"
MAX_INTERPOLATION_DAYS="7"
BUFFER_DAYS=""
XY_CHUNK_SIZE="512"
TIME_CHUNK_SIZE="32"
NUM_WORKERS="32"

# Comma-separated worker IDs to rerun (Slurm array syntax also works, e.g. "25,29" or "25-29")
FAILED_WORKER_IDS="25,29"

# Resume options
CLEAN_PARTIAL_FILES="1"
SUBMIT_FINALIZE="1"

mkdir -p ./logs

if [[ ! -d "${OUTPUT_ZARR}" ]]; then
    echo "ERROR: OUTPUT_ZARR does not exist: ${OUTPUT_ZARR}"
    exit 1
fi

if [[ -z "${FAILED_WORKER_IDS}" ]]; then
    echo "ERROR: FAILED_WORKER_IDS is empty"
    exit 1
fi

if [[ "${CLEAN_PARTIAL_FILES}" == "1" ]]; then
    echo "Cleaning leftover Zarr partial files under:"
    echo "  ${OUTPUT_ZARR}"
    partial_count_before=$(find "${OUTPUT_ZARR}" -type f -name '*.partial' | wc -l)
    echo "Found ${partial_count_before} partial files before cleanup"
    find "${OUTPUT_ZARR}" -type f -name '*.partial' -delete
    partial_count_after=$(find "${OUTPUT_ZARR}" -type f -name '*.partial' | wc -l)
    echo "Remaining partial files after cleanup: ${partial_count_after}"
fi

common_export=(
    "START_DATE=${START_DATE}"
    "END_DATE=${END_DATE}"
    "BASE_PATH=${BASE_PATH}"
    "OUTPUT_ZARR=${OUTPUT_ZARR}"
    "MAX_INTERPOLATION_DAYS=${MAX_INTERPOLATION_DAYS}"
    "XY_CHUNK_SIZE=${XY_CHUNK_SIZE}"
    "TIME_CHUNK_SIZE=${TIME_CHUNK_SIZE}"
    "NUM_WORKERS=${NUM_WORKERS}"
    "OVERWRITE_ZARR=0"
    "DRY_RUN_CHUNK_PLAN=0"
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

echo "Submitting resume worker array for IDs: ${FAILED_WORKER_IDS}"
resume_worker_job_id=$(sbatch --parsable \
    --array="${FAILED_WORKER_IDS}" \
    --export="$(join_export worker)" \
    run_interpolate_new.sh)
echo "resume_worker_job_id=${resume_worker_job_id}"

if [[ "${SUBMIT_FINALIZE}" == "1" ]]; then
    echo "Submitting finalize job (afterok:${resume_worker_job_id})"
    finalize_job_id=$(sbatch --parsable \
        --dependency="afterok:${resume_worker_job_id}" \
        --export="$(join_export finalize)" \
        run_interpolate_new.sh)
    echo "finalize_job_id=${finalize_job_id}"
else
    echo "SUBMIT_FINALIZE=0, not submitting finalize job."
fi

echo "Resume workflow submitted."
echo "  workers:  ${resume_worker_job_id} (array ${FAILED_WORKER_IDS})"
if [[ "${SUBMIT_FINALIZE}" == "1" ]]; then
    echo "  finalize: ${finalize_job_id}"
fi
