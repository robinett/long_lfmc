#!/bin/bash
# Submit a parallel-safe Daymet zarr conversion workflow:
#   1) init zarr store (writes first month + preallocates full time axis)
#   2) worker array writes disjoint time regions in parallel
#   3) finalize metadata

set -euo pipefail

ROOT="/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_regrid"
OUT="/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_all_vars.zarr"
COORD_DIR="/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_queue_coord"
NUM_WORKERS="16"
OVERWRITE_OUT="1"
REBUILD_INDEX="1"
MAX_MONTHS=""
MAKE_QA_PLOTS="1"
QA_PLOT_SEED="42"
QA_PLOT_BASE_DIR="/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/qc_plots"

if [[ "${NUM_WORKERS}" -lt 1 ]]; then
    echo "NUM_WORKERS must be >= 1"
    exit 1
fi

mkdir -p ./logs

common_export=(
    "ROOT=${ROOT}"
    "OUT=${OUT}"
    "COORD_DIR=${COORD_DIR}"
    "OVERWRITE_OUT=${OVERWRITE_OUT}"
    "REBUILD_INDEX=${REBUILD_INDEX}"
    "MAKE_QA_PLOTS=${MAKE_QA_PLOTS}"
    "QA_PLOT_SEED=${QA_PLOT_SEED}"
    "QA_PLOT_BASE_DIR=${QA_PLOT_BASE_DIR}"
)

if [[ -n "${MAX_MONTHS}" ]]; then
    common_export+=("MAX_MONTHS=${MAX_MONTHS}")
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
    run_daymet_to_zarr_parallel.sh)
echo "init_job_id=${init_job_id}"

array_max=$((NUM_WORKERS - 1))
echo "Submitting worker array: 0-${array_max}"
worker_job_id=$(sbatch --parsable \
    --dependency="afterok:${init_job_id}" \
    --array="0-${array_max}" \
    --export="$(join_export worker)" \
    run_daymet_to_zarr_parallel.sh)
echo "worker_job_id=${worker_job_id}"

echo "Submitting finalize job"
finalize_job_id=$(sbatch --parsable \
    --dependency="afterok:${worker_job_id}" \
    --export="$(join_export finalize)" \
    run_daymet_to_zarr_parallel.sh)
echo "finalize_job_id=${finalize_job_id}"

echo "Submitted workflow:"
echo "  init:     ${init_job_id}"
echo "  workers:  ${worker_job_id} (array size ${NUM_WORKERS})"
echo "  finalize: ${finalize_job_id}"
