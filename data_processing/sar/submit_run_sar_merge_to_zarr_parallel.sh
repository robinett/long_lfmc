#!/bin/bash
# Submit SAR merge workflow:
#   1) init output zarr
#   2) worker array writes disjoint time chunks in parallel
#   3) finalize metadata + QA plots

set -euo pipefail

VH_ZARR="/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full.zarr"
VV_ZARR="/scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full_vv.zarr"
OUTPUT_ZARR="/scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_all_vars.zarr"
COORD_DIR="/scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_merge_queue_coord"
TIME_BLOCK_DAYS="16"
NUM_WORKERS="16"
OVERWRITE_OUT="1"
MAKE_QA_PLOTS="1"
QA_PLOT_SEED="42"
QA_PLOT_BASE_DIR="/scratch/users/trobinet/long_lfmc/final_lfmc/sar/qc_plots"
MAX_CLAIMS=""

if [[ "${NUM_WORKERS}" -lt 1 ]]; then
    echo "NUM_WORKERS must be >= 1"
    exit 1
fi

mkdir -p ./logs

common_export=(
    "VH_ZARR=${VH_ZARR}"
    "VV_ZARR=${VV_ZARR}"
    "OUTPUT_ZARR=${OUTPUT_ZARR}"
    "COORD_DIR=${COORD_DIR}"
    "TIME_BLOCK_DAYS=${TIME_BLOCK_DAYS}"
    "OVERWRITE_OUT=${OVERWRITE_OUT}"
    "MAKE_QA_PLOTS=${MAKE_QA_PLOTS}"
    "QA_PLOT_SEED=${QA_PLOT_SEED}"
    "QA_PLOT_BASE_DIR=${QA_PLOT_BASE_DIR}"
)

if [[ -n "${MAX_CLAIMS}" ]]; then
    common_export+=("MAX_CLAIMS=${MAX_CLAIMS}")
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
    run_sar_merge_to_zarr_parallel.sh)
echo "init_job_id=${init_job_id}"

array_max=$((NUM_WORKERS - 1))
echo "Submitting worker array: 0-${array_max}"
worker_job_id=$(sbatch --parsable \
    --dependency="afterok:${init_job_id}" \
    --array="0-${array_max}" \
    --export="$(join_export worker)" \
    run_sar_merge_to_zarr_parallel.sh)
echo "worker_job_id=${worker_job_id}"

echo "Submitting finalize job"
finalize_job_id=$(sbatch --parsable \
    --dependency="afterok:${worker_job_id}" \
    --export="$(join_export finalize)" \
    run_sar_merge_to_zarr_parallel.sh)
echo "finalize_job_id=${finalize_job_id}"

echo "Submitted workflow:"
echo "  init:     ${init_job_id}"
echo "  workers:  ${worker_job_id} (array size ${NUM_WORKERS})"
echo "  finalize: ${finalize_job_id}"
