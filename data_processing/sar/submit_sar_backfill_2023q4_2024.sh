#!/usr/bin/env bash
# Submit SAR backfill pipeline:
#   1) download/regrid missing VH (late-2023 gap + 2024)
#   2) download/regrid missing VV (2024)
#   3) append VH source zarr
#   4) append VV source zarr
#   5) rebuild combined sar_all_vars.zarr (parallel init/workers/finalize)

set -euo pipefail

mkdir -p ./logs

# -------- User settings --------
CHUNK_MONTHS="1"
NUM_MERGE_WORKERS="16"

VH_START_DATE="2023-10-04"
VH_END_DATE="2024-12-31"
VV_START_DATE="2024-01-01"
VV_END_DATE="2024-12-31"

VH_RAW_DIR="/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_raw_daily"
VV_RAW_DIR="/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_raw_daily_vv"

VH_SOURCE_ZARR="/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full.zarr"
VV_SOURCE_ZARR="/scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full_vv.zarr"

MERGE_OUTPUT_ZARR="/scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_all_vars.zarr"
MERGE_COORD_DIR="/scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_merge_queue_coord"
MERGE_TIME_BLOCK_DAYS="16"
MERGE_OVERWRITE_OUT="1"
MAKE_QA_PLOTS="1"
QA_PLOT_SEED="42"
QA_PLOT_BASE_DIR="/scratch/users/trobinet/long_lfmc/final_lfmc/sar/plots"
SKIP_EXISTING="1"
# ------------------------------

if [[ "${NUM_MERGE_WORKERS}" -lt 1 ]]; then
    echo "NUM_MERGE_WORKERS must be >= 1"
    exit 1
fi

months_between_inclusive() {
    local start_date="$1"
    local end_date="$2"
    python3 - "$start_date" "$end_date" <<'PY'
import sys
from datetime import date

s = date.fromisoformat(sys.argv[1])
e = date.fromisoformat(sys.argv[2])
if e < s:
    print(0)
else:
    print((e.year - s.year) * 12 + (e.month - s.month) + 1)
PY
}

chunk_count_for_range() {
    local start_date="$1"
    local end_date="$2"
    local chunk_months="$3"
    local n_months
    n_months="$(months_between_inclusive "$start_date" "$end_date")"
    python3 - "$n_months" "$chunk_months" <<'PY'
import math, sys
n_months = int(sys.argv[1])
chunk = int(sys.argv[2])
print(max(1, math.ceil(n_months / chunk)))
PY
}

join_export() {
    local mode="$1"
    shift
    local arr=("ALL" "MODE=${mode}" "$@")
    local out=""
    local first=1
    local item
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

vh_chunks="$(chunk_count_for_range "${VH_START_DATE}" "${VH_END_DATE}" "${CHUNK_MONTHS}")"
vv_chunks="$(chunk_count_for_range "${VV_START_DATE}" "${VV_END_DATE}" "${CHUNK_MONTHS}")"
vh_array_max=$((vh_chunks - 1))
vv_array_max=$((vv_chunks - 1))

echo "Submitting VH download/regrid array (${vh_chunks} chunks)"
vh_download_job_id=$(sbatch --parsable \
    --array="0-${vh_array_max}" \
    --export="$(join_export worker \
        START_DATE=${VH_START_DATE} \
        END_DATE=${VH_END_DATE} \
        CHUNK_MONTHS=${CHUNK_MONTHS} \
        POLARIZATION=VH \
        OUT_DIR=${VH_RAW_DIR} \
        OUT_VAR_NAME=vh_backscatter \
        SKIP_EXISTING=${SKIP_EXISTING})" \
    run_get_sar_raw_backfill.sh)
echo "vh_download_job_id=${vh_download_job_id}"

echo "Submitting VV download/regrid array (${vv_chunks} chunks)"
vv_download_job_id=$(sbatch --parsable \
    --array="0-${vv_array_max}" \
    --export="$(join_export worker \
        START_DATE=${VV_START_DATE} \
        END_DATE=${VV_END_DATE} \
        CHUNK_MONTHS=${CHUNK_MONTHS} \
        POLARIZATION=VV \
        OUT_DIR=${VV_RAW_DIR} \
        OUT_VAR_NAME=vv_backscatter \
        SKIP_EXISTING=${SKIP_EXISTING})" \
    run_get_sar_raw_backfill.sh)
echo "vv_download_job_id=${vv_download_job_id}"

echo "Submitting VH source-zarr append"
vh_append_job_id=$(sbatch --parsable \
    --dependency="afterok:${vh_download_job_id}" \
    --export="$(join_export append \
        RAW_DIR=${VH_RAW_DIR} \
        OUT_ZARR=${VH_SOURCE_ZARR} \
        VAR_NAME=vh_backscatter \
        START_DATE=${VH_START_DATE} \
        END_DATE=${VH_END_DATE} \
        APPEND=1)" \
    run_raw_sar_to_zarr_append.sh)
echo "vh_append_job_id=${vh_append_job_id}"

echo "Submitting VV source-zarr append"
vv_append_job_id=$(sbatch --parsable \
    --dependency="afterok:${vv_download_job_id}" \
    --export="$(join_export append \
        RAW_DIR=${VV_RAW_DIR} \
        OUT_ZARR=${VV_SOURCE_ZARR} \
        VAR_NAME=vv_backscatter \
        START_DATE=${VV_START_DATE} \
        END_DATE=${VV_END_DATE} \
        APPEND=1)" \
    run_raw_sar_to_zarr_append.sh)
echo "vv_append_job_id=${vv_append_job_id}"

merge_dep="afterok:${vh_append_job_id}:${vv_append_job_id}"

echo "Submitting SAR merge init"
merge_init_job_id=$(sbatch --parsable \
    --dependency="${merge_dep}" \
    --export="$(join_export init \
        VH_ZARR=${VH_SOURCE_ZARR} \
        VV_ZARR=${VV_SOURCE_ZARR} \
        OUTPUT_ZARR=${MERGE_OUTPUT_ZARR} \
        COORD_DIR=${MERGE_COORD_DIR} \
        TIME_BLOCK_DAYS=${MERGE_TIME_BLOCK_DAYS} \
        OVERWRITE_OUT=${MERGE_OVERWRITE_OUT} \
        MAKE_QA_PLOTS=${MAKE_QA_PLOTS} \
        QA_PLOT_SEED=${QA_PLOT_SEED} \
        QA_PLOT_BASE_DIR=${QA_PLOT_BASE_DIR})" \
    run_sar_merge_to_zarr_parallel.sh)
echo "merge_init_job_id=${merge_init_job_id}"

merge_array_max=$((NUM_MERGE_WORKERS - 1))
echo "Submitting SAR merge worker array: 0-${merge_array_max}"
merge_worker_job_id=$(sbatch --parsable \
    --dependency="afterok:${merge_init_job_id}" \
    --array="0-${merge_array_max}" \
    --export="$(join_export worker \
        VH_ZARR=${VH_SOURCE_ZARR} \
        VV_ZARR=${VV_SOURCE_ZARR} \
        OUTPUT_ZARR=${MERGE_OUTPUT_ZARR} \
        COORD_DIR=${MERGE_COORD_DIR} \
        TIME_BLOCK_DAYS=${MERGE_TIME_BLOCK_DAYS} \
        OVERWRITE_OUT=${MERGE_OVERWRITE_OUT} \
        MAKE_QA_PLOTS=${MAKE_QA_PLOTS} \
        QA_PLOT_SEED=${QA_PLOT_SEED} \
        QA_PLOT_BASE_DIR=${QA_PLOT_BASE_DIR})" \
    run_sar_merge_to_zarr_parallel.sh)
echo "merge_worker_job_id=${merge_worker_job_id}"

echo "Submitting SAR merge finalize"
merge_finalize_job_id=$(sbatch --parsable \
    --dependency="afterok:${merge_worker_job_id}" \
    --export="$(join_export finalize \
        VH_ZARR=${VH_SOURCE_ZARR} \
        VV_ZARR=${VV_SOURCE_ZARR} \
        OUTPUT_ZARR=${MERGE_OUTPUT_ZARR} \
        COORD_DIR=${MERGE_COORD_DIR} \
        TIME_BLOCK_DAYS=${MERGE_TIME_BLOCK_DAYS} \
        OVERWRITE_OUT=${MERGE_OVERWRITE_OUT} \
        MAKE_QA_PLOTS=${MAKE_QA_PLOTS} \
        QA_PLOT_SEED=${QA_PLOT_SEED} \
        QA_PLOT_BASE_DIR=${QA_PLOT_BASE_DIR})" \
    run_sar_merge_to_zarr_parallel.sh)
echo "merge_finalize_job_id=${merge_finalize_job_id}"

echo "Submitted SAR backfill pipeline:"
echo "  VH download:    ${vh_download_job_id} (array ${vh_chunks})"
echo "  VV download:    ${vv_download_job_id} (array ${vv_chunks})"
echo "  VH append:      ${vh_append_job_id}"
echo "  VV append:      ${vv_append_job_id}"
echo "  Merge init:     ${merge_init_job_id}"
echo "  Merge workers:  ${merge_worker_job_id} (array ${NUM_MERGE_WORKERS})"
echo "  Merge finalize: ${merge_finalize_job_id}"
