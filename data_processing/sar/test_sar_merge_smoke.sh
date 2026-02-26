#!/bin/bash
set -euo pipefail

mkdir -p ./logs

source ~/uv_activations/activate_lfmc_process_py312.sh

OUT=/scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_all_vars_smoketest.zarr
COORD=/scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_merge_queue_coord_smoketest

python3 -u sar_merge_to_zarr_parallel.py \
    --mode init \
    --overwrite-out \
    --vh-path /oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full.zarr \
    --vv-path /scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full_vv.zarr \
    --out "${OUT}" \
    --coord-dir "${COORD}" \
    --time-block-days 16

python3 -u sar_merge_to_zarr_parallel.py \
    --mode worker \
    --max-claims 1 \
    --vh-path /oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full.zarr \
    --vv-path /scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full_vv.zarr \
    --out "${OUT}" \
    --coord-dir "${COORD}" \
    --time-block-days 16

python3 -u sar_merge_to_zarr_parallel.py \
    --mode finalize \
    --vh-path /oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full.zarr \
    --vv-path /scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full_vv.zarr \
    --out "${OUT}" \
    --coord-dir "${COORD}" \
    --time-block-days 16

python3 -u sar_zarr_qc_plots.py \
    --zarr-path "${OUT}" \
    --out-dir /scratch/users/trobinet/long_lfmc/final_lfmc/sar/qc_plots/sar_smoketest_qc \
    --seed 42
