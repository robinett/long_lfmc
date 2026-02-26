#!/bin/bash
set -euo pipefail

mkdir -p ./logs

source ~/uv_activations/activate_lfmc_process_py312.sh

python3 -u sar_merge_to_zarr_parallel.py \
    --mode init \
    --dry-run \
    --vh-path /oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full.zarr \
    --vv-path /scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full_vv.zarr \
    --out /scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_all_vars_dryrun_placeholder.zarr \
    --coord-dir /scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_merge_queue_coord_dryrun \
    --time-block-days 16
