#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"
cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh

python3 "${script_dir}/upload_source_coop.py" \
    --config_path "${script_dir}/source_coop_transfer_configs.yaml" \
    --dataset_key viewer_3857_assets
