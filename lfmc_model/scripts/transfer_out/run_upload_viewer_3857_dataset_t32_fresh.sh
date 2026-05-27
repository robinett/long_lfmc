#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"
config_path="${script_dir}/source_coop_transfer_configs.yaml"

cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh

python3 "${script_dir}/upload_source_coop.py" \
    --config_path "${config_path}" \
    --dataset_key viewer_3857_lfmc_maps_t32 \
    --transfer_mode fresh-prefix \
    --upload_backend boto3 \
    --verify_mode sample \
    --no-delete_extra_remote_files
