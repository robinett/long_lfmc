#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"
cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

config_path="${script_dir}/source_coop_transfer_configs.yaml"
dataset_key="example_lfmc_maps"

python3 "${script_dir}/upload_source_coop.py" \
    --config_path "${config_path}" \
    --dataset_key "${dataset_key}"
