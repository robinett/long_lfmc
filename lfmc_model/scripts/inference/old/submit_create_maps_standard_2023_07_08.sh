#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
config_path="${script_dir}/map_configs_standard_2023_07_08.yaml"

cd "${script_dir}"

echo "Submitting standard inference test with config:"
echo "  ${config_path}"

CONFIG_PATH="${config_path}" bash "${script_dir}/submit_create_maps_ensemble.sh"
