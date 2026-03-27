#!/usr/bin/env bash

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
config_path="${script_dir}/map_configs_shared_dm128_2023_07_08.yaml"
model_env="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"

source ~/.bashrc
source "${model_env}"

set -euo pipefail

max_tiles="${MAX_TILES:-}"
manifest_only="${MANIFEST_ONLY:-0}"

cmd=(
    python3 "${script_dir}/create_maps.py"
    --config_path "${config_path}"
)

if [[ -n "${max_tiles}" ]]; then
    cmd+=(--max_tiles "${max_tiles}")
fi

if [[ "${manifest_only}" == "1" ]]; then
    cmd+=(--manifest_only)
fi

echo "Running shared dm128 two-month map workflow"
echo "config_path=${config_path}"
if [[ -n "${max_tiles}" ]]; then
    echo "max_tiles=${max_tiles}"
fi
if [[ "${manifest_only}" == "1" ]]; then
    echo "manifest_only=1"
fi

"${cmd[@]}"
