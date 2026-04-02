#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
config_path="${script_dir}/map_configs_multisource_fusion_clim20_2023_07_08.yaml"

cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

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

echo "Running multisource-fusion clim20 July/August 2023 map workflow"
echo "config_path=${config_path}"
if [[ -n "${max_tiles}" ]]; then
    echo "max_tiles=${max_tiles}"
fi
if [[ "${manifest_only}" == "1" ]]; then
    echo "manifest_only=1"
fi

"${cmd[@]}"
